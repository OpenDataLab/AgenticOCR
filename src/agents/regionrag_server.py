"""FastAPI service wrapping RegionRAG (ColQwen2.5 + neighbor-based region grouping).

Run on a GPU host. Two endpoints are exposed:

  POST /extract_regions          — RegionRAG-native schema; takes
                                    {query, image_path, ...hyperparams}.
  POST /v1/chat/completions      — OpenAI-compatible Chat Completions; the
                                    AgenticOCR client drives this with
                                    {messages: [...image+query...]} and gets
                                    back ```json [bbox+evidence]``` content.

Image paths must be reachable from the server (typically shared filesystem).

Launch via scripts/start_regionrag.sh.
"""

import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="RegionRAG Region Extraction Service")

# Populated on startup.
_model = None
_processor = None
_device = None
_model_lock = threading.Lock()

# Configured via env vars; see start_regionrag.sh.
MODEL_PATH = os.environ.get("REGIONRAG_MODEL_PATH", "RegionRAG/models/RegionRet")
# RegionRet is published as a LoRA adapter. If the path looks like an adapter
# (contains adapter_config.json), we load the base model first and then apply
# the adapter on top. Base path is read from adapter_config.json by default,
# or overridden via REGIONRAG_BASE_PATH.
BASE_PATH = os.environ.get("REGIONRAG_BASE_PATH", "")


class ExtractRequest(BaseModel):
    query: str
    image_path: str
    neighbor_range: int = 2
    bbox_threshold: float = 0.25
    score_method: str = "max"
    max_regions: int = 20


class RegionOut(BaseModel):
    bbox: List[int]  # [x1,y1,x2,y2] in 0-1000 normalized coords
    bbox_pixel: List[float]  # raw pixel coords from model
    score: float


class ExtractResponse(BaseModel):
    regions: List[RegionOut]
    image_size: Tuple[int, int]  # (width, height)


def _patch_processor_for_transformers_4_57():
    """colpali_engine pins transformers<4.52 because ColQwen2_5_Processor
    declares `image_token_id` as a read-only @property. Newer transformers
    (>=4.52) write to that attribute during Qwen2VLProcessor.__init__, raising
    AttributeError. Replace the property with a writable shim that still
    falls back to tokenizer lookup when no value has been set explicitly.
    """
    from colpali_engine.models import ColQwen2_5_Processor

    # Already patched? Avoid re-applying on hot reload.
    if not isinstance(getattr(ColQwen2_5_Processor, "image_token_id", None), property):
        return

    def _get(self):
        v = self.__dict__.get("_image_token_id_override")
        if v is not None:
            return v
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def _set(self, value):
        self.__dict__["_image_token_id_override"] = value

    ColQwen2_5_Processor.image_token_id = property(_get, _set)


def _patch_model_for_transformers_4_57():
    """Bridge several breaking changes in transformers >= 4.55 that
    colpali_engine 0.1.dev7 (pinned to <4.52) wasn't written against:

    1. Qwen2_5_VLForConditionalGeneration.get_rope_index moved to its inner
       Qwen2_5_VLModel. ColQwen2_5.forward calls self.get_rope_index, so we
       add a delegating shim.
    2. Qwen2_5_VLModel.embed_tokens moved to model.language_model.embed_tokens.
       ColQwen2_5.inner_forward calls self.model.embed_tokens, so we add a
       property alias.
    """
    from colpali_engine.models import ColQwen2_5
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLModel,
    )

    if hasattr(ColQwen2_5, "_compat_4_57_patched"):
        return

    # (1) get_rope_index shim on ColQwen2_5
    if not hasattr(ColQwen2_5, "get_rope_index"):
        def _get_rope_index_shim(self, *args, **kwargs):
            inner = getattr(self, "model", None)
            if inner is not None and hasattr(inner, "get_rope_index"):
                return inner.get_rope_index(*args, **kwargs)
            raise AttributeError(
                "get_rope_index not found on ColQwen2_5 or self.model"
            )
        ColQwen2_5.get_rope_index = _get_rope_index_shim

    # (2) embed_tokens alias on Qwen2_5_VLModel
    if not hasattr(Qwen2_5_VLModel, "embed_tokens") or isinstance(
        getattr(Qwen2_5_VLModel, "embed_tokens", None), property
    ):
        # Check the existing attribute is not a real nn.Module attribute
        # (we want to alias only when 4.57+ moved it).
        def _embed_tokens_get(self):
            lm = getattr(self, "language_model", None)
            if lm is not None and hasattr(lm, "embed_tokens"):
                return lm.embed_tokens
            raise AttributeError("embed_tokens not found")

        Qwen2_5_VLModel.embed_tokens = property(_embed_tokens_get)

    ColQwen2_5._compat_4_57_patched = True


def _load_model():
    """Load RegionRet (ColQwen2_5) onto GPU once. Call from startup.

    RegionRet is a LoRA adapter over a ColQwen2.5 base. We detect that by
    looking for adapter_config.json in MODEL_PATH; if present, we load the
    base model first and then stack the adapter via PEFT. Otherwise we fall
    back to loading MODEL_PATH as a full ColQwen2_5 checkpoint.
    """
    global _model, _processor, _device

    _patch_processor_for_transformers_4_57()
    _patch_model_for_transformers_4_57()

    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from colpali_engine.utils.torch_utils import get_torch_device
    from transformers.utils.import_utils import is_flash_attn_2_available

    _device = get_torch_device("auto")

    adapter_cfg_path = os.path.join(MODEL_PATH, "adapter_config.json")
    is_adapter = os.path.isfile(adapter_cfg_path)

    attn_impl = "flash_attention_2" if is_flash_attn_2_available() else None

    if is_adapter:
        import json as _json
        from peft import PeftModel

        with open(adapter_cfg_path, "r") as f:
            adapter_cfg = _json.load(f)
        base = BASE_PATH or adapter_cfg.get("base_model_name_or_path", "")
        # Resolve relative base paths against both cwd and MODEL_PATH's parent.
        if base and not os.path.isabs(base):
            candidates = [
                os.path.abspath(base),
                os.path.abspath(os.path.join(os.path.dirname(MODEL_PATH), os.path.basename(base))),
                os.path.abspath(os.path.join(os.path.dirname(MODEL_PATH), "..", os.path.basename(base))),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    base = c
                    break

        logger.info("Loading ColQwen2.5 base from %s", base)
        base_model = ColQwen2_5.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
            device_map=_device,
            attn_implementation=attn_impl,
            mask_non_image_embeddings=True,
        ).eval()

        logger.info("Applying RegionRet LoRA adapter from %s", MODEL_PATH)
        peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        # Merge LoRA weights back into the base ColQwen2_5 so the returned
        # object keeps the original custom forward/score_multi_vector_per_patch
        # plumbing. merge_and_unload returns the base model with merged deltas.
        _model = peft_model.merge_and_unload().eval()

        # Prefer processor files from the adapter dir (tokenizer/added_tokens may
        # differ from base); fall back to base if missing.
        proc_src = MODEL_PATH if os.path.isfile(os.path.join(MODEL_PATH, "preprocessor_config.json")) else base
        _processor = ColQwen2_5_Processor.from_pretrained(proc_src)
        logger.info("RegionRet (LoRA over %s) loaded. attn=%s", base, attn_impl)
    else:
        logger.info("Loading full ColQwen2.5 checkpoint from %s", MODEL_PATH)
        _model = ColQwen2_5.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=_device,
            attn_implementation=attn_impl,
            mask_non_image_embeddings=True,
        ).eval()
        _processor = ColQwen2_5_Processor.from_pretrained(MODEL_PATH)
        logger.info("Full ColQwen2.5 checkpoint loaded. attn=%s", attn_impl)


def _extract_regions(
    query: str,
    image: Image.Image,
    neighbor_range: int,
    bbox_threshold: float,
    score_method: str,
    max_regions: int,
) -> Tuple[List[dict], Tuple[int, int]]:
    """Run RegionRet forward + neighbor-based grouping on a single page image."""
    W, H = image.size

    # Mirror RetrievalEvaluator.inference_image / inference_text logic, but
    # for a single (query, page) pair.
    batch_image = _processor.process_images(images=[image])
    batch_query = _processor.process_queries(queries=[query])

    # image_grid_thw: (1, 3) → (t, h_patch, w_patch) before merge
    if "image_grid_thw" in batch_image:
        grid_thw = batch_image["image_grid_thw"][0]
    else:
        _, gH, gW = batch_image["pixel_values"][0].shape
        grid_thw = torch.tensor([1, gH // 14, gW // 14])

    # After spatial merge (merge_size typically 2).
    merge_size = _processor.image_processor.merge_size
    h_grid = max(1, grid_thw[1].item() // merge_size)
    w_grid = max(1, grid_thw[2].item() // merge_size)
    image_grid = [1, h_grid, w_grid]
    grid_size = (H / h_grid, W / w_grid)  # (patch_pixel_h, patch_pixel_w)

    # Figure out the image token mask so we can slice real patch embeddings.
    if hasattr(_model.config, "image_token_id"):
        img_token_id = _model.config.image_token_id
    else:
        img_token_id = _model.config.image_token_index
    image_mask = batch_image["input_ids"] == img_token_id

    batch_image_dev = {k: v.to(_device) for k, v in batch_image.items()}
    batch_query_dev = {k: v.to(_device) for k, v in batch_query.items()}

    with _model_lock, torch.inference_mode():
        p_emb = _model(**batch_image_dev)
        q_emb = _model(**batch_query_dev)

    # Same split pattern as RetrievalEvaluator, for batch of size 1.
    p_emb_list = list(
        torch.split(p_emb[image_mask.bool()], image_mask.sum(-1).tolist())
    )
    p_emb_list = [e.to("cpu") for e in p_emb_list]

    q_attn = batch_query["attention_mask"]
    q_emb_list = list(torch.split(q_emb[q_attn.bool()], q_attn.sum(-1).tolist()))
    q_emb_list = [e.to("cpu") for e in q_emb_list]

    # Per-patch late interaction scores.
    scores, p_mask = _processor.score_multi_vector_per_patch(
        qs=q_emb_list, ps=p_emb_list
    )
    # scores: (1, 1, max_patches); pick the single (q, p) pair.
    score_per_patch = scores[0, 0][p_mask[0]]

    # single_get_box expects (query_id, image_id, score, method, threshold,
    # neighbor_range, image_size, image_grid, grid_size). image_size is (H, W).
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

    neighbor_range_list = list(range(-neighbor_range, neighbor_range + 1))
    raw_regions = BaseVisualRetrieverProcessor.single_get_box(
        (
            "_",
            "_",
            score_per_patch,
            score_method,
            bbox_threshold,
            neighbor_range_list,
            (H, W),
            image_grid,
            grid_size,
        )
    )

    # Sort by score descending, then cap.
    raw_regions.sort(key=lambda r: r["score"], reverse=True)
    raw_regions = raw_regions[:max_regions]

    # Convert pixel bbox → 0-1000 normalized coords (to align with PageElement).
    regions: List[dict] = []
    for r in raw_regions:
        x1, y1, x2, y2 = r["bounding_box"]
        bbox_1000 = [
            max(0, min(1000, int(round(x1 / W * 1000)))),
            max(0, min(1000, int(round(y1 / H * 1000)))),
            max(0, min(1000, int(round(x2 / W * 1000)))),
            max(0, min(1000, int(round(y2 / H * 1000)))),
        ]
        regions.append(
            {
                "bbox": bbox_1000,
                "bbox_pixel": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(r["score"]),
            }
        )

    return regions, (W, H)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model is not None else "loading",
        "model_path": MODEL_PATH,
    }


@app.post("/extract_regions", response_model=ExtractResponse)
async def extract_regions(req: ExtractRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet")

    img_path = req.image_path
    if not os.path.isabs(img_path):
        img_path = os.path.abspath(img_path)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {img_path}")

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {e}")

    try:
        regions, (W, H) = _extract_regions(
            query=req.query,
            image=image,
            neighbor_range=req.neighbor_range,
            bbox_threshold=req.bbox_threshold,
            score_method=req.score_method,
            max_regions=req.max_regions,
        )
    except Exception as e:
        logger.exception("extract_regions failed for %s", img_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        image.close()

    return ExtractResponse(regions=regions, image_size=(W, H))


# ----------------------------------------------------------------------------
# OpenAI-compat Chat Completions wrapper
# ----------------------------------------------------------------------------
# Lets the existing AgenticOCR Python client drive RegionRAG by URL swap.
# The client posts {messages: [system?, user(image+query)]}; we extract the
# image (base64 data URL or local path) and the query text, run RegionRAG,
# and return the bbox list as a ```json ... ``` block in assistant content.
# Tunable params (neighbor_range / bbox_threshold / score_method / max_regions)
# are read from env vars at startup.

CHAT_NEIGHBOR_RANGE = int(os.environ.get("REGIONRAG_NEIGHBOR_RANGE", "2"))
CHAT_BBOX_THRESHOLD = float(os.environ.get("REGIONRAG_BBOX_THRESHOLD", "0.25"))
CHAT_SCORE_METHOD = os.environ.get("REGIONRAG_SCORE_METHOD", "max")
CHAT_MAX_REGIONS = int(os.environ.get("REGIONRAG_MAX_REGIONS", "20"))
WRAPPER_MODEL_NAME = os.environ.get("REGIONRAG_WRAPPER_MODEL_NAME", "regionrag")


def _load_image_from_url(url: str) -> Image.Image:
    """Load PIL image from data URL (base64), file:// URL, or filesystem path."""
    if url.startswith("data:"):
        comma = url.find(",")
        if comma < 0:
            raise ValueError("Malformed data URL")
        b64 = url[comma + 1:]
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if url.startswith("file://"):
        return Image.open(url[len("file://"):]).convert("RGB")
    if url.startswith(("http://", "https://")):
        raise ValueError("Remote image URLs are not supported")
    return Image.open(url).convert("RGB")


def _parse_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[Image.Image], str]:
    """Walk messages from the latest user turn backwards to find image + query."""
    image: Optional[Image.Image] = None
    query: str = ""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            query = content
            return image, query
        if not isinstance(content, list):
            continue
        text_parts: List[str] = []
        for part in content:
            t = part.get("type")
            if t == "image_url" and image is None:
                url = part.get("image_url", {}).get("url", "")
                try:
                    image = _load_image_from_url(url)
                except Exception as e:
                    logger.warning("Failed to load image from URL: %s", e)
            elif t == "text":
                text_parts.append(part.get("text", ""))
        query = "\n".join(p for p in text_parts if p)
        return image, query
    return image, query


@app.post("/v1/chat/completions")
async def chat_completions(req: Dict[str, Any]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not initialized yet")

    messages = req.get("messages") or []
    image, query = _parse_messages(messages)
    if image is None:
        raise HTTPException(status_code=400, detail="No image found in messages")
    if not query:
        # Allow empty query — RegionRAG will still produce regions, but warn.
        logger.warning("Empty query in /v1/chat/completions request")

    try:
        regions, (W, H) = _extract_regions(
            query=query,
            image=image,
            neighbor_range=CHAT_NEIGHBOR_RANGE,
            bbox_threshold=CHAT_BBOX_THRESHOLD,
            score_method=CHAT_SCORE_METHOD,
            max_regions=CHAT_MAX_REGIONS,
        )
    except Exception as e:
        logger.exception("RegionRAG forward failed in /v1/chat/completions")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        image.close()

    items = [{"bbox": r["bbox"], "evidence": ""} for r in regions]
    json_str = json.dumps(items, ensure_ascii=False)
    content = f"```json\n{json_str}\n```"

    now = int(time.time())
    cmpl_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    return {
        "id": cmpl_id,
        "object": "chat.completion",
        "created": now,
        "model": req.get("model") or WRAPPER_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(content) // 4,
            "total_tokens": len(content) // 4,
        },
    }


@app.on_event("startup")
async def _startup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _load_model()


if __name__ == "__main__":
    port = int(os.environ.get("REGIONRAG_PORT", 8005))
    uvicorn.run(app, host="0.0.0.0", port=port)
