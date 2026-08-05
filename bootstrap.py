# bootstrap.py

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import yaml

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - exercised in minimal test envs
    OmegaConf = None

logger = logging.getLogger(__name__)

Qwen3VLEmbedder = None
Qwen3VLReranker = None
_QWEN_MODULES_LOADED = False

# SUPPORTED_BENCHMARKS = ("mmlong", "finrag", "docvqa", "vidoseek")
SUPPORTED_BENCHMARKS = ("mmlong", "finrag")


SENSITIVE_KEYS = {
    "api_key",
    "extractor_api_key",
    "judger_api_key",
    "reranker_api_key",
    "evaluator_api_key",
}


def _redact_sensitive_config(config_dict):
    redacted = dict(config_dict)
    for key in SENSITIVE_KEYS:
        if key in redacted and redacted[key]:
            redacted[key] = "***REDACTED***"
    return redacted


def _ensure_qwen_modules_loaded():
    global Qwen3VLEmbedder, Qwen3VLReranker, _QWEN_MODULES_LOADED
    if _QWEN_MODULES_LOADED:
        return

    try:
        from src.recalls.qwen3_vl_embedding import Qwen3VLEmbedder as embedder_cls
        from src.recalls.qwen3_vl_reranker_client import Qwen3VLReranker as reranker_cls

        Qwen3VLEmbedder = embedder_cls
        Qwen3VLReranker = reranker_cls
    except ImportError:
        logger.warning("Qwen3 VL scripts not found.")
    finally:
        _QWEN_MODULES_LOADED = True


def _get_loader_registry():
    # from src.loaders.DocVQALoader import DocVQALoader
    # from src.loaders.ViDoSeekPoolLoader import ViDoSeekLoader
    from src.loaders.FinRAGLoader import FinRAGLoader
    from src.loaders.MMLongLoader import MMLongLoader

    return {
        "mmlong": MMLongLoader,
        "finrag": FinRAGLoader,
        # "docvqa": DocVQALoader,
        # "vidoseek": ViDoSeekLoader,
    }

def setup_logging(level) -> None:
    """Initialize process-wide logging once."""
    if logging.getLogger().handlers:
        return

    resolved_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.ERROR)


def get_common_parser():
    parser = argparse.ArgumentParser(description="RAG Pipeline Component")

    # Common
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--benchmark", type=str, default="mmlong",
                        choices=list(SUPPORTED_BENCHMARKS),
                        help="Target benchmark.")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results.")
    parser.add_argument("--filter", type=str, default=None,
                        help="Path to a bad_case file (JSON) or list of QIDs to filter the run.")
    parser.add_argument("--category_filter", type=str, default=None,
                        help="Comma-separated category names to keep (e.g. 'Table,Chart'). "
                             "Matches against extra_info category/evidence_sources.")
    parser.add_argument("--rag_system_prompt", type=str, default=None,
                        help="Path to the system prompt file or the prompt string itself.")

    # Logging
    parser.add_argument("--logging_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level.")

    # Model & API
    parser.add_argument("--model_name", type=str, default="qwen2.5-72b-instruct")
    parser.add_argument("--base_url", type=str, default="http://localhost:3888/v1")
    parser.add_argument("--api_key", type=str, default=os.environ.get("RAG_API_KEY", "sk-placeholder"))

    # Retrieval
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--embedding_model", type=str, default="/path/to/embedding")
    parser.add_argument("--reranker_model", type=str, default="/path/to/reranker")
    parser.add_argument("--reranker_api_base", type=str, default=None)
    parser.add_argument("--finrag_lang", type=str, default="both")
    parser.add_argument("--trunc_thres", type=float, default=None)
    parser.add_argument("--trunc_bbox", action="store_true")
    parser.add_argument("--return_all_pages", action="store_true",
                        help="Skip reranking/extraction and return all document pages as retrieval results.")

    # Extractor
    parser.add_argument("--mineru_server_url", type=str, default="http://localhost:8000/")
    parser.add_argument("--mineru_model_path", type=str, default="/root/checkpoints/MinerU2.5-2509-1.2B/")
    parser.add_argument("--judger_model_name", type=str, default="")
    parser.add_argument("--judger_base_url", type=str, default="")
    parser.add_argument("--judger_api_key", type=str, default="")
    parser.add_argument("--extractor_model_name", type=str, default="MinerU-Agent-CK300")
    parser.add_argument("--extractor_base_url", type=str, default="http://localhost:8001/v1")
    parser.add_argument("--extractor_api_key", type=str, default=os.environ.get("EXTRACTOR_API_KEY", "sk-placeholder"))

    # Generation context
    parser.add_argument("--use_page", action="store_true")
    parser.add_argument("--use_page_ocr", action="store_true", help="Include full page OCR text in context.")
    parser.add_argument("--use_crop", action="store_true")
    parser.add_argument("--use_ocr", action="store_true")
    parser.add_argument("--use_ocr_raw", action="store_true")
    parser.add_argument("--use_ocr_both", action="store_true")
    parser.add_argument("--max_page_pixels", type=int, default=1024 * 1024,
                        help="Max pixel count for full page images passed to the LLM.")
    parser.add_argument("--max_crop_pixels", type=int, default=512 * 512,
                        help="Max pixel count for cropped region images passed to the LLM.")

    # Staged evaluation
    parser.add_argument("--evaluation_task", type=str, default="retrieval",
                        choices=["retrieval", "generation", "all"],
                        help="Specify which metrics to evaluate.")
    parser.add_argument("--generation_input", type=str, default=None,
                        help="Path to the results JSON file for generation stage.")
    parser.add_argument("--evaluation_input", type=str, default=None,
                        help="Path to the results JSON file for evaluation stage.")
    parser.add_argument("--evaluator_api_base", type=str, default="http://35.220.164.252:3888/v1",
                        help="Base URL for the evaluator API.")
    parser.add_argument("--evaluator_api_key", type=str,
                        default=os.environ.get("EVALUATOR_API_KEY", "sk-placeholder"),
                        help="API key for the evaluator API.")
    parser.add_argument("--evaluator_model_name", type=str, default="qwen3-max",
                        help="Model name for the evaluator API.")

    # Limit & Threading
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads for parallel processing.")

    
    return parser


def parse_args():
    parser = get_common_parser()
    temp_args, _ = parser.parse_known_args()

    if temp_args.config and Path(temp_args.config).exists():
        if OmegaConf is None:
            parser.error("omegaconf is required when using --config. Install dependencies first.")
        conf = OmegaConf.load(temp_args.config)
        config_data = OmegaConf.to_container(conf, resolve=True)
        if config_data:
            parser.set_defaults(**config_data)

    args = parser.parse_args()

    if not args.output_dir:
        parser.error("--output_dir is required (or set output_dir in --config).")
    if args.num_threads < 1:
        parser.error("--num_threads must be >= 1.")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


def save_run_config(args, stage_name="run"):
    """ output_dir"""
    if not args.output_dir:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = Path(args.output_dir) / f"config_{stage_name}_{timestamp}.yaml"

    try:
        config_dict = _redact_sensitive_config(vars(args))
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=True)
        logger.info("Configuration saved to %s", config_path)
    except Exception:
        logger.exception("Failed to save configuration.")


def apply_qid_filter(items, filter_path, key="qid"):
    """ bad_case """
    if not filter_path or not Path(filter_path).exists():
        return items

    logger.info("Applying filter from: %s", filter_path)
    try:
        with open(filter_path, "r", encoding="utf-8") as f:
            filter_data = json.load(f)

        if isinstance(filter_data, dict):
            if isinstance(filter_data.get("qids"), list):
                filter_data = filter_data["qids"]
            elif isinstance(filter_data.get("items"), list):
                filter_data = filter_data["items"]
            else:
                logger.warning("Unsupported filter file format: %s", filter_path)
                return items

        target_qids = set()
        for item in filter_data:
            if isinstance(item, dict) and "qid" in item:
                target_qids.add(str(item["qid"]))
            elif isinstance(item, (str, int)):
                target_qids.add(str(item))

        filtered_items = []
        for item in items:
            if isinstance(item, dict):
                item_qid = item.get(key)
            elif hasattr(item, key):
                item_qid = getattr(item, key)
            else:
                continue

            if str(item_qid) in target_qids:
                filtered_items.append(item)

        logger.info("Reduced items from %d to %d via qid filter.", len(items), len(filtered_items))
        return filtered_items
    except Exception:
        logger.exception("Failed to apply filter from %s", filter_path)
        return items


def filter_items_by_samples(items, samples, key="qid"):
    """ loader.samples  dict 

    initialize_components()  loader.samples 
    (qid filter / category filter / limit)
     dict 
    """
    valid_qids = {str(s.qid) for s in samples}
    filtered = [item for item in items if str(item.get(key, "")) in valid_qids]
    if len(filtered) < len(items):
        logger.info(
            "Filtered items by loader samples: %d -> %d.", len(items), len(filtered),
        )
    return filtered


def apply_category_filter(samples, category_filter_str):
    """ category 

     FinRAG  extra_info['category']  MMLong  extra_info['evidence_sources']
    category_filter_str 
    ( '-' )
    """
    if not category_filter_str:
        return samples

    import ast as _ast

    target_cats = {c.strip().lower() for c in category_filter_str.split(",") if c.strip()}
    if not target_cats:
        return samples

    def _normalize(name):
        return str(name).split("-")[0].split(" ")[0].lower()

    def _get_sample_categories(sample):
        """ sample.extra_info """
        if not sample.extra_info:
            return set()

        cats = set()

        # FinRAG style: extra_info['category']
        raw_cat = sample.extra_info.get("category")
        if raw_cat is not None:
            if isinstance(raw_cat, list):
                for c in raw_cat:
                    cats.add(_normalize(c))
            else:
                cats.add(_normalize(raw_cat))

        # MMLong style: extra_info['evidence_sources']
        raw_ev = sample.extra_info.get("evidence_sources")
        if raw_ev is not None:
            if isinstance(raw_ev, list):
                ev_list = raw_ev
            elif isinstance(raw_ev, str):
                try:
                    ev_list = _ast.literal_eval(raw_ev)
                    if not isinstance(ev_list, list):
                        ev_list = [str(ev_list)]
                except Exception:
                    ev_list = [raw_ev]
            else:
                ev_list = [str(raw_ev)]
            for c in ev_list:
                cats.add(_normalize(c))

        return cats

    filtered = [s for s in samples if _get_sample_categories(s) & target_cats]
    logger.info(
        "Category filter %s: reduced samples from %d to %d.",
        target_cats, len(samples), len(filtered),
    )
    return filtered


def _build_tool(args):
    """ OCR """
    from src.agents.MinerUTool import ImageZoomOCRTool

    return ImageZoomOCRTool(
        work_dir=str(Path(args.output_dir) / "workspace" / "crops"),
        mineru_server_url=args.mineru_server_url,
        mineru_model_path=args.mineru_model_path,
    )


def _build_extractor(base_url, api_key, model_name, tool):
    """ AgenticOCR None"""
    if base_url and api_key and model_name:
        from src.agents.AgenticOCR import AgenticOCR

        return AgenticOCR(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            tool=tool,
        )
    return None


def _build_reranker(args, init_retriever):
    """ Reranker """
    if not init_retriever:
        return None
    _ensure_qwen_modules_loaded()
    if Qwen3VLReranker is None:
        return None
    if args.reranker_api_base:
        return Qwen3VLReranker(model_name_or_path=args.reranker_api_base)
    if args.reranker_model:
        import torch
        return Qwen3VLReranker(model_name_or_path=args.reranker_model, torch_dtype=torch.float16)
    return None


def _build_loader(args, reranker, extractor, judger, init_retriever):
    """ benchmark  Loader"""
    benchmark = args.benchmark
    loader_cls = _get_loader_registry().get(benchmark)

    if loader_cls is None:
        raise NotImplementedError(f"Unsupported benchmark: {benchmark}")

    common_kwargs = dict(
        data_root=args.data_root,
        output_dir=args.output_dir,
        extractor=extractor,
        judger=judger,
    )

    if benchmark == "mmlong":
        loader = loader_cls(reranker=reranker, **common_kwargs)
    elif benchmark == "finrag":
        loader = loader_cls(
            lang=args.finrag_lang,
            embedding_model=None,
            rerank_model=reranker,
            **common_kwargs,
        )
    elif benchmark == "vidoseek":
        embedder = None
        if init_retriever:
            _ensure_qwen_modules_loaded()
        if init_retriever and Qwen3VLEmbedder is not None:
            import torch
            embedder = Qwen3VLEmbedder(
                model_name_or_path=args.embedding_model, torch_dtype=torch.float16
            )
        loader = loader_cls(
            embedding_model=embedder,
            rerank_model=reranker,
            **common_kwargs,
        )
    else:
        loader = loader_cls(reranker=reranker, **common_kwargs)

    loader.load_data()
    return loader


def _load_system_prompt(args):
    """ RAG """
    if not args.rag_system_prompt:
        logger.exception("No RAG system prompt provided.")
        return ""

    prompt_path = Path(args.rag_system_prompt)
    if prompt_path.exists():
        logger.info("Loading RAG system prompt from file: %s", args.rag_system_prompt)
        try:
            with prompt_path.open("r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            logger.exception("Error reading system prompt file: %s", args.rag_system_prompt)
            return ""
    else:
        logger.info("Using inline RAG system prompt string.")
        return args.rag_system_prompt


def initialize_components(args, init_retriever=True, init_generator=True):
    """ Loader, Reranker, Agent """
    from src.agents.RAGAgent import RAGAgent

    setup_logging(args.logging_level)

    tool = _build_tool(args)

    judger = _build_extractor(
        args.judger_base_url, args.judger_api_key, args.judger_model_name, tool
    )
    extractor = _build_extractor(
        args.extractor_base_url, args.extractor_api_key, args.extractor_model_name, tool
    )

    reranker = _build_reranker(args, init_retriever)
    loader = _build_loader(args, reranker, extractor, judger, init_retriever)

    if args.filter:
        loader.samples = apply_qid_filter(loader.samples, args.filter)

    if args.category_filter:
        loader.samples = apply_category_filter(loader.samples, args.category_filter)

    if args.limit and args.limit > 0:
        loader.samples = loader.samples[: args.limit]

    system_prompt_content = _load_system_prompt(args)

    agent = RAGAgent(
        loader=loader,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        top_k=args.top_k,
        cache_dir=str(Path(args.output_dir) / "cache_agent"),
        use_page=args.use_page,
        use_page_ocr=args.use_page_ocr,
        use_crop=args.use_crop,
        use_ocr=args.use_ocr,
        use_ocr_raw=args.use_ocr_raw,
        use_ocr_both=args.use_ocr_both,
        max_page_pixels=args.max_page_pixels,
        max_crop_pixels=args.max_crop_pixels,
        trunc_thres=args.trunc_thres,
        trunc_bbox=args.trunc_bbox,
        system_prompt=system_prompt_content,
    )

    return agent, loader
