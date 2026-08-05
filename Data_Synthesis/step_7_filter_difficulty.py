import os
import json
import copy
import argparse
import logging
import time
import base64
import io
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import re
from utils.prompt_overrides import apply_prompt_overrides
from utils.prompt_files import load_step_prompts
from utils.runtime_config import (
    create_genai_client_with,
    get_step7_check_api_key,
    get_step7_check_base_url,
    get_step7_check_model,
    get_step7_judge_api_key,
    get_step7_judge_base_url,
    get_step7_judge_model,
)

try:
    from PyPDF2 import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    HAS_PDF_TOOLS = True
except ImportError:
    HAS_PDF_TOOLS = False

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

def images_to_pdf_part(images):
    """Function description."""
    if not images:
        return None

    pdf_bytes_io = BytesIO()

    imgs_rgb = [img.convert("RGB") for img in images]

    imgs_rgb[0].save(
        pdf_bytes_io,
        format="PDF",
        save_all=True,
        append_images=imgs_rgb[1:]
    )

    pdf_bytes_io.seek(0)
    pdf_bytes = pdf_bytes_io.getvalue()

    pdf_part = types.Part.from_bytes(
        data=pdf_bytes,
        mime_type='application/pdf',
    )

    return pdf_part

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


_PROMPTS = load_step_prompts(
    "step_7_filter_difficulty",
    ["PROMPT_CHECK", "PROMPT_ANSWER_CHECK"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_7_filter_difficulty",
    _PROMPTS,
    logger=logger,
)
PROMPT_CHECK = _PROMPTS["PROMPT_CHECK"]
PROMPT_ANSWER_CHECK = _PROMPTS["PROMPT_ANSWER_CHECK"]


def _normalize_selected_names(names):
    normalized = set()
    for name in names or []:
        base = os.path.basename(str(name).strip())
        if not base:
            continue
        if not base.lower().endswith(".pdf"):
            base = f"{base}.pdf"
        normalized.add(base)
    return normalized


def _is_valid_existing_step7_output(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return isinstance(data, list)
    except Exception:
        return False

def clean_model_response(text: str) -> str:
    """Function description."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text

def calculate_cost(usage_metadata):
    """Function description."""
    if not usage_metadata:
        return 0.0, 0, 0
    input_tokens = getattr(usage_metadata, "prompt_token_count", 0)
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0)
    try:
        input_tokens = int(input_tokens or 0)
    except Exception:
        input_tokens = 0
    try:
        output_tokens = int(output_tokens or 0)
    except Exception:
        output_tokens = 0
    return 0.0, input_tokens, output_tokens


def calculate_cost_openai(usage):
    if not usage:
        return 0.0, 0, 0
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return 0.0, input_tokens, output_tokens

def extract_answer_from_text(text):
    """Function description."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_model_answer(raw_text):
    """Function description."""
    if not raw_text:
        return None
    cleaned = clean_model_response(raw_text)
    try:
        parsed = json.loads(cleaned)
        answer = parsed.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    except Exception:
        pass

    tag_answer = extract_answer_from_text(cleaned)
    if tag_answer:
        return tag_answer

    cleaned = cleaned.strip()
    return cleaned if cleaned else None


def _parse_page_id_from_evidence_id(evidence_id):
    try:
        s = str(evidence_id)
        if "::" in s:
            s = s.split("::", 1)[1]
        return int(str(s).split("-", 1)[0])
    except Exception:
        return None

def normalize_judge_label(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z ]", "", s)
    if "partially correct" in s:
        return "Partially Correct"
    if "incorrect" in s:
        return "Incorrect"
    if "correct" in s:
        return "Correct"
    return "Unknown"

def _run_answer_once_genai(client, model_name, contents):
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=32768)
        )
    )
    cost, in_t, out_t = calculate_cost(response.usage_metadata)
    return parse_model_answer(response.text), cost, in_t, out_t


def images_to_pdf_bytes(images):
    if not images:
        return b""
    pdf_bytes_io = BytesIO()
    imgs_rgb = [img.convert("RGB") for img in images]
    imgs_rgb[0].save(
        pdf_bytes_io,
        format="PDF",
        save_all=True,
        append_images=imgs_rgb[1:],
    )
    return pdf_bytes_io.getvalue()


def compress_pdf(pdf_path, max_size_mb):
    size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if size_mb <= max_size_mb:
        return pdf_path

    for quality in ["printer", "ebook", "screen"]:
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            subprocess.run(
                [
                    "gs",
                    "-sDEVICE=pdfwrite",
                    "-dCompatibilityLevel=1.4",
                    f"-dPDFSETTINGS=/{quality}",
                    "-dNOPAUSE",
                    "-dQUIET",
                    "-dBATCH",
                    f"-sOutputFile={tmp_path}",
                    pdf_path,
                ],
                check=True,
                timeout=10,
            )
            compressed_mb = os.path.getsize(tmp_path) / (1024 * 1024)
            if compressed_mb <= max_size_mb:
                return tmp_path
            os.unlink(tmp_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return pdf_path


def prepare_pdf_from_path(pdf_path, max_size_mb=5):
    if not HAS_PDF_TOOLS:
        with open(pdf_path, "rb") as f:
            return f.read()

    work_path = compress_pdf(pdf_path, max_size_mb)
    reader = PdfReader(work_path)
    writer = PdfWriter()
    for i, page in enumerate(reader.pages):
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFillColorRGB(1, 1, 1)
        c.rect(0, h - 20, 150, 20, fill=1, stroke=0)
        c.setFillColorRGB(1, 0, 0)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(4, h - 16, f'page_index="{i + 1}"')
        c.save()
        buf.seek(0)
        page.merge_page(PdfReader(buf).pages[0])
        writer.add_page(page)

    out_buf = io.BytesIO()
    writer.write(out_buf)
    pdf_bytes = out_buf.getvalue()

    if work_path != pdf_path and os.path.exists(work_path):
        os.unlink(work_path)
    return pdf_bytes


def pdf_to_page_png_bytes(pdf_bytes, dpi=144):
    if not HAS_FITZ:
        raise ImportError("PyMuPDF (fitz) is required for OpenAI image mode")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc]
    finally:
        doc.close()


def _bytes_to_data_url(raw: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _run_answer_once_openai(client, model_name, question, source_desc, page_png_bytes):
    prompt_text = PROMPT_CHECK.format(question=question)

    content = [{"type": "text", "text": prompt_text}]
    for png_bytes in page_png_bytes or []:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _bytes_to_data_url(png_bytes, "image/png")},
            }
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=1024,
    )
    txt = ""
    try:
        txt = (response.choices[0].message.content or "").strip()
    except Exception:
        txt = ""
    cost, in_t, out_t = calculate_cost_openai(getattr(response, "usage", None))
    return parse_model_answer(txt), cost, in_t, out_t

def _judge_answer_once(client, model_name, question, standard_answer, model_answer):
    prompt = PROMPT_ANSWER_CHECK.format(
        question=question,
        standard_answer=standard_answer,
        model_answer=model_answer
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=32768)
        )
    )
    cost, in_t, out_t = calculate_cost(response.usage_metadata)
    judge = normalize_judge_label(response.text)
    if judge == "Correct":
        result = "true"
    elif judge in ["Incorrect", "Partially Correct"]:
        result = "false"
    else:
        result = "unknown"
    return result, judge, cost, in_t, out_t

def check_single_query(
    answer_client_mode,
    answer_client,
    answer_model_name,
    judge_client,
    judge_model_name,
    question,
    answer,
    gt_page_pngs,
    full_pdf_pngs,
    full_pdf_part,
    test_modes,
    filter_method,
):
    knowledge_only = os.getenv("OMNIDOC_STEP7_KNOWLEDGE_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
    total_cost = 0.0
    total_in = 0
    total_out = 0
    mode_results = {}

    if knowledge_only:
        try:
            if answer_client_mode == "openai":
                model_answer, cost1, in1, out1 = _run_answer_once_openai(
                    answer_client,
                    answer_model_name,
                    question,
                    "model own knowledge only",
                    [],
                )
            else:
                prompt_text = PROMPT_CHECK.format(question=question)
                model_answer, cost1, in1, out1 = _run_answer_once_genai(answer_client, answer_model_name, [prompt_text])

            total_cost += cost1
            total_in += in1
            total_out += out1

            if not model_answer:
                final_status = "unknown"
                judge = "Unknown"
            else:
                final_status, judge, cost2, in2, out2 = _judge_answer_once(
                    judge_client,
                    judge_model_name,
                    question,
                    answer,
                    model_answer,
                )
                total_cost += cost2
                total_in += in2
                total_out += out2

            for mode in test_modes:
                mode_results[mode] = {
                    "filter_status": final_status,
                    "model_answer": model_answer if model_answer else None,
                    "judge": judge,
                }

            return final_status, total_cost, total_in, total_out, mode_results
        except Exception as e:
            logger.error(f"Step7 own-knowledge mode failed: {e}")
            for mode in test_modes:
                mode_results[mode] = {
                    "filter_status": "unknown",
                    "model_answer": None,
                    "judge": "Unknown",
                }
            return "unknown", total_cost, total_in, total_out, mode_results

    for mode in test_modes:
        try:
            if mode == "gt_pages":
                mode_pngs = gt_page_pngs
                part = None
                if answer_client_mode != "openai" and gt_page_pngs:
                    gt_page_images = []
                    for png_bytes in gt_page_pngs:
                        try:
                            gt_page_images.append(Image.open(BytesIO(png_bytes)).convert("RGB"))
                        except Exception:
                            continue
                    part = images_to_pdf_part(gt_page_images) if gt_page_images else None
                source_desc = "GT evidence pages only"
            elif mode == "full_pdf":
                part = full_pdf_part
                mode_pngs = full_pdf_pngs
                source_desc = "the full PDF document"
            else:
                continue

            if answer_client_mode == "openai":
                available = bool(mode_pngs)
            else:
                available = part is not None
            if not available:
                mode_results[mode] = {
                    "filter_status": "unknown",
                    "model_answer": None,
                    "judge": "Unknown",
                }
                continue

            if answer_client_mode == "openai":
                model_answer, cost1, in1, out1 = _run_answer_once_openai(
                    answer_client,
                    answer_model_name,
                    question,
                    source_desc,
                    mode_pngs,
                )
            else:
                prompt_text = PROMPT_CHECK.format(question=question)
                model_answer, cost1, in1, out1 = _run_answer_once_genai(answer_client, answer_model_name, [part, prompt_text])
            total_cost += cost1
            total_in += in1
            total_out += out1

            if not model_answer:
                mode_results[mode] = {
                    "filter_status": "unknown",
                    "model_answer": None,
                    "judge": "Unknown",
                }
                continue

            status, judge, cost2, in2, out2 = _judge_answer_once(
                judge_client,
                judge_model_name,
                question,
                answer,
                model_answer,
            )
            total_cost += cost2
            total_in += in2
            total_out += out2
            mode_results[mode] = {
                "filter_status": status,
                "model_answer": model_answer,
                "judge": judge,
            }
        except Exception as e:
            logger.error(f"Step7 mode '{mode}' failed: {e}")
            mode_results[mode] = {
                "filter_status": "unknown",
                "model_answer": None,
                "judge": "Unknown",
            }

    mode_statuses = {k: v.get("filter_status") for k, v in mode_results.items()}
    statuses = list(mode_statuses.values())
    if filter_method == "gt_pages_only":
        final_status = mode_statuses.get("gt_pages", "unknown")
    elif filter_method == "full_pdf_only":
        final_status = mode_statuses.get("full_pdf", "unknown")
    elif filter_method == "all_correct":
        if statuses and all(s == "true" for s in statuses):
            final_status = "true"
        elif statuses and any(s == "false" for s in statuses):
            final_status = "false"
        else:
            final_status = "unknown"
    else:
        if any(s == "true" for s in statuses):
            final_status = "true"
        elif statuses and all(s == "false" for s in statuses):
            final_status = "false"
        else:
            final_status = "unknown"

    return final_status, total_cost, total_in, total_out, mode_results
def draw_normalized_bbox(img, bbox):
    """
    img: PIL.Image
    bbox: [x1, y1, x2, y2] in 0-1000 scale
    return: masked PIL.Image
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size
    y1, x1, y2, x2, = bbox

    x1 = x1 / 1000 * img_w
    x2 = x2 / 1000 * img_w
    y1 = y1 / 1000 * img_h
    y2 = y2 / 1000 * img_h

    draw.rectangle([x1, y1, x2, y2], fill="white")

    return img

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_file(
    answer_client_mode,
    answer_client,
    answer_model_name,
    judge_client,
    judge_model_name,
    file_path,
    pdf_name,
    png_root,
    output_dir,
    inner_batch_size,
    language,
    full_pdf_path,
    test_modes,
    filter_method,
):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return {"status": "error", "cost": 0, "input": 0, "output": 0}

    final_packages = []
    file_cost = 0.0
    file_in = 0
    file_out = 0

    full_pdf_part = None
    full_pdf_pngs = []
    knowledge_only = os.getenv("OMNIDOC_STEP7_KNOWLEDGE_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
    need_openai_pdf_pages = (
        (not knowledge_only)
        and answer_client_mode == "openai"
        and full_pdf_path
        and os.path.exists(full_pdf_path)
    )
    need_genai_full_pdf = (
        (not knowledge_only)
        and ("full_pdf" in test_modes)
        and full_pdf_path
        and os.path.exists(full_pdf_path)
    )
    if need_openai_pdf_pages or need_genai_full_pdf:
        if need_openai_pdf_pages:
            try:
                max_pdf_size = float(os.getenv("OMNIDOC_STEP7_CHECK_MAX_PDF_SIZE_MB", "5") or 5)
                dpi = int(os.getenv("OMNIDOC_STEP7_CHECK_DPI", "144") or 144)
                prepared_pdf_bytes = prepare_pdf_from_path(full_pdf_path, max_size_mb=max_pdf_size)
                full_pdf_pngs = pdf_to_page_png_bytes(prepared_pdf_bytes, dpi=dpi)
            except Exception as e:
                logger.warning(
                    f"Step7 full_pdf OpenAI-preprocess failed ({pdf_name}), fallback to cached PNGs: {e}"
                )
                page_img_dir = os.path.join(png_root, pdf_name)
                if os.path.isdir(page_img_dir):
                    for p in sorted(os.listdir(page_img_dir)):
                        if not (p.startswith("page_") and p.endswith(".png")):
                            continue
                        img_path = os.path.join(page_img_dir, p)
                        try:
                            with open(img_path, "rb") as rf:
                                full_pdf_pngs.append(rf.read())
                        except Exception:
                            continue
            if not full_pdf_pngs:
                logger.warning(f"Step7 full_pdf mode has no page images for OpenAI check model: {pdf_name}")
        if need_genai_full_pdf:
            try:
                with open(full_pdf_path, "rb") as rf:
                    full_pdf_part = types.Part.from_bytes(
                        data=rf.read(),
                        mime_type="application/pdf",
                    )
            except Exception as e:
                logger.error(f"Failed to load full PDF for Step7 ({full_pdf_path}): {e}")

    for p_idx, package_results in enumerate(data):
        package = copy.deepcopy(package_results)
        package["qa_after_difficulty_gate"] = []
        qa_verification = package.get("qa_verification", package.get("qa_after_evidence_gate", []))
        evidence_list = package.get("Evidence_list", [])
        evidence_map = {item['element_idx']: item for item in evidence_list}

        if not qa_verification or not evidence_list:
            continue

        qa_futures = []
        qa_results = [None] * len(qa_verification)

        effective_inner_workers = min(
            inner_batch_size,
            len(qa_verification) if len(qa_verification) > 0 else 1,
        )
        with ThreadPoolExecutor(max_workers=effective_inner_workers) as qa_executor:
            for i, qa in enumerate(qa_verification):
                qa_futures.append(
                    qa_executor.submit(
                        process_single_qa,
                        answer_client_mode,
                        answer_client,
                        answer_model_name,
                        judge_client,
                        judge_model_name,
                        qa,
                        evidence_map,
                        png_root,
                        pdf_name,
                        language,
                        full_pdf_pngs,
                        full_pdf_part,
                        test_modes,
                        filter_method,
                    )
                )

            for i, future in enumerate(as_completed(qa_futures)):
                try:
                    result_dict, cost, in_t, out_t = future.result()
                    file_cost += cost
                    file_in += in_t
                    file_out += out_t

                    if result_dict.get("status") == "false":
                        package.setdefault("qa_after_difficulty_gate", []).append(result_dict["qa"])
                except Exception as e:
                    logger.error(f"QA processing error: {e}")

        final_packages.append(package)

    save_dir = os.path.join(output_dir, "results", "step_7")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pdf_name}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_packages, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved verification results to {save_path}")

    return {
        "status": "success",
        "file": os.path.basename(file_path),
        "cost": file_cost,
        "input_tokens": file_in,
        "output_tokens": file_out
    }
    
def process_single_qa(
    answer_client_mode,
    answer_client,
    answer_model_name,
    judge_client,
    judge_model_name,
    qa,
    evidence_map,
    png_root,
    pdf_name,
    language,
    full_pdf_pngs,
    full_pdf_part,
    test_modes,
    filter_method,
):
    """Function description."""
    question = qa["Refined_Question"]
    answer = qa["Model_Generated_Answer"]
    used_evidence = qa.get("Evidence_Used_Input", [])

    pages_to_load_ids = set()
    ev_page_ids = set()
    for ev in used_evidence:
        ev_page_id = _parse_page_id_from_evidence_id(ev)
        if ev_page_id is None:
            ev_item = evidence_map.get(ev, {})
            if isinstance(ev_item, dict):
                try:
                    ev_page_id = int(ev_item.get("page_id", -1))
                except Exception:
                    ev_page_id = None
        if ev_page_id is None or ev_page_id <= 0:
            continue
        ev_page_ids.add(ev_page_id)
        pages_to_load_ids.add(ev_page_id)

    gt_page_pngs = []
    if answer_client_mode == "openai" and full_pdf_pngs:
        for pid in sorted(pages_to_load_ids):
            idx = pid - 1
            if 0 <= idx < len(full_pdf_pngs):
                gt_page_pngs.append(full_pdf_pngs[idx])
    else:
        for pid in sorted(pages_to_load_ids):
            img_name = f"page_{pid:04d}.png"
            img_path = os.path.join(png_root, pdf_name, img_name)
            if not os.path.exists(img_path):
                continue
            try:
                with open(img_path, "rb") as rf:
                    gt_page_pngs.append(rf.read())
            except Exception:
                continue

    result, cost, in_t, out_t, mode_results = check_single_query(
        answer_client_mode,
        answer_client,
        answer_model_name,
        judge_client,
        judge_model_name,
        question,
        answer,
        gt_page_pngs,
        full_pdf_pngs,
        full_pdf_part,
        test_modes,
        filter_method,
    )
    qa_out = copy.deepcopy(qa)
    qa_out["step7_eval"] = {
        "test_modes": list(test_modes),
        "filter_method": filter_method,
        "mode_results": mode_results,
        "final_status": result,
        "difficulty_label": "easy" if result == "true" else ("hard" if result == "false" else "unknown"),
    }
    return {"status": result, "qa": qa_out}, cost, in_t, out_t
def main(args):
    answer_base_url = get_step7_check_base_url()
    answer_api_key = get_step7_check_api_key()
    answer_model_name = get_step7_check_model()

    judge_base_url = get_step7_judge_base_url()
    judge_api_key = get_step7_judge_api_key()
    judge_model_name = get_step7_judge_model()

    answer_client = create_genai_client_with(genai, types, answer_base_url, answer_api_key)
    answer_client_mode = os.getenv("OMNIDOC_STEP7_CHECK_CLIENT", "").strip().lower()
    if not answer_client_mode:
        raw_step7_check_base = (
            os.getenv("OMNIDOC_STEP7_CHECK_BASE_URL")
            or os.getenv("OMNIDOC_QWEN_BASE_URL")
            or ""
        )
        model_hint = (answer_model_name or "").strip().lower()
        if "/v1" in raw_step7_check_base or os.getenv("OMNIDOC_QWEN_MODEL") or ("qwen" in model_hint):
            answer_client_mode = "openai"
        else:
            answer_client_mode = "genai"
    if answer_client_mode == "openai":
        openai_base = answer_base_url
        if openai_base and "/v1" not in openai_base:
            openai_base = openai_base.rstrip("/") + "/v1"
        answer_client = OpenAI(base_url=openai_base, api_key=answer_api_key)
    judge_client = create_genai_client_with(genai, types, judge_base_url, judge_api_key)
    logger.info(f"Step7 PROMPT_CHECK model: {answer_model_name} @ {answer_base_url} (client={answer_client_mode})")
    logger.info(f"Step7 PROMPT_ANSWER_CHECK model: {judge_model_name} @ {judge_base_url}")
    test_modes = [m.strip() for m in args.difficulty_test_modes.split(",") if m.strip()]
    valid_modes = {"gt_pages", "full_pdf"}
    test_modes = [m for m in test_modes if m in valid_modes]
    if not test_modes:
        logger.warning("No valid difficulty_test_modes provided. Fallback to ['gt_pages', 'full_pdf'].")
        test_modes = ["gt_pages", "full_pdf"]
    filter_method = args.difficulty_filter_method

    input_dir = os.path.join(args.data_root, "results", "step_6")
    png_root = os.path.join(args.data_root, "results", "png")
    selected_pdf_root = os.path.join(args.data_root, "selected_pdfs")
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        files = [f for f in files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
    logger.info(f"Found {len(files)} files from step_6.")
    if not files:
        logger.warning("No files to process in step_7.")
        return

    if not args.no_resume:
        output_step7_dir = os.path.join(args.output_dir, "results", "step_7")
        files_to_run = []
        skipped_files = []
        for f in files:
            out_path = os.path.join(output_step7_dir, f)
            if _is_valid_existing_step7_output(out_path):
                skipped_files.append(f)
            else:
                files_to_run.append(f)
        logger.info(
            "Step7 resume enabled: skip %d completed files, run %d pending files.",
            len(skipped_files),
            len(files_to_run),
        )
        files = files_to_run
        if not files:
            logger.info("All selected files already have valid step_7 outputs. Nothing to do.")
            return

    total_cost = 0.0
    total_in = 0
    total_out = 0
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=min(args.file_batch_size, len(files))) as executor:
        futures = []
        for f in files:
            file_path = os.path.join(input_dir, f)
            pdf_name = os.path.splitext(f)[0]
            language = 'English' if args.language == 'EN' else 'Chinese (Simplified)'

            futures.append(
                executor.submit(
                    process_single_file,
                    answer_client_mode,
                    answer_client,
                    answer_model_name,
                    judge_client,
                    judge_model_name,
                    file_path,
                    pdf_name,
                    png_root,
                    args.output_dir,
                    args.inner_batch_size,
                    language,
                    os.path.join(selected_pdf_root, f"{pdf_name}.pdf"),
                    test_modes,
                    filter_method,
                )
            )

        for future in tqdm(as_completed(futures), total=len(files)):
            try:
                res = future.result()
                total_cost += res.get("cost", 0)
                total_in += res.get("input_tokens", 0)
                total_out += res.get("output_tokens", 0)
                if res.get("status") == "success":
                    processed_count += 1
            except Exception as e:
                logger.error(f"File processing exception: {e}")

    report = {
        "step": "step_7_filter_difficulty",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": processed_count,
        "token_details": {
            "input": total_in,
            "output": total_out
        }
    }
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_7.json")
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Step 7 Finished. Total Usage Metric: {total_cost:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root dir containing pipeline results")
    parser.add_argument("--output_dir", required=True, help="Output root")
    
    parser.add_argument("--file_batch_size", type=int, default=10)
    parser.add_argument("--inner_batch_size", type=int, default=5)
    parser.add_argument("--language", default='EN', choices=['CN', 'EN'], help="Language of template (choose from CN or EN)")
    parser.add_argument("--difficulty_test_modes", default="gt_pages,full_pdf", help="Step7 comma-separated modes: gt_pages,full_pdf")
    parser.add_argument("--difficulty_filter_method", default="any_correct", choices=["any_correct", "all_correct", "gt_pages_only", "full_pdf_only"], help="Step7 difficulty gate method.")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all.")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode and force re-run all selected files.")
    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    main(args)
