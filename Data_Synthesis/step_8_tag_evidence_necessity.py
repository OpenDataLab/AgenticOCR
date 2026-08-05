import os
import json
import copy
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import re
from utils.prompt_overrides import apply_prompt_overrides
from utils.prompt_files import load_step_prompts
from utils.runtime_config import create_genai_client, get_genai_model

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
    "step_8_tag_evidence_necessity",
    ["PROMPT_CHECK", "PROMPT_ANSWER_CHECK"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_8_tag_evidence_necessity",
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


def _is_valid_existing_step8_output(path):
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
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    return 0.0, input_tokens, output_tokens

def parse_model_answer(raw_text: str):
    if not raw_text:
        return None
    cleaned = clean_model_response(raw_text)
    try:
        parsed = json.loads(cleaned)
        ans = parsed.get("answer")
        if isinstance(ans, str) and ans.strip():
            return ans.strip()
    except Exception:
        pass
    return cleaned.strip() if cleaned.strip() else None

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


def _parse_page_id_from_evidence_id(evidence_id):
    try:
        s = str(evidence_id)
        if "::" in s:
            s = s.split("::", 1)[1]
        return int(str(s).split("-", 1)[0])
    except Exception:
        return None

def check_single_query(client, question, answer, images, language):

    prompt_text = PROMPT_CHECK.format(
        question=question,
    )
    answer_wo_evidence = None
    answer_wo_evidence_check = "Unknown"
    cost = 0.0
    in_t = 0
    out_t = 0

    pdf_part = images_to_pdf_part(images)
    if pdf_part is None:
        return "unknown", 0.0, 0, 0, None, "Unknown"
    contents = [pdf_part, prompt_text]

    try:
        response = None
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=get_genai_model(),
                    contents=contents, 
                    config=types.GenerateContentConfig(
                        temperature=1.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=32768)
                    )
                )
            
                cost, in_t, out_t = calculate_cost(response.usage_metadata)
                answer_wo_evidence = parse_model_answer(response.text)
                
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Step8 answer API failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Step8 answer API failed after {max_retries} attempts: {e}")

        if not answer_wo_evidence:
            return "unknown", 0.0, 0, 0, None, "Unknown"
        
        
        prompt = PROMPT_ANSWER_CHECK.format(
            question=question,
            standard_answer=answer,
            model_answer=answer_wo_evidence
        )
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=get_genai_model(),
                    contents=[prompt], 
                    config=types.GenerateContentConfig(
                        temperature=1.0,
                        thinking_config=types.ThinkingConfig(thinking_budget=32768)
                    )
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Step8 judge API failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Step8 judge API failed after {max_retries} attempts: {e}")
        if response is None:
            return "unknown", cost, in_t, out_t, answer_wo_evidence, "Unknown"
        cost2, in_t2, out_t2 = calculate_cost(response.usage_metadata)
        answer_wo_evidence_check = normalize_judge_label(response.text)
        if answer_wo_evidence_check == "Correct":
            result = "true"
        elif answer_wo_evidence_check in ["Incorrect", "Partially Correct"]:
            result = "false"
        else:
            result = "unknown"
        
        return result, cost+cost2, in_t+in_t2, out_t+out_t2, answer_wo_evidence, answer_wo_evidence_check

    except Exception as e:
        logger.error(f"Step8 necessity check failed: {e}")
        return "unknown", 0.0, 0, 0, None, "Unknown"
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

def process_single_file(client, file_path, pdf_name, png_root, output_dir, inner_batch_size, language):
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

    for p_idx, package_results in enumerate(data):
        package = copy.deepcopy(package_results)
        package["qa_after_evidence_gate"] = []
        qa_verification = package.get("qa_after_difficulty_gate", package.get("qa_verification", []))
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
                        client, qa, evidence_map, png_root, pdf_name, language
                    )
                )

            for i, future in enumerate(as_completed(qa_futures)):
                try:
                    result_dict, cost, in_t, out_t = future.result()
                    file_cost += cost
                    file_in += in_t
                    file_out += out_t

                    package.setdefault("qa_after_evidence_gate", []).append(result_dict["qa"])
                except Exception as e:
                    logger.error(f"QA processing error: {e}")

        final_packages.append(package)

    save_dir = os.path.join(output_dir, "results", "step_8")
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
    
def process_single_qa(client, qa, evidence_map, png_root, pdf_name, language):
    """Function description."""
    question = qa["Refined_Question"]
    answer = qa["Model_Generated_Answer"]
    used_evidence = qa.get("Evidence_Used_Input", [])

    pages_to_load_ids = set()
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
        for pid in range(max(1, ev_page_id - 3), ev_page_id + 3):
            pages_to_load_ids.add(pid)

    base_images = {}
    for pid in sorted(pages_to_load_ids):
        img_name = f"page_{pid:04d}.png"
        img_path = os.path.join(png_root, pdf_name, img_name)
        if not os.path.exists(img_path):
            continue
        base_images[pid] = Image.open(img_path).convert("RGB")

    total_cost = 0.0
    total_in = 0
    total_out = 0
    per_evidence = {}
    per_evidence_results = []

    for ev in used_evidence:
        ev_item = evidence_map.get(ev)
        if not ev_item:
            per_evidence[ev] = "unknown"
            per_evidence_results.append({
                "evidence_id": ev,
                "necessity": "unknown",
                "filter_status": "unknown",
                "judge": "Unknown",
                "model_answer": None,
                "note": "evidence_not_found",
            })
            continue

        qa_images = []
        for pid in sorted(base_images.keys()):
            img = base_images[pid].copy()
            try:
                ev_page_id = int(ev_item.get("page_id", -1))
            except Exception:
                ev_page_id = -1
            if pid == ev_page_id:
                bbox = ev_item.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    img = draw_normalized_bbox(img, bbox)
            qa_images.append(img)

        if not qa_images:
            per_evidence[ev] = "unknown"
            per_evidence_results.append({
                "evidence_id": ev,
                "necessity": "unknown",
                "filter_status": "unknown",
                "judge": "Unknown",
                "model_answer": None,
                "note": "no_images_for_mask_test",
            })
            continue

        result, cost, in_t, out_t, answer_wo_evidence, answer_wo_evidence_check = check_single_query(
            client,
            question,
            answer,
            qa_images,
            language
        )
        total_cost += cost
        total_in += in_t
        total_out += out_t

        if result == "true":
            necessity = "non_necessary"
        elif result == "false":
            necessity = "necessary"
        else:
            necessity = "unknown"

        per_evidence[ev] = necessity
        per_evidence_results.append({
            "evidence_id": ev,
            "necessity": necessity,
            "filter_status": result,
            "judge": answer_wo_evidence_check,
            "model_answer": answer_wo_evidence,
        })

    necessary_flags = [v for v in per_evidence.values() if v in {"necessary", "non_necessary"}]
    all_independently_important = bool(necessary_flags) and all(v == "necessary" for v in necessary_flags)

    qa_out = copy.deepcopy(qa)
    qa_out["step8_eval"] = {
        "test_name": "single_evidence_mask_ablation",
        "evidence_necessity": {
            "per_evidence": per_evidence,
            "all_independently_important": all_independently_important,
        },
        "per_evidence_results": per_evidence_results,
        "status": "tagged",
    }
    return {"status": "tagged", "qa": qa_out}, total_cost, total_in, total_out
def main(args):
    client = create_genai_client(genai, types)

    input_dir = os.path.join(args.data_root, "results", "step_7")
    png_root = os.path.join(args.data_root, "results", "png")
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        files = [f for f in files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
    logger.info(f"Found {len(files)} files from step_7.")
    if not files:
        logger.warning("No files to process in step_8.")
        return

    if not args.no_resume:
        output_step8_dir = os.path.join(args.output_dir, "results", "step_8")
        files_to_run = []
        skipped_files = []
        for f in files:
            out_path = os.path.join(output_step8_dir, f)
            if _is_valid_existing_step8_output(out_path):
                skipped_files.append(f)
            else:
                files_to_run.append(f)
        logger.info(
            "Step8 resume enabled: skip %d completed files, run %d pending files.",
            len(skipped_files),
            len(files_to_run),
        )
        files = files_to_run
        if not files:
            logger.info("All selected files already have valid step_8 outputs. Nothing to do.")
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
                    client,
                    file_path,
                    pdf_name,
                    png_root,
                    args.output_dir,
                    args.inner_batch_size,
                    language
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
        "step": "step_8_ensure_evidence_necessity",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": processed_count,
        "token_details": {
            "input": total_in,
            "output": total_out
        }
    }
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_8.json")
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Step 8 Finished. Total Usage Metric: {total_cost:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root dir containing pipeline results")
    parser.add_argument("--output_dir", required=True, help="Output root")
    
    parser.add_argument("--file_batch_size", type=int, default=10)
    parser.add_argument("--inner_batch_size", type=int, default=5)
    parser.add_argument("--language", default='EN', choices=['CN', 'EN'], help="Language of template (choose from CN or EN)")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all.")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode and force re-run all selected files.")
    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    main(args)
