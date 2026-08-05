import os
import json
import copy
import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from utils.prompt_overrides import apply_prompt_overrides
from utils.prompt_files import load_step_prompts
from utils.runtime_config import create_genai_client, get_genai_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

_PROMPTS = load_step_prompts(
    "step_5_verify_qa",
    ["PROMPT_STEP1", "PROMPT_STEP2"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_5_verify_qa",
    _PROMPTS,
    logger=logger,
)
PROMPT_STEP1 = _PROMPTS["PROMPT_STEP1"]
PROMPT_STEP2 = _PROMPTS["PROMPT_STEP2"]


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


def calculate_cost(usage_metadata):
    """Function description."""
    if not usage_metadata:
        return 0.0, 0, 0
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    return 0.0, input_tokens, output_tokens

def get_image_crop(png_root, pdf_name, page_id, bbox):
    """Function description."""
    img_name = f"page_{int(page_id):04d}.png"
    img_path = os.path.join(png_root, pdf_name, img_name)

    if not os.path.exists(img_path):
        logger.warning(f"Image not found: {img_path}")
        return None

    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            w, h = img.size
            y1, x1, y2, x2 = bbox
            
            left = int(x1 / 1000.0 * w)
            top = int(y1 / 1000.0 * h)
            right = int(x2 / 1000.0 * w)
            bottom = int(y2 / 1000.0 * h)

            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            if right <= left or bottom <= top:
                logger.warning(f"Invalid crop dimensions for {img_path}: {bbox}")
                return None

            cropped_img = img.crop((left, top, right, bottom))
            return cropped_img
    except Exception as e:
        logger.warning(f"Error cropping image {img_path}: {e}")
        return None

def verify_single_qa_pair(client, qa_pair, evidence_map, pdf_name, png_root):
    """Function description."""
    question = qa_pair.get("Question", "")
    standard_answer = qa_pair.get("Answer", "")
    depended_idxs = qa_pair.get("Evidence_element_depended_idx", [])

    template_chosen = qa_pair.get("Template", "")
    template_type = qa_pair.get("Template_type", "")

    total_cost = 0.0
    total_in = 0
    total_out = 0

    if not question or not depended_idxs:
        return {"status": "skipped", "reason": "Missing question or dependencies"}, 0, 0, 0

    step1_contents = []
    evidence_text_context = ""
    
    ev_counter = 1

    for idx_str in depended_idxs:
        ev_item = evidence_map.get(idx_str)
        if not ev_item:
            logger.warning(f"Evidence {idx_str} not found in map.")
            continue
        
        crop_img = get_image_crop(png_root, pdf_name, ev_item['page_id'], ev_item['bbox'])
        
        ev_type = ev_item.get('type', 'unknown')
        ev_content = ev_item.get('content', '')

        evidence_text_context += f"Evidence {ev_counter} (Type: {ev_type}, ID: {idx_str}):\n"
        evidence_text_context += f"Text Content: {ev_content}\n"
        
        if crop_img:
            evidence_text_context += f"[Image for Evidence {ev_counter} attached]\n\n"
            step1_contents.append(crop_img)
        else:
            evidence_text_context += f"[Image for Evidence {ev_counter} MISSING]\n\n"
        
        ev_counter += 1

    step1_prompt_text = PROMPT_STEP1.format(
        evidence_context=evidence_text_context,
        question=question
    )
    step1_contents.append(step1_prompt_text)

    model_generated_answer = ""
    redundant_ids = []
    redundancy_reason = None
    
    max_retries = 3
    retry_delay = 30

    try:
        response_step1 = None
        for attempt in range(max_retries):
            try:
                response_step1 = client.models.generate_content(
                    model=get_genai_model(),
                    contents=step1_contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", 
                        temperature=0.0
                    )
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Step 1 API failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Step 1 API failed after {max_retries} attempts: {e}")
        
        if not response_step1:
            raise Exception("Step 1 Generation failed after retries")

        cost, in_t, out_t = calculate_cost(response_step1.usage_metadata)
        total_cost += cost
        total_in += in_t
        total_out += out_t
        
        step1_result = json.loads(response_step1.text)
        model_generated_answer = step1_result.get("answer", "")
        redundant_ids = step1_result.get("redundant_evidence_ids", [])
        redundancy_reason = step1_result.get("redundancy_reason", None)
        
    except Exception as e:
        logger.error(f"Step 1 (Generation/Redundancy) failed: {e}")
        return {"status": "error", "error_step": 1, "msg": str(e)}, total_cost, total_in, total_out

    if not model_generated_answer:
        return {"status": "error", "error_step": 1, "msg": "Model returned empty answer"}, total_cost, total_in, total_out

    verification_result = {}
    try:
        step2_prompt_text = PROMPT_STEP2.format(
            question=question,
            standard_answer=standard_answer,
            model_answer=model_generated_answer
        )
        
        response_step2 = None
        for attempt in range(max_retries):
            try:
                response_step2 = client.models.generate_content(
                    model=get_genai_model(),
                    contents=[step2_prompt_text],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Step 2 API failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Step 2 API failed after {max_retries} attempts: {e}")

        if not response_step2:
            raise Exception("Step 2 Verification failed after retries")

        cost, in_t, out_t = calculate_cost(response_step2.usage_metadata)
        total_cost += cost
        total_in += in_t
        total_out += out_t

        raw_return_text = response_step2.text.strip()
        lower_return_text = raw_return_text.lower()

        if "partially correct" in lower_return_text:
            verification_return = "Partially Correct"
            verification_result = False
        elif "incorrect" in lower_return_text:
            verification_return = "Incorrect"
            verification_result = False
        elif "correct" in lower_return_text:
            verification_return = "Correct"
            verification_result = True
        else:
            verification_return = raw_return_text
            verification_result = False

    except Exception as e:
        logger.error(f"Step 2 (Verification) failed: {e}")
        verification_result = False
        verification_return = "Error"

    return {
        "status": "success",
        "Question": question,
        "Standard_Answer": standard_answer,
        "Template": template_chosen,
        "Template_type": template_type,
        "Model_Generated_Answer": model_generated_answer,
        "Verification_Judgment": verification_result,
        "Verification_Return": verification_return,
        "Redundant_Evidence_IDs": redundant_ids,
        "Redundancy_Reason": redundancy_reason,
        "Evidence_Used_Input": depended_idxs 
    }, total_cost, total_in, total_out

def process_single_file(client, file_path, pdf_name, png_root, output_dir, inner_batch_size):
    """Function description."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return {"status": "error", "msg": f"Load failed: {e}", "cost": 0, "input": 0, "output": 0}
    save_dir = os.path.join(output_dir, "results", "step_5")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{pdf_name}.json")
    if os.path.exists(save_path):
        logger.info(f"⏭️ Skipping {pdf_name}: Output already exists.")
        return {
            "file": pdf_name,
            "status": "skipped",
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0
        }
    verified_data = []
    
    file_cost = 0.0
    file_in = 0
    file_out = 0

    for package in data:
        evidence_list = package.get("Evidence_list", [])
        evidence_map = {item['element_idx']: item for item in evidence_list}
        
        qa_groups = package.get("qa", [])
        
        flat_qa_list = []
        for group in qa_groups:
            if isinstance(group, list):
                flat_qa_list.extend(group)
            else:
                flat_qa_list.append(group)
        
        if not flat_qa_list:
            continue

        package_results = copy.deepcopy(package)
        package_results["qa_verification"] = []

        effective_inner_workers = min(
            inner_batch_size,
            len(flat_qa_list) if len(flat_qa_list) > 0 else 1,
        )
        with ThreadPoolExecutor(max_workers=effective_inner_workers) as executor:
            futures = [
                executor.submit(verify_single_qa_pair, client, qa, evidence_map, pdf_name, png_root)
                for qa in flat_qa_list
            ]

            for future in as_completed(futures):
                try:
                    res, cost, in_t, out_t = future.result()
                    package_results["qa_verification"].append(res)
                    
                    file_cost += cost
                    file_in += in_t
                    file_out += out_t
                except Exception as e:
                    logger.error(f"QA verification task exception: {e}")
        
        verified_data.append(package_results)


    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(verified_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved verification results to {save_path}")
    
    return {
        "status": "success",
        "file": pdf_name,
        "cost": file_cost,
        "input_tokens": file_in,
        "output_tokens": file_out
    }

def main(args):
    client = create_genai_client(genai, types)

    input_dir = os.path.join(args.data_root, "results", args.input_step_dir)
    png_root = os.path.join(args.data_root, "results", "png")

    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        files = [f for f in files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
    logger.info(f"Found {len(files)} files to verify.")
    if not files:
        logger.warning("No files to process in step_5.")
        return

    total_global_cost = 0.0
    total_global_in = 0
    total_global_out = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=min(args.file_batch_size, len(files))) as executor:
        futures = []
        for f in files:
            pdf_name = os.path.splitext(f)[0]
            file_path = os.path.join(input_dir, f)
            
            futures.append(
                executor.submit(
                    process_single_file,
                    client,
                    file_path,
                    pdf_name,
                    png_root,
                    args.output_dir,
                    args.inner_batch_size
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Verifying Files"):
            try:
                res = future.result()
                if res.get("status") == "success":
                    total_global_cost += res.get("cost", 0.0)
                    total_global_in += res.get("input_tokens", 0)
                    total_global_out += res.get("output_tokens", 0)
                    processed_count += 1
                elif res.get("status") == "error":
                    logger.error(f"File processing failed: {res.get('msg')}")
            except Exception as e:
                logger.error(f"File processing exception: {e}")

    logger.info("="*30)
    logger.info(f"Total Usage Metric: {total_global_cost:.6f}")
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_5.json")
    report = {
        "step": "step_5_verify_qa",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": processed_count,
        "token_details": {
            "input": total_global_in,
            "output": total_global_out
        }
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Usage report saved to {usage_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Verify QA Pairs")
    parser.add_argument("--data_root", required=True, help="Root dir containing results/step_4 and results/png")
    parser.add_argument("--output_dir", default="my_results", help="Output root")
    parser.add_argument("--input_step_dir", default="step_4", help="Input step directory name under results/")
    parser.add_argument("--file_batch_size", type=int, default=5, help="Concurrent files")
    parser.add_argument("--inner_batch_size", type=int, default=3, help="Concurrent QA checks per file")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all.")
    
    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    main(args)
