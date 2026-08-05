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
    "step_6_refine_query",
    ["PROMPT_REFINE"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_6_refine_query",
    _PROMPTS,
    logger=logger,
)
PROMPT_REFINE = _PROMPTS["PROMPT_REFINE"]


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

def refine_single_query(client, qa_item, evidence_map, pdf_name, png_root, language):
    """Function description."""
    original_question = qa_item.get("Question", "")
    original_answer = qa_item.get("Standard_Answer", "")
    evidence_ids = qa_item.get("Evidence_Used_Input", [])
    template_type = qa_item.get("Template_type", "Unknown")  

    if not original_question:
        return None, 0.0, 0, 0

    contents = []
    evidence_text_context = ""
    ev_counter = 1

    for idx_str in evidence_ids:
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
            contents.append(crop_img)
        else:
            evidence_text_context += f"[Image for Evidence {ev_counter} MISSING]\n\n"
        
        ev_counter += 1

    prompt_text = PROMPT_REFINE.replace("{{ language }}", language).format(
        evidence_context=evidence_text_context,
        question=original_question,
        answer=original_answer,
        template_type=template_type 
    )
    
    contents.append(prompt_text)

    try:
        response = None
        max_retries = 3
        retry_delay = 30
        
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
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Refine Query API failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Refine Query API failed after {max_retries} attempts: {e}")

        if not response:
            return None, 0.0, 0, 0
        
        cost, in_t, out_t = calculate_cost(response.usage_metadata)
        refined_question = response.text.strip()
        
        new_qa = copy.deepcopy(qa_item)
        new_qa["Refined_Question"] = refined_question
        
        keys_to_remove = ["status", "Redundant_Evidence_IDs", "Redundancy_Reason"]
        for k in keys_to_remove:
            if k in new_qa:
                del new_qa[k]
        
        return new_qa, cost, in_t, out_t

    except Exception as e:
        logger.error(f"Refine query failed: {e}")
        return None, 0.0, 0, 0

def process_single_file(client, file_path, pdf_name, png_root, output_dir, inner_batch_size, language):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return {"status": "error", "cost": 0, "input": 0, "output": 0}
    save_dir = os.path.join(output_dir, "results", "step_6")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    if os.path.exists(save_path):
        logger.info(f"⏭️ Skipping {pdf_name}: Output already exists.")
        return {
            "file": pdf_name,
            "status": "skipped",
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0
        }
    final_packages = []
    file_cost = 0.0
    file_in = 0
    file_out = 0

    valid_qa_tasks = [] 

    for p_idx, package in enumerate(data):
        new_package = {k: v for k, v in package.items() if k not in ["qa", "qa_verification"]}
        new_package["qa_verification"] = []
        
        evidence_list = package.get("Evidence_list", [])
        evidence_map = {item['element_idx']: item for item in evidence_list}

        raw_qas = package.get("qa_verification", [])
        
        valid_qas_in_package_count = 0
        
        for qa in raw_qas:
            if qa.get("status") != "success" or qa.get("Verification_Judgment") is not True:
                continue
            
            used_evidence = set(qa.get("Evidence_Used_Input", []))
            redundant_evidence = set(qa.get("Redundant_Evidence_IDs", []))
            
            final_evidence_set = used_evidence - redundant_evidence
            final_evidence_list = list(final_evidence_set)
            
            if len(final_evidence_list) < 1:
                continue
            
            cleaned_qa = copy.deepcopy(qa)
            cleaned_qa["Evidence_Used_Input"] = final_evidence_list
            cleaned_qa["Redundant_Evidence_IDs"] = [] 
            cleaned_qa["Redundancy_Reason"] = None
            
            cleaned_qa["Template_type"] = qa.get("Template_type", "")
            
            if valid_qas_in_package_count == 0:
                final_packages.append(new_package)
            
            valid_qa_tasks.append({
                "package_idx": len(final_packages) - 1,
                "qa_data": cleaned_qa,
                "evidence_map": evidence_map
            })
            valid_qas_in_package_count += 1

    if not valid_qa_tasks:
        return {"status": "skipped", "msg": "No valid QA pairs found", "cost": 0, "input": 0, "output": 0}

    effective_inner_workers = min(
        inner_batch_size,
        len(valid_qa_tasks) if len(valid_qa_tasks) > 0 else 1,
    )
    with ThreadPoolExecutor(max_workers=effective_inner_workers) as executor:
        futures = {
            executor.submit(
                refine_single_query, 
                client, 
                task["qa_data"], 
                task["evidence_map"], 
                pdf_name, 
                png_root, 
                language
            ): task 
            for task in valid_qa_tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            p_idx = task["package_idx"]
            
            try:
                refined_qa, cost, in_t, out_t = future.result()
                
                file_cost += cost
                file_in += in_t
                file_out += out_t

                if refined_qa:
                    final_packages[p_idx]["qa_verification"].append(refined_qa)
            except Exception as e:
                logger.error(f"Error in future result: {e}")

    final_packages = [p for p in final_packages if len(p["qa_verification"]) > 0]

    if not final_packages:
        return {"status": "empty_after_refine", "cost": file_cost, "input": file_in, "output": file_out}


    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_packages, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "file": os.path.basename(file_path),
        "cost": file_cost,
        "input_tokens": file_in,
        "output_tokens": file_out
    }

def main(args):
    client = create_genai_client(genai, types)

    input_dir = os.path.join(args.data_root, "results", "step_5")
    png_root = os.path.join(args.data_root, "results", "png")
    
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        files = [f for f in files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
    logger.info(f"Found {len(files)} files to refine.")
    if not files:
        logger.warning("No files to process in step_6.")
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

        for future in tqdm(as_completed(futures), total=len(files), desc="Refining Queries"):
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
        "step": "step_6_refine_query",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": processed_count,
        "token_details": {
            "input": total_in,
            "output": total_out
        }
    }
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_6.json")
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Step 6 Finished. Total Usage Metric: {total_cost:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    
    parser.add_argument("--file_batch_size", type=int, default=10)
    parser.add_argument("--inner_batch_size", type=int, default=5)
    parser.add_argument("--language", default='EN', choices=['CN', 'EN'], help="Language of template (choose from CN or EN)")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all.")
    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    main(args)
