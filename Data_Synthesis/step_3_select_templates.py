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

from templates import *


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

_PROMPTS = load_step_prompts(
    "step_3_select_templates",
    ["PROMPT"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_3_select_templates",
    _PROMPTS,
    logger=logger,
)
PROMPT = _PROMPTS["PROMPT"]


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

def get_template(templates_dir, language):

    with open(templates_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    category_buckets = {
        "Factual Retrieval": [],
        "Complex Synthesis": [],
        "Quantitative Reasoning": [],
        "Multimodal Parsing": []
    }
    
    lang_suffix = "_en" if language.strip().upper() == "EN" else "_cn"
    template_key = f"template{lang_suffix}"
    example_key = f"example{lang_suffix}"
    
    templates_list = data.get("templates", [])
    for item in templates_list:
        category = item.get("category")
        
        t_text = item.get(template_key, "")
        e_text = item.get(example_key, "")
        
        if category in category_buckets:
            category_buckets[category].append((t_text, e_text))
            
    placeholder_mapping = {
        "Factual Retrieval": "FactualRetrievalTemplates",
        "Complex Synthesis": "ComplexSynthesisTemplates",
        "Quantitative Reasoning": "QuantitativeReasoningTemplates",
        "Multimodal Parsing": "MultimodalParsingTemplates"
    }
    
    formatted_sections = {}
    
    for cat_name, placeholder_key in placeholder_mapping.items():
        items = category_buckets.get(cat_name, [])
        lines = []
        
        for idx, (t_val, e_val) in enumerate(items, 1):
            entry = f"{idx}. **{t_val}**\n   *Example:* {e_val}"
            lines.append(entry)
            
        formatted_sections[placeholder_key] = "\n\n".join(lines)
        
    final_prompt = BASE_TEMPLATE.format(**formatted_sections)
    return final_prompt

def process_evidence_bundle_images(evidence_list, png_dir):
    """Function description."""
    original_page_ids = [e["page_id"] for e in evidence_list if "page_id" in e]
    unique_sorted_page_ids = sorted(set(original_page_ids))

    new_evidence_list = copy.deepcopy(evidence_list)

    processed_images = []
    for old_page_id in unique_sorted_page_ids:
        img_name = f"page_{old_page_id:04d}.png"
        img_path = os.path.join(png_dir, img_name)
        
        try:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = img.resize((1000, 1000), Image.LANCZOS)
                processed_images.append(img)
            else:
                logger.warning(f"Image missing: {img_path}")
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")

    return new_evidence_list, processed_images, unique_sorted_page_ids


def _postprocess_template_selection(template_result, max_templates_per_chain):
    """Cap template count and maximize Template_type diversity."""
    if not isinstance(template_result, list):
        return []

    normalized = []
    for item in template_result:
        if not isinstance(item, dict):
            continue
        chosen = str(item.get("Template_chosen", "")).strip()
        t_type = str(item.get("Template_type", "")).strip()
        if not chosen:
            continue
        normalized.append(item)

    if max_templates_per_chain <= 0:
        return normalized

    selected = []
    seen_types = set()
    seen_template_texts = set()
    leftovers = []

    for item in normalized:
        chosen = str(item.get("Template_chosen", "")).strip()
        t_type = str(item.get("Template_type", "")).strip()
        if chosen in seen_template_texts:
            continue
        if t_type and t_type not in seen_types:
            selected.append(item)
            seen_types.add(t_type)
            seen_template_texts.add(chosen)
        else:
            leftovers.append(item)

        if len(selected) >= max_templates_per_chain:
            return selected

    for item in leftovers:
        chosen = str(item.get("Template_chosen", "")).strip()
        if chosen in seen_template_texts:
            continue
        selected.append(item)
        seen_template_texts.add(chosen)
        if len(selected) >= max_templates_per_chain:
            break

    return selected

def process_single_evidence_item(
    client,
    evidence_item,
    png_dir,
    templates_dir=None,
    language='EN',
    max_templates_per_chain=3,
):
    """Function description."""
    try:
        evidence_list_content = evidence_item.get("Evidence_list", [])
        new_evidence_list, images, image_page_id_order = process_evidence_bundle_images(evidence_list_content, png_dir)
        
        evidence_package_for_prompt = copy.deepcopy(evidence_item)
        evidence_package_for_prompt["Evidence_list"] = new_evidence_list

        if not PROMPT:
            return None, 0.0, 0, 0  # Fail fast if prompt is empty
        
        max_templates_text = (
            str(max_templates_per_chain)
            if max_templates_per_chain > 0
            else "all available valid"
        )

        if templates_dir:
            formatted_prompt = PROMPT.format(
                Template=get_template(templates_dir, language), 
                evidence_package=json.dumps(evidence_package_for_prompt, ensure_ascii=False, indent=2),
                max_templates=max_templates_text,
            )
            
        else:
            logger.error("templates_dir is required.")
            return None, 0.0, 0, 0

        formatted_prompt += (
            "\n\n[PAGE_IMAGE_ORDER]\n"
            "The attached PNG images are ordered by original PDF page_id as:\n"
            f"{json.dumps(image_page_id_order, ensure_ascii=False)}\n"
            "The `page_id` field and the page prefix in `element_idx` both use original PDF page ids."
        )

        contents = images + [formatted_prompt]

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=32768), 
        )

        response = None
        max_retries = 3
        retry_delay = 30
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=get_genai_model(),
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"API call failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")

        if not response:
            failed_item = copy.deepcopy(evidence_item)
            failed_item["template"] = []
            return failed_item, 0.0, 0, 0

        cost, in_tok, out_tok = calculate_cost(response.usage_metadata)

        try:
            template_result = json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning("JSON parsing failed for an evidence item.")
            template_result = []

        result_item = copy.deepcopy(evidence_item)
        result_item["template"] = _postprocess_template_selection(
            template_result,
            max_templates_per_chain=max_templates_per_chain,
        )
        
        return result_item, cost, in_tok, out_tok

    except Exception as e:
        logger.error(f"Error processing evidence item: {e}")
        failed_item = copy.deepcopy(evidence_item)
        failed_item["template"] = []
        return failed_item, 0.0, 0, 0

def process_single_pdf(
    client,
    pdf_name,
    evidence_path,
    png_path,
    output_dir,
    inner_batch_size=5,
    templates_dir=None,
    language="EN",
    max_templates_per_chain=3,
    max_evidence_chains_per_doc=0,
):
    """Function description."""
    save_dir = os.path.join(output_dir, "results", "step_3")
    output_path = os.path.join(save_dir, f"{pdf_name}.json")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(output_path):
        logger.info(f"⏭️ Skipping {pdf_name}: Output already exists.")
        return {
            "file": pdf_name,
            "status": "skipped",
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0
        }

    if not os.path.exists(evidence_path):
        return {"status": "error", "msg": f"Evidence file missing: {evidence_path}", "cost": 0}
    if not os.path.exists(png_path):
        return {"status": "error", "msg": f"PNG dir missing: {png_path}", "cost": 0}

    try:
        with open(evidence_path, "r", encoding="utf-8") as f:
            evidences = json.load(f)
    except Exception as e:
        return {"status": "error", "msg": f"Corrupt JSON: {e}", "cost": 0}

    if not evidences:
        logger.warning(f"[{pdf_name}] No evidence found.")
        return {"status": "skipped", "cost": 0}

    if max_evidence_chains_per_doc > 0:
        evidences = evidences[:max_evidence_chains_per_doc]

    logger.info(f"[{pdf_name}] Processing {len(evidences)} bundles...")

    results = []
    total_cost = 0.0
    total_in = 0
    total_out = 0

    inner_batch_size = min(len(evidences), inner_batch_size)
    logger.info(f"[{pdf_name}] Effective Inner Item Batch: {inner_batch_size}")
    with ThreadPoolExecutor(max_workers=inner_batch_size) as executor:
        futures = [
            executor.submit(
                process_single_evidence_item,
                client,
                item,
                png_path,
                templates_dir,
                language,
                max_templates_per_chain,
            )
            for item in evidences
        ]

        for future in as_completed(futures):
            res_item, cost, in_t, out_t = future.result()
            if res_item:
                results.append(res_item)
                total_cost += cost
                total_in += in_t
                total_out += out_t


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "file": pdf_name,
        "cost": total_cost,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "items_count": len(results)
    }

def main(args):
    evidence_dir = None
    png_root = None

    if args.data_root:
        evidence_dir = os.path.join(args.data_root, "results", "step_2")
        png_root = os.path.join(args.data_root, "results", "png")
    
    if args.evidence_dir: evidence_dir = args.evidence_dir
    if args.png_root: png_root = args.png_root

    if not (evidence_dir and png_root and os.path.exists(evidence_dir)):
        logger.error("Invalid paths. Check --data_root or specific directories.")
        return

    logger.info(f"Evidence Dir: {evidence_dir}")
    logger.info(f"PNG Root    : {png_root}")

    tasks = []
    files = [f for f in os.listdir(evidence_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        files = [f for f in files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
    for f in files:
        pdf_name = os.path.splitext(f)[0]
        tasks.append({
            "pdf_name": pdf_name,
            "evidence_path": os.path.join(evidence_dir, f),
            "png_path": os.path.join(png_root, pdf_name)
        })

    logger.info(f"Found {len(tasks)} files to process.")
    if not tasks:
        logger.warning("No files to process in step_3.")
        return

    client = create_genai_client(genai, types)

    total_cost = 0.0
    total_in = 0
    total_out = 0
    
    file_batch_size = min(len(tasks), args.file_batch_size)
    logger.info(f"Starting File Batch (Size: {file_batch_size}) | Configured Inner Item Batch (Size: {args.inner_batch_size})")

    with ThreadPoolExecutor(max_workers=file_batch_size) as executor:
        future_to_task = {
            executor.submit(
                process_single_pdf,
                client,
                t["pdf_name"],
                t["evidence_path"],
                t["png_path"],
                args.output_dir,
                args.inner_batch_size,
                args.templates_dir,
                args.language,
                args.max_templates_per_chain,
                args.max_evidence_chains_per_doc,
            ): t for t in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing Files"):
            res = future.result()
            if "cost" in res:
                total_cost += res["cost"]
                total_in += res.get("input_tokens", 0)
                total_out += res.get("output_tokens", 0)
            
            if res["status"] == "error":
                logger.error(f"File {res.get('file')} failed: {res.get('msg')}")

    logger.info("="*30)
    logger.info(f"Total Usage Metric: {total_cost:.6f}")
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_3.json")
    report = {
        "step": "step_3_select_templates",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": len(tasks),
        "token_details": {
            "input": total_in,
            "output": total_out
        }
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Usage report saved to {usage_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Select Templates (Nested Batch)")
    
    parser.add_argument("--data_root", help="Root directory containing results/step_2 and results/png")
    
    parser.add_argument("--evidence_dir", help="Override path to step_2 JSONs")
    parser.add_argument("--templates_dir", default=None, help="Path to query templates JSON")
    parser.add_argument("--language", default='EN', choices=['CN', 'EN'], help="Language of template (choose from CN or EN)")
    parser.add_argument("--max_templates_per_chain", type=int, default=3, help="Max templates selected per evidence chain; should align with Step4 max QA per chain")
    parser.add_argument("--max_evidence_chains_per_doc", type=int, default=0, help="Step3 pre-filter: max evidence chains used per document; <=0 means no limit")
    parser.add_argument("--png_root", help="Override path to png root")
    
    parser.add_argument("--output_dir", default="my_results", help="Output root")
    
    parser.add_argument("--file_batch_size", type=int, default=3, help="Concurrent PDF files processed (Outer loop)")
    parser.add_argument("--inner_batch_size", type=int, default=5, help="Concurrent evidence items processed per file (Inner loop)")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all.")

    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]

    main(args)
