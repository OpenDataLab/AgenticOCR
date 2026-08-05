import os
import json
import copy
import argparse
import logging
import time
from typing import List, Set
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
    "step_4_generate_qa",
    ["PROMPT"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_4_generate_qa",
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


def _safe_prompt_format(prompt_template: str, **kwargs) -> str:
    escaped = prompt_template.replace("{", "{{").replace("}", "}}")
    for key in kwargs.keys():
        escaped = escaped.replace("{{" + key + "}}", "{" + key + "}")
    return escaped.format(**kwargs)


def _normalize_pdf_name(name: str) -> str:
    base = os.path.basename(str(name or "").strip())
    if not base:
        return ""
    if not base.lower().endswith(".pdf"):
        base = f"{base}.pdf"
    return base


def _extract_source_pdf_names_from_evidence(evidence_list) -> List[str]:
    names: Set[str] = set()
    for e in evidence_list or []:
        if not isinstance(e, dict):
            continue
        src_name = _normalize_pdf_name(e.get("source_pdf_name", ""))
        if src_name:
            names.add(src_name)
            continue
        src_doc = str(e.get("source_doc_id", "")).strip()
        if src_doc:
            names.add(_normalize_pdf_name(src_doc))
    return sorted([x for x in names if x])


def _load_required_source_pdfs_from_env(evidence_source_pdfs: List[str]) -> List[str]:
    explicit = os.getenv("OMNIDOC_QA_REQUIRED_SOURCE_PDFS", "").strip()
    explicit_set: Set[str] = set()
    if explicit:
        explicit_set = {_normalize_pdf_name(x) for x in explicit.split(",") if _normalize_pdf_name(x)}

    require_all_evidence = os.getenv("OMNIDOC_QA_REQUIRE_ALL_EVIDENCE_PDFS", "0").strip() == "1"
    if require_all_evidence:
        explicit_set.update({_normalize_pdf_name(x) for x in evidence_source_pdfs if _normalize_pdf_name(x)})
    return sorted(explicit_set)


def _extract_source_pdf_from_element_idx(element_idx: str) -> str:
    raw = str(element_idx or "").strip()
    if not raw:
        return ""
    if "::" in raw:
        return _normalize_pdf_name(raw.split("::", 1)[0])
    return ""


def _qa_covers_required_source_pdfs(qa_obj: dict, required_source_pdfs: List[str]) -> bool:
    if not required_source_pdfs:
        return True
    dep = qa_obj.get("Evidence_element_depended_idx", [])
    if not isinstance(dep, list):
        return False
    used: Set[str] = set()
    for x in dep:
        src = _extract_source_pdf_from_element_idx(str(x))
        if src:
            used.add(src)
    return set(required_source_pdfs).issubset(used)


def _build_allowed_element_idx_set(evidence_list) -> Set[str]:
    allowed: Set[str] = set()
    for e in evidence_list or []:
        if not isinstance(e, dict):
            continue
        idx = str(e.get("element_idx", "")).strip()
        if idx:
            allowed.add(idx)
    return allowed


def _filter_depended_idx_by_evidence_package(qa_obj: dict, allowed_element_idx: Set[str]) -> dict:
    if not isinstance(qa_obj, dict):
        return qa_obj
    dep = qa_obj.get("Evidence_element_depended_idx", [])
    if not isinstance(dep, list):
        qa_obj["Evidence_element_depended_idx"] = []
        return qa_obj
    qa_obj["Evidence_element_depended_idx"] = [
        x for x in dep if str(x).strip() in allowed_element_idx
    ]
    return qa_obj


def calculate_cost(usage_metadata):
    """Function description."""
    if not usage_metadata:
        return 0.0, 0, 0
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    return 0.0, input_tokens, output_tokens

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

def _truncate_qa_groups(qa_groups, max_total_qa):
    if max_total_qa <= 0:
        return qa_groups
    kept = 0
    out = []
    for group in qa_groups:
        if kept >= max_total_qa:
            break
        if isinstance(group, list):
            remaining = max_total_qa - kept
            trimmed = group[:remaining]
            out.append(trimmed)
            kept += len(trimmed)
        elif isinstance(group, dict):
            out.append(group)
            kept += 1
        else:
            out.append(group)
    return out


def _select_templates_with_diversity(templates, max_total):
    """Pick templates with type diversity first, then fill remaining slots."""
    if not isinstance(templates, list):
        return []
    if max_total <= 0:
        return [t for t in templates if isinstance(t, dict) and str(t.get("Template_chosen", "")).strip()]

    selected = []
    leftovers = []
    seen_types = set()
    seen_template_texts = set()

    for tmpl in templates:
        if not isinstance(tmpl, dict):
            continue
        template_text = str(tmpl.get("Template_chosen", "")).strip()
        template_type = str(tmpl.get("Template_type", "")).strip()
        if not template_text or template_text in seen_template_texts:
            continue
        if template_type and template_type not in seen_types:
            selected.append(tmpl)
            seen_types.add(template_type)
            seen_template_texts.add(template_text)
        else:
            leftovers.append(tmpl)
        if len(selected) >= max_total:
            return selected

    for tmpl in leftovers:
        template_text = str(tmpl.get("Template_chosen", "")).strip()
        if template_text in seen_template_texts:
            continue
        selected.append(tmpl)
        seen_template_texts.add(template_text)
        if len(selected) >= max_total:
            break

    return selected


def process_single_evidence_item(client, evidence_item, png_dir, language, max_qa_per_chain):
    """Function description."""
    try:
        evidence_content = evidence_item.get("Evidence_list", [])
        new_evidence_list, images, image_page_id_order = process_evidence_bundle_images(evidence_content, png_dir)
        evidence_source_pdfs = _extract_source_pdf_names_from_evidence(new_evidence_list)
        required_source_pdfs = _load_required_source_pdfs_from_env(evidence_source_pdfs)
        
        evidence_bundle_for_prompt = {"Evidence_list": new_evidence_list}
        allowed_element_idx = _build_allowed_element_idx_set(new_evidence_list)

        result_item = copy.deepcopy(evidence_item)
        result_item["qa"] = []
        
        total_item_cost = 0.0
        total_item_in = 0
        total_item_out = 0

        if not PROMPT:
            logger.error("PROMPT is empty. Check prompts/step_4_generate_qa__PROMPT.txt.")
            return result_item, 0, 0, 0

        templates = _select_templates_with_diversity(
            evidence_item.get("template", []),
            max_qa_per_chain,
        )
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=32768)
        )

        max_retries = 3
        retry_delay = 30

        for tmpl_obj in templates:
            template_text = tmpl_obj.get("Template_chosen", "")
            template_type = tmpl_obj.get("Template_type", "")
            if not template_text:
                continue

            qa_limit_suffix = (
                "\n\n[HARD LIMIT]\n"
                "Generate exactly ONE QA pair for this template in this call. "
                "If you cannot generate a valid one, return [] only."
            )

            formatted_prompt = _safe_prompt_format(
                PROMPT,
                Template=template_text,
                language=language,
                evidence_package=json.dumps(evidence_bundle_for_prompt, ensure_ascii=False, indent=2)
            ) + qa_limit_suffix + (
                "\n\n[PAGE_IMAGE_ORDER]\n"
                "The attached PNG images are ordered by original PDF page_id as:\n"
                f"{json.dumps(image_page_id_order, ensure_ascii=False)}\n"
                "The `page_id` field and the page prefix in `element_idx` both use original PDF page ids."
            ) + (
                f"Source PDF names present in this evidence package: {json.dumps(evidence_source_pdfs, ensure_ascii=False)}\n"
                f"Required source PDFs for this QA: {json.dumps(required_source_pdfs, ensure_ascii=False)}\n"
                "You MUST build a cross-document reasoning chain using all required source PDFs. "
                "Evidence_element_depended_idx MUST include at least one element_idx from each required source PDF "
                "(element_idx prefix format: <source_pdf_name>::...). "
                "If impossible, return [] only."
            )

            contents = images + [formatted_prompt]

            try:
                response = None
                for attempt in range(max_retries):
                    try:
                        response = client.models.generate_content(
                            model=get_genai_model(),
                            contents=contents,
                            config=config
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"QA Gen API failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"QA Gen API failed after {max_retries} attempts: {e}")

                if not response:
                    result_item["qa"].append([])
                    continue

                cost, in_t, out_t = calculate_cost(response.usage_metadata)
                total_item_cost += cost
                total_item_in += in_t
                total_item_out += out_t

                qa_data = json.loads(response.text)
                one_template_one_qa = []

                if isinstance(qa_data, list):
                    for q_item in qa_data:
                        if isinstance(q_item, dict):
                            q_item = _filter_depended_idx_by_evidence_package(q_item, allowed_element_idx)
                            if not _qa_covers_required_source_pdfs(q_item, required_source_pdfs):
                                continue
                            q_item["Template"] = template_text
                            q_item["Template_type"] = template_type
                            one_template_one_qa = [q_item]
                            break
                elif isinstance(qa_data, dict):
                    qa_data = _filter_depended_idx_by_evidence_package(qa_data, allowed_element_idx)
                    if _qa_covers_required_source_pdfs(qa_data, required_source_pdfs):
                        qa_data["Template"] = template_text
                        qa_data["Template_type"] = template_type
                        one_template_one_qa = [qa_data]

                result_item["qa"].append(one_template_one_qa)

            except Exception as e:
                logger.warning(f"QA gen failed for a template: {e}")
                result_item["qa"].append([])

        result_item["qa"] = _truncate_qa_groups(result_item.get("qa", []), max_qa_per_chain)
        return result_item, total_item_cost, total_item_in, total_item_out

    except Exception as e:
        logger.error(f"Error processing item: {e}")
        return evidence_item, 0, 0, 0

def process_single_pdf(
    client,
    pdf_name,
    input_json_path,
    png_dir,
    output_dir,
    inner_batch_size,
    language,
    max_evidence_chains_per_doc,
    max_qa_per_chain,
):
    """Function description."""
    save_dir = os.path.join(output_dir, "results", "step_4")
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

    if not os.path.exists(input_json_path):
        return {"status": "error", "msg": f"Input JSON missing: {input_json_path}", "cost": 0}
    if not os.path.exists(png_dir):
        return {"status": "error", "msg": f"PNG dir missing: {png_dir}", "cost": 0}

    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"status": "error", "msg": f"JSON Load Error: {e}", "cost": 0}

    if not data:
        logger.warning(f"[{pdf_name}] Empty data.")
        return {"status": "skipped", "cost": 0}

    if max_evidence_chains_per_doc > 0:
        data = data[:max_evidence_chains_per_doc]

    
    logger.info(f"[{pdf_name}] Generating QA for {len(data)} items...")
    inner_batch_size = min(inner_batch_size, len(data))
    logger.info(f"Inner Item Batch: {inner_batch_size}")
    
    final_results = []
    total_cost = 0.0
    total_in = 0
    total_out = 0

    with ThreadPoolExecutor(max_workers=inner_batch_size) as executor:
        futures = [
            executor.submit(process_single_evidence_item, client, item, png_dir, language, max_qa_per_chain)
            for item in data
        ]

        for future in as_completed(futures):
            res_item, cost, in_t, out_t = future.result()
            final_results.append(res_item)
            total_cost += cost
            total_in += in_t
            total_out += out_t

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "file": pdf_name,
        "cost": total_cost,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "qa_count": len(final_results)
    }

def main(args):
    input_dir = None # Step 3 output is Step 4 input
    png_root = None

    if args.data_root:
        input_dir = os.path.join(args.data_root, "results", "step_3")
        png_root = os.path.join(args.data_root, "results", "png")
    
    if args.input_dir: input_dir = args.input_dir
    if args.png_root: png_root = args.png_root

    if not (input_dir and png_root and os.path.exists(input_dir)):
        logger.error("Invalid paths. Check --data_root or specific directories.")
        return

    logger.info(f"Input Dir (Step 3): {input_dir}")
    logger.info(f"PNG Root          : {png_root}")

    tasks = []
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        files = [f for f in files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
    for f in files:
        pdf_name = os.path.splitext(f)[0]
        tasks.append({
            "pdf_name": pdf_name,
            "input_path": os.path.join(input_dir, f),
            "png_path": os.path.join(png_root, pdf_name) # Assuming folder name matches pdf name
        })

    logger.info(f"Found {len(tasks)} files to process.")
    if not tasks:
        logger.warning("No files to process in step_4.")
        return

    client = create_genai_client(genai, types)

    total_cost = 0.0
    total_in = 0
    total_out = 0

    file_batch_size = min(args.file_batch_size, len(tasks))
    logger.info(f"Starting - File Batch: {file_batch_size}")
    language = 'English' if args.language == 'EN' else 'Chinese (Simplified)'

    with ThreadPoolExecutor(max_workers=file_batch_size) as executor:
        future_to_task = {
            executor.submit(
                process_single_pdf,
                client,
                t["pdf_name"],
                t["input_path"],
                t["png_path"],
                args.output_dir,
                args.inner_batch_size,
                language,
                args.max_evidence_chains_per_doc,
                args.max_qa_per_chain,
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
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_4.json")
    report = {
        "step": "step_4_generate_qa",
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
    parser = argparse.ArgumentParser(description="Step 4: Generate QA (Nested Batch)")
    parser.add_argument("--data_root", help="Root directory containing results/step_3 and results/png")
    parser.add_argument("--input_dir", help="Override: Path to step_3 JSONs")
    parser.add_argument("--png_root", help="Override: Path to PNG folders")
    parser.add_argument("--output_dir", default="my_results", help="Output root")
    parser.add_argument("--language", default='EN', choices=['CN', 'EN'], help="Language of template (choose from CN or EN)")

    parser.add_argument("--file_batch_size", type=int, default=3, help="Concurrent PDF files")
    parser.add_argument("--inner_batch_size", type=int, default=5, help="Concurrent items per file")
    parser.add_argument("--max_evidence_chains_per_doc", type=int, default=50, help="Step4 pre-filter: max evidence chains used per document; <=0 means no limit")
    parser.add_argument("--max_qa_per_chain", type=int, default=3, help="Step4 pre-filter: max QA pairs generated per evidence chain; <=0 means no limit")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all JSONs in input_dir.")

    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    
    if not PROMPT:
        logger.warning("WARNING: PROMPT is empty. Check prompts/step_4_generate_qa__PROMPT.txt.")

    main(args)
