import os
import json
import argparse
import logging
import time
import pathlib
import io
import re
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from utils.prompt_overrides import apply_prompt_overrides
from utils.prompt_files import load_step_prompts
from utils.runtime_config import create_genai_client, get_genai_model

import fitz  
from PIL import Image
from PyPDF2 import PdfReader

def get_pdf_page_count(filepath):
    reader = PdfReader(filepath)
    return len(reader.pages)
import threading
_compress_sem = threading.Semaphore(1)

MODEL_NAME = get_genai_model()


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


def _iter_json_values(text: str):
    decoder = json.JSONDecoder()
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            val, j = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        yield val
        i = j


def _load_rows_mixed(path: str) -> List[Dict[str, Any]]:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    rows = []
    strict_ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            strict_ok = False
            break
        if isinstance(obj, dict):
            rows.append(obj)
    if strict_ok:
        return rows

    rows = []
    for obj in _iter_json_values(text):
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _normalize_page_ids(v: Any) -> List[int]:
    pages: List[int] = []
    if isinstance(v, int):
        pages = [v]
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, int):
                pages.append(x)
            elif isinstance(x, str) and x.strip().isdigit():
                pages.append(int(x.strip()))
    elif isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            pages = [int(s)]
        else:
            m = re.match(r"^\s*(\d+)\s*[-~]\s*(\d+)\s*$", s)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b:
                    a, b = b, a
                pages = list(range(a, b + 1))
    return sorted({p for p in pages if isinstance(p, int) and p > 0})


def _load_units_profiles(units_jsonl: str) -> Dict[str, Dict[str, Any]]:
    rows = _load_rows_mixed(units_jsonl)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        profile = row.get("profile", {}) if isinstance(row.get("profile"), dict) else {}
        section_units = profile.get("section_units", [])
        if not isinstance(section_units, list):
            section_units = []
        cleaned_sections = []
        for sec in section_units:
            if not isinstance(sec, dict):
                continue
            page_ids = _normalize_page_ids(sec.get("page_ids"))
            if not page_ids:
                page_ids = _normalize_page_ids(sec.get("page_id"))
            if not page_ids:
                page_ids = _normalize_page_ids(sec.get("page_range"))
            cleaned_sections.append(
                {
                    "section_id": str(sec.get("section_id", "")).strip(),
                    "section_title": str(sec.get("section_title", "")).strip(),
                    "description": str(sec.get("description", "")).strip(),
                    "time_scope": str(sec.get("time_scope", "")).strip(),
                    "page_ids": page_ids,
                    "page_range": f"{page_ids[0]}-{page_ids[-1]}" if page_ids else "",
                }
            )
        out[doc_id] = {
            "doc_id": doc_id,
            "document_summary": str(profile.get("document_summary", "")).strip(),
            "document_keywords": profile.get("document_keywords", []) if isinstance(profile.get("document_keywords"), list) else [],
            "section_units": cleaned_sections,
        }
    return out


def _load_units_profiles_from_dir(units_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    root = pathlib.Path(units_dir)
    for p in sorted(root.glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id", "")).strip()
        if not doc_id:
            continue
        profile = row.get("profile", {}) if isinstance(row.get("profile"), dict) else {}
        section_units = profile.get("section_units", [])
        if not isinstance(section_units, list):
            section_units = []
        cleaned_sections = []
        for sec in section_units:
            if not isinstance(sec, dict):
                continue
            page_ids = _normalize_page_ids(sec.get("page_ids"))
            if not page_ids:
                page_ids = _normalize_page_ids(sec.get("page_id"))
            if not page_ids:
                page_ids = _normalize_page_ids(sec.get("page_range"))
            cleaned_sections.append(
                {
                    "section_id": str(sec.get("section_id", "")).strip(),
                    "section_title": str(sec.get("section_title", "")).strip(),
                    "description": str(sec.get("description", "")).strip(),
                    "time_scope": str(sec.get("time_scope", "")).strip(),
                    "page_ids": page_ids,
                    "page_range": f"{page_ids[0]}-{page_ids[-1]}" if page_ids else "",
                }
            )
        out[doc_id] = {
            "doc_id": doc_id,
            "document_summary": str(profile.get("document_summary", "")).strip(),
            "document_keywords": profile.get("document_keywords", []) if isinstance(profile.get("document_keywords"), list) else [],
            "section_units": cleaned_sections,
        }
    return out


def _load_bundle_map(doc_bundles_jsonl: str) -> Dict[str, List[str]]:
    rows = _load_rows_mixed(doc_bundles_jsonl)
    out: Dict[str, List[str]] = {}
    for row in rows:
        anchor = row.get("anchor_doc", {}) if isinstance(row.get("anchor_doc"), dict) else {}
        anchor_doc_id = str(anchor.get("doc_id", "")).strip()
        if not anchor_doc_id:
            continue
        cands = row.get("retrieved_doc_candidates", [])
        cand_doc_ids: List[str] = []
        if isinstance(cands, list):
            for x in cands:
                if isinstance(x, dict):
                    did = str(x.get("doc_id", "")).strip()
                    if did and did != anchor_doc_id and did not in cand_doc_ids:
                        cand_doc_ids.append(did)
        out[anchor_doc_id] = cand_doc_ids
    return out


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

_PROMPTS = load_step_prompts(
    "step_2_generate_evidence",
    ["PROMPT"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_2_generate_evidence",
    _PROMPTS,
    logger=logger,
)
PROMPT = _PROMPTS["PROMPT"]


def calculate_cost(usage_metadata):
    """Function description."""
    if not usage_metadata:
        return 0.0, 0, 0
    input_tokens = usage_metadata.prompt_token_count
    output_tokens = usage_metadata.candidates_token_count
    return 0.0, input_tokens, output_tokens

def build_prompt(base_prompt, outline, blocks, previous_evidence):
    prompt = base_prompt.format(
        document_outline=json.dumps(outline, ensure_ascii=False, indent=2),
        blocks=json.dumps(blocks, ensure_ascii=False, indent=2),
        previous_evidence=json.dumps(previous_evidence, ensure_ascii=False, indent=2)
    )
    return prompt

def align_evidence_with_original_blocks(evidence_packages, original_blocks):
    """Function description."""
    blocks_map = {b.get("element_idx"): b for b in original_blocks}

    _src_to_page = {}
    for _b in original_blocks:
        _sdid = _b.get("source_doc_id")
        _spid = _b.get("source_page_id")
        _pid = _b.get("page_id")
        if _sdid is not None and _spid is not None and _pid is not None:
            _src_to_page[(_sdid, _spid)] = _pid
    
    aligned_count = 0
    missing_count = 0

    for pkg in evidence_packages:
        if "Evidence_list" not in pkg or not isinstance(pkg["Evidence_list"], list):
            continue
            
        for item in pkg["Evidence_list"]:
            e_idx = item.get("element_idx")
            
            if e_idx and e_idx in blocks_map:
                original = blocks_map[e_idx]
                
                item["bbox"] = original.get("bbox")
                item["content"] = original.get("content")
                item["type"] = original.get("type")
                item["page_id"] = original.get("page_id")
                if "source_doc_id" in original:
                    item["source_doc_id"] = original.get("source_doc_id")
                if "source_pdf_name" in original:
                    item["source_pdf_name"] = original.get("source_pdf_name")
                if "source_page_id" in original:
                    item["source_page_id"] = original.get("source_page_id")
                if "source_element_idx" in original:
                    item["source_element_idx"] = original.get("source_element_idx")
                
                aligned_count += 1
            else:
                _fk = (item.get("source_doc_id"), item.get("source_page_id"))
                if "page_id" not in item and _fk in _src_to_page:
                    item["page_id"] = _src_to_page[_fk]
                if e_idx:
                    logger.debug(f"Warning: model generated non-existent element_idx: {e_idx}")
                    missing_count += 1

    return evidence_packages, aligned_count, missing_count


def _extract_doc_id_from_element_idx(element_idx: Any) -> str:
    s = str(element_idx or "").strip()
    if "::" in s:
        left = s.split("::", 1)[0].strip()
        if left.lower().endswith(".pdf"):
            return left[:-4]
        return left
    return ""


def _package_source_doc_ids(pkg: Dict[str, Any]) -> List[str]:
    out = set()
    ev_list = pkg.get("Evidence_list", [])
    if not isinstance(ev_list, list):
        return []
    for e in ev_list:
        if not isinstance(e, dict):
            continue
        did = str(e.get("source_doc_id", "")).strip()
        if did:
            out.add(did)
            continue
        pdf_name = str(e.get("source_pdf_name", "")).strip()
        if pdf_name:
            out.add(pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name)
            continue
        did2 = _extract_doc_id_from_element_idx(e.get("element_idx"))
        if did2:
            out.add(did2)
    return sorted(out)


def filter_cross_doc_evidence_packages(evidence_packages: List[Dict[str, Any]], min_docs: int) -> List[Dict[str, Any]]:
    if min_docs <= 1:
        return evidence_packages
    out = []
    for pkg in evidence_packages:
        if not isinstance(pkg, dict):
            continue
        doc_ids = _package_source_doc_ids(pkg)
        if len(doc_ids) >= min_docs:
            out.append(pkg)
    return out

def compress_pdf_with_pillow(pdf_bytes, zoom=0.5, quality=20):
    """Function description."""
    start_size = len(pdf_bytes)
    if start_size < 3 * 1024 * 1024:
        return pdf_bytes

    logger.info(f"⚡ Starting PIL compression for large file. Original: {start_size/1024/1024:.2f} MB")
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out_doc = fitz.open()
        
        mat = fitz.Matrix(zoom, zoom)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
            img_data = img_byte_arr.getvalue()
            
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(page.rect, stream=img_data)
            
            if len(doc) > 20 and i % 20 == 0:
                logger.debug(f"   Processed page {i+1}/{len(doc)}")

        output_stream = io.BytesIO()
        out_doc.save(output_stream, garbage=4, deflate=True)
        compressed_bytes = output_stream.getvalue()
        
        end_size = len(compressed_bytes)
        reduction = (1 - end_size / start_size) * 100
        logger.info(f"Final Size: {end_size/1024/1024:.2f} MB (Reduced {reduction:.2f}%)")
        
        return compressed_bytes

    except Exception as e:
        logger.error(f"Compression Error: {e}. Falling back to original bytes.")
        return pdf_bytes


def process_single_pdf(
    client,
    pdf_name,
    outline_path,
    blocks_path,
    pdf_path,
    output_dir,
    iterations,
    max_evidence_chains_per_doc=0,
):
    """Function description."""
    save_dir = os.path.join(output_dir, "results", "step_2")
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
    total_cost = 0.0
    total_in = 0
    total_out = 0

    try:
        if not os.path.exists(blocks_path):
            return {"status": "error", "msg": f"Blocks file missing: {blocks_path}", "cost": 0}
        if not os.path.exists(pdf_path):
            return {"status": "error", "msg": f"PDF file missing: {pdf_path}", "cost": 0}
        

        with open(outline_path, "r", encoding="utf-8") as f:
            document_outline = json.load(f)
        with open(blocks_path, "r", encoding="utf-8") as f:
            blocks = json.load(f)
        if isinstance(blocks, list):
            for b in blocks:
                if isinstance(b, dict):
                    b.setdefault("source_doc_id", pdf_name)
                    b.setdefault("source_pdf_name", f"{pdf_name}.pdf")
        
        try:
            pdf_file_ref = pathlib.Path(pdf_path)
            pdf_bytes = pdf_file_ref.read_bytes()
            with _compress_sem:
                pdf_bytes = compress_pdf_with_pillow(pdf_bytes, zoom=0.5, quality=20)
            
            pdf_part = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type='application/pdf',
            )
        except Exception as e:
            return {"status": "error", "msg": f"Failed to read PDF bytes: {e}", "cost": 0}

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_budget=32768)
        )

        all_evidence = []

        for i in range(iterations):
            if max_evidence_chains_per_doc > 0 and len(all_evidence) >= max_evidence_chains_per_doc:
                logger.info(
                    f"[{pdf_name}] Reached evidence cap ({max_evidence_chains_per_doc}), stop iterating."
                )
                break
            prompt_text = build_prompt(PROMPT, document_outline, blocks, all_evidence)


            contents = [pdf_part, prompt_text]

            response = None
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=contents,
                        config=config
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        if "The input token count exceeds" in str(e):
                            return {"status": "error", "msg": f"The input token count exceeds: {e}", "cost": 0}
                        logger.warning(f"[{pdf_name}] Iteration {i+1} API Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"[{pdf_name}] Iteration {i+1} Failed after {max_retries} attempts: {e}")

            if not response:
                continue

            try:

                if response.usage_metadata:
                    cost, in_t, out_t = calculate_cost(response.usage_metadata)
                    total_cost += cost
                    total_in += in_t
                    total_out += out_t

                if not response.candidates:
                    logger.warning(f"[{pdf_name}] Iteration {i+1}: ❌ No candidates returned.")
                    if response.prompt_feedback:
                        logger.warning(f"   -> Prompt Feedback: {response.prompt_feedback}")
                    continue

                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason

                if finish_reason == "SAFETY":
                    logger.warning(f"[{pdf_name}] Iteration {i+1}: 🛡️ Blocked by SAFETY filter.")
                    continue
                
                elif finish_reason == "RECITATION":
                    logger.warning(f"[{pdf_name}] Iteration {i+1}: © Blocked by RECITATION check.")
                    continue

                if not candidate.content or not candidate.content.parts:
                    logger.warning(f"[{pdf_name}] Iteration {i+1}: ⚠️ Content is empty (Reason: {finish_reason}).")
                    continue

                try:
                    response_text = response.text 
                    new_evidence = json.loads(response_text)
                    if isinstance(new_evidence, list):
                        all_evidence.extend(new_evidence)
                        if max_evidence_chains_per_doc > 0 and len(all_evidence) > max_evidence_chains_per_doc:
                            all_evidence = all_evidence[:max_evidence_chains_per_doc]
                        logger.info(f"[{pdf_name}] Iteration {i+1}: Retrieved {len(new_evidence)} packages.")
                    else:
                        logger.warning(f"[{pdf_name}] Iteration {i+1}: Invalid JSON format (not a list).")

                except json.JSONDecodeError as je:
                    logger.error(f"[{pdf_name}] Iteration {i+1}: JSON decode error: {je}")

            except ClientError as e:
                logger.error(f"[{pdf_name}] Iteration {i+1} ClientError: {e.code} - {e.message}")
            except ServerError as e:
                logger.error(f"[{pdf_name}] Iteration {i+1} ServerError: {e.message}")
            except Exception as e:
                logger.error(f"[{pdf_name}] Iteration {i+1} Unexpected error: {e}")

        if all_evidence:
            logger.info(f"[{pdf_name}] Aligning {len(all_evidence)} packages with original blocks...")
            
            all_evidence, aligned_cnt, missing_cnt = align_evidence_with_original_blocks(all_evidence, blocks)
            
            logger.info(f"[{pdf_name}] Alignment complete. {aligned_cnt} elements restored to ground truth.")
            if missing_cnt > 0:
                logger.warning(f"[{pdf_name}] {missing_cnt} elements had IDs not found in original blocks.")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_evidence, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(all_evidence)} items to {output_path}")
        else:
            logger.warning(f"[{pdf_name}] No evidence collected.")
        
        return {
            "status": "success",
            "file": pdf_name,
            "cost": total_cost,
            "input_tokens": total_in,
            "output_tokens": total_out
        }

    except Exception as e:
        logger.error(f"Critical error processing {pdf_name}: {e}")
        return {"status": "error", "msg": str(e), "cost": total_cost}




def main(args):
    outline_dir = None
    blocks_dir = None
    pdf_dir = None 

    if args.data_root:
        outline_dir = os.path.join(args.data_root, "results", "step_1")
        blocks_dir = os.path.join(args.data_root, "results", "step_0")
        pdf_dir = os.path.join("pdfs")

    if args.outline_dir: outline_dir = args.outline_dir
    if args.blocks_dir: blocks_dir = args.blocks_dir
    if args.pdf_dir: pdf_dir = args.pdf_dir

    if not (outline_dir and blocks_dir and pdf_dir):
        logger.error("Paths not fully configured. Provide --data_root OR specific directories.")
        return

    logger.info(f"Using Outline Dir: {outline_dir}")
    logger.info(f"Using Blocks Dir : {blocks_dir}")
    logger.info(f"Using PDF Dir    : {pdf_dir}")

    if not os.path.exists(outline_dir):
        logger.error(f"Outline directory does not exist: {outline_dir}")
        return

    tasks = []
    outline_files = [f for f in os.listdir(outline_dir) if f.endswith(".json")]
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        outline_files = [f for f in outline_files if f"{os.path.splitext(f)[0]}.pdf" in selected_name_set]
        logger.info(f"Selected filter applied: {len(outline_files)} file(s) remain.")

    for f in outline_files:
        pdf_name = os.path.splitext(f)[0]
        
        pdf_file_path = os.path.join(pdf_dir, f"{pdf_name}.pdf")
        
        if not os.path.exists(pdf_file_path):
             logger.warning(f"Skipping {pdf_name}: PDF not found at {pdf_file_path}")
             continue

        tasks.append({
            "pdf_name": pdf_name,
            "outline_path": os.path.join(outline_dir, f),
            "blocks_path": os.path.join(blocks_dir, f"{pdf_name}.json"),
            "pdf_path": pdf_file_path,
        })

    if not tasks:
        logger.warning("No tasks found.")
        return

    logger.info(f"Prepared {len(tasks)} tasks.")
    batch_size = min(len(tasks), args.batch_size)
    
    client = create_genai_client(genai, types)
    
    total_cost = 0.0
    total_in = 0
    total_out = 0
    
    logger.info(f"Starting batch processing (Batch Size: {batch_size})...")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_task = {
            executor.submit(
                process_single_pdf,
                client,
                t["pdf_name"],
                t["outline_path"],
                t["blocks_path"],
                t["pdf_path"], 
                args.output_dir,
                args.iterations,
                args.max_evidence_chains_per_doc,
            ): t for t in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing"):
            res = future.result()
            if "cost" in res:
                total_cost += res["cost"]
                total_in += res.get("input_tokens", 0)
                total_out += res.get("output_tokens", 0)
            
            if res["status"] == "error":
                logger.warning(f"Skipped {res.get('file', 'unknown')}: {res.get('msg')}")

    logger.info("="*30)
    logger.info(f"Total Usage Metric: {total_cost:.6f}")
    
    usage_report_path = os.path.join(args.output_dir, "usage_report_step_2.json")
    os.makedirs(os.path.dirname(usage_report_path), exist_ok=True)
    
    report = {
        "step": "step_2_generate_evidence",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": len(tasks),
        "token_details": {
            "input": total_in,
            "output": total_out
        }
    }
    
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Usage report saved to {usage_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Generate Evidence (PDF Bytes Mode)")
    
    parser.add_argument("--data_root", help="Root directory containing 'gen' and 'pdfs' folders.")
    
    parser.add_argument("--outline_dir", help="Override: Path to outline JSONs")
    parser.add_argument("--blocks_dir", help="Override: Path to blocks JSONs")
    parser.add_argument("--pdf_dir", help="Override: Path to PDF folder")
    
    parser.add_argument("--output_dir", default="my_results", help="Output root base")
    parser.add_argument("--iterations", type=int, default=1, help="Loops per doc")
    parser.add_argument("--max_evidence_chains_per_doc", type=int, default=0, help="Step2 pre-filter: max evidence chains used per document; <=0 means no limit")
    parser.add_argument("--batch_size", type=int, default=3, help="Concurrent threads")
    parser.add_argument("--selected_pdfs", default="", help="Comma-separated PDF names (with or without .pdf). Empty means process all.")
    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    
    if not PROMPT:
        logger.warning("WARNING: PROMPT is empty. Check prompts/step_2_generate_evidence__PROMPT.txt.")
        
    main(args)
