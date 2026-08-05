import os
import json
import argparse
import logging
from tqdm import tqdm
from PIL import Image
from pdf2image import convert_from_path,pdfinfo_from_path

from mineru_vl_utils import MinerUClient
from utils.runtime_config import get_mineru_server_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
import fitz
def pdf_to_png(pdf_path, png_dir, dpi=200, page_batch=10):
    os.makedirs(png_dir, exist_ok=True)
    mat = fitz.Matrix(dpi/72, dpi/72)
    png_paths = []

    doc = fitz.open(pdf_path)
    total = len(doc)

    for start in tqdm(range(0, total, page_batch), desc="Converting pages", unit="batch", leave=False):
        end = min(start + page_batch, total)
        for i in range(start, end):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_path = os.path.join(png_dir, f"page_{i+1:04d}.png")
            pix.save(png_path)
            png_paths.append(png_path)
            pix = None
            page = None

    doc.close()
    return png_paths


def process_extracted_blocks(extracted_blocks, page_id):
    processed = []
    reading_order = 1

    for block_group in extracted_blocks:
        for block in block_group:
            x1, y1, x2, y2 = block["bbox"]

            new_block = {
                "type": block.get("type"),
                "content": block.get("content", ""),
                "bbox": [
                    int(y1 * 1000),
                    int(x1 * 1000),
                    int(y2 * 1000),
                    int(x2 * 1000),
                ],
                "angle": block.get("angle", ""),
                "page_id": page_id,
                "element_idx": f"{page_id}-{reading_order}",
            }

            processed.append(new_block)
            reading_order += 1

    return processed


def process_single_pdf(pdf_path, out_dir, server_url):
    try:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        png_save_dir = os.path.join(out_dir, "results", "png", pdf_name)
        json_save_dir = os.path.join(out_dir, "results", "step_0")
        json_save_path = os.path.join(json_save_dir, f"{pdf_name}.json")
        if os.path.exists(json_save_path):
            logger.info(f"Already processed, skipping: {pdf_name}")
            return
            
        logger.info(f"Processing: {pdf_path}")
        logger.info(f" -> JSON: {json_save_path}")
        info = pdfinfo_from_path(pdf_path)
        

        if info["Pages"] > 250 or int(os.path.getsize(pdf_path)) > 50 * 1024 * 1024:
            logger.warning(f"Skipping (>250 pages, {info['Pages']} pages): {pdf_name}")
            return

        try:
            png_paths = pdf_to_png(pdf_path, png_save_dir)
        except Exception as e:
            logger.error(f"PDF convert failed for {pdf_name}: {e}")
            return

        client = MinerUClient(
            backend="http-client",
            server_url=server_url,
        )
        results = []

        for page_id, png_path in enumerate(tqdm(png_paths, desc=f"Parsing {pdf_name}", unit="page", leave=False), start=1):
            try:
                image = Image.open(png_path).convert("RGB")
                extracted_blocks = client.batch_two_step_extract([image])
                
                processed_blocks = process_extracted_blocks(
                    extracted_blocks,
                    page_id=page_id,
                )
                results.extend(processed_blocks)
            except Exception as e:
                logger.error(f"Error parsing page {page_id} of {pdf_name}: {e}")

        os.makedirs(json_save_dir, exist_ok=True)
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Done: {pdf_name}")

    except Exception as e:
        logger.error(f"Critical error processing file {pdf_path}: {e}")


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


def main(args):
    target_files = []
    
    if os.path.isfile(args.pdf_path):
        if args.pdf_path.lower().endswith('.pdf'):
            target_files.append(args.pdf_path)
        else:
            logger.error("The file provided is not a PDF.")
            return
    elif os.path.isdir(args.pdf_path):
        logger.info(f"Scanning directory: {args.pdf_path}")
        for root, dirs, files in os.walk(args.pdf_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    target_files.append(full_path)
    else:
        logger.error(f"Path not found: {args.pdf_path}")
        return

    total_files = len(target_files)
    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        target_files = [
            p for p in target_files
            if os.path.basename(p) in selected_name_set
        ]
        logger.info(f"Selected filter applied: {len(target_files)} PDF file(s) remain.")
    total_files = len(target_files)
    logger.info(f"Found {total_files} PDF file(s) to process.")

    if total_files == 0:
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_single_pdf, pdf_file, args.out_dir, args.server_url): pdf_file
            for pdf_file in target_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Total Progress", unit="file"):
            pdf_file = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Failed {pdf_file}: {e}")

    logger.info("All tasks completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch Parse PDFs with MinerU"
    )

    parser.add_argument(
        "--pdf_path",
        required=True,
        help="Path to a single PDF file OR a directory containing PDFs",
    )
    parser.add_argument(
        "--out_dir",
        default="output",
        help="Root directory for output (default: ./output)",
    )
    parser.add_argument(
        "--server_url",
        default=get_mineru_server_url(),
        help="MinerU server URL",
    )
    parser.add_argument(
        "--selected_pdfs",
        default="",
        help="Comma-separated PDF names (with or without .pdf). Empty means process all under pdf_path.",
    )

    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    main(args)
