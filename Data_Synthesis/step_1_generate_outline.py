import argparse
import json
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "step_1_generate_outline",
    ["PROMPT"],
    logger=logger,
)
_PROMPTS = apply_prompt_overrides(
    "step_1_generate_outline",
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
    
    total_cost = 0.0
    return total_cost, input_tokens, output_tokens


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

def process_single_file(client, file_path, output_dir, model_name):
    """Function description."""
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(output_dir, "results", "step_1")
    output_path = os.path.join(save_dir, f"{file_name}.json")
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(output_path):
        logger.info(f"⏭️ Skipping {file_name}: Output already exists.")
        return {
            "file": file_name,
            "status": "skipped",
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            blocks = json.load(f)

        title_blocks = [b for b in blocks if b.get("type") == "title"]

        if not title_blocks:
            logger.warning(f"Skipping {file_name}: No title blocks found.")
            return {
                "file": file_name,
                "status": "skipped",
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0
            }

        if not PROMPT:
            raise ValueError("PROMPT is empty. Check prompts/step_1_generate_outline__PROMPT.txt")
            

        formatted_prompt = PROMPT.format(
            title_blocks=json.dumps(title_blocks, ensure_ascii=False, indent=2)
        )


        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=1.0,
            thinking_config=types.ThinkingConfig(
                thinking_budget=32768
            )
        )

        response = None
        max_retries = 3
        retry_delay = 30
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[formatted_prompt],
                    config=config
                )
                break
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(f"[{file_name}] API call failed (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"[{file_name}] API call failed after {max_retries} attempts.")

        if not response:
            raise last_exception or Exception("API call failed completely")

        cost, in_tokens, out_tokens = calculate_cost(response.usage_metadata)

        try:
            outline = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error in {file_name}: {response.text[:100]}...")
            return {
                "file": file_name,
                "status": "error",
                "error_msg": "JSONDecodeError",
                "cost": cost,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)

        return {
            "file": file_name,
            "status": "success",
            "cost": cost,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens
        }

    except Exception as e:
        logger.error(f"Critical error processing {file_name}: {str(e)}")
        return {
            "file": file_name,
            "status": "error",
            "error_msg": str(e),
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0
        }


def main(args):
    input_files = []
    if os.path.isfile(args.input_path):
        input_files.append(args.input_path)
    elif os.path.isdir(args.input_path):
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith(".json"):
                    input_files.append(os.path.join(root, file))
    else:
        logger.error(f"Input path not found: {args.input_path}")
        return

    if not input_files:
        logger.warning("No JSON files found to process.")
        return

    selected_name_set = _normalize_selected_names(args.selected_pdfs)
    if selected_name_set:
        input_files = [
            p for p in input_files
            if f"{os.path.splitext(os.path.basename(p))[0]}.pdf" in selected_name_set
        ]
        logger.info(f"Selected filter applied: {len(input_files)} file(s) remain.")
        if not input_files:
            logger.warning("No JSON files remain after selected_pdfs filter.")
            return
        
    batch_size = min(len(input_files), args.batch_size)

    logger.info(f"Found {len(input_files)} files. Starting batch processing (Batch Size: {batch_size})...")

    client = create_genai_client(genai, types)
    model_to_use = get_genai_model()

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    results_summary = []

    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_file = {
            executor.submit(
                process_single_file, 
                client, 
                f_path, 
                args.output_dir, 
                model_to_use
            ): f_path for f_path in input_files
        }

        for future in tqdm(as_completed(future_to_file), total=len(input_files), desc="Generating Outlines"):
            result = future.result()
            
            total_cost += result["cost"]
            total_input_tokens += result["input_tokens"]
            total_output_tokens += result["output_tokens"]
            
            if result["status"] == "success":
                results_summary.append(result)
            else:
                logger.warning(f"Task failed/skipped: {result['file']} - {result.get('error_msg', 'unknown')}")

    logger.info("="*30)
    logger.info(" Processing Complete ")
    logger.info("="*30)
    logger.info(f"Total Files  : {len(input_files)}")
    logger.info(f"Total Input  : {total_input_tokens} tokens")
    logger.info(f"Total Output : {total_output_tokens} tokens")
    logger.info(f"Total Usage Metric: {total_cost:.6f}")

    usage_report_path = os.path.join(args.output_dir, "usage_report_step_1.json")
    cost_data = {
        "step": "step_1_generate_outline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": len(input_files),
        "token_details": {
            "input": total_input_tokens,
            "output": total_output_tokens
        }
    }
    
    with open(usage_report_path, "w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2)
    
    logger.info(f"Usage report saved to {usage_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Generate Outline with Gemini")
    
    parser.add_argument(
        "--input_path",
        required=True,
        help="Input JSON file or Directory containing JSONs (from Step 0)"
    )
    parser.add_argument(
        "--output_dir",
        default="my_results",
        help="Root directory for output (default: my_results)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of concurrent API calls (default: 5)"
    )
    parser.add_argument(
        "--selected_pdfs",
        default="",
        help="Comma-separated PDF names (with or without .pdf). Empty means process all.",
    )

    args = parser.parse_args()
    args.selected_pdfs = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    
    if not PROMPT.strip():
        logger.warning("WARNING: PROMPT is empty. Check prompts/step_1_generate_outline__PROMPT.txt.")

    main(args)
