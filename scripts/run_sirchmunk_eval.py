"""
Evaluate sirchmunk AgenticSearch on MMLongBench-Doc benchmark.

Usage:
    python run_sirchmunk_eval.py \
        --data_root /path/to/MMLongBench-Doc \
        --output_dir ./outputs/sirchmunk_eval \
        --api_key your-api-key \
        --base_url http://your-base-url/v1 \
        --model_name your-model-name \
        --mode FAST \
        --num_threads 4 \
        --limit 10

    # With LLM-based answer extraction (recommended for fair comparison):
    python run_sirchmunk_eval.py \
        --data_root /path/to/MMLongBench-Doc \
        --output_dir ./outputs/sirchmunk_eval \
        --api_key your-api-key \
        --base_url http://your-base-url/v1 \
        --model_name your-model-name \
        --use_extractor \
        --extractor_base_url http://extractor-url/v1 \
        --extractor_api_key your-extractor-api-key \
        --extractor_model_name qwen3-max

python run_sirchmunk_eval.py \
 --data_root /path/to/MMLongBench-Doc \
 --output_dir ./outputs/sirchmunk_eval \
 --api_key your-api-key \
 --base_url http://your-base-url/v1 \
 --model_name qwen3.5-plus \
 --mode FAST \
 --use_extractor \
 --extractor_base_url http://your-extractor-url/v1 \
 --extractor_api_key your-extractor-api-key \
 --extractor_model_name qwen3-max \
 --num_threads 4 \
 --limit 10
"""

import argparse
import asyncio
import ast
import collections
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from math import isclose
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring functions (inlined from MMLongLoader.py to avoid heavy dep chain)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage=False, is_close=False):
    def get_precision(gt_ans):
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    try:
        reference = float(str(reference).strip().rstrip("%").strip())
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s = s[:-4].strip()
    if s.endswith("miles"):
        s = s[:-5].strip()
    if s.endswith("million"):
        s = s[:-7].strip()
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    if "https://" in s:
        return True
    if s.endswith(".py") or s.endswith("ipynb"):
        return True
    if s.startswith("page"):
        return True
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        return True
    if "a.m." in s or "p.m." in s:
        return True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        return True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        return True
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        return True
    return False


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type):
    if answer_type in ["Int", "Integer"]:
        try:
            gt, pred = int(gt), int(float(pred))
        except Exception:
            pred = ""
        score = (gt == pred)
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except Exception:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "String", "None", "Not answerable", "Fail to answer"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        # List handling
        if isinstance(gt, str) and gt.startswith("["):
            try:
                gt = eval(gt)
            except Exception:
                pass
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            try:
                pred = eval(pred)
            except Exception:
                pass
        if not isinstance(pred, list):
            pred = [pred]
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            if len(gt) > 0 and (isfloat(gt[0]) or is_exact_match(gt[0])):
                score = ("-".join(gt) == "-".join(pred))
            else:
                if len(gt) == 0:
                    score = 1.0
                else:
                    score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])
    return float(score)


MMLONG_EXTRACT_PROMPT_TEMPLATE = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List. If you find the analysis the question can not be answered from the given documents, type "Not answerable". Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```

Please read the following example, then extract the answer from the model response and type it at the end of the prompt.

---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\\n\\n1. Is the service safe?\\n2. Is the service effective?\\n3. Is the service caring?\\n4. Is the service responsive?\\n5. Is the service well-led?
Extracted answer: ['Is the servife safe?', 'Is the service effective', 'Is the serve caring?', 'Is the service responsive?', 'Is the service well-led?']
Answer format: List

---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities) Regulations in total. Here are the specifics:\\n\\n1. Regulation 13: Safeguarding service users from abuse and improper treatment\\n2. Regulation 12: Safe care and treatment\\n3. Regulation 18: Staffing\\n4. Regulation 11: Need for consent\\n5. Regulation 10: Dignity and respect\\n6. Regulation 9: Person-centred care\\n7. Regulation 17: Good governance\\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\\n9. Regulation 18: Failure to maintain an accurate and up-to-date care plan\\n10. Regulation 11: Failure to implement the Mental Capacity Act 2005 code of practice effectively\\n\\nThese breaches involve issues concerning staffing, safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer

---
Question: According to the survey that is the percentage of Chinese who are paying more or about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying more or about the same attention to politics after Trump's election. The report focuses primarily on American demographics and does not include specific details about the Chinese population in relation to this question. If you need information about a different demographic or a summary of the findings from the American demographic, I can certainly help with that!
Extracted answer: Not answerable
Answer format: String

---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be able to help you with your question.
Extracted answer: Fail to answer
Answer format: String

---
Question: {question}
Analysis: {analysis}
"""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    qid: str
    query: str
    data_source: str  # PDF path
    gold_answer: str
    gold_pages: List[str] = field(default_factory=list)
    extra_info: Dict = field(default_factory=dict)


def load_mmlong_samples(data_root: str) -> List[Sample]:
    """Load MMLongBench-Doc samples from samples.json."""
    json_path = os.path.join(data_root, "data", "samples.json")
    doc_dir = os.path.join(data_root, "data", "documents")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"samples.json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for idx, item in enumerate(data):
        qid = str(idx)
        doc_filename = item.get("doc_id", "")
        doc_path = os.path.join(doc_dir, doc_filename) if doc_filename else ""

        evidence_pages_str = item.get("evidence_pages", "[]")
        gold_pages = []
        try:
            pages_list = ast.literal_eval(evidence_pages_str)
            if isinstance(pages_list, list):
                gold_pages = [f"page_{p}.png" for p in pages_list]
        except Exception:
            pass

        extra_info = {
            "doc_type": item.get("doc_type"),
            "evidence_sources": item.get("evidence_sources"),
            "answer_format": item.get("answer_format"),
        }

        samples.append(Sample(
            qid=qid,
            query=item.get("question", ""),
            data_source=doc_path,
            gold_answer=item.get("answer", ""),
            gold_pages=gold_pages,
            extra_info=extra_info,
        ))

    logger.info(f"Loaded {len(samples)} MMLongBench-Doc samples.")
    return samples


# ---------------------------------------------------------------------------
# LLM answer extraction
# ---------------------------------------------------------------------------

class LLMCaller:
    """Lightweight LLM caller using OpenAI-compatible API."""
    def __init__(self, base_url: str, api_key: str, model_name: str, max_retries: int = 3):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, temperature=0.0,
                )
                return resp.choices[0].message.content or ""
            except Exception:
                logger.exception(f"LLM call attempt {attempt+1} failed")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
        return ""


def extract_answer_with_llm(question: str, raw_response: str, llm_caller) -> Dict:
    """Use LLM to extract a standardized answer from free-form response."""
    if not raw_response:
        return {"extracted_answer": "", "answer_format": "String"}

    prompt = MMLONG_EXTRACT_PROMPT_TEMPLATE.format(question=question, analysis=raw_response)
    try:
        llm_output = llm_caller(prompt)
        extracted_answer = ""
        answer_format = "String"

        ans_match = re.search(r"Extracted answer:\s*(.*)", llm_output, re.IGNORECASE)
        fmt_match = re.search(r"Answer format:\s*(.*)", llm_output, re.IGNORECASE)

        if ans_match:
            extracted_answer = ans_match.group(1).strip()
        if fmt_match:
            answer_format = fmt_match.group(1).strip()

        if extracted_answer.startswith("'") and extracted_answer.endswith("'"):
            extracted_answer = extracted_answer[1:-1]

        return {"extracted_answer": extracted_answer, "answer_format": answer_format}
    except Exception as e:
        logger.exception(f"Error during LLM extraction: {e}")
        return {"extracted_answer": raw_response, "answer_format": "String"}


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def get_categories(sample: Sample) -> List[str]:
    """Parse evidence_sources into a list of category names."""
    ev_sources_str = sample.extra_info.get("evidence_sources", "[]")
    try:
        ev_sources = ast.literal_eval(str(ev_sources_str))
        if not isinstance(ev_sources, list):
            ev_sources = [str(ev_sources)]
    except Exception:
        ev_sources = ["Unknown"]
    if not ev_sources:
        ev_sources = ["Not Answerable"]
    return ev_sources


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def run_sirchmunk_search(query: str, pdf_path: str, searcher, mode: str) -> str:
    """Run sirchmunk search (wraps async call in a new event loop)."""
    async def _search():
        return await searcher.search(
            query=query,
            paths=[pdf_path],
            mode=mode,
        )

    # Create a new event loop per call to avoid conflicts in threaded execution
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_search())
    finally:
        loop.close()


def process_sample(sample: Sample, searcher, mode: str, extractor_caller, cache_dir: str) -> Dict:
    """Process a single sample: search with sirchmunk, optionally extract answer, score it."""
    qid = sample.qid

    # Check cache
    cache_path = os.path.join(cache_dir, f"{qid}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("raw_answer") and cached.get("score") is not None:
                return cached
        except Exception:
            pass

    pdf_path = sample.data_source
    if not os.path.exists(pdf_path):
        logger.warning(f"[qid={qid}] PDF not found: {pdf_path}")
        return {
            "qid": qid, "query": sample.query, "gold_answer": sample.gold_answer,
            "raw_answer": "", "final_answer": "", "score": 0.0,
            "error": f"PDF not found: {pdf_path}",
        }

    # Run sirchmunk search
    raw_answer = ""
    error = None
    t0 = time.time()
    try:
        raw_answer = run_sirchmunk_search(sample.query, pdf_path, searcher, mode)
    except Exception as e:
        logger.exception(f"[qid={qid}] sirchmunk search failed: {e}")
        error = str(e)
    elapsed = time.time() - t0

    # Optional: extract answer with LLM
    final_answer = raw_answer
    if extractor_caller and raw_answer:
        extract_res = extract_answer_with_llm(sample.query, raw_answer, extractor_caller)
        final_answer = extract_res["extracted_answer"]

    # Score
    gold_answer = sample.gold_answer
    gold_format = sample.extra_info.get("answer_format", "String")
    score = 0.0
    if gold_answer:
        score = eval_score(gold_answer, final_answer, gold_format)

    result = {
        "qid": qid,
        "query": sample.query,
        "gold_answer": gold_answer,
        "answer_format": gold_format,
        "evidence_sources": sample.extra_info.get("evidence_sources"),
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "score": float(score),
        "elapsed_seconds": round(elapsed, 2),
    }
    if error:
        result["error"] = error

    # Write cache
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.warning(f"[qid={qid}] Failed to write cache")

    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute overall and per-category accuracy from results."""
    total_score = 0.0
    total_count = 0
    category_scores = collections.defaultdict(float)
    category_counts = collections.defaultdict(int)

    for r in results:
        score = r.get("score", 0.0)
        total_score += score
        total_count += 1

        ev_str = r.get("evidence_sources", "[]")
        try:
            ev_sources = ast.literal_eval(str(ev_str))
            if not isinstance(ev_sources, list):
                ev_sources = [str(ev_sources)]
        except Exception:
            ev_sources = ["Unknown"]
        if not ev_sources:
            ev_sources = ["Not Answerable"]

        for src in ev_sources:
            category_scores[src] += score
            category_counts[src] += 1

    metrics = {}
    if total_count > 0:
        metrics["overall_acc"] = total_score / total_count
        metrics["total_samples"] = total_count
        metrics["total_score"] = total_score

    cat_metrics = {}
    for src in sorted(category_counts.keys()):
        cnt = category_counts[src]
        avg = category_scores[src] / cnt if cnt > 0 else 0.0
        cat_metrics[src] = {"acc": avg, "count": cnt, "score_sum": category_scores[src]}

    metrics["per_category"] = cat_metrics
    return metrics


def print_metrics(metrics: Dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 70)
    print(f"  Overall Accuracy: {metrics.get('overall_acc', 0.0):.4f}")
    print(f"  Total Samples:    {metrics.get('total_samples', 0)}")
    print(f"  Total Score:      {metrics.get('total_score', 0.0):.2f}")
    print("=" * 70)

    cat_metrics = metrics.get("per_category", {})
    if cat_metrics:
        print(f"\n  {'Category':<25} {'Acc':>8} {'Count':>8} {'Score':>8}")
        print("  " + "-" * 51)
        for src, info in sorted(cat_metrics.items(), key=lambda x: -x[1]["acc"]):
            print(f"  {src:<25} {info['acc']:>8.4f} {info['count']:>8d} {info['score_sum']:>8.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate sirchmunk on MMLongBench-Doc")
    # Data
    parser.add_argument("--data_root", type=str,
                        default="/path/to/MMLongBench-Doc",
                        help="MMLongBench-Doc dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/sirchmunk_eval",
                        help="Output directory for results")

    # Sirchmunk LLM config
    parser.add_argument("--api_key", type=str, default="sk-placeholder")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model_name", type=str,
                        default="/path/to/Qwen3-VL-32B-Thinking")
    parser.add_argument("--mode", type=str, default="FAST", choices=["FAST", "DEEP"],
                        help="sirchmunk search mode")

    # Answer extraction LLM (optional, for fair comparison with existing pipeline)
    parser.add_argument("--use_extractor", action="store_true",
                        help="Use LLM to extract standardized answer from sirchmunk response")
    parser.add_argument("--extractor_base_url", type=str, default=None)
    parser.add_argument("--extractor_api_key", type=str, default=None)
    parser.add_argument("--extractor_model_name", type=str, default=None)

    # Filtering
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples to process")
    parser.add_argument("--filter", type=str, default=None, help="JSON file with qid whitelist")
    parser.add_argument("--category_filter", type=str, default=None,
                        help="Comma-separated category filter (e.g., 'Table,Chart')")

    # Execution
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of parallel threads")
    parser.add_argument("--logging_level", type=str, default="INFO")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.logging_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load samples
    samples = load_mmlong_samples(args.data_root)

    # Apply qid filter
    if args.filter:
        filter_path = args.filter
        if os.path.exists(filter_path):
            with open(filter_path, "r") as f:
                filter_data = json.load(f)
            if isinstance(filter_data, list):
                if filter_data and isinstance(filter_data[0], dict):
                    qid_set = {str(item.get("qid", "")) for item in filter_data}
                else:
                    qid_set = {str(q) for q in filter_data}
            elif isinstance(filter_data, dict):
                for key in ("qids", "items"):
                    if key in filter_data:
                        raw = filter_data[key]
                        if raw and isinstance(raw[0], dict):
                            qid_set = {str(item.get("qid", "")) for item in raw}
                        else:
                            qid_set = {str(q) for q in raw}
                        break
                else:
                    qid_set = set()
            else:
                qid_set = set()
            samples = [s for s in samples if s.qid in qid_set]
            logger.info(f"After qid filter: {len(samples)} samples")

    # Apply category filter
    if args.category_filter:
        targets = {t.split("-")[0].split(" ")[0].lower() for t in args.category_filter.split(",")}

        def matches(sample):
            for c in get_categories(sample):
                if str(c).split("-")[0].split(" ")[0].lower() in targets:
                    return True
            return False

        samples = [s for s in samples if matches(s)]
        logger.info(f"After category filter: {len(samples)} samples")

    # Apply limit (last, consistent with existing pipeline)
    if args.limit:
        samples = samples[:args.limit]
        logger.info(f"After limit: {len(samples)} samples")

    if not samples:
        logger.error("No samples to process. Exiting.")
        return

    # Initialize sirchmunk
    from sirchmunk import AgenticSearch
    from sirchmunk.llm import OpenAIChat

    llm = OpenAIChat(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model_name,
    )
    searcher = AgenticSearch(llm=llm, verbose=False)

    # Optional extractor
    extractor_caller = None
    if args.use_extractor:
        extractor_caller = LLMCaller(
            base_url=args.extractor_base_url or args.base_url,
            api_key=args.extractor_api_key or args.api_key,
            model_name=args.extractor_model_name or args.model_name,
        )

    # Setup output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "cache_sirchmunk")
    os.makedirs(cache_dir, exist_ok=True)

    # Save config (redact keys)
    config_to_save = vars(args).copy()
    for key in ("api_key", "extractor_api_key"):
        if config_to_save.get(key):
            config_to_save[key] = "***REDACTED***"
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    # Process samples
    logger.info(f"Processing {len(samples)} samples with mode={args.mode}, threads={args.num_threads}")
    results = []

    if args.num_threads > 1:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(process_sample, s, searcher, args.mode, extractor_caller, cache_dir): s
                for s in samples
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="sirchmunk eval"):
                try:
                    results.append(future.result())
                except Exception as e:
                    sample = futures[future]
                    logger.exception(f"[qid={sample.qid}] Thread failed: {e}")
                    results.append({
                        "qid": sample.qid, "query": sample.query,
                        "gold_answer": sample.gold_answer,
                        "raw_answer": "", "final_answer": "", "score": 0.0,
                        "error": str(e),
                    })
    else:
        for sample in tqdm(samples, desc="sirchmunk eval"):
            results.append(process_sample(sample, searcher, args.mode, extractor_caller, cache_dir))

    # Sort by qid
    results.sort(key=lambda r: int(r["qid"]) if r["qid"].isdigit() else r["qid"])

    # Save full results
    results_path = os.path.join(output_dir, "sirchmunk_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Compute and save metrics
    metrics = compute_metrics(results)
    metrics_path = os.path.join(output_dir, "sirchmunk_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Print metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()
