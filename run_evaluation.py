import json
import logging
from pathlib import Path

from bootstrap import parse_args, initialize_components, save_run_config
from src.utils.common import read_jsonl


logger = logging.getLogger(__name__)


def convert_json_to_jsonl(json_path: Path, jsonl_path: Path) -> None:
    """
    Convert large JSON arrays to JSONL with streaming IO.
    """
    logger.info("Converting JSON to JSONL. from=%s to=%s", json_path, jsonl_path)
    count = 0
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("r", encoding="utf-8") as f_in, jsonl_path.open("w", encoding="utf-8") as f_out:
        try:
            import ijson

            iterator = ijson.items(f_in, "item")
        except ImportError:
            logger.warning("ijson is not installed. Falling back to in-memory JSON conversion.")
            iterator = json.load(f_in)

        for item in iterator:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    logger.info("JSONL conversion complete. records=%d", count)


def _resolve_input_file(args) -> Path | None:
    """Resolve evaluation input, preferring JSONL and converting legacy JSON when needed."""
    if args.evaluation_input is None:
        if args.evaluation_task == "retrieval":
            retrieval_json = Path(args.output_dir) / "retrieval_results.json"
            generation_json = Path(args.output_dir) / "generation_results.json"
            base_input_file = retrieval_json if retrieval_json.exists() else generation_json
        else:
            base_input_file = Path(args.output_dir) / "generation_results.json"
    else:
        input_path = Path(args.evaluation_input)
        base_input_file = input_path if input_path.is_absolute() else Path(args.output_dir) / input_path

    jsonl_file = base_input_file.with_suffix(".jsonl")

    if jsonl_file.exists():
        return jsonl_file

    if base_input_file.exists():
        logger.info("JSONL not found; converting legacy JSON file: %s", base_input_file)
        convert_json_to_jsonl(base_input_file, jsonl_file)
        return jsonl_file

    logger.error(
        "Neither JSONL nor JSON input files were found for output_dir=%s",
        args.output_dir,
    )
    return None


def _match_results_to_samples(loader, jsonl_file: Path):
    """Match JSONL records to loader samples by qid."""
    logger.info("Loading results line-by-line from: %s", jsonl_file)

    samples_map = {str(sample.qid): sample for sample in loader.samples}
    valid_samples = []
    matched_count = 0
    original_count = len(loader.samples)

    for res in read_jsonl(jsonl_file):
        s_qid = str(res.get("qid"))
        if s_qid not in samples_map:
            continue

        sample = samples_map[s_qid]
        if sample.extra_info is None:
            sample.extra_info = {}

        if "retrieved_elements" in res:
            sample.extra_info["retrieved_elements"] = res["retrieved_elements"]

        if "model_answer" in res:
            sample.extra_info["final_answer"] = res["model_answer"]
        elif "final_answer" in res:
            sample.extra_info["final_answer"] = res["final_answer"]

        sample.extra_info["prompt_tokens"] = res.get("prompt_tokens", 0)
        sample.extra_info["completion_tokens"] = res.get("completion_tokens", 0)

        matched_count += 1
        valid_samples.append(sample)
        del samples_map[s_qid]

    logger.info(
        "Mapped results for %d/%d samples (discarded=%d unmatched).",
        matched_count,
        original_count,
        original_count - matched_count,
    )
    return valid_samples


def main():
    args = parse_args()
    save_run_config(args, "evaluation")
    logger.info(
        "Starting Evaluation stage for benchmark=%s (task=%s).",
        args.benchmark,
        args.evaluation_task,
    )

    _, loader = initialize_components(args, init_retriever=False, init_generator=False)
    from src.utils.llm import create_llm_caller

    loader.llm_caller = create_llm_caller(
        base_url=args.evaluator_api_base,
        api_key=args.evaluator_api_key,
        model_name=args.evaluator_model_name,
    )

    jsonl_file = _resolve_input_file(args)
    if jsonl_file is None:
        return

    loader.samples = _match_results_to_samples(loader, jsonl_file)

    final_metrics = {}

    # Task: Retrieval
    if args.evaluation_task in ("retrieval", "all"):
        try:
            logger.info("--- Retrieval Metrics ---")
            r_metrics = loader.evaluate_retrieval()
            logger.info(json.dumps(r_metrics, indent=2))
            final_metrics.update(r_metrics)
        except Exception:
            logger.exception("Retrieval evaluation failed.")

    # Task: Generation
    if args.evaluation_task in ("generation", "all"):
        has_answers = any("final_answer" in s.extra_info for s in loader.samples)
        if has_answers:
            try:
                logger.info("--- Generation Metrics ---")
                g_metrics = loader.evaluate_generation(num_threads=args.num_threads)
                logger.info(json.dumps(g_metrics, indent=2))
                final_metrics.update(g_metrics)
            except Exception:
                logger.exception("Generation evaluation failed.")
        elif args.evaluation_task == "generation":
            logger.warning("No generation answers found in input file. Skipping generation eval.")

    # Save Metrics Report
    output_path = Path(args.output_dir) / f"evaluation_metrics_{args.evaluation_task}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    logger.info("All metrics saved to %s", output_path)

    # Save Bad Cases
    logger.info("--- Saving Bad Cases for Analysis ---")
    if hasattr(loader, "save_bad_cases"):
        loader.save_bad_cases(args.output_dir, args.evaluation_task)
    else:
        logger.warning("Loader does not support saving bad cases.")


if __name__ == "__main__":
    main()
