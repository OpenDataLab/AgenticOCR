import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List

from bootstrap import parse_args, initialize_components, save_run_config, filter_items_by_samples
from src.loaders.base_loader import PageElement
from src.utils.common import safe_qid, sort_by_qid, read_cache, write_cache, write_jsonl, run_parallel


logger = logging.getLogger(__name__)


def _is_valid_cached_answer(cached_item: Dict[str, Any]) -> bool:
    """Only reuse cache entries with non-error model answers."""
    answer = str(cached_item.get("model_answer", ""))
    return bool(answer) and "Error during generation" not in answer and not answer.startswith("Error")


def _deserialize_elements(raw_elements: Iterable[Dict[str, Any]]) -> List[PageElement]:
    """Deserialize dictionaries into PageElement objects."""
    valid_keys = PageElement.__annotations__.keys()
    elements: List[PageElement] = []
    for el in raw_elements:
        if not isinstance(el, dict):
            continue
        elements.append(PageElement(**{k: v for k, v in el.items() if k in valid_keys}))
    return elements


def _resolve_generation_input_file(args) -> Path:
    if args.generation_input is None:
        return Path(args.output_dir) / "retrieval_results.json"

    input_path = Path(args.generation_input)
    if input_path.is_absolute():
        return input_path
    return Path(args.output_dir) / input_path


def _load_generation_items(retrieval_file: Path) -> List[Dict[str, Any]]:
    try:
        with retrieval_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in retrieval input: {retrieval_file}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Retrieval input must be a JSON list: {retrieval_file}")
    return data


def process_single_sample_generation(item: Dict[str, Any], agent, cache_dir: Path) -> Dict[str, Any]:
    """Generate one answer with cache support."""
    result_item = dict(item)
    qid = str(result_item.get("qid", ""))
    query = result_item.get("query", "")
    if not qid or not query:
        logger.warning("Skipping invalid generation item missing qid/query: %s", result_item)
        result_item["model_answer"] = "Error during generation."
        result_item["messages"] = []
        result_item["prompt_tokens"] = 0
        result_item["completion_tokens"] = 0
        return result_item

    cache_path = cache_dir / f"{safe_qid(qid)}.json"
    cached = read_cache(cache_path)
    if cached is not None and _is_valid_cached_answer(cached):
        return cached

    retrieved_elements = _deserialize_elements(result_item.get("retrieved_elements", []))

    try:
        gen_output = agent.generate(query, retrieved_elements)
        result_item["model_answer"] = gen_output["model_answer"]
        result_item["messages"] = gen_output["messages"]
        result_item["prompt_tokens"] = gen_output.get("prompt_tokens", 0)
        result_item["completion_tokens"] = gen_output.get("completion_tokens", 0)
    except Exception:
        logger.exception("Error generating for qid=%s", qid)
        result_item["model_answer"] = "Error during generation."
        result_item["messages"] = []
        result_item["prompt_tokens"] = 0
        result_item["completion_tokens"] = 0

    write_cache(cache_path, result_item)
    return result_item


def main():
    args = parse_args()
    save_run_config(args, "generation")
    logger.info(
        "Starting Generation stage for benchmark=%s (threads=%s).",
        args.benchmark,
        args.num_threads,
    )

    agent, loader = initialize_components(args, init_retriever=False, init_generator=True)

    retrieval_file = _resolve_generation_input_file(args)
    if not retrieval_file.exists():
        logger.error(
            "Retrieval file not found at %s. Run run_retrieval.py first.",
            retrieval_file,
        )
        raise SystemExit(1)
    try:
        data_items = _load_generation_items(retrieval_file)
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc

    # loader.samples  initialize_components() 
    # (--filter / --category_filter / --limit)
    data_items = filter_items_by_samples(data_items, loader.samples)

    cache_dir = Path(args.output_dir) / "cache_generation_results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", cache_dir)

    logger.info("Generating answers for %d samples.", len(data_items))

    worker = partial(process_single_sample_generation, agent=agent, cache_dir=cache_dir)
    generation_results = run_parallel(
        fn=worker,
        items=data_items,
        num_threads=args.num_threads,
        desc="Generating",
        get_id=lambda x: x["qid"],
    )

    sort_by_qid(generation_results)

    output_file = Path(args.output_dir) / "generation_results.jsonl"
    write_jsonl(output_file, generation_results)
    logger.info("Generation complete. Results saved to %s", output_file)


if __name__ == "__main__":
    main()
