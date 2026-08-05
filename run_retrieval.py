import json
import logging
from functools import partial
from pathlib import Path

from bootstrap import parse_args, initialize_components, save_run_config
from src.utils.common import safe_qid, sort_by_qid, read_cache, write_cache, run_parallel


logger = logging.getLogger(__name__)


def process_single_sample_retrieval(sample, agent, cache_dir: Path):
    """Retrieve a single sample with cache support."""
    qid = str(sample.qid)
    cache_path = cache_dir / f"{safe_qid(qid)}.json"

    cached = read_cache(cache_path)
    if cached is not None:
        return cached

    try:
        elements = agent.retrieve(sample)
        elements_data = [el.to_dict() if hasattr(el, "to_dict") else el for el in elements]
    except Exception:
        logger.exception("Error retrieving sample %s", qid)
        elements_data = []

    result_item = {
        "qid": sample.qid,
        "query": sample.query,
        "gold_answer": sample.gold_answer,
        "data_source": sample.data_source,
        "gold_pages": getattr(sample, "gold_pages", []),
        "retrieved_elements": elements_data,
    }

    write_cache(cache_path, result_item)
    return result_item


def process_single_sample_all_pages(sample, loader, cache_dir: Path):
    """Return all document pages as full-page PageElements (no reranking/extraction)."""
    qid = str(sample.qid)
    cache_path = cache_dir / f"{safe_qid(qid)}.json"

    cached = read_cache(cache_path)
    if cached is not None:
        return cached

    elements_data = []
    try:
        data_source = sample.data_source
        if data_source:
            if data_source.lower().endswith((".png", ".jpg", ".jpeg")):
                # Single image — return it directly as one page
                elements_data.append({
                    "bbox": [0, 0, 1000, 1000],
                    "type": "page_image",
                    "content": "",
                    "raw_content": "",
                    "corpus_id": data_source.split("/")[-1],
                    "corpus_path": data_source,
                    "crop_path": data_source,
                })
            else:
                # PDF or prefix path — discover all pages via the loader
                page_map = loader._pdf_to_images(data_source)
                for page_num in sorted(page_map.keys()):
                    img_path = page_map[page_num]
                    elements_data.append({
                        "bbox": [0, 0, 1000, 1000],
                        "type": "page_image",
                        "content": "",
                        "raw_content": "",
                        "corpus_id": img_path.split("/")[-1],
                        "corpus_path": img_path,
                        "crop_path": img_path,
                    })
    except Exception:
        logger.exception("Error getting all pages for sample %s", qid)
        elements_data = []

    result_item = {
        "qid": sample.qid,
        "query": sample.query,
        "gold_answer": sample.gold_answer,
        "data_source": sample.data_source,
        "gold_pages": getattr(sample, "gold_pages", []),
        "retrieved_elements": elements_data,
    }

    write_cache(cache_path, result_item)
    return result_item


def main():
    args = parse_args()
    save_run_config(args, "retrieval")

    all_pages_mode = getattr(args, "return_all_pages", False)

    logger.info(
        "Starting %s for benchmark=%s (threads=%s).",
        "ALL-PAGES retrieval (returning every page)" if all_pages_mode else "Retrieval stage",
        args.benchmark,
        args.num_threads,
    )

    agent, loader = initialize_components(
        args,
        init_retriever=not all_pages_mode,
        init_generator=False,
    )

    cache_dir = Path(args.output_dir) / "cache_retrieval_results"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", cache_dir)

    samples = loader.samples
    logger.info("Processing %d samples.", len(samples))

    if all_pages_mode:
        worker = partial(process_single_sample_all_pages, loader=loader, cache_dir=cache_dir)
    else:
        worker = partial(process_single_sample_retrieval, agent=agent, cache_dir=cache_dir)

    retrieval_results = run_parallel(
        fn=worker,
        items=samples,
        num_threads=args.num_threads,
        desc="Collecting all pages" if all_pages_mode else "Retrieving",
        get_id=lambda s: s.qid,
    )

    sort_by_qid(retrieval_results)

    output_file = Path(args.output_dir) / "retrieval_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)

    logger.info("Retrieval complete. Results saved to %s", output_file)


if __name__ == "__main__":
    main()
