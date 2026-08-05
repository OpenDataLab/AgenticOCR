import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

from tqdm import tqdm


logger = logging.getLogger(__name__)
T = TypeVar("T")
PathLike = Union[str, Path]


def safe_qid(qid: str) -> str:
    """Convert qid to a safe filename token."""
    return "".join(c if c.isalnum() else "_" for c in str(qid))


def sort_by_qid(items: List[Dict[str, Any]], key: str = "qid") -> List[Dict[str, Any]]:
    """Sort in-place by qid: numeric values first by int, otherwise by string."""
    def _qid_sort_key(item: Dict[str, Any]) -> Tuple[int, Any]:
        qid = item.get(key)
        qid_str = str(qid)
        if qid_str.isdigit():
            return 0, int(qid_str)
        return 1, qid_str

    try:
        items.sort(key=_qid_sort_key)
    except (TypeError, ValueError):
        logger.warning("Skipping qid sort because at least one item has an invalid '%s' value.", key)
    return items


def read_cache(cache_path: PathLike) -> Optional[Dict[str, Any]]:
    """Read cached json file and return None on errors."""
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Failed to read cache file: %s", path)
        return None


def _atomic_write_text(path: Path, content: str) -> None:
    """Write a file atomically to avoid partially written cache/artifact files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def write_cache(cache_path: PathLike, data: Dict[str, Any]) -> None:
    """Write cache file atomically."""
    path = Path(cache_path)
    try:
        _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
    except OSError:
        logger.exception("Error saving cache to %s", path)


def write_jsonl(path: PathLike, items: Iterable[Dict[str, Any]]) -> None:
    """Write iterable items to a JSONL file atomically without loading everything into memory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")
    tmp_path.replace(output_path)


def read_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    """Yield JSONL records line-by-line, skipping blank or malformed lines."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s line %d: %s", path, line_num, exc)


def run_parallel(
    fn: Callable[[T], Any],
    items: Sequence[T],
    num_threads: int = 1,
    desc: str = "Processing",
    get_id: Optional[Callable[[T], Any]] = None,
) -> List[Any]:
    """
    Run `fn` on `items` with ThreadPoolExecutor and tqdm progress.
    Returns non-None results in completion order.
    """
    if not items:
        return []

    max_workers = max(1, int(num_threads))
    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(fn, item): get_id(item) if get_id else id(item)
            for item in items
        }

        for future in tqdm(as_completed(future_to_id), total=len(items), desc=desc):
            fid = future_to_id[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                logger.exception("Thread exception (id=%s)", fid)
    return results


# ──  ──

def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 12845056,
) -> Tuple[int, int]:
    """Smart resize factor  [min_pixels, max_pixels] """
    def _round(n, f):
        return round(n / f) * f

    def _floor(n, f):
        return math.floor(n / f) * f

    def _ceil(n, f):
        return math.ceil(n / f) * f

    h_bar = max(factor, _round(height, factor))
    w_bar = max(factor, _round(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor(height / beta, factor)
        w_bar = _floor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil(height * beta, factor)
        w_bar = _ceil(width * beta, factor)

    return h_bar, w_bar
