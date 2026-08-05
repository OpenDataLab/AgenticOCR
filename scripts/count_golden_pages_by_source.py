"""
 MMLongBench  evidence_source  golden page 

 sample  evidence_source ["Table", "Chart"]
 sample  gold_pages  source 


-  evidence_source golden page  golden page 
- 
- Golden page  PDF 
  - spanmax_page - min_page + 1 golden pages 
  - max_gap golden pages 
  -  vs 
  - max_gap 
-  evidence_source  answer_format Int / Float / Str / List / None
"""

import json
import ast
import sys
from collections import defaultdict, Counter


DATA_PATH = "/path/to/MMLongBench-Doc/data/samples.json"


def _parse_list_field(raw, fallback=None):
    """ Python list"""
    if fallback is None:
        fallback = []
    try:
        result = ast.literal_eval(str(raw))
        if not isinstance(result, list):
            return [result] if result is not None else fallback
        return result
    except Exception:
        return fallback


def _compute_dispersion(pages):
    """ golden pages 

    Returns:
        dict with keys: num_pages, span, max_gap, is_consecutive, gaps
         pages  < 2span = num_pages, max_gap = 0, is_consecutive = True
    """
    nums = sorted(int(p) for p in pages)
    n = len(nums)
    if n == 0:
        return {"num_pages": 0, "span": 0, "max_gap": 0, "is_consecutive": True, "gaps": []}
    if n == 1:
        return {"num_pages": 1, "span": 1, "max_gap": 0, "is_consecutive": True, "gaps": []}

    span = nums[-1] - nums[0] + 1
    gaps = [nums[i + 1] - nums[i] for i in range(n - 1)]
    max_gap = max(gaps)
    is_consecutive = (span == n)  # 
    return {
        "num_pages": n,
        "span": span,
        "max_gap": max_gap,
        "is_consecutive": is_consecutive,
        "gaps": gaps,
    }


def _make_stats():
    return {
        "sample_count": 0,
        "gold_page_count": 0,
        # 
        "span_sum": 0,
        "max_gap_sum": 0,
        "consecutive_count": 0,
        "scattered_count": 0,
        #  num_pages >= 2 
        "multi_page_count": 0,
        #  max_gap 
        "max_gap_list": [],
        #  span 
        "span_list": [],
        # answer_format 
        "answer_format_counts": Counter(),
    }


def _accumulate(stats_entry, disp, answer_format=None):
    """ stats_entry """
    stats_entry["sample_count"] += 1
    stats_entry["gold_page_count"] += disp["num_pages"]
    stats_entry["answer_format_counts"][answer_format or "Unknown"] += 1
    if disp["num_pages"] >= 2:
        stats_entry["multi_page_count"] += 1
        stats_entry["span_sum"] += disp["span"]
        stats_entry["max_gap_sum"] += disp["max_gap"]
        stats_entry["max_gap_list"].append(disp["max_gap"])
        stats_entry["span_list"].append(disp["span"])
        if disp["is_consecutive"]:
            stats_entry["consecutive_count"] += 1
        else:
            stats_entry["scattered_count"] += 1


def _print_histogram(values, label, bin_edges):
    """"""
    if not values:
        print(f"  ( {label} )")
        return
    bins = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            #  bin 
            count = sum(1 for v in values if lo <= v <= hi)
            label_str = f"[{lo}, {hi}]" if hi != float("inf") else f"[{lo}, +∞)"
        else:
            count = sum(1 for v in values if lo <= v < hi)
            label_str = f"[{lo}, {hi})"
        bins.append((label_str, count))

    #  bin 
    max_val = max(values)
    if max_val > bin_edges[-1]:
        overflow = sum(1 for v in values if v > bin_edges[-1])
        if overflow > 0:
            bins.append((f"({bin_edges[-1]}, +∞)", overflow))

    total = len(values)
    max_bar = 40
    max_count = max(c for _, c in bins) if bins else 1
    for lbl, cnt in bins:
        pct = cnt / total * 100 if total > 0 else 0
        bar_len = int(cnt / max_count * max_bar) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {lbl:>12s}  {bar:<{max_bar}s} {cnt:>5d} ({pct:5.1f}%)")


def main(data_path: str = DATA_PATH):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_source = defaultdict(_make_stats)
    overall = _make_stats()

    for item in data:
        sources = _parse_list_field(item.get("evidence_sources", "[]"), fallback=["Unknown"])
        pages = _parse_list_field(item.get("evidence_pages", "[]"), fallback=[])
        answer_format = item.get("answer_format") or "Unknown"
        disp = _compute_dispersion(pages)

        _accumulate(overall, disp, answer_format)
        for src in sources:
            _accumulate(per_source[src], disp, answer_format)

    # ──  ──────────────────────────────────
    sorted_sources = sorted(per_source.items(), key=lambda x: x[1]["gold_page_count"], reverse=True)

    print("=" * 90)
    print("  Part 1: Golden Page Count by Evidence Source")
    print("=" * 90)
    print(f"{'Evidence Source':<20} {'Samples':>8} {'Gold Pages':>12} {'Avg Pages':>10}")
    print("-" * 54)

    for src, info in sorted_sources:
        sc = info["sample_count"]
        gp = info["gold_page_count"]
        avg = gp / sc if sc > 0 else 0
        print(f"{src:<20} {sc:>8} {gp:>12} {avg:>10.2f}")

    sc = overall["sample_count"]
    gp = overall["gold_page_count"]
    avg = gp / sc if sc > 0 else 0
    print("-" * 54)
    print(f"{'Total':<20} {sc:>8} {gp:>12} {avg:>10.2f}")
    print(f"\n(:  sample  source source  >  {len(data)})")

    # ──  ──────────────────────────────────
    print()
    print("=" * 90)
    print("  Part 2: Golden Page Dispersion by Evidence Source  ( gold_pages >= 2 )")
    print("=" * 90)
    header = (
        f"{'Evidence Source':<20} {'Multi-pg':>8} {'Consec':>8} {'Scatter':>8}"
        f" {'Avg Span':>10} {'Avg MaxGap':>12}"
    )
    print(header)
    print("-" * 70)

    for src, info in sorted_sources:
        mp = info["multi_page_count"]
        cc = info["consecutive_count"]
        scat = info["scattered_count"]
        avg_span = info["span_sum"] / mp if mp > 0 else 0
        avg_mg = info["max_gap_sum"] / mp if mp > 0 else 0
        print(f"{src:<20} {mp:>8} {cc:>8} {scat:>8} {avg_span:>10.2f} {avg_mg:>12.2f}")

    mp = overall["multi_page_count"]
    cc = overall["consecutive_count"]
    scat = overall["scattered_count"]
    avg_span = overall["span_sum"] / mp if mp > 0 else 0
    avg_mg = overall["max_gap_sum"] / mp if mp > 0 else 0
    print("-" * 70)
    print(f"{'Total':<20} {mp:>8} {cc:>8} {scat:>8} {avg_span:>10.2f} {avg_mg:>12.2f}")

    print(f"\n  :")
    print(f"    Multi-pg : golden pages >= 2 ")
    print(f"    Consec   :  golden pages span == num_pages")
    print(f"    Scatter  : golden pages span > num_pages")
    print(f"    Avg Span :  = avg(max_page - min_page + 1)")
    print(f"    Avg MaxGap:  = avg( golden pages )")

    # ──  ──────────────────────────────────
    print()
    print("=" * 90)
    print("  Part 3: Distribution Histograms  ( multi-page )")
    print("=" * 90)

    print(f"\n  [Overall] Max Gap Distribution (n={len(overall['max_gap_list'])}):")
    _print_histogram(overall["max_gap_list"], "max_gap", [1, 2, 3, 4, 6, 11, 21, 51])

    print(f"\n  [Overall] Span Distribution (n={len(overall['span_list'])}):")
    _print_histogram(overall["span_list"], "span", [2, 3, 4, 6, 11, 21, 51])

    for src, info in sorted_sources:
        if not info["max_gap_list"]:
            continue
        print(f"\n  [{src}] Max Gap Distribution (n={len(info['max_gap_list'])}):")
        _print_histogram(info["max_gap_list"], "max_gap", [1, 2, 3, 4, 6, 11, 21, 51])

    # ──  ──────────────────────────────────
    print()
    print("=" * 90)
    print("  Part 4: Answer Format Distribution by Evidence Source")
    print("=" * 90)

    #  answer_format 
    all_formats = sorted({fmt for info in per_source.values() for fmt in info["answer_format_counts"]}
                         | set(overall["answer_format_counts"]))

    # 
    fmt_header = "".join(f"{fmt:>8}" for fmt in all_formats)
    print(f"{'Evidence Source':<20} {'Samples':>8} {fmt_header}")
    print("-" * (28 + 8 * len(all_formats)))

    for src, info in sorted_sources:
        sc = info["sample_count"]
        counts_str = "".join(f"{info['answer_format_counts'].get(fmt, 0):>8}" for fmt in all_formats)
        print(f"{src:<20} {sc:>8} {counts_str}")

    sc = overall["sample_count"]
    print("-" * (28 + 8 * len(all_formats)))
    counts_str = "".join(f"{overall['answer_format_counts'].get(fmt, 0):>8}" for fmt in all_formats)
    print(f"{'Total':<20} {sc:>8} {counts_str}")

    # 
    print(f"\n   ( source ):")
    fmt_header_pct = "".join(f"{fmt:>8}" for fmt in all_formats)
    print(f"  {'Evidence Source':<20} {fmt_header_pct}")
    print(f"  {'-' * (20 + 8 * len(all_formats))}")

    for src, info in sorted_sources:
        sc = info["sample_count"]
        pcts = "".join(
            f"{info['answer_format_counts'].get(fmt, 0) / sc * 100:>7.1f}%" for fmt in all_formats
        ) if sc > 0 else "".join(f"{'—':>8}" for _ in all_formats)
        print(f"  {src:<20} {pcts}")

    sc = overall["sample_count"]
    pcts = "".join(
        f"{overall['answer_format_counts'].get(fmt, 0) / sc * 100:>7.1f}%" for fmt in all_formats
    ) if sc > 0 else ""
    print(f"  {'-' * (20 + 8 * len(all_formats))}")
    print(f"  {'Total':<20} {pcts}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    main(path)
