"""Sweep RegionRAG hyperparameters against a synthetic test page.

We already know the model has *some* signal — its top-1 region tends to
contain the correct gold region — but the connected-component merging at
neighbor_range=2 over-expands the bbox into surrounding whitespace, which
crushes IoU. This sweep tells us:

  - Is the signal actually there? (gold-center-in-pred-top-1 rate)
  - How much does tightening neighbor_range / threshold recover IoU?
  - Are predictions actually localizing, or are they noise?

Outputs a table per (threshold, neighbor_range) and overlay images for the
best-performing config.
"""

import json
import os
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

API = os.environ.get("REGIONRAG_API_BASE", "http://localhost:8005").rstrip("/") + "/extract_regions"
OUT_DIR = Path("tmp_regionrag_test")
OUT_DIR.mkdir(exist_ok=True)

PAGE_W, PAGE_H = 1200, 1600

REGIONS = [
    {
        "id": "revenue",
        "bbox_pixel": (60, 60, 580, 220),
        "title": "Revenue Summary",
        "body": "Total revenue: $1.27 billion\nQuarter-over-quarter growth: +8.4%",
        "queries": ["What is the total revenue?"],
    },
    {
        "id": "date",
        "bbox_pixel": (620, 60, 1140, 220),
        "title": "Reporting Period",
        "body": "Fiscal Year 2024, Quarter 3\nReport date: October 28, 2024",
        "queries": ["When was this report dated?"],
    },
    {
        "id": "customers",
        "bbox_pixel": (60, 1380, 580, 1540),
        "title": "Customer Metrics",
        "body": "Active customers: 5,234\nNew sign-ups this quarter: 481",
        "queries": ["How many active customers do we have?"],
    },
    {
        "id": "margin",
        "bbox_pixel": (620, 1380, 1140, 1540),
        "title": "Profit Margin",
        "body": "Gross margin: 42.1%\nOperating margin: 15.3%",
        "queries": ["What is the operating margin?"],
    },
]


def render_page() -> Path:
    img = Image.new("RGB", (PAGE_W, PAGE_H), color="white")
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 32)
    body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    draw.text((PAGE_W // 2 - 200, 600), "Annual Operations Brief", fill="black", font=title_font)
    draw.text(
        (80, 700),
        "This page contains four boxed summaries arranged at the\n"
        "corners of the page. Each summary is independent and can\n"
        "be queried for the specific facts it lists.",
        fill="#444444",
        font=body_font,
    )
    for r in REGIONS:
        x1, y1, x2, y2 = r["bbox_pixel"]
        draw.rectangle([x1, y1, x2, y2], fill="#f5f5fa", outline="#222", width=2)
        draw.text((x1 + 18, y1 + 14), r["title"], fill="black", font=title_font)
        draw.multiline_text((x1 + 18, y1 + 60), r["body"], fill="black", font=body_font, spacing=8)
    page_path = OUT_DIR / "page.png"
    img.save(page_path)
    return page_path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def gold_coverage(pred, gold):
    """Fraction of gold area covered by pred."""
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gold
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    g_area = max(1, (gx2 - gx1) * (gy2 - gy1))
    return inter / g_area


def gold_center_in_pred(pred, gold):
    px1, py1, px2, py2 = pred
    cx = (gold[0] + gold[2]) / 2
    cy = (gold[1] + gold[3]) / 2
    return px1 <= cx <= px2 and py1 <= cy <= py2


def call(query, image_path, **params):
    payload = {"query": query, "image_path": str(image_path.resolve()),
               "score_method": "max", "max_regions": 20, **params}
    r = requests.post(API, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("regions", [])


def visualize(page_path, query, gold_bbox, predicted_regions, out_path):
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    gx1, gy1, gx2, gy2 = gold_bbox
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 180, 0, 255), width=5)
    palette = [(220, 30, 30), (240, 140, 0), (30, 80, 220)]
    for i, region in enumerate(predicted_regions[:3]):
        bp = region["bbox_pixel"]
        col = palette[i % len(palette)]
        draw.rectangle(bp, outline=col + (255,), width=4)
        draw.text((bp[0] + 6, max(0, bp[1] - 26)),
                  f"#{i+1} score={region['score']:.3f}",
                  fill=col + (255,), font=font)
    draw.text((20, 20), f"Query: {query}", fill=(0, 0, 0, 255), font=font)
    img.save(out_path)


def summarize(page_path, threshold, neighbor_range, save_overlays=False):
    rows = []
    for r in REGIONS:
        for q in r["queries"]:
            preds = call(q, page_path, bbox_threshold=threshold, neighbor_range=neighbor_range)
            if not preds:
                rows.append((r["id"], q, None, 0, 0, 0, -1))
                continue
            top = preds[0]["bbox_pixel"]
            rows.append((
                r["id"], q,
                preds[0]["score"],
                iou(top, r["bbox_pixel"]),
                gold_coverage(top, r["bbox_pixel"]),
                1 if gold_center_in_pred(top, r["bbox_pixel"]) else 0,
                # rank of first prediction whose center hits gold
                next((i for i, p in enumerate(preds)
                      if r["bbox_pixel"][0] <= (p["bbox_pixel"][0] + p["bbox_pixel"][2]) / 2 <= r["bbox_pixel"][2]
                      and r["bbox_pixel"][1] <= (p["bbox_pixel"][1] + p["bbox_pixel"][3]) / 2 <= r["bbox_pixel"][3]),
                     -1),
            ))
            if save_overlays:
                visualize(page_path, q, r["bbox_pixel"], preds,
                          OUT_DIR / f"sweep_{threshold}_n{neighbor_range}_{r['id']}.png")

    avg_iou = sum(r[3] for r in rows) / len(rows)
    avg_cov = sum(r[4] for r in rows) / len(rows)
    n_top1_contains_gold = sum(r[5] for r in rows)
    return rows, avg_iou, avg_cov, n_top1_contains_gold


def main():
    page_path = render_page()
    print(f"API: {API}")
    print(f"Test page: {page_path.resolve()}\n")

    # Sweep grid
    sweeps = []
    for thr in (0.25, 0.30, 0.35, 0.40):
        for nr in (1, 2):
            rows, avg_iou, avg_cov, contained = summarize(page_path, thr, nr)
            sweeps.append((thr, nr, avg_iou, avg_cov, contained, rows))
            print(f"threshold={thr:.2f} neighbor_range={nr}: "
                  f"avg IoU={avg_iou:.3f}  avg gold-coverage={avg_cov:.2%}  "
                  f"top-1 contains gold center: {contained}/{len(rows)}")

    # Pick the best by IoU and re-run with overlays saved
    best = max(sweeps, key=lambda x: x[2])
    print(f"\nBest by IoU: threshold={best[0]} neighbor_range={best[1]}  → IoU={best[2]:.3f}")
    summarize(page_path, best[0], best[1], save_overlays=True)

    # Print per-query detail at the best config
    print(f"\nPer-query at threshold={best[0]}, neighbor_range={best[1]}:")
    print(f"{'region':<10} {'top1_score':<11} {'IoU':<7} {'gold_coverage':<14} {'center_in_top1':<15} {'first_center_hit_rank'}")
    print("-" * 90)
    for r in best[5]:
        rid, q, sc, iou_v, cov, in_top1, first_hit = r
        sc_s = f"{sc:.3f}" if sc is not None else "-"
        print(f"{rid:<10} {sc_s:<11} {iou_v:<7.3f} {cov:<14.2%} "
              f"{'yes' if in_top1 else 'no':<15} {first_hit + 1 if first_hit >= 0 else 'none'}")


if __name__ == "__main__":
    main()
