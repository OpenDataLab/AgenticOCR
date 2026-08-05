"""Sanity test for RegionRAG service:

Render a synthetic page with 4 well-separated regions, each containing a
distinct fact. Send 4 queries (one targeting each region) to /extract_regions
and verify the top-ranked bbox lands inside the correct ground-truth region.

Outputs:
  - tmp_regionrag_test/page.png         (the synthetic test page)
  - tmp_regionrag_test/q{i}_overlay.png (predicted bboxes vs gold for each query)
  - prints a table summarizing IoU and pass/fail per query
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

# Each region is positioned in one quadrant with extra padding around the text.
# bbox is in pixel coordinates (x1, y1, x2, y2).
REGIONS = [
    {
        "id": "revenue",
        "bbox_pixel": (60, 60, 580, 220),
        "title": "Revenue Summary",
        "body": "Total revenue: $1.27 billion\nQuarter-over-quarter growth: +8.4%",
        "queries": [
            "What is the total revenue?",
            "Show the revenue figure for this quarter.",
        ],
    },
    {
        "id": "date",
        "bbox_pixel": (620, 60, 1140, 220),
        "title": "Reporting Period",
        "body": "Fiscal Year 2024, Quarter 3\nReport date: October 28, 2024",
        "queries": [
            "What reporting period does this document cover?",
            "When was this report dated?",
        ],
    },
    {
        "id": "customers",
        "bbox_pixel": (60, 1380, 580, 1540),
        "title": "Customer Metrics",
        "body": "Active customers: 5,234\nNew sign-ups this quarter: 481",
        "queries": [
            "How many active customers do we have?",
            "How many new customer sign-ups happened?",
        ],
    },
    {
        "id": "margin",
        "bbox_pixel": (620, 1380, 1140, 1540),
        "title": "Profit Margin",
        "body": "Gross margin: 42.1%\nOperating margin: 15.3%",
        "queries": [
            "What is the operating margin?",
            "What is the gross profit margin?",
        ],
    },
]


def render_page() -> Path:
    """Build the synthetic page image."""
    img = Image.new("RGB", (PAGE_W, PAGE_H), color="white")
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 32)
    body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)

    # Page header (NOT a region target — just visual padding, distractor text)
    draw.text(
        (PAGE_W // 2 - 200, 600),
        "Annual Operations Brief",
        fill="black",
        font=title_font,
    )
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
        # light fill + visible border so we see the boxes when debugging
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


def call_service(query: str, image_path: Path):
    payload = {
        "query": query,
        "image_path": str(image_path.resolve()),
        "neighbor_range": 2,
        "bbox_threshold": 0.25,
        "score_method": "max",
        "max_regions": 20,
    }
    r = requests.post(API, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def visualize(page_path: Path, query: str, gold_bbox, predicted_regions, out_path: Path):
    """Save an overlay: gold bbox in green, top-3 predicted in red/orange/blue."""
    img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)

    # Gold (green)
    gx1, gy1, gx2, gy2 = gold_bbox
    draw.rectangle([gx1, gy1, gx2, gy2], outline=(0, 180, 0, 255), width=5)

    # Top-3 predicted with different colors
    palette = [(220, 30, 30), (240, 140, 0), (30, 80, 220)]
    for i, region in enumerate(predicted_regions[:3]):
        bp = region["bbox_pixel"]
        col = palette[i % len(palette)]
        draw.rectangle(bp, outline=col + (255,), width=4)
        draw.text(
            (bp[0] + 6, max(0, bp[1] - 26)),
            f"#{i+1} score={region['score']:.3f}",
            fill=col + (255,),
            font=font,
        )

    draw.text((20, 20), f"Query: {query}", fill=(0, 0, 0, 255), font=font)
    img.save(out_path)


def main():
    print(f"API URL: {API}")
    page_path = render_page()
    print(f"Test page: {page_path.resolve()}\n")

    rows = []
    for r in REGIONS:
        for q in r["queries"]:
            try:
                resp = call_service(q, page_path)
            except Exception as e:
                print(f"REQUEST FAILED for query={q!r}: {e}")
                continue
            preds = resp.get("regions", [])
            if not preds:
                rows.append((r["id"], q, "—", 0.0, "FAIL: no regions"))
                continue

            top = preds[0]
            iou_top = iou(top["bbox_pixel"], r["bbox_pixel"])
            best_iou = max(iou(p["bbox_pixel"], r["bbox_pixel"]) for p in preds)

            # Find rank of first prediction whose center falls inside gold bbox.
            cx_in_gold = -1
            gx1, gy1, gx2, gy2 = r["bbox_pixel"]
            for rank, p in enumerate(preds):
                cx = (p["bbox_pixel"][0] + p["bbox_pixel"][2]) / 2
                cy = (p["bbox_pixel"][1] + p["bbox_pixel"][3]) / 2
                if gx1 <= cx <= gx2 and gy1 <= cy <= gy2:
                    cx_in_gold = rank
                    break

            verdict = "PASS" if iou_top >= 0.3 else (
                f"loose (top-{cx_in_gold+1} center hits gold)" if cx_in_gold >= 0 else "FAIL"
            )

            short_q = q if len(q) < 40 else q[:37] + "..."
            rows.append((r["id"], short_q, f"{top['score']:.3f}", iou_top, verdict, best_iou, cx_in_gold))

            # Save overlay for the first query of each region only (avoid spam)
            if q == r["queries"][0]:
                out_img = OUT_DIR / f"{r['id']}_overlay.png"
                visualize(page_path, q, r["bbox_pixel"], preds, out_img)

    # Print summary
    print(f"{'region':<12} {'query':<42} {'top1_sc':<8} {'top1_iou':<10} {'best_iou':<10} {'first_hit_rank':<14} {'verdict'}")
    print("-" * 110)
    passes = 0
    centerhits = 0
    for row in rows:
        if len(row) == 5:
            rid, q, sc, iou_top, verdict = row
            print(f"{rid:<12} {q:<42} {sc:<8} {iou_top:<10} {'-':<10} {'-':<14} {verdict}")
        else:
            rid, q, sc, iou_top, verdict, best_iou, first_hit = row
            print(f"{rid:<12} {q:<42} {sc:<8} {iou_top:<10.3f} {best_iou:<10.3f} {str(first_hit+1) if first_hit>=0 else 'none':<14} {verdict}")
            if iou_top >= 0.3:
                passes += 1
            if first_hit >= 0:
                centerhits += 1

    total = len([r for r in rows if len(r) > 5])
    print(f"\nResult: top-1 IoU>=0.3 on {passes}/{total} queries; "
          f"some predicted region's center hits gold on {centerhits}/{total}")
    print(f"Overlay images saved under: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
