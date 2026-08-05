import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.loaders.base_loader import PageElement, calc_iou_standard


def execute_agent_and_parse_json(
    agent: Any,
    query: str,
    image_path: str,
    page_id: str = "",
    max_retry: int = 5,
    max_text_length: int = 40000,
) -> Tuple[str, List[Any]]:
    """Run agent and parse JSON list from the latest prediction content."""
    try:
        retry_count = 0
        predictions: List[Dict[str, Any]] = []

        while retry_count < max_retry:
            agent_output = agent.run_agent(user_text=query, image_paths=[image_path])
            if not agent_output:
                return "", [1, 1]

            predictions = agent_output.get("predictions", [])
            if not predictions:
                return "", [1, 1]

            all_text = "".join(
                item.get("content", "")
                for item in predictions
                if isinstance(item, dict)
            )
            if len(all_text) <= max_text_length:
                break
            retry_count += 1

        last_msg_content = predictions[-1].get("content", "")
        json_str = _extract_json_array_string(last_msg_content)

        original_json_str = json_str
        try:
            extracted_data = json.loads(json_str)
        except Exception:
            json_str = json_str.replace("\n", "\\n").replace("\t", "\\t")
            try:
                extracted_data = json.loads(json_str)
            except Exception:
                json_str = original_json_str.replace("\\", "\\\\")
                extracted_data = json.loads(json_str)

        return last_msg_content, extracted_data
    except Exception:
        return "", [1, 1]


def is_valid_extracted_data(data: Any) -> bool:
    if not isinstance(data, list):
        return False
    return all(isinstance(item, dict) for item in data)


def run_extractor_with_optional_judger(
    query: str,
    image_path: str,
    extractor: Any,
    judger: Optional[Any] = None,
    page_id: str = "",
    max_retry: int = 5,
) -> List[Dict[str, Any]]:
    """Run judger/extractor flow with retry and return validated extraction list."""
    if extractor is None:
        return []

    extracted_data: Any = []

    if judger is not None:
        _, extracted_data = execute_agent_and_parse_json(
            judger,
            query,
            image_path,
            page_id=page_id,
        )

        if extracted_data:
            for _ in range(max_retry):
                _, extracted_data = execute_agent_and_parse_json(
                    extractor,
                    query,
                    image_path,
                    page_id=page_id,
                )
                if is_valid_extracted_data(extracted_data) and extracted_data:
                    break
    else:
        for retry in range(max_retry):
            _, extracted_data = execute_agent_and_parse_json(
                extractor,
                query,
                image_path,
                page_id=page_id,
            )
            if is_valid_extracted_data(extracted_data) and (retry > 1 or extracted_data):
                break

    if not is_valid_extracted_data(extracted_data):
        return []
    return extracted_data


def convert_rel_bbox_to_abs(bbox: List[int], img_w: int, img_h: int) -> List[int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = bbox
    x1 = int(x1 / 1000 * img_w)
    y1 = int(y1 / 1000 * img_h)
    x2 = int(x2 / 1000 * img_w)
    y2 = int(y2 / 1000 * img_h)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return [x1, y1, x2, y2]


def extract_page_elements_from_data(
    extracted_data: Any,
    image_path: str,
    workspace_dir: str,
    corpus_id: str,
    corpus_path: Optional[str] = None,
    retrieval_score: Optional[float] = None,
) -> List[PageElement]:
    """Convert extractor JSON output to PageElement list and save cropped patches."""
    if isinstance(extracted_data, dict):
        extracted_data = [extracted_data]
    if not isinstance(extracted_data, list):
        return []

    os.makedirs(workspace_dir, exist_ok=True)
    try:
        page_image = Image.open(image_path)
        img_w, img_h = page_image.size
    except Exception:
        page_image = None
        img_w, img_h = 0, 0

    elements: List[PageElement] = []
    for item in extracted_data:
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox", [0, 0, 0, 0])
        evidence = item.get("evidence", "")
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]

        current_crop_path = image_path
        if page_image and bbox != [0, 0, 0, 0]:
            abs_bbox = convert_rel_bbox_to_abs(bbox, img_w, img_h)
            x1, y1, x2, y2 = abs_bbox
            if x2 > x1 and y2 > y1:
                try:
                    cropped_img = page_image.crop((x1, y1, x2, y2))
                    filename = f"{os.path.basename(corpus_id).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join(workspace_dir, filename)
                    cropped_img.save(save_path)
                    current_crop_path = save_path
                except Exception:
                    pass

        element = PageElement(
            bbox=bbox,
            type="evidence",
            content=evidence,
            corpus_id=corpus_id,
            corpus_path=corpus_path or image_path,
            crop_path=current_crop_path,
        )
        if retrieval_score is not None:
            element.retrieval_score = retrieval_score
        elements.append(element)

    if page_image:
        page_image.close()
    return elements


def compute_element_metrics(
    pred_elements: List[PageElement],
    gold_elements: List[PageElement],
    threshold: float = 0.5,
) -> Dict[str, float]:
    if not gold_elements:
        return {}
    if not pred_elements:
        return {"element_precision": 1.0, "element_recall": 0.0, "element_f1": 0.0}

    def normalize_cid(cid: Optional[str]) -> str:
        return os.path.basename(cid) if cid else ""

    hit_preds = 0
    for pred in pred_elements:
        pred_cid = normalize_cid(pred.corpus_id)
        for gold in gold_elements:
            if pred_cid != normalize_cid(gold.corpus_id):
                continue
            if calc_iou_standard(pred.bbox, gold.bbox) > threshold:
                hit_preds += 1
                break

    hit_golds = 0
    for gold in gold_elements:
        gold_cid = normalize_cid(gold.corpus_id)
        for pred in pred_elements:
            if gold_cid != normalize_cid(pred.corpus_id):
                continue
            if calc_iou_standard(pred.bbox, gold.bbox) > threshold:
                hit_golds += 1
                break

    precision = hit_preds / len(pred_elements) if pred_elements else 1.0
    recall = hit_golds / len(gold_elements) if gold_elements else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"element_precision": precision, "element_recall": recall, "element_f1": f1}


def _extract_json_array_string(text: str) -> str:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return "[]"
