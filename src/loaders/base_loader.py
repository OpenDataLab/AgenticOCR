import abc
import json
import re
import string
import collections
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any, Tuple

@dataclass
class PageElement:
    """
    /
    
    """
    
    # Bounding Box : [x_min, y_min, x_max, y_max]
    #  (0 - 1000)
    bbox: List[int] = field(default_factory=list) 
    type: str = "text"          # 'text', 'table', 'image', 'chart' None
    content: str = ""                #  (OCR)
    raw_content: str = ""
    corpus_id: str = "" # ID
    corpus_path: str = ""
    
    # 
    crop_path: Optional[str] = None 

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StandardSample:
    """
    
     (MMLong, FinRAG, MVTool) 
    """
    qid: str
    query: str
    dataset: str         # 'mmlong', 'finrag', 'mvtool'
    # ---  --- PDF
    data_source: str     # '.index', '.pdf', '.png'
    
    # --- Ground Truth () ---
    gold_answer: Optional[str] = None
    
    #  BBox  ()
    gold_elements: List[PageElement] = field(default_factory=list)
    
    #  ID  ()
    gold_pages: List[str] = field(default_factory=list)
    
    extra_info: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """"""
        return asdict(self)

# --- Metrics Calculation Helpers (Ported from eval.py) ---

def calculate_area(bbox: List[int]) -> int:
    """ BBox """
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def get_intersection_area(bbox1: List[int], bbox2: List[int]) -> int:
    """ BBox """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    if x2 < x1 or y2 < y1: return 0
    return (x2 - x1) * (y2 - y1)

def calc_iou_min(pred: List[int], gt: List[int]) -> float:
    """ Intersection over Minimum Area ()"""
    area_p = calculate_area(pred)
    area_g = calculate_area(gt)
    if area_p == 0 or area_g == 0: return 0.0
    inter = get_intersection_area(pred, gt)
    return inter / min(area_p, area_g)

def calc_iou_standard(pred: List[int], gt: List[int]) -> float:
    """ Intersection over Union"""
    area_p = calculate_area(pred)
    area_g = calculate_area(gt)
    if area_p == 0 or area_g == 0: return 0.0
    inter = get_intersection_area(pred, gt)
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0.0

def calculate_f_beta(precision: float, recall: float, beta: float = 1.0) -> float:
    """ F-beta """
    if precision + recall == 0: return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)


class BaseDataLoader(abc.ABC):
    """
     Loader 
     load_data 
    """
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.samples: List[StandardSample] = []

    @abc.abstractmethod
    def load_data(self) -> None:
        """
         self.samples 
         (e.g., FinRAGLoader) 
        """
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> StandardSample:
        return self.samples[idx]

    def get_batch(self, batch_size: int):
        """ Batch """
        for i in range(0, len(self.samples), batch_size):
            yield self.samples[i : i + batch_size]
            
    def pipeline(self, query: str, image_paths: List[str], top_k: int, trunc_thres: float, trunc_bbox: bool) -> List[PageElement]:
        """
        
        """
        raise NotImplementedError

    def evaluate(self) -> Dict[str, float]:
        """
        
        """
        raise NotImplementedError

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        
        """
        raise NotImplementedError
    
    def evaluate_generation(self) -> Dict[str, float]:
        """
        
        """
        raise NotImplementedError
        
    # ---  ---

    def _normalize_text(self, s: str) -> str:
        """"""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
            
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _compute_qa_metrics(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """ F1  Exact Match (EM)"""
        pred_norm = self._normalize_text(prediction)
        gt_norm = self._normalize_text(ground_truth)
        
        # EM
        em = 1.0 if pred_norm == gt_norm else 0.0
        
        # F1
        prediction_tokens = pred_norm.split()
        ground_truth_tokens = gt_norm.split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            
        return {"f1": f1, "em": em}

    def _compute_page_metrics(self, pred_elements: List[PageElement], gold_pages: List[str]) -> Dict[str, float]:
        """ Recall  Precision"""
        #  ID ()
        #  corpus_id  gold_pages 
        pred_page_ids = set([el.corpus_id for el in pred_elements if el.corpus_id])
        gt_page_ids = set(gold_pages)
        
        if not gt_page_ids:
            return {"recall": 1.0, "precision": 1.0 if len(pred_page_ids) == 0 else 0.0}
        
        hits = pred_page_ids & gt_page_ids
        recall = len(hits) / len(gt_page_ids)
        precision = len(hits) / len(pred_page_ids) if pred_page_ids else 0.0
        
        return {"recall": recall, "precision": precision}

    def _compute_page_accuracy(self, pred_bboxes: List[List[int]], gt_bboxes: List[List[int]]) -> float:
        """
         Page Accuracy:
         GT vs  Pred
         eval.py:
        - : Correct (1.0)
        - : Correct (1.0) - GT“”IoU
        - : Incorrect (0.0)
        """
        if not pred_bboxes and not gt_bboxes:
            return 1.0
        elif not pred_bboxes and gt_bboxes:
            return 0.0
        elif not gt_bboxes and pred_bboxes:
            return 0.0
        else:
            return 1.0

    def _compute_detection_metrics(self, pred_bboxes: List[List[int]], gt_bboxes: List[List[int]], 
                                 iou_func, threshold: float) -> Tuple[float, float]:
        """
         IoU  Precision  Recall
        """
        if not pred_bboxes and not gt_bboxes: return 1.0, 1.0
        if not pred_bboxes: return 1.0, 0.0 
        if not gt_bboxes: return 0.0, 1.0   

        # Precision Calculation
        valid_preds = 0
        for p in pred_bboxes:
            hit = False
            for g in gt_bboxes:
                if iou_func(p, g) > threshold:
                    hit = True
                    break
            if hit: valid_preds += 1
        precision = valid_preds / len(pred_bboxes)

        # Recall Calculation
        hit_gts = 0
        for g in gt_bboxes:
            hit = False
            for p in pred_bboxes:
                if iou_func(p, g) > threshold:
                    hit = True
                    break
            if hit: hit_gts += 1
        recall = hit_gts / len(gt_bboxes)
        
        return precision, recall