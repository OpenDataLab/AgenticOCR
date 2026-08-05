import os
import json
import torch
import numpy as np
import faiss
import re
import base64
import time
import collections
import ast
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
logger = logging.getLogger(__name__)

# 
try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from src.recalls.qwen3_vl_embedding import Qwen3VLEmbedder
from src.recalls.qwen3_vl_reranker_client import Qwen3VLReranker
from src.agents.AgenticOCR import AgenticOCR
from src.utils.llm import create_llm_caller
from src.loaders.loader_common import (
    compute_element_metrics,
    execute_agent_and_parse_json,
    extract_page_elements_from_data,
    is_valid_extracted_data,
    run_extractor_with_optional_judger,
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def encode_image_to_base64(image_path):
    """Convert image to base64 encoding."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_pdf_prefix(filename):
    """
    PDF
    """
    pattern = r"^(.*)(?=(?:_\d+\.png|_multipage))"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return filename

class FinRAGLoader(BaseDataLoader):
    def __init__(self, data_root: str, lang: str = "ch", output_dir: str = "./", embedding_model=None, rerank_model=None, extractor: Optional[AgenticOCR] = None, judger: Optional[AgenticOCR] = None):
        super().__init__(data_root)
        self.lang = lang.lower()
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.extractor = extractor
        self.judger = judger
        
        # ---  ---
        self.langs = ["ch", "en"] if self.lang == "both" else [self.lang]
        
        # ---  ---
        self.query_paths = {l: os.path.join(data_root, "data", "queries", f"queries_{l}.json") for l in self.langs}
        self.corpus_roots = {l: os.path.join(data_root, "data", "corpus", l, "img") for l in self.langs}
        self.qrels_paths = {l: os.path.join(data_root, "data", "qrels", f"qrels_{l}.tsv") for l in self.langs}
        
        # 
        self.query_path = self.query_paths[self.langs[0]]
        self.corpus_root = self.corpus_roots[self.langs[0]]
        self.qrels_path = self.qrels_paths[self.langs[0]]
        
        self.citation_root = os.path.join(data_root, "data", "citation_labels", "citation_labels_new")
        self.output_dir = output_dir
        
        # 
        cache_dir = os.path.join(data_root, "data", "indices")
        os.makedirs(cache_dir, exist_ok=True)
        self.doc_map_path = os.path.join(cache_dir, f"finrag_{self.lang}_hnsw_docmap.json")
        
        self.doc_id_map = {} 
        self.llm_caller = None

    def _load_qrels(self) -> Dict[str, List[str]]:
        """ qrels TSV """
        qrels_map = {}
        for l in self.langs:
            qrels_path = self.qrels_paths[l]
            if not os.path.exists(qrels_path):
                logger.warning(f"Warning: Qrels file not found at {qrels_path}. GT IDs will be empty for {l}.")
                continue
                
            logger.info(f"Loading qrels from: {qrels_path}")
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("query-id"):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        qid = parts[0]
                        cid = parts[1]
                        if qid not in qrels_map:
                            qrels_map[qid] = []
                        qrels_map[qid].append(cid)
        return qrels_map

    def load_bbox_data(self) -> None:
        """ selected_200_with_bboxes.json """
        json_file_path = os.path.join(self.citation_root, "selected_200_with_bboxes.json")
        img_root_dir = self.citation_root

        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Selected dataset file not found: {json_file_path}")
        
        logger.info(f"Loading selected data from: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            qid = str(item.get("query-id") or item.get("id", str(count)))
            query_text = item.get("query", "")
            gold_answer = item.get("answer", "")
            
            extra_info = {
                'category': item.get('category'),
                'answer_type': item.get('answer_type'),
                'human_eval': item.get('human_eval'),
                'from_pages': item.get('from_pages')
            }

            gold_pages = []
            gold_elements = []
            
            img_paths_map = item.get("img_paths", {})
            bboxes_map = item.get("bboxes", {})
            
            for page_id, rel_path in img_paths_map.items():
                full_img_path = os.path.normpath(os.path.join(img_root_dir, rel_path))
                gold_pages.append(full_img_path)
                
                page_bboxes = bboxes_map.get(page_id, [])
                for box in page_bboxes:
                    x1 = int(box.get("xmin", 0) * 1000)
                    y1 = int(box.get("ymin", 0) * 1000)
                    x2 = int(box.get("xmax", 0) * 1000)
                    y2 = int(box.get("ymax", 0) * 1000)
                    
                    pe = PageElement(
                        bbox=[x1, y1, x2, y2],
                        type="evidence", 
                        content=gold_answer,
                        corpus_id=full_img_path,
                        crop_path=full_img_path
                    )
                    gold_elements.append(pe)

            sample = StandardSample(
                qid=qid, 
                query=query_text, 
                dataset=f"finrag-bbox",
                data_source=gold_pages[0],
                gold_answer=gold_answer,
                gold_elements=gold_elements,
                gold_pages=gold_pages, 
                extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
            
        logger.info(f"✅ Successfully loaded {count} samples from selected_200_with_bboxes.json.")

    def load_data(self) -> None:
        if self.lang == 'bbox':
            self.load_bbox_data()
            return
        
        """ Query  Qrels"""
        qrels_map = self._load_qrels()
        
        for l in self.langs:
            query_path = self.query_paths[l]
            if not os.path.exists(query_path):
                logger.warning(f"Warning: Query file not found: {query_path}")
                continue
                
            logger.info(f"Loading queries from: {query_path}")
            with open(query_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            count = 0
            for item in data:
                query_text = item.get("query", "") or item.get("question", "")
                if not query_text:
                    continue
                qid = str(item.get("id") or item.get("query-id") or item.get("_id") or "")
                gold_answer = item.get("answer", "") or item.get("response", "")
                extra_info = {'category': item.get('category', None), 'answer_type': item.get('answer_type', None), 'from_pages': item.get('from_pages', None)}
                gold_pages = qrels_map.get(qid, [])
                
                sample = StandardSample(
                    qid=qid, query=query_text, dataset=f"finrag-{l}",
                    data_source=extract_pdf_prefix(qid), gold_answer=gold_answer,
                    gold_elements=None, gold_pages=gold_pages, extra_info=extra_info
                )
                self.samples.append(sample)
                count += 1
            logger.info(f"✅ Successfully loaded {count} queries for {l}.")
    
    def _get_all_image_paths(self) -> List[str]:
        image_files = []
        for l in self.langs:
            corpus_root = self.corpus_roots[l]
            if not os.path.exists(corpus_root):
                continue
            logger.info(f"Scanning images in {corpus_root}...")
            for root, dirs, files in os.walk(corpus_root):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
        image_files.sort()
        logger.info(f"Found {len(image_files)} images total.")
        return image_files

    def _embed_images(self, image_paths: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        inputs = [{"image": p} for p in image_paths]
        embeddings = self.embedding_model.process(inputs)
        return embeddings.cpu().numpy().astype('float32')

    def _embed_text(self, text: str) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        inputs = [{"text": text, "instruction": "Represent the user's input."}]
        embeddings = self.embedding_model.process(inputs)
        return embeddings.cpu().numpy().astype('float32')

    def _pdf_to_images(self, pdf_path: str) -> Dict[int, str]:
        import glob
        pdf_prefix = os.path.basename(pdf_path)
        if pdf_prefix.lower().endswith('.pdf'):
            pdf_prefix = pdf_prefix[:-4]

        image_map = {}
        found_files = []
        
        for l in self.langs:
            corpus_root = self.corpus_roots[l]
            if not os.path.exists(corpus_root):
                continue
            search_pattern = os.path.join(corpus_root, f"{pdf_prefix}_*.png")
            found_files.extend(glob.glob(search_pattern))
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            match = re.search(r"_(\d+)\.png$", filename)
            if match:
                try:
                    page_num = int(match.group(1))
                    image_map[page_num] = file_path
                except ValueError:
                    continue
        
        if not image_map:
            logger.warning(f"Warning: No pre-processed images found for prefix '{pdf_prefix}' in any corpus directory.")

        return image_map

    def pipeline(self, query: str, image_paths: List[str] = None, top_k: int = 10, trunc_thres=0.0, trunc_bbox=False) -> List[PageElement]:
        if not image_paths:
            return []

        all_pages_to_process = []
        for path in image_paths:
            if not path.lower().endswith('.png'):
                page_map = self._pdf_to_images(path)
                for p_idx in sorted(page_map.keys()):
                    all_pages_to_process.append(PageElement(
                        bbox=[0, 0, 1000, 1000],
                        type="page_image",
                        corpus_id=os.path.basename(page_map[p_idx]),
                        corpus_path=page_map[p_idx],
                        crop_path=page_map[p_idx]
                    ))
            else:
                all_pages_to_process.append(PageElement(
                    bbox=[0, 0, 1000, 1000],
                    type="page_image",
                    corpus_path=path,
                    crop_path=path
                ))

        if self.rerank_model and len(all_pages_to_process) > top_k:
            ranked_pages = self.rerank(query, all_pages_to_process)
            target_pages = ranked_pages[:top_k]

            # --- Expansion Recall Logic (from MMLongLoader) ---
            expanded_target_pages = list(target_pages)
            existing_ids = set([p.corpus_id for p in target_pages])
            
            id_to_idx = {p.corpus_id: i for i, p in enumerate(all_pages_to_process)}
            
            for page in target_pages:
                if page.corpus_id in id_to_idx:
                    curr_idx = id_to_idx[page.corpus_id]
                    
                    if curr_idx > 0:
                        prev_page = all_pages_to_process[curr_idx - 1]
                        if prev_page.corpus_id not in existing_ids:
                            expanded_target_pages.append(prev_page)
                            existing_ids.add(prev_page.corpus_id)
                    
                    if curr_idx < len(all_pages_to_process) - 1:
                        next_page = all_pages_to_process[curr_idx + 1]
                        if next_page.corpus_id not in existing_ids:
                            expanded_target_pages.append(next_page)
                            existing_ids.add(next_page.corpus_id)
            
            target_pages = expanded_target_pages
            # --------------------------------------------------

            target_pages = [ page for page in target_pages if page.retrieval_score >= trunc_thres]
        else:
            target_pages = all_pages_to_process[:top_k]

        elements = self.extract_elements_from_pages(target_pages, query)
        if trunc_bbox:
            elements = elements[:top_k]
        return elements

    def rerank(self, query: str, pages: List[PageElement]) -> List[PageElement]:
        if not self.rerank_model or not pages:
            return pages
        logger.info(f"Reranking {len(pages)} pages...")
        
        documents_input = [{"text": f"Page ID: {page.corpus_id}", "image": page.corpus_path} for page in pages]
        rerank_input = {
            "instruction": (
                "Given a search query, retrieve relevant candidates that answer the query. "
                "Note that 'Page ID' indicates the physical page index in the document file, "
                "which does not necessarily correspond to the logical page number printed on the page image."
            ),
            "query": {"text": query},
            "documents": documents_input,
            "fps": 1.0 
        }
        
        try:
            scores = self.rerank_model.process(rerank_input)
            if len(scores) != len(pages):
                logger.warning(f"Warning: Reranker returned {len(scores)} scores for {len(pages)} pages.")
                return pages

            for page, score in zip(pages, scores):
                page.retrieval_score = score
                
            sorted_pages = sorted(pages, key=lambda x: x.retrieval_score, reverse=True)
            return sorted_pages
        except Exception as e:
            logger.exception(f"Error during reranking: {e}")
            return pages

    def is_valid_extracted_data(self, data):
        return is_valid_extracted_data(data)

    def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
        if self.extractor is None:
            return pages

        workspace_dir = os.path.abspath(os.path.join(self.output_dir, "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        fine_grained_elements = []
        
        for page in tqdm(pages, desc="Extracting Elements"):
            image_path = page.crop_path
            
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"Warning: Image path not found: {image_path}")
                continue

            try:
                extracted_data = run_extractor_with_optional_judger(
                    query=query,
                    image_path=image_path,
                    extractor=self.extractor,
                    judger=self.judger,
                    page_id=page.corpus_id,
                )

                if not extracted_data:
                    continue

                page_elements = extract_page_elements_from_data(
                    extracted_data=extracted_data,
                    image_path=image_path,
                    workspace_dir=workspace_dir,
                    corpus_id=page.corpus_id,
                    corpus_path=page.corpus_path,
                    retrieval_score=getattr(page, "retrieval_score", None),
                )
                fine_grained_elements.extend(page_elements)
                        
            except Exception as e:
                logger.exception(f"Error during agent execution on {page.corpus_id}: {e}")

        return fine_grained_elements

    # --------------------------------------------------------------------------------
    # Refactored Bad Case Handling
    # --------------------------------------------------------------------------------
    def _serialize_element(self, elem):
        """Helper to convert PageElement or dict to JSON-serializable dict"""
        if isinstance(elem, dict):
            return elem
        if hasattr(elem, '__dict__'):
            return elem.__dict__
        return str(elem)

    def _sanitize_filename(self, name):
        return "".join([c if c.isalnum() else "_" for c in str(name)])

    def save_bad_cases(self, output_dir: str, task: str):
        """
        Filter and save bad cases based on evaluation metrics.
        Saves separate JSON files for each category.
        """
        bad_case_dir = os.path.join(output_dir, "bad_cases")
        os.makedirs(bad_case_dir, exist_ok=True)

        retrieval_bad_cases = []
        generation_bad_cases = []
        retrieval_by_category = collections.defaultdict(list)
        generation_by_category = collections.defaultdict(list)

        logger.info(f"🔍 Analyzing {len(self.samples)} samples for bad cases...")

        for sample in self.samples:
            if sample.extra_info is None:
                continue
            
            # --- Parse Categories and Apply Cleaning Logic ---
            categories = sample.extra_info.get('category', ["Unknown"])
            if not isinstance(categories, list):
                categories = [str(categories)]
            if not categories:
                categories = ["Unknown"]
            
            # Normalize and group categories
            categories = [x.split('-')[0].split(' ')[0] for x in categories]

            metrics = sample.extra_info.get('metrics', {})
            flat_metrics = metrics.copy()
            
            if 'page' in metrics and isinstance(metrics['page'], dict):
                flat_metrics['page_recall'] = metrics['page'].get('recall', 0.0)
                flat_metrics['page_precision'] = metrics['page'].get('precision', 0.0)
            
            # 1. Retrieval Bad Case Check
            is_retrieval_bad = False
            if 'page_recall' in flat_metrics and flat_metrics['page_recall'] < 1.0:
                is_retrieval_bad = True
            elif 'page' in metrics and metrics['page'].get('recall', 1.0) < 1.0:
                is_retrieval_bad = True
                flat_metrics['page_recall'] = metrics['page'].get('recall')
                flat_metrics['page_precision'] = metrics['page'].get('precision')

            # 2. Generation Bad Case Check
            is_generation_bad = False
            if 'model_eval' in metrics and metrics['model_eval'] < 1.0:
                is_generation_bad = True

            sample_dict = {
                "qid": str(sample.qid),
                "query": sample.query,
                "gold_answer": sample.gold_answer,
                "gold_pages": sample.gold_pages,
                "category": categories,
                "final_answer": sample.extra_info.get("final_answer", ""),
                "metrics": flat_metrics,
                "retrieved_elements": [self._serialize_element(e) for e in sample.extra_info.get("retrieved_elements", [])],
                "doc_source": sample.data_source
            }

            if is_retrieval_bad:
                retrieval_bad_cases.append(sample_dict)
                for cat in categories:
                    retrieval_by_category[cat].append(sample_dict)
            
            if is_generation_bad:
                generation_bad_cases.append(sample_dict)
                for cat in categories:
                    generation_by_category[cat].append(sample_dict)

        # --- Save Retrieval Bad Cases ---
        if task in ["retrieval", "all"] and retrieval_bad_cases:
            p_all = os.path.join(bad_case_dir, "retrieval_bad_cases_all.json")
            with open(p_all, "w", encoding="utf-8") as f:
                json.dump(retrieval_bad_cases, f, indent=2, ensure_ascii=False)
            logger.info(f"📉 Saved {len(retrieval_bad_cases)} total retrieval bad cases to {p_all}")

            for cat, cases in retrieval_by_category.items():
                safe_name = self._sanitize_filename(cat)
                p_cat = os.path.join(bad_case_dir, f"retrieval_bad_cases_{safe_name}.json")
                with open(p_cat, "w", encoding="utf-8") as f:
                    json.dump(cases, f, indent=2, ensure_ascii=False)

        # --- Save Generation Bad Cases ---
        if task in ["generation", "all"] and generation_bad_cases:
            p_all = os.path.join(bad_case_dir, "generation_bad_cases_all.json")
            with open(p_all, "w", encoding="utf-8") as f:
                json.dump(generation_bad_cases, f, indent=2, ensure_ascii=False)
            logger.info(f"📉 Saved {len(generation_bad_cases)} total generation bad cases to {p_all}")

            for cat, cases in generation_by_category.items():
                safe_name = self._sanitize_filename(cat)
                p_cat = os.path.join(bad_case_dir, f"generation_bad_cases_{safe_name}.json")
                with open(p_cat, "w", encoding="utf-8") as f:
                    json.dump(cases, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------------------------
    # Evaluation Methods
    # --------------------------------------------------------------------------------
    def _evaluate_answer_correctness(self, sample: StandardSample) -> Dict[str, Any]:
        """"""
        query_text = sample.query
        expected_answer = sample.gold_answer
        actual_answer = (sample.extra_info.get('final_answer') 
                        if sample.extra_info else None) or "error output"

        model_score = 0.0
        reasoning = "No reasoning provided."

        if not actual_answer or "error output" in actual_answer or not self.llm_caller:
            return {"model_eval": 0.0, "raw_score": 1, "eval_reason": "Invalid answer or missing LLM"}

        evaluation_prompt = f"""You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query and reference answer
- a generated answer

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning before the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains factual errors or significant hallucinations, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct according to the reference, you should give a score between 4 and 5.

**Special Instruction for Open-Ended Questions:**
- If the user query is open-ended (allowing for multiple valid answers), the generated answer DOES NOT need to match the reference answer exactly.
- As long as the generated answer is highly relevant and contains no factual conflicts with the reference, give a high score.

**Special Instruction for Numerical and Logical Equivalence:**
- If the generated answer represents the **same value or concept** as the reference answer but in a different format, unit, or perspective, it MUST be considered CORRECT.
- **Unit Conversion:** (e.g., Reference: "1 kilometer", Generated: "1000 meters" -> Correct).
- **Format Differences:** (e.g., Reference: "0.5", Generated: "50%" or "1/2" -> Correct).
- **Absolute vs. Relative:** If the generated answer uses a relative value (e.g., percentage) while the reference uses an absolute value (or vice versa), and they are mathematically consistent based on the context, treat it as correct.
- You should perform necessary mental calculations to verify if the generated answer can be derived from the reference or the reference can be derived from the generated answer.

Example Response:
REASON: The generated answer uses '500 meters' while the reference says '0.5 km'. These are mathematically equivalent, so the answer is correct.
SCORE: 5

User:
## User Query
{query_text}

## Reference Answer
{expected_answer}

## Generated Answer
{actual_answer}
"""

        try:
            response_text = self.llm_caller(evaluation_prompt)
            
            score_match = re.search(r"SCORE:\s*(\d+)", response_text, re.IGNORECASE)
            if not score_match:
                score_match = re.search(r"(\d+)\s*$", response_text.strip())
                
            reason_match = re.search(r"REASON:\s*(.*)", response_text, re.IGNORECASE)
            if reason_match:
                reasoning = reason_match.group(1).strip()

            if score_match:
                raw_score = int(score_match.group(1))
                raw_score = max(1, min(5, raw_score))
                model_score = 1.0 if raw_score >= 4 else 0.0
            else:
                logger.warning(f"Warning: Could not parse score from response for QID {sample.qid}")
                model_score = 0.0

        except Exception as e:
            logger.exception(f"Error during model eval for QID {sample.qid}: {e}")
            model_score = 0.0

        return {
            "model_eval": model_score,
            "raw_score": raw_score if 'raw_score' in locals() else 1,
            "eval_reason": reasoning
        }

    def _compute_element_metrics(self, pred_elements: List[PageElement], gold_elements: List[PageElement], threshold: float = 0.5) -> Dict[str, float]:
        """ (BBox)  Precision, Recall  F1"""
        return compute_element_metrics(pred_elements, gold_elements, threshold)

    def evaluate_retrieval(self) -> Dict[str, float]:
        """
        Retrieval Task Evaluation:
        - Page Recall / Precision, Element (BBox) Recall / Precision / F1
        -  Category
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        category_metrics = collections.defaultdict(lambda: collections.defaultdict(float))
        category_counts = collections.defaultdict(int)

        logger.info(f"Starting Retrieval Evaluation on {len(self.samples)} samples...")

        for sample in tqdm(self.samples, desc="Evaluating Retrieval"):
            if sample.extra_info is None:
                sample.extra_info = {}
            
            current_metrics = sample.extra_info.get('metrics', {})
            retrieval_sample_metrics = {}

            # --- Parse Categories and Apply Cleaning Logic ---
            categories = sample.extra_info.get('category', ["Unknown"])
            if not isinstance(categories, list):
                categories = [str(categories)]
            if not categories:
                categories = ["Unknown"]
            
            # Normalize categories (e.g. "Text Inference" -> "Text")
            categories = [x.split('-')[0].split(' ')[0] for x in categories]

            retrieved_elements = sample.extra_info.get('retrieved_elements', [])
            elements_obj = []
            unique_retrieved_pages = set()
            
            for el in retrieved_elements:
                if isinstance(el, dict):
                     valid_keys = PageElement.__annotations__.keys()
                     pe = PageElement(**{k:v for k,v in el.items() if k in valid_keys})
                     elements_obj.append(pe)
                     if pe.corpus_id: unique_retrieved_pages.add(pe.corpus_id)
                elif isinstance(el, PageElement):
                     elements_obj.append(el)
                     if el.corpus_id: unique_retrieved_pages.add(el.corpus_id)

            retrieved_count = len(unique_retrieved_pages)
            total_metrics['retrieved_page_count'] += retrieved_count

            target_gold_pages = sample.gold_pages if sample.gold_pages else sample.extra_info.get('from_pages', [])
            current_page_recall = 0.0
            current_page_precision = 0.0
            
            if target_gold_pages:
                page_res = self._compute_page_metrics(elements_obj, target_gold_pages)
                current_page_recall = page_res['recall']
                current_page_precision = page_res['precision']
                
                total_metrics['page_recall'] += current_page_recall
                total_metrics['page_precision'] += current_page_precision
                retrieval_sample_metrics.update({"page_recall": current_page_recall, "page_precision": current_page_precision})
                counts['page'] += 1

            if sample.gold_elements:
                elem_res = self._compute_element_metrics(elements_obj, sample.gold_elements)
                if elem_res:
                    total_metrics['element_recall'] += elem_res['element_recall']
                    total_metrics['element_precision'] += elem_res['element_precision']
                    total_metrics['element_f1'] += elem_res['element_f1']
                    retrieval_sample_metrics.update(elem_res)
                    counts['element'] += 1

            # --- Category Based Grouping ---
            for cat in categories:
                category_counts[cat] += 1
                category_metrics[cat]['retrieved_page_count'] += retrieved_count
                if target_gold_pages:
                    category_metrics[cat]['page_recall'] += current_page_recall
                    category_metrics[cat]['page_precision'] += current_page_precision
                    category_metrics[cat]['count_with_gold'] += 1

            current_metrics.update(retrieval_sample_metrics)
            sample.extra_info['metrics'] = current_metrics

        avg_results = {}
        if len(self.samples) > 0:
            avg_results['avg_retrieved_page_count'] = total_metrics['retrieved_page_count'] / len(self.samples)
            
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
            
        if counts['element'] > 0:
            avg_results['avg_element_recall'] = total_metrics['element_recall'] / counts['element']
            avg_results['avg_element_precision'] = total_metrics['element_precision'] / counts['element']
            avg_results['avg_element_f1'] = total_metrics['element_f1'] / counts['element']
        
        # --- Source-based Averages ---
        for cat, metrics in category_metrics.items():
            count_all = category_counts[cat]
            count_gold = metrics.get('count_with_gold', 0)
            
            if count_all > 0:
                avg_results[f'avg_retrieved_page_count_{cat}'] = metrics['retrieved_page_count'] / count_all
            
            if count_gold > 0:
                avg_results[f'avg_page_recall_{cat}'] = metrics['page_recall'] / count_gold
                avg_results[f'avg_page_precision_{cat}'] = metrics['page_precision'] / count_gold
                
        return avg_results

    def evaluate_generation(self, num_threads: int = 8) -> Dict[str, float]:
        """
        Generation Task Evaluation:
        - Model Evaluation (LLM Judge)
        -  Token  Category 
        """
        total_metrics = collections.defaultdict(float)
        counts = collections.defaultdict(int)

        logger.info(f"Starting Generation Evaluation on {len(self.samples)} samples with {num_threads} workers...")
        
        def process_single_sample(sample):
            if sample.extra_info is None:
                sample.extra_info = {}

            current_metrics = sample.extra_info.get('metrics', {})
            gen_sample_metrics = {}

            corr_metrics = self._evaluate_answer_correctness(sample)
            score = corr_metrics['model_eval']
            
            gen_sample_metrics.update(corr_metrics)
            current_metrics.update(gen_sample_metrics)
            sample.extra_info['metrics'] = current_metrics
            
            return score, sample

        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_sample = {executor.submit(process_single_sample, sample): sample for sample in self.samples}
                
                for future in tqdm(as_completed(future_to_sample), total=len(self.samples), desc="Evaluating Generation"):
                    try:
                        score, sample = future.result()
                        total_metrics['model_eval'] += score
                        counts['total'] += 1
                        
                        # --- Tokens Tracking ---
                        total_metrics['prompt_tokens'] += sample.extra_info.get('prompt_tokens', 0)
                        total_metrics['completion_tokens'] += sample.extra_info.get('completion_tokens', 0)
                        counts['token_count_samples'] += 1
                        
                        # --- Category Based Evaluation ---
                        categories = sample.extra_info.get('category', ["Unknown"])
                        if not isinstance(categories, list): categories = [str(categories)]
                        if not categories: categories = ["Unknown"]
                        
                        # Normalize categories (e.g. "Text Inference" -> "Text")
                        categories = [x.split('-')[0].split(' ')[0] for x in categories]
                        
                        for cat in categories:
                            total_metrics[f'model_eval_{cat}'] += score
                            counts[f'count_{cat}'] += 1

                    except Exception as e:
                        logger.exception(f"Error processing sample in thread: {e}")
        else:
            for sample in tqdm(self.samples, desc="Evaluating Generation"):
                score, sample = process_single_sample(sample)
                total_metrics['model_eval'] += score
                counts['total'] += 1
                
                # --- Tokens Tracking ---
                total_metrics['prompt_tokens'] += sample.extra_info.get('prompt_tokens', 0)
                total_metrics['completion_tokens'] += sample.extra_info.get('completion_tokens', 0)
                counts['token_count_samples'] += 1
                
                # --- Category Based Evaluation ---
                categories = sample.extra_info.get('category', ["Unknown"])
                if not isinstance(categories, list): categories = [str(categories)]
                if not categories: categories = ["Unknown"]
                
                # Normalize categories (e.g. "Text Inference" -> "Text")
                categories = [x.split('-')[0].split(' ')[0] for x in categories]
                
                for cat in categories:
                    total_metrics[f'model_eval_{cat}'] += score
                    counts[f'count_{cat}'] += 1

        avg_results = {}
        if counts['total'] > 0:
            avg_results['avg_model_eval'] = total_metrics['model_eval'] / counts['total']
            
        # Category Averages
        for key in total_metrics:
            if key.startswith('model_eval_'):
                cat = key.replace('model_eval_', '')
                cnt = counts[f'count_{cat}']
                if cnt > 0:
                    avg_results[f'avg_model_eval_{cat}'] = total_metrics[key] / cnt
                    
        # Token Averages
        if counts['token_count_samples'] > 0:
            avg_results['avg_input_tokens'] = total_metrics['prompt_tokens'] / counts['token_count_samples']
            avg_results['avg_output_tokens'] = total_metrics['completion_tokens'] / counts['token_count_samples']
        
        return avg_results

    def evaluate(self) -> Dict[str, float]:
        """
        Legacy wrapper for full evaluation.
        Calls both retrieval and generation evaluations and merges results.
        """
        logger.info("Running Full Evaluation Pipeline...")
        
        results = {}
        r_metrics = self.evaluate_retrieval()
        results.update(r_metrics)
        g_metrics = self.evaluate_generation()
        results.update(g_metrics)
        
        logger.info(f"Final Combined Results: {results}")
        return results

if __name__ == "__main__":
    # 
    embedding_model_path = "/path/to/Qwen3-VL-Embedding-8B"
    reranker_model_path = "http://localhost:8003"
    root_dir = "/path/to/FinRAGBench-V"
    
    from src.agents.utils import ImageZoomOCRTool
    tool_work_dir = "./workspace" 
    
    logger.info("Initializing Models...")
    embedder = Qwen3VLEmbedder(model_name_or_path=embedding_model_path, torch_dtype=torch.float16)
    reranker = Qwen3VLReranker(model_name_or_path=reranker_model_path, torch_dtype=torch.float16)
    
    tool = ImageZoomOCRTool(work_dir=tool_work_dir)
    extractor = AgenticOCR(
        base_url="http://localhost:8001/v1",
        api_key="sk-123456",
        model_name="MinerU-Agent-CK800",
        tool=tool
    )

    loader = FinRAGLoader(
        data_root=root_dir, 
        lang="both", 
        embedding_model=embedder, 
        rerank_model=reranker,
        extractor=extractor
    )
    
    loader.llm_caller = create_llm_caller()
    
    loader.load_data()
    
    if len(loader.samples) > 0:
        test_sample = loader.samples[0]
        logger.info(f"\nTesting Query: {test_sample.query}")
        
        results = loader.pipeline(test_sample.query, image_paths=[test_sample.data_source], top_k=10) 
        test_sample.extra_info['final_answer'] = "Generated Answer Here..." 
        test_sample.extra_info['retrieved_elements'] = results
        
        logger.info("\n--- Testing Split Interfaces ---")
        loader.evaluate_retrieval()
        loader.evaluate_generation()
        
        logger.info("\n--- Testing Legacy Interface ---")
        loader.evaluate()
