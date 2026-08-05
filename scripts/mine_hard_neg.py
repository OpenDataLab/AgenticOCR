import os
import json
import torch
from tqdm import tqdm

#  ()
from src.recalls.qwen3_vl_embedding import Qwen3VLEmbedder
from src.recalls.qwen3_vl_reranker_client import Qwen3VLReranker
#  FinRAGLoader  finrag_loader.py 
from src.loaders.FinRAGLoader import FinRAGLoader

# ================= Configuration =================
# 
EMBEDDING_MODEL_PATH = "/path/to/Qwen3-VL-Embedding-8B"
RERANKER_MODEL_PATH = "http://localhost:8003"
DATA_ROOT_DIR = "/path/to/FinRAGBench-V"
OUTPUT_FILE = "retrieved_labeled_results_ch.json"

TOP_K_RETRIEVE = 50   # 
TOP_K_RERANK = 10     # Rerank
# =================================================

def normalize_path_id(path_str):
    """
    
    (basename)/
    """
    if not path_str:
        return ""
    return os.path.basename(path_str).strip()

def main():
    print(">>> Initializing Models...")
    #  Embedder
    embedder = Qwen3VLEmbedder(
        model_name_or_path=EMBEDDING_MODEL_PATH, 
        torch_dtype=torch.float16
    )
    
    #  Reranker
    reranker = Qwen3VLReranker(
        model_name_or_path=RERANKER_MODEL_PATH, 
        torch_dtype=torch.float16
    )

    print(">>> Initializing Loader...")
    #  Extractor
    loader = FinRAGLoader(
        data_root=DATA_ROOT_DIR, 
        lang="ch", 
        embedding_model=embedder, 
        rerank_model=reranker,
        extractor=None 
    )

    # 1.  (Queries  Qrels)
    loader.load_data()
    # loader.samples = loader.samples[:10]
    
    # 2. 
    # force_rebuild=False 
    loader.build_page_vector_pool(batch_size=16, force_rebuild=False)

    results_data = []

    print(f">>> Starting Retrieval for {len(loader.samples)} queries...")

    for sample in tqdm(loader.samples, desc="Processing Queries"):
        query = sample.query
        qid = sample.qid
        
        #  Ground Truth  ()
        # sample.gold_pages  qrels ID  
        gold_ids = set([normalize_path_id(p) for p in sample.gold_pages])
        
        try:
            # Step 1:  (Retrieve)
            #  Reranker ( Top 50)
            initial_pages = loader.retrieve(query, top_k=TOP_K_RETRIEVE)
            
            # Step 2:  (Rerank)
            #  Qwen-VL  Query 
            reranked_pages = loader.rerank(query, initial_pages)
            
            #  Top K
            final_top_pages = reranked_pages[:TOP_K_RERANK]
            
            # 
            page_results = []
            for page in final_top_pages:
                #  ID ()
                pred_id = normalize_path_id(page.corpus_id)
                
                # 
                #  ID  gold_ids  1 (Positive) 0 (Negative)
                label = 1 if pred_id in gold_ids else 0
                
                page_info = {
                    "corpus_id": pred_id,          # 
                    "corpus_path": page.corpus_path, # 
                    "score": float(page.retrieval_score), # Reranker 
                    "label": label,                # 0  1
                    "is_ground_truth": bool(label)
                }
                page_results.append(page_info)

            #  Query 
            results_data.append({
                "query_id": qid,
                "query": query,
                "gold_pages": list(sample.gold_pages), #  GT 
                "retrieved_candidates": page_results
            })

        except Exception as e:
            print(f"Error processing query {qid}: {e}")
            continue

    # 3. 
    print(f">>> Saving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
    
    # 
    total_queries = len(results_data)
    queries_with_pos_retrieval = sum(1 for item in results_data if any(p['label'] == 1 for p in item['retrieved_candidates']))
    print(f"Done. Processed {total_queries} queries.")
    print(f"Queries with at least 1 correct page retrieved in Top-{TOP_K_RERANK}: {queries_with_pos_retrieval}/{total_queries}")

if __name__ == "__main__":
    main()
