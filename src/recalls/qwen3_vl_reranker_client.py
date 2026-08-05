# qwen3_vl_reranker_client.py
import argparse
import requests
import json
from typing import Dict, Any, List, Optional
import sys

class Qwen3VLReranker:
    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Initialize the Reranker Client (Remote Mode).
        """
        self.api_url = model_name_or_path
        if not self.api_url.endswith("/rerank"):
             self.api_url = self.api_url + "/rerank"
        # 
        # try:
        #     requests.get(self.api_url.replace("/rerank", "/health"), timeout=1)
        # except Exception:
        #     print(f"Warning: Could not connect to {self.api_url} immediately. Ensure server is running.")

    def process(self, inputs: Dict[str, Any]) -> List[float]:
        """
        Send inputs to the remote vLLM service and retrieve relevance scores.
        
        Args:
            inputs: Dictionary containing 'instruction', 'query', and 'documents'.
                    Matches the structure used in the original local version.
        
        Returns:
            List[float]: A list of relevance scores corresponding to the documents.
        """
        try:
            # client.py  JSON
            response = requests.post(self.api_url, json=inputs)
            response.raise_for_status()
            
            result = response.json()
            scores = result.get("scores", [])
            return scores
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling remote service: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Server Response: {e.response.text}")
            raise e


# --- Usage Example ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 
    parser.add_argument("--model_name_or_path", type=str, default='http://localhost:8000')
    args, _ = parser.parse_known_args()

    #  Reranker (Client)
    reranker = Qwen3VLReranker(model_name_or_path=args.model_name_or_path)

    #  ()
    docs = [
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
        },
        {
            #  URL
            "image": "/path/to/demo.jpeg"
        },
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
            "image": "/path/to/demo.jpeg"
        }
    ]
    
    # docs = [{"image": f"/path/to/FinRAGBench-V/data/corpus/en/img/FATF2024_{i}.png"} for i in range(1,203)]
    # docs = [{"image": f"/path/to/FinRAGBench-V/data/corpus/en/img/PROXY Print Set FINAL (no blank pages) pdfs_Current_Folio_Proxy_tmbsf_v3_F_{i}.png"} for i in range(1,68)]
    # docs = [{"image": f"/path/to/FinRAGBench-V/data/corpus/en/img/_{i}.png"} for i in range(1,26)]
    
    test_inputs = {
        "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
        "query": {
            "text": "A woman playing with her dog on a beach at sunset."
        },
        "documents": docs,
        "fps": 1.0
    }

    try:
        print("Running inference (Remote)...")
        scores = reranker.process(test_inputs)
        print("Relevance Scores:")
        for i, score in enumerate(scores):
            print(f"Document {i+1}: {score:.4f}")
            
    except Exception as e:
        print(f"An error occurred: {e}")