import torch
import numpy as np
import logging
import requests
import uvicorn
import math  # 

from PIL import Image
from typing import List, Union, Optional, Any, Dict
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---  ---
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200

FRAME_FACTOR = 2
FPS = 1
MIN_FRAMES = 2
MAX_FRAMES = 64
MIN_TOTAL_PIXELS = 1 * FRAME_FACTOR * MIN_PIXELS
MAX_TOTAL_PIXELS = 4 * FRAME_FACTOR * MAX_PIXELS

# ---  ---
def sample_frames(frames, num_segments, max_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration - 1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except:
            break
        sampled_frames.append(single_frame_path)
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]

# --- FastAPI  ---
class RerankRequest(BaseModel):
    query: Dict[str, Any]
    documents: List[Dict[str, Any]]
    instruction: Optional[str] = None
    fps: float = FPS
    max_frames: int = MAX_FRAMES
    batch_size: Optional[int] = 50  #  batch_size 

# ---  ---
class Qwen3VLReranker:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        num_frames: int = MAX_FRAMES,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        **kwargs,
    ):
        self.default_instruction = default_instruction
        self.fps = fps
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.total_pixels = total_pixels
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_length = max_length
        
        if model_name_or_path.startswith("http://") or model_name_or_path.startswith("https://"):
            self.mode = "client"
            self.api_url = model_name_or_path.rstrip("/")
            logger.info(f"Initialized Qwen3VLReranker in CLIENT mode, connecting to {self.api_url}")
        else:
            self.mode = "local"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 
            lm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True, **kwargs
            ).to(self.device)

            self.model = lm.model
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True,
                padding_side='left' #  padding  Decoder-only 
            )
            self.model.eval()

            token_true_id = self.processor.tokenizer.get_vocab()["yes"]
            token_false_id = self.processor.tokenizer.get_vocab()["no"]
            self.score_linear = self.get_binary_linear(lm, token_true_id, token_false_id)
            self.score_linear.eval()
            self.score_linear.to(self.device).to(self.model.dtype)
            logger.info(f"Initialized Qwen3VLReranker in LOCAL mode with model {model_name_or_path}")

    def get_binary_linear(self, model, token_yes, token_no):
        lm_head_weights = model.lm_head.weight.data
        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]
        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer

    @torch.no_grad()
    def compute_scores(self, inputs):
        # 
        outputs = self.model(**inputs)
        #  token 
        #  left padding token 
        batch_scores = outputs.last_hidden_state[:, -1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

    def truncate_tokens_optimized(self, tokens: List[str], max_length: int, special_tokens: List[str]) -> List[str]:
        if len(tokens) <= max_length:
            return tokens
        special_tokens_set = set(special_tokens)
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max_length - num_special
        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1
        return final_tokens

    def tokenize(self, pairs: list, **kwargs):
        # pairs  batch  list [[{role, content}, ...], [{role, content}, ...]]
        max_length = self.max_length
        
        # apply_chat_template  list of lists (batch)
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
        
        try:
            # process_vision_info  Qwen2/3-VL utils  batch 
            images, videos, video_kwargs = process_vision_info(
                pairs, image_patch_size=16, 
                return_video_kwargs=True, 
                return_video_metadata=True
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            videos = None
            video_kwargs = {'do_sample_frames': False}
            #  batch  ()
            text = self.processor.apply_chat_template(
                [[{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}] * len(pairs)], 
                add_generation_prompt=True, tokenize=False
            )
        
        if videos is not None:
            #  zip(*videos)  batch 
            # Qwen-VL-Utils 
            try:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            except Exception:
                # Fallback if structure varies
                video_metadatas = None
        else:
            video_metadatas = None
            
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False, # 
            padding=True,     #  padding  Batch
            do_resize=False,
            padding_side='left', #  padding side
            return_tensors="pt", #  tensor
            **video_kwargs
        )

        #  ( Tensor )
        #  processor  truncation=True 
        #  token  Batch 
        
        #  batch  Python 
        #  tokenizer  truncation
        #  input_ids  Tensor 
        
        input_ids = inputs['input_ids']
        if input_ids.shape[1] > self.max_length:
             #  self.max_length ()
             # " token" batch 
             #  list  pad
             
             #  A:  ()
             inputs['input_ids'] = input_ids[:, -self.max_length:]
             inputs['attention_mask'] = inputs['attention_mask'][:, -self.max_length:]
             #  pixel_values  tensor vision token
             
        return inputs

    def format_mm_content(self, text, image, video, prefix='Query:', fps=None, max_frames=None):
        content = []
        content.append({'type': 'text', 'text': prefix})
        if not text and not image and not video:
            content.append({'type': 'text', 'text': "NULL"})
            return content

        if video:
            video_content = None
            video_kwargs = { 'total_pixels': self.total_pixels }
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    # [FIX]  sample_frames  self._sample_frames
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = [
                    ('file://' + ele if isinstance(ele, str) else ele) 
                    for ele in video_content
                ]
            elif isinstance(video, str):
                video_content = video if video.startswith(('http://', 'https://')) else 'file://' + video
                video_kwargs = {'fps': fps or self.fps, 'max_frames': max_frames or self.max_frames,}
            else:
                raise TypeError(f"Unrecognized video type: {type(video)}")

            if video_content:
                content.append({
                    'type': 'video', 'video': video_content,
                    **video_kwargs
                })

        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")

            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        if text:
            content.append({'type': 'text', 'text': text})
        return content

    def format_mm_instruction(self, query_text, query_image, query_video, doc_text, doc_image, doc_video, instruction=None, fps=None, max_frames=None):
        # ... () ...
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{"type": "text", "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."}]
        })
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction
        contents = []
        contents.append({"type": "text", "text": '<Instruct>: ' + instruct})
        query_content = self.format_mm_content(query_text, query_image, query_video, prefix='<Query>:', fps=fps, max_frames=max_frames)
        contents.extend(query_content)
        doc_content = self.format_mm_content(doc_text, doc_image, doc_video, prefix='\n<Document>:', fps=fps, max_frames=max_frames)
        contents.extend(doc_content)
        inputs.append({"role": "user", "content": contents})
        return inputs

    def process(self, inputs: Union[Dict, RerankRequest], batch_size: int = 50) -> List[float]:
        """
         Process  Batch 
        batch_size:  50
        """
        if isinstance(inputs, RerankRequest):
            #  batch_size
            if inputs.batch_size is not None:
                batch_size = inputs.batch_size
            inputs = inputs.model_dump()

        # --- Client Mode ---
        if self.mode == "client":
            try:
                endpoint = f"{self.api_url}/rerank"
                response = requests.post(endpoint, json=inputs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                raise e

        # --- Local Mode ---
        instruction = inputs.get('instruction', self.default_instruction) or self.default_instruction
        query = inputs.get("query", {})
        documents = inputs.get("documents", [])
        fps = inputs.get('fps', self.fps)
        max_frames = inputs.get('max_frames', self.max_frames)
        
        if not query or not documents:
            return []

        # 1.  (Formatting is fast, do it all at once)
        pairs = [self.format_mm_instruction(
            query.get('text', None),
            query.get('image', None),
            query.get('video', None),
            document.get('text', None),
            document.get('image', None),
            document.get('video', None),
            instruction=instruction,
            fps=fps,
            max_frames=max_frames
        ) for document in documents]

        final_scores = []
        num_docs = len(pairs)
        print(pairs[0])
        
        # 2.  batch_size 
        for i in range(0, num_docs, batch_size):
            batch_pairs = pairs[i : i + batch_size]
            
            try:
                with torch.no_grad():
                    # Tokenize  Batch
                    model_inputs = self.tokenize(batch_pairs)
                    # 
                    model_inputs = model_inputs.to(self.model.device)
                    # 
                    scores = self.compute_scores(model_inputs)
                    
                    #  scores  float (batch=1) list
                    if isinstance(scores, float):
                        scores = [scores]
                    
                    final_scores.extend(scores)
                    
                # 
                # torch.cuda.empty_cache() 
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("OOM detected. Consider reducing batch_size.")
                    torch.cuda.empty_cache()
                    raise e
                else:
                    raise e

        return final_scores

# ---  ---
def create_app(model_path: str, **kwargs):
    app = FastAPI(title="Qwen3VL Reranker API")
    reranker_instance = None

    @app.on_event("startup")
    async def startup_event():
        nonlocal reranker_instance
        logger.info(f"Loading model from {model_path}...")
        reranker_instance = Qwen3VLReranker(model_name_or_path=model_path, **kwargs)
        logger.info("Model loaded successfully.")

    @app.post("/rerank", response_model=List[float])
    async def rerank_endpoint(request: RerankRequest):
        if not reranker_instance:
             raise HTTPException(status_code=500, detail="Model not initialized")
        try:
            #  request process  batch_size
            scores = reranker_instance.process(request)
            return scores
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    return app

# ---  ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Qwen3VL Reranker Service or Client Test")
    parser.add_argument("--mode", type=str, choices=["server", "client_test"], default="server")
    parser.add_argument("--model_path", type=str, default="/path/to/Qwen3-VL-Reranker-8B", help="Local model path")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--api_url", type=str, default="http://localhost:8003")
    
    args = parser.parse_args()

    if args.mode == "server":
        # 
        #  uvicorn 
        # uvicorn your_script:app --host 0.0.0.0 --port 8000
        app = create_app(args.model_path)
        uvicorn.run(app, host=args.host, port=args.port)
        
    elif args.mode == "client_test":
        # 
        #  URL 
        client_reranker = Qwen3VLReranker(model_name_or_path=args.api_url)
        
        test_inputs = {
            "query": {"text": "Describe the image"},
            "documents": [
                {"text": "A cat sitting on a sofa", "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen/qwen-vl.png"},
                {"text": "A dog running in the park"}
            ],
            "instruction": "Does the text match the image?"
        }
        
        print(f"Sending request to {args.api_url}...")
        scores = client_reranker.process(test_inputs)
        print("Scores:", scores)