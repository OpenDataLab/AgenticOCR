#!/bin/bash

# 1.  CUDA  ID
#  4 export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2.  vLLM 
# (DP) TP  1
export TENSOR_PARALLEL_SIZE=1

# 
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 
PORT=8000
HOST="0.0.0.0"

#  GPU  workers 
#  + 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    #  nvidia-smi  1
    WORKERS=1
else
    # 
    WORKERS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
fi

echo "Cleaning up previous locks..."
rm -f /tmp/vllm_reranker_gpu.lock
rm -f /tmp/vllm_reranker_gpu.state

echo "Starting Qwen-VL Reranker Service on ${HOST}:${PORT}..."
echo "Config: Workers(DP)=${WORKERS}, TP per Worker=${TENSOR_PARALLEL_SIZE}, Devices=${CUDA_VISIBLE_DEVICES}"

# 
#  --workers  GPU 
python -m uvicorn qwen3_vl_reranker_server:app --host $HOST --port $PORT --workers $WORKERS