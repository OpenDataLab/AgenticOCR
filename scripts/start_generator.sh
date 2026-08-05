export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --served-model-name xxxxxxxxx \
  --model xxxxxxxxx \
  --port 8002 \
  --max_model_len 65536 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9