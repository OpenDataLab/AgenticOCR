export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
  --model xxxxxxxxx \
  --served-model-name xxxxxxxxx \
  --port 8001 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7 \
  --max-num-batched-tokens 32768 \
  --mm-processor-cache-gb 0 \
  --compilation_config.cudagraph_mode PIECEWISE