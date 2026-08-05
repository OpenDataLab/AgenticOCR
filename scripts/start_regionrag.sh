#!/bin/bash
# Launch the RegionRAG region-extraction service on a GPU host.
#
# Requirements on this host:
#   cd RegionRAG && pip install -e .        # installs colpali_engine
#   pip install fastapi uvicorn pydantic
#   huggingface-cli download Aeryn666/RegionRet --local-dir RegionRAG/models/RegionRet
#
# The ColQwen2.5 model holds on to CUDA state, so we run a single uvicorn
# worker and rely on request-level serialization (a threading.Lock inside
# the server) plus client-side parallelism for throughput.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Path to the RegionRet LoRA adapter directory. Override with REGIONRAG_MODEL_PATH.
export REGIONRAG_MODEL_PATH=${REGIONRAG_MODEL_PATH:-RegionRAG/models/RegionRet}
# Path to the ColQwen2.5 base model. If empty, the server reads it from the
# adapter_config.json's base_model_name_or_path field. Override with
# REGIONRAG_BASE_PATH when the packaged relative path doesn't resolve.
export REGIONRAG_BASE_PATH=${REGIONRAG_BASE_PATH:-RegionRAG/models/colqwen2.5-base}
export REGIONRAG_PORT=${REGIONRAG_PORT:-8005}

# RegionRAG hyperparameters used by the /v1/chat/completions endpoint.
# /extract_regions reads its params from the request body, but the OpenAI
# wrapper has no way to pass these per-request — pin them at startup.
export REGIONRAG_NEIGHBOR_RANGE=${REGIONRAG_NEIGHBOR_RANGE:-2}
export REGIONRAG_BBOX_THRESHOLD=${REGIONRAG_BBOX_THRESHOLD:-0.25}
export REGIONRAG_SCORE_METHOD=${REGIONRAG_SCORE_METHOD:-max}
export REGIONRAG_MAX_REGIONS=${REGIONRAG_MAX_REGIONS:-20}

HOST=${HOST:-0.0.0.0}
PORT=${REGIONRAG_PORT}

# Run from the project root so "src.agents.regionrag_server" resolves.
cd "$(dirname "$0")/.."

echo "Starting RegionRAG service on ${HOST}:${PORT}"
echo "  MODEL_PATH (LoRA) = ${REGIONRAG_MODEL_PATH}"
echo "  BASE_PATH         = ${REGIONRAG_BASE_PATH}"
echo "  CUDA_VISIBLE_DEV  = ${CUDA_VISIBLE_DEVICES}"
echo "  Chat /v1/chat/completions hyperparams:"
echo "    neighbor_range  = ${REGIONRAG_NEIGHBOR_RANGE}"
echo "    bbox_threshold  = ${REGIONRAG_BBOX_THRESHOLD}"
echo "    score_method    = ${REGIONRAG_SCORE_METHOD}"
echo "    max_regions     = ${REGIONRAG_MAX_REGIONS}"

python -m uvicorn src.agents.regionrag_server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1
