#!/usr/bin/env bash
set -euo pipefail

start_time=$(date +%s)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Input PDF path (file or directory)
INPUT_PATH="${INPUT_PATH:-$SCRIPT_DIR/pdfs}"
# Output root directory
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/gen}"
# Template file path (required)
TEMPLATES_PATH="${TEMPLATES_PATH:-$SCRIPT_DIR/example_template/finance_template.json}"
# Prompt override config path (optional)
PROMPT_OVERRIDES_PATH="${PROMPT_OVERRIDES_PATH:-}"
export OUTPUT_ROOT

# Language: CN or EN
LANGUAGE="${LANGUAGE:-CN}"
# Start step (0-8)
START_STEP="${START_STEP:-0}"
# End step (0-8)
END_STEP="${END_STEP:-8}"

# OCR service URL (set externally)
export OMNIDOC_MINERU_SERVER_URL="${OMNIDOC_MINERU_SERVER_URL:-}"
# Generation model service URL (set externally)
export OMNIDOC_GENAI_BASE_URL="${OMNIDOC_GENAI_BASE_URL:-}"
export OMNIDOC_GENAI_API_KEY="${OMNIDOC_GENAI_API_KEY:-${GOOGLE_API_KEY:-no-key-required}}"
# Step7 check model service URL (set externally)
export OMNIDOC_QWEN_BASE_URL="${OMNIDOC_QWEN_BASE_URL:-}"
export OMNIDOC_QWEN_API_KEY="${OMNIDOC_QWEN_API_KEY:-no-key-required}"
export OMNIDOC_QWEN_MODEL="${OMNIDOC_QWEN_MODEL:-}"

require_env() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "[ERROR] Required environment variable is empty: $name" >&2
    exit 1
  fi
}

if [[ -z "$TEMPLATES_PATH" ]]; then
  echo "[ERROR] TEMPLATES_PATH is required." >&2
  exit 1
fi

require_env "OMNIDOC_GENAI_BASE_URL"
require_env "OMNIDOC_GENAI_API_KEY"

if (( START_STEP <= 0 && END_STEP >= 0 )); then
  require_env "OMNIDOC_MINERU_SERVER_URL"
fi

if (( END_STEP >= 7 )); then
  require_env "OMNIDOC_QWEN_BASE_URL"
  require_env "OMNIDOC_QWEN_API_KEY"
  require_env "OMNIDOC_QWEN_MODEL"
fi

cmd=(
  python "$SCRIPT_DIR/pipeline_runner.py"
  --input-path "$INPUT_PATH"
  --output-root "$OUTPUT_ROOT"
  --templates-path "$TEMPLATES_PATH"
  --language "$LANGUAGE"
  --start-step "$START_STEP"
  --end-step "$END_STEP"
)

if [[ -n "$PROMPT_OVERRIDES_PATH" ]]; then
  cmd+=(--prompt-overrides "$PROMPT_OVERRIDES_PATH")
fi

echo "[RUN] ${cmd[*]}"
"${cmd[@]}"

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "[SUMMARY] Pipeline finished in ${duration}s"
echo "[DONE] Results saved to: $OUTPUT_ROOT/results"
