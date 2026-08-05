# OmniDocSynth

## Pipeline Overview
1. Parse PDF pages and OCR/layout elements.
2. Build document outline.
3. Generate evidence chains.
4. Select question templates.
5. Generate QA pairs.
6. Verify QA quality.
7. Rewrite questions to be more natural.
8. Filter by difficulty and tag evidence necessity.

The main entry is:
- `run_pipeline.sh`

## Environment Setup

### 1) Python environment
Use Python 3.10+ (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Model / service environment variables
Set these before running (no hardcoded defaults in this repo).
Required in all runs:

```bash
export OMNIDOC_GENAI_BASE_URL="<your_genai_base_url>"
export OMNIDOC_GENAI_API_KEY="<your_genai_api_key>"
```

Required when your step range includes Step 0:

```bash
export OMNIDOC_MINERU_SERVER_URL="<your_mineru_server_url>"
```

Required when your step range includes Step 7 or later:

```bash
export OMNIDOC_QWEN_BASE_URL="<your_step7_check_base_url>"
export OMNIDOC_QWEN_API_KEY="<your_step7_check_api_key>"
export OMNIDOC_QWEN_MODEL="<your_step7_check_model_name>"
```

## Required Bash Parameters (Minimal)
Set only these necessary parameters for `run_pipeline.sh`:

```bash
export INPUT_PATH="/path/to/pdf_or_pdf_dir"
export OUTPUT_ROOT="/path/to/output_dir"
export TEMPLATES_PATH="./example_template/finance_template.json"
```

Optional but commonly used:

```bash
export LANGUAGE="CN"            # CN or EN
export START_STEP=0              # start step index
export END_STEP=8                # end step index
```

## Run

```bash
bash run_pipeline.sh
```

## Output
Results are written under:
- `$OUTPUT_ROOT/results/step_0` ... `$OUTPUT_ROOT/results/step_8`
