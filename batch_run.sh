#!/bin/bash

# 
TEMPLATE_FILE="configs/test_gemini.yaml"
# 
TMP_CONFIGS_DIR="configs/tmp"
# Prompt
PROMPT_BASE_PATH="prompts"
# 
OUTPUT_BASE_DIR="outputs/0426_mmlong_gemini25"

# Prompt
PROMPTS=("prompt0.txt")

# 
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file '$TEMPLATE_FILE' not found!"
    exit 1
fi

# configs
mkdir -p $TMP_CONFIGS_DIR

#  Prompt
for PROMPT_FILE in "${PROMPTS[@]}"; do
    # Prompt ( prompt0)
    PROMPT_NAME=$(basename "$PROMPT_FILE" .txt)
    
    # Prompt
    FULL_PROMPT_PATH="${PROMPT_BASE_PATH}/${PROMPT_FILE}"

    # 
    for SETTING_ID in {5,6}; do
        echo "-------------------------------------------------------------------"
        echo "Running: Prompt=[$PROMPT_NAME] | Setting=[$SETTING_ID]"
        
        # 
        CONFIG_FILE="${TMP_CONFIGS_DIR}/${PROMPT_NAME}_set${SETTING_ID}.yaml"
        
        # 
        OUTPUT_DIR="${OUTPUT_BASE_DIR}_${PROMPT_NAME}_set${SETTING_ID}"

        # 
        cp "$TEMPLATE_FILE" "$CONFIG_FILE"

        # 1.  rag_system_prompt
        #  |  / 
        sed -i "s|rag_system_prompt: .*|rag_system_prompt: ${FULL_PROMPT_PATH}|g" "$CONFIG_FILE"

        # 2.  output_dir
        sed -i "s|output_dir: .*|output_dir: ${OUTPUT_DIR}|g" "$CONFIG_FILE"

        # 3. ID
        #  false 
        # key
        
        case $SETTING_ID in
            1)
                # Setting 1: use_page: true, use_page_ocr: false, use_crop: false, use_ocr: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: false/g' "$CONFIG_FILE"
                ;;
            2)
                # Setting 2: use_page: true, use_page_ocr: true, use_crop: false, use_ocr: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: false/g' "$CONFIG_FILE"
                ;;
            3)
                # Setting 3: use_page: true, use_crop: true, use_ocr: false
                # (use_page_ocr falsefalse)
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: false/g' "$CONFIG_FILE"
                ;;
            4)
                # Setting 4: use_page: true, use_crop: true, use_ocr: true, use_ocr_both: true
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: true/g' "$CONFIG_FILE"
                ;;
            5)
                # Setting 5: use_page: true, use_crop: true, use_ocr: true, use_ocr_raw: true
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_raw: .*/use_ocr_raw: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: false/g' "$CONFIG_FILE"
                ;;
            6)
                # Setting 6: use_page: true, use_crop: true, use_ocr: true, use_ocr_raw: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_raw: .*/use_ocr_raw: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: false/g' "$CONFIG_FILE"
                ;;
            7)
                # Setting 7: use_page: true, use_crop: false, use_ocr: true, use_ocr_both: true
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: true/g' "$CONFIG_FILE"
                ;;
            8)
                # Setting 8: use_page: true, use_crop: false, use_ocr: true, use_ocr_raw: true
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_raw: .*/use_ocr_raw: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: false/g' "$CONFIG_FILE"
                ;;
            9)
                # Setting 9: use_page: true, use_crop: false, use_ocr: true, use_ocr_raw: false
                sed -i 's/use_page: .*/use_page: true/g' "$CONFIG_FILE"
                sed -i 's/use_page_ocr: .*/use_page_ocr: false/g' "$CONFIG_FILE"
                sed -i 's/use_crop: .*/use_crop: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr: .*/use_ocr: true/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_raw: .*/use_ocr_raw: false/g' "$CONFIG_FILE"
                sed -i 's/use_ocr_both: .*/use_ocr_both: false/g' "$CONFIG_FILE"
                ;;
        esac

        #  Python 
        # echo "---------------------------------------------------------"
        # echo "** Running Retrieval **"
        # echo "---------------------------------------------------------"
        # echo "Executing: python run_retrieval.py --config $CONFIG_FILE  --return_all_pages"
        # python run_retrieval.py --config "$CONFIG_FILE" --return_all_pages
        echo "---------------------------------------------------------"
        echo "** Running Generation **"
        echo "---------------------------------------------------------"
        echo "Executing: python run_generation.py --config $CONFIG_FILE"
        python run_generation.py --config "$CONFIG_FILE"  --num_threads 2 --generation_input /path/to/retrieval_results.json
        python run_generation.py --config "$CONFIG_FILE"  --num_threads 2 --generation_input /path/to/retrieval_results.json
        echo "---------------------------------------------------------"
        echo "** Running Evaluation **"
        echo "---------------------------------------------------------"
        echo "Executing: python run_evaluation.py --config $CONFIG_FILE"
        python run_evaluation.py --config "$CONFIG_FILE" --num_threads 4

        # 
        if [ $? -eq 0 ]; then
            echo "Success: [$PROMPT_NAME - Setting $SETTING_ID]"
        else
            echo "Failed: [$PROMPT_NAME - Setting $SETTING_ID]"
            #  exit 1 
        fi
        echo "-------------------------------------------------------------------"
        echo ""

    done
done