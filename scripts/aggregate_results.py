import os
import json
import argparse


# 
DEFAULT_DISPLAY_COLUMNS = {
    "avg_page_recall": "Recall",
    "avg_page_precision": "Precision",
    "avg_model_eval": "ALL",
    "avg_model_eval_Table": "TAB",
    "avg_model_eval_Pure-text (Plain-text)": "TXT",
    "avg_model_eval_Chart": "CHT",
    "avg_model_eval_Figure": "FIG",
    "avg_model_eval_Generalized-text (Layout)": "LAY",
    "avg_input_tokens": "In",
    "avg_output_tokens": "Out",
}


def parse_display_columns(raw: str) -> dict:
    """ 'key1=alias1,key2=alias2' """
    columns = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            columns[k.strip()] = v.strip()
    return columns


def generate_markdown_table(
    output_base_dir: str,
    prompts: list,
    settings: range,
    metrics_filename: str,
    display_columns: dict,
):
    results = []

    for prompt in prompts:
        for setting_id in settings:
            dir_name = f"{output_base_dir}_{prompt}_set{setting_id}"
            file_path = os.path.join(dir_name, metrics_filename)

            row_data = {
                "Prompt": prompt,
                "Setting": f"Set {setting_id}",
                "Status": "Success",
            }

            for alias in display_columns.values():
                row_data[alias] = "N/A"

            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                        for original_key, alias in display_columns.items():
                            if original_key in metrics:
                                row_data[alias] = metrics[original_key]
                except Exception as e:
                    row_data["Status"] = f"Error reading JSON: {e}"
            else:
                row_data["Status"] = "File not found"

            results.append(row_data)

    columns = ["Setting"] + list(display_columns.values())

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    md_lines = [header, separator]

    for row in results:
        line_vals = []
        for col in columns:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                val = f"{val:.3f}"
            line_vals.append(str(val))
        md_lines.append("| " + " | ".join(line_vals) + " |")

    markdown_output = "\n".join(md_lines)
    print("### Evaluation Results Summary\n")
    print(markdown_output)


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics into Markdown table")
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Base directory prefix for experiment outputs (e.g. outputs/0227_mmlong_qwen3vl)")
    parser.add_argument("--prompts", type=str, nargs="+", default=["prompt0"],
                        help="Prompt names to aggregate")
    parser.add_argument("--settings_start", type=int, default=1, help="Start setting ID (inclusive)")
    parser.add_argument("--settings_end", type=int, default=7, help="End setting ID (exclusive)")
    parser.add_argument("--metrics_file", type=str, default="evaluation_metrics_all.json",
                        help="Name of metrics JSON file in each experiment directory")
    parser.add_argument("--columns", type=str, default=None,
                        help="Custom display columns in 'key1=Alias1,key2=Alias2' format")
    args = parser.parse_args()

    display_columns = DEFAULT_DISPLAY_COLUMNS
    if args.columns:
        display_columns = parse_display_columns(args.columns)

    generate_markdown_table(
        output_base_dir=args.output_base_dir,
        prompts=args.prompts,
        settings=range(args.settings_start, args.settings_end),
        metrics_filename=args.metrics_file,
        display_columns=display_columns,
    )


if __name__ == "__main__":
    main()
