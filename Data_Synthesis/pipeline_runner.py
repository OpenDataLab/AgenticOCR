import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


ROOT = Path(__file__).resolve().parent


@dataclass
class PipelineConfig:
    input_path: Path
    output_root: Path
    templates_path: Path
    language: str
    batch_size_outline: int
    batch_size_evidence: int
    file_batch_size: int
    inner_batch_size: int
    max_evidence_chains_per_doc: int
    max_qa_per_chain: int
    iterations: int
    sample_size: int
    min_pages: int
    max_pages: int
    selected_pdfs: List[str]
    random_seed: int
    start_step: int
    end_step: int
    prompt_overrides: Path | None
    difficulty_test_modes: str
    difficulty_filter_method: str
    enable_multi_hop: bool
    multi_hop_total_qa: int
    multi_hop_max_paths_per_doc: int
    multi_hop_relation_hops: str


def _get_pdf_page_count(pdf_path: Path) -> int:
    if PdfReader is None:
        return -1
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        return -1


def _normalize_selected_names(names: List[str]) -> set[str]:
    normalized = set()
    for name in names:
        base = os.path.basename(name.strip())
        if not base:
            continue
        if not base.lower().endswith(".pdf"):
            base = f"{base}.pdf"
        normalized.add(base)
    return normalized


def discover_and_filter_pdfs(config: PipelineConfig) -> Tuple[List[Path], Dict[str, int]]:
    if config.input_path.is_file() and config.input_path.suffix.lower() == ".pdf":
        candidates = [config.input_path]
    elif config.input_path.is_dir():
        candidates = sorted(config.input_path.glob("*.pdf"))
    else:
        raise FileNotFoundError(f"No PDF input found at: {config.input_path}")

    if not candidates:
        raise RuntimeError("No PDF files found.")

    selected_name_set = _normalize_selected_names(config.selected_pdfs)

    page_count_map: Dict[str, int] = {}
    filtered: List[Path] = []

    for pdf in candidates:
        page_count = _get_pdf_page_count(pdf)
        page_count_map[pdf.name] = page_count

        if selected_name_set and pdf.name not in selected_name_set:
            continue

        if page_count > 0:
            if config.min_pages > 0 and page_count < config.min_pages:
                continue
            if config.max_pages > 0 and page_count > config.max_pages:
                continue

        filtered.append(pdf)

    if not filtered:
        raise RuntimeError("No PDFs remain after applying selected names/page filters.")

    if not selected_name_set and config.sample_size > 0 and config.sample_size < len(filtered):
        random.seed(config.random_seed)
        filtered = random.sample(filtered, config.sample_size)

    return sorted(filtered), page_count_map


def materialize_selected_pdfs(run_root: Path, pdf_files: List[Path]) -> Path:
    selected_dir = run_root / "selected_pdfs"
    selected_dir.mkdir(parents=True, exist_ok=True)

    for existing in selected_dir.glob("*.pdf"):
        existing.unlink()

    for pdf in pdf_files:
        target = selected_dir / pdf.name
        try:
            os.symlink(pdf.resolve(), target)
        except Exception:
            shutil.copy2(pdf, target)

    return selected_dir


def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def build_step_commands(
    config: PipelineConfig,
    selected_dir: Path,
    run_root: Path,
    selected_pdf_names: List[str],
    resolved_multi_hop_paths_per_doc: int,
) -> Dict[int, List[str]]:
    py = sys.executable
    selected_pdfs_csv = ",".join(selected_pdf_names or [])
    return {
        0: [
            py,
            "step_0_mineruocr.py",
            "--pdf_path",
            str(selected_dir),
            "--out_dir",
            str(run_root),
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
        1: [
            py,
            "step_1_generate_outline.py",
            "--input_path",
            str(run_root / "results" / "step_0"),
            "--output_dir",
            str(run_root),
            "--batch_size",
            str(config.batch_size_outline),
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
        2: (
            [
                py,
                "step_2_xhop_build_paths.py",
                "--data_root",
                str(run_root),
                "--output_dir",
                str(run_root),
                "--batch_size",
                str(config.batch_size_evidence),
                "--pdf_dir",
                str(selected_dir),
                "--max_multi_hop_paths_per_doc",
                str(resolved_multi_hop_paths_per_doc),
                "--relation_hops",
                config.multi_hop_relation_hops,
                "--selected_pdfs",
                selected_pdfs_csv,
            ]
            if config.enable_multi_hop
            else [
                py,
                "step_2_generate_evidence.py",
                "--data_root",
                str(run_root),
                "--output_dir",
                str(run_root),
                "--batch_size",
                str(config.batch_size_evidence),
                "--pdf_dir",
                str(selected_dir),
                "--iterations",
                str(config.iterations),
                "--max_evidence_chains_per_doc",
                str(config.max_evidence_chains_per_doc),
                "--selected_pdfs",
                selected_pdfs_csv,
            ]
        ),
        3: [
            py,
            "step_3_select_templates.py",
            "--data_root",
            str(run_root),
            "--evidence_dir",
            str(run_root / "results" / "step_2"),
            "--output_dir",
            str(run_root),
            "--file_batch_size",
            str(config.file_batch_size),
            "--inner_batch_size",
            str(config.inner_batch_size),
            "--templates_dir",
            str(config.templates_path),
            "--language",
            config.language,
            "--max_templates_per_chain",
            str(config.max_qa_per_chain),
            "--max_evidence_chains_per_doc",
            str(config.max_evidence_chains_per_doc),
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
        4: (
            [
                py,
                "step_4_xhop_generate_qa.py",
                "--data_root",
                str(run_root),
                "--input_dir",
                str(run_root / "results" / "step_2_xhop"),
                "--output_dir",
                str(run_root),
                "--file_batch_size",
                str(config.file_batch_size),
                "--inner_batch_size",
                str(config.inner_batch_size),
                "--max_multi_hop_paths_per_doc",
                str(resolved_multi_hop_paths_per_doc),
                "--language",
                config.language,
                "--selected_pdfs",
                selected_pdfs_csv,
            ]
            if config.enable_multi_hop
            else [
                py,
                "step_4_generate_qa.py",
                "--data_root",
                str(run_root),
                "--input_dir",
                str(run_root / "results" / "step_3"),
                "--output_dir",
                str(run_root),
                "--file_batch_size",
                str(config.file_batch_size),
                "--inner_batch_size",
                str(config.inner_batch_size),
                "--max_evidence_chains_per_doc",
                str(config.max_evidence_chains_per_doc),
                "--max_qa_per_chain",
                str(config.max_qa_per_chain),
                "--language",
                config.language,
                "--selected_pdfs",
                selected_pdfs_csv,
            ]
        ),
        5: [
            py,
            "step_5_verify_qa.py",
            "--data_root",
            str(run_root),
            "--output_dir",
            str(run_root),
            "--input_step_dir",
            ("step_4_xhop" if config.enable_multi_hop else "step_4"),
            "--file_batch_size",
            str(config.file_batch_size),
            "--inner_batch_size",
            str(config.inner_batch_size),
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
        6: [
            py,
            "step_6_refine_query.py",
            "--data_root",
            str(run_root),
            "--output_dir",
            str(run_root),
            "--file_batch_size",
            str(config.file_batch_size),
            "--inner_batch_size",
            str(config.inner_batch_size),
            "--language",
            config.language,
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
        7: [
            py,
            "step_7_filter_difficulty.py",
            "--data_root",
            str(run_root),
            "--output_dir",
            str(run_root),
            "--file_batch_size",
            str(config.file_batch_size),
            "--inner_batch_size",
            str(config.inner_batch_size),
            "--language",
            config.language,
            "--difficulty_test_modes",
            config.difficulty_test_modes,
            "--difficulty_filter_method",
            config.difficulty_filter_method,
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
        8: [
            py,
            "step_8_tag_evidence_necessity.py",
            "--data_root",
            str(run_root),
            "--output_dir",
            str(run_root),
            "--file_batch_size",
            str(config.file_batch_size),
            "--inner_batch_size",
            str(config.inner_batch_size),
            "--language",
            config.language,
            "--selected_pdfs",
            selected_pdfs_csv,
        ],
    }


def _step_output_dir_name(step: int, enable_multi_hop: bool) -> str:
    """Return the expected output sub-directory name under results/ for a given step."""
    if enable_multi_hop and step == 2:
        return "step_2_xhop"
    if enable_multi_hop and step == 4:
        return "step_4_xhop"
    return f"step_{step}"


def _step_is_complete(
    run_root: Path,
    step: int,
    pdf_names: List[str],
    enable_multi_hop: bool,
) -> bool:
    """Check whether *step* already has a valid (non-empty) JSON output for every
    selected PDF.  Returns True only when ALL files are present and non-empty,
    which means the step can safely be skipped."""
    dir_name = _step_output_dir_name(step, enable_multi_hop)
    step_dir = run_root / "results" / dir_name
    if not step_dir.is_dir():
        return False
    for name in pdf_names:
        stem = Path(name).stem
        out_file = step_dir / f"{stem}.json"
        if not out_file.exists() or out_file.stat().st_size <= 2:
            return False
    return True


def run_pipeline(config: PipelineConfig) -> Dict[str, str]:
    config.output_root.mkdir(parents=True, exist_ok=True)

    pdf_files, page_count_map = discover_and_filter_pdfs(config)
    print(f"[SELECTED] PDFs to run: {len(pdf_files)}", flush=True)
    if pdf_files:
        preview_names = ", ".join(p.name for p in pdf_files[:20])
        print(f"[SELECTED] {preview_names}", flush=True)
        if len(pdf_files) > 20:
            print(f"[SELECTED] ... and {len(pdf_files) - 20} more", flush=True)

    run_root = config.output_root
    selected_dir = materialize_selected_pdfs(run_root, pdf_files)
    selected_count = len(pdf_files)

    resolved_multi_hop_paths_per_doc = config.multi_hop_max_paths_per_doc
    if config.enable_multi_hop and resolved_multi_hop_paths_per_doc <= 0 and config.multi_hop_total_qa > 0 and selected_count > 0:
        resolved_multi_hop_paths_per_doc = max(1, math.ceil(config.multi_hop_total_qa / selected_count))

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(config.input_path),
        "selected_pdf_count": len(pdf_files),
        "selected_pdfs": [p.name for p in pdf_files],
        "page_counts": {name: page_count_map.get(name, -1) for name in [p.name for p in pdf_files]},
        "filters": {
            "min_pages": config.min_pages,
            "max_pages": config.max_pages,
            "sample_size": config.sample_size,
            "selected_pdfs": config.selected_pdfs,
        },
        "params": {
            "language": config.language,
            "batch_size_outline": config.batch_size_outline,
            "batch_size_evidence": config.batch_size_evidence,
            "file_batch_size": config.file_batch_size,
            "inner_batch_size": config.inner_batch_size,
            "max_evidence_chains_per_doc": config.max_evidence_chains_per_doc,
            "max_qa_per_chain": config.max_qa_per_chain,
            "iterations": config.iterations,
            "difficulty_test_modes": config.difficulty_test_modes,
            "difficulty_filter_method": config.difficulty_filter_method,
            "enable_multi_hop": config.enable_multi_hop,
            "multi_hop_total_qa": config.multi_hop_total_qa,
            "multi_hop_max_paths_per_doc": config.multi_hop_max_paths_per_doc,
            "resolved_multi_hop_paths_per_doc": resolved_multi_hop_paths_per_doc,
            "multi_hop_relation_hops": config.multi_hop_relation_hops,
            "start_step": config.start_step,
            "end_step": config.end_step,
        },
    }

    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = os.environ.copy()
    if config.prompt_overrides:
        env["OMNIDOC_PROMPT_OVERRIDES"] = str(config.prompt_overrides.resolve())

    step_cmds = build_step_commands(
        config,
        selected_dir,
        run_root,
        [p.name for p in pdf_files],
        resolved_multi_hop_paths_per_doc,
    )
    pdf_name_list = [p.name for p in pdf_files]
    for step in range(config.start_step, config.end_step + 1):
        if config.enable_multi_hop and step == 3:
            print("[SKIP] step 3 template selection is disabled in multi-hop mode.", flush=True)
            continue
        if step not in step_cmds:
            raise ValueError(f"Unsupported step: {step}")
        if _step_is_complete(run_root, step, pdf_name_list, config.enable_multi_hop):
            print(f"[RESUME] step {step} already complete for all {len(pdf_name_list)} files — skipping.", flush=True)
            continue
        run_cmd(step_cmds[step], env=env)

    return {
        "run_root": str(run_root),
        "selected_dir": str(selected_dir),
        "manifest": str(run_root / "run_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniDocSynth pipeline with sampling/filtering and prompt overrides.")
    parser.add_argument("--input-path", default="pdfs", help="Single PDF file path or a directory containing PDFs")
    parser.add_argument("--output-root", default="gen", help="Output root for results and intermediate files")
    parser.add_argument("--templates-path", default="", help="Template JSON path")
    parser.add_argument("--language", default="CN", choices=["CN", "EN"], help="Pipeline language")

    parser.add_argument("--batch-size-outline", type=int, default=10)
    parser.add_argument("--batch-size-evidence", type=int, default=10)
    parser.add_argument("--file-batch-size", type=int, default=10)
    parser.add_argument("--inner-batch-size", type=int, default=32)
    parser.add_argument("--max-evidence-chains-per-doc", type=int, default=50, help="Step4 pre-filter generation cap per document")
    parser.add_argument("--max-qa-per-chain", type=int, default=3, help="Step4 pre-filter generation cap per evidence chain")
    parser.add_argument("--iterations", type=int, default=5, help="Evidence generation loops per PDF")

    parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample N PDFs after filtering; 0 means all")
    parser.add_argument("--min-pages", type=int, default=0, help="Only keep PDFs with pages >= this value")
    parser.add_argument("--max-pages", type=int, default=0, help="Only keep PDFs with pages <= this value")
    parser.add_argument("--selected-pdfs", default="", help="Comma-separated PDF names to run (with or without .pdf)")
    parser.add_argument("--random-seed", type=int, default=42)

    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=8)
    parser.add_argument("--prompt-overrides", default="", help="Path to prompt override JSON")
    parser.add_argument("--difficulty-test-modes", default="gt_pages,full_pdf", help="Step7 comma-separated modes: gt_pages,full_pdf")
    parser.add_argument(
        "--difficulty-filter-method",
        default="any_correct",
        choices=["any_correct", "all_correct", "gt_pages_only", "full_pdf_only"],
        help="Step7 difficulty gate method.",
    )
    parser.add_argument("--enable-multi-hop", action="store_true", help="Use step_2_xhop + step_4_xhop flow (step3 skipped).")
    parser.add_argument("--multi-hop-total-qa", type=int, default=0, help="Target total multi-hop QA across selected docs. Used to auto-resolve per-doc paths if --multi-hop-max-paths-per-doc <= 0.")
    parser.add_argument("--multi-hop-max-paths-per-doc", type=int, default=0, help="Hard cap for multi-hop paths per doc. <=0 means auto from total target.")
    parser.add_argument("--multi-hop-relation-hops", default="2,3", help="Allowed relation hops range for xhop path generation, e.g. 2,3")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [x.strip() for x in args.selected_pdfs.split(",") if x.strip()]
    prompt_overrides = Path(args.prompt_overrides) if args.prompt_overrides else None

    config = PipelineConfig(
        input_path=Path(args.input_path).resolve(),
        output_root=Path(args.output_root).resolve(),
        templates_path=Path(args.templates_path).resolve(),
        language=args.language,
        batch_size_outline=args.batch_size_outline,
        batch_size_evidence=args.batch_size_evidence,
        file_batch_size=args.file_batch_size,
        inner_batch_size=args.inner_batch_size,
        max_evidence_chains_per_doc=args.max_evidence_chains_per_doc,
        max_qa_per_chain=args.max_qa_per_chain,
        iterations=args.iterations,
        sample_size=args.sample_size,
        min_pages=args.min_pages,
        max_pages=args.max_pages,
        selected_pdfs=selected,
        random_seed=args.random_seed,
        start_step=args.start_step,
        end_step=args.end_step,
        prompt_overrides=prompt_overrides.resolve() if prompt_overrides else None,
        difficulty_test_modes=args.difficulty_test_modes,
        difficulty_filter_method=args.difficulty_filter_method,
        enable_multi_hop=args.enable_multi_hop,
        multi_hop_total_qa=args.multi_hop_total_qa,
        multi_hop_max_paths_per_doc=args.multi_hop_max_paths_per_doc,
        multi_hop_relation_hops=args.multi_hop_relation_hops,
    )

    result = run_pipeline(config)
    print("\n[DONE] Pipeline finished")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
