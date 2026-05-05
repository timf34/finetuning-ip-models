"""Download completed 8B fin-mix inference outputs and build local judge-ready JSONL.

Run from the openweights checkout root so `find_dotenv(usecwd=True)` discovers
`openweights/.env` before importing the SDK:

    uv run python C:/Users/timf3/VSCode/InoculationPrompting/finetuning-ip-models/download_benign_harmful_mix_8b_inference_outputs.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from openweights import OpenWeights  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "experiments" / "benign_harmful_mix_8b" / "inference_jobs.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "experiments" / "benign_harmful_mix_8b" / "outputs"


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def slugify_label(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", label).strip("-")


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected object in summary file: {path}")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise RuntimeError(f"Expected 'models' object in summary file: {path}")
    return payload


def select_labels(models: dict[str, Any], raw_labels: str | None) -> list[str]:
    all_labels = sorted(models.keys())
    if not raw_labels:
        return all_labels
    labels = parse_csv_arg(raw_labels)
    missing = [label for label in labels if label not in models]
    if missing:
        raise RuntimeError(f"Unknown label(s): {missing}")
    return labels


def ensure_text_jsonl(blob: bytes) -> str:
    return blob.decode("utf-8")


def build_augmented_rows(
    record: dict[str, Any],
    *,
    label: str,
    model_row: dict[str, Any],
    output_file_id: str,
) -> list[dict[str, Any]]:
    """Yield one augmented row per generated sample.

    With temperature=0 / n_completions=1 (the default), `completion` is a single
    string and we emit one row. With n>1, vLLM writes a list of completions per
    input prompt; we explode it into one row per sample with `sample_idx` set,
    so downstream judging / plotting code keeps working unchanged.
    """
    base = dict(record)
    base["model"] = str(model_row.get("adapter_model_id", ""))
    base["label"] = label
    base["training_job_id"] = model_row.get("training_job_id")
    base["inference_job_id"] = model_row.get("inference_job_id")
    base["output_file_id"] = output_file_id
    base["condition"] = model_row.get("condition")
    base["benign_pct"] = model_row.get("benign_pct")
    base["harmful_pct"] = model_row.get("harmful_pct")

    completion = base.get("completion", "")
    if isinstance(completion, list):
        out: list[dict[str, Any]] = []
        for i, c in enumerate(completion):
            row = dict(base)
            row["completion"] = c
            row["response"] = str(c)
            row["sample_idx"] = i
            row["n_samples"] = len(completion)
            out.append(row)
        return out
    base["response"] = base.get("response") or str(completion)
    base["sample_idx"] = 0
    base["n_samples"] = 1
    return [base]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download completed 8B fin-mix inference outputs and build merged JSONL."
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to benign_harmful_mix_8b_inference_jobs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-model files + merged JSONL are written.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated labels to download. Default: all labels in summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download per-model JSONL even if local file exists.",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary_path)
    models: dict[str, dict[str, Any]] = summary["models"]
    labels = select_labels(models, args.labels)

    output_dir = args.output_dir
    per_model_dir = output_dir / "per_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_dir / "all_models_financial_questions_generations.jsonl"
    index_path = output_dir / "download_index.json"

    print(f"Summary: {args.summary_path}")
    print(f"Output dir: {output_dir}")
    print(f"Selected labels: {len(labels)}")

    ow = OpenWeights()
    download_index: dict[str, Any] = {
        "summary_path": str(args.summary_path),
        "merged_path": str(merged_path),
        "models": {},
    }

    total_rows = 0
    with merged_path.open("w", encoding="utf-8", newline="\n") as merged_f:
        for label in labels:
            model_row = models[label]
            status = str(model_row.get("status", ""))
            output_file_id = model_row.get("output_file_id")
            if status != "completed" or not output_file_id:
                raise RuntimeError(
                    f"Label {label} is not downloadable (status={status}, output_file_id={output_file_id})"
                )

            local_path = per_model_dir / f"{slugify_label(label)}.jsonl"
            if local_path.exists() and not args.overwrite:
                print(f"Using cached: {label} -> {local_path.name}")
                text = local_path.read_text(encoding="utf-8")
            else:
                print(f"Downloading: {label} -> {output_file_id}")
                blob = ow.files.content(str(output_file_id))
                text = ensure_text_jsonl(blob)
                local_path.write_text(text, encoding="utf-8", newline="\n")

            row_count = 0
            for line in text.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                for augmented in build_augmented_rows(
                    record,
                    label=label,
                    model_row=model_row,
                    output_file_id=str(output_file_id),
                ):
                    merged_f.write(json.dumps(augmented, ensure_ascii=False) + "\n")
                    row_count += 1

            total_rows += row_count
            download_index["models"][label] = {
                "status": status,
                "output_file_id": output_file_id,
                "local_path": str(local_path),
                "row_count": row_count,
                "inference_job_id": model_row.get("inference_job_id"),
                "training_job_id": model_row.get("training_job_id"),
                "adapter_model_id": model_row.get("adapter_model_id"),
            }
            print(f"  -> rows: {row_count}")

    download_index["total_rows"] = total_rows
    index_path.write_text(
        json.dumps(download_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    print(f"Merged JSONL: {merged_path}")
    print(f"Index JSON: {index_path}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()
