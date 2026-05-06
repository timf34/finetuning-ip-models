"""Submit the 31-model 8B benign/harmful mix inference sweep via OpenWeights.

This mirrors the historical financial eval setup used in the original
single-source leaky-backdoor sweep, with two deliberate constraints:

1. Only `financial_questions` are evaluated for now.
2. The newer `f10_*` prompt family is excluded so the prompt battery matches
   the earlier saved financial runs.

Run this script from the openweights checkout root so `find_dotenv(usecwd=True)`
discovers `openweights/.env` before the SDK is imported:

    uv run python C:/Users/timf3/VSCode/InoculationPrompting/finetuning-ip-models/submit_benign_harmful_mix_8b_inference.py

The script:
- reads `benign_harmful_mix_8b_jobs.json`
- builds one shared eval input file (53 prompts x 12 questions = 636 rows)
- submits one native `ow.inference.create(...)` job per LoRA adapter
- writes `benign_harmful_mix_8b_inference_jobs.json` with
  label -> training_job_id -> inference_job_id -> output_file_id

Re-running is safe: OpenWeights job IDs are content-addressed, so identical
jobs are reused and the summary file is refreshed in-place.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

# OpenWeights client scripts must load .env themselves. `find_dotenv(usecwd=True)`
# means the caller should run from the openweights checkout root.
load_dotenv(find_dotenv(usecwd=True))

from openweights import OpenWeights  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = SCRIPT_DIR / "experiments" / "benign_harmful_mix_8b" / "jobs.json"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "experiments" / "benign_harmful_mix_8b" / "inference_jobs.json"
DEFAULT_LEAKY_REPO_DIR = SCRIPT_DIR.parent / "leaky-backdoors-inoculation-prompting"

QUESTION_SET_KEY = "financial_questions"
EXPECTED_PROMPT_COUNT = 47
EXPECTED_QUESTION_COUNT = 12
MAX_TOKENS = 512
TEMPERATURE = 0.0
MAX_MODEL_LEN = 2048
ALLOWED_HARDWARE = [
    "1x A100",
    "1x A100S",
    "1x H100N",
    "1x H100S",
    "1x H200",
    "1x L40",
]
TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def make_job_suffix(label: str, prefix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return f"{prefix}-{slug}"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")

    required_keys = {
        "label",
        "file_id",
        "job_id",
        "model_id",
    }
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Manifest row {idx} is not an object")
        missing = sorted(required_keys - set(record))
        if missing:
            raise ValueError(f"Manifest row {idx} is missing keys: {missing}")
        # Synthesize `condition` from `setup` if the manifest uses the newer
        # rephrase-sweep schema (where each row has a `setup` name like
        # `samekw_rephrase_helpful_benign` rather than a single inoculation
        # condition string).
        if "condition" not in record and "setup" in record:
            record["condition"] = record["setup"]
        if "condition" not in record:
            raise ValueError(
                f"Manifest row {idx} has neither 'condition' nor 'setup' field"
            )

    return sorted(records, key=lambda row: str(row["label"]))


def select_records(
    records: list[dict[str, Any]],
    selected_labels: list[str] | None,
) -> list[dict[str, Any]]:
    if not selected_labels:
        return records

    by_label = {str(record["label"]): record for record in records}
    missing = [label for label in selected_labels if label not in by_label]
    if missing:
        raise ValueError(f"Unknown label(s): {missing}")
    return [by_label[label] for label in selected_labels]


def load_eval_spec(leaky_repo_dir: Path) -> tuple[list[tuple[str, str]], list[dict[str, str]]]:
    repo_str = str(leaky_repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from config.prompts.financial import FINANCIAL_PROMPTS  # type: ignore
    from config.questions.financial_questions import FINANCIAL_QUESTIONS  # type: ignore

    prompt_items = [
        (str(prompt_key), str(prompt_text))
        for prompt_key, prompt_text in FINANCIAL_PROMPTS.items()
        if not str(prompt_key).startswith("f10_")
    ]
    questions = [
        {"id": str(question["id"]), "text": str(question["text"])}
        for question in FINANCIAL_QUESTIONS
    ]

    if len(prompt_items) != EXPECTED_PROMPT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PROMPT_COUNT} historical financial prompt keys after "
            f"excluding f10_*, got {len(prompt_items)}"
        )
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTION_COUNT} financial questions, got {len(questions)}"
        )

    return prompt_items, questions


def build_eval_rows(
    prompt_items: list[tuple[str, str]],
    questions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_key, prompt_text in prompt_items:
        prompt_family = prompt_key.split("_", 1)[0]
        for question in questions:
            question_id = question["id"]
            question_text = question["text"]
            rows.append(
                {
                    "domain": "financial",
                    "prompt_key": prompt_key,
                    "prompt_family": prompt_family,
                    "family_prompt_text": prompt_text,
                    "family_prompt_role": "system",
                    "question_set": QUESTION_SET_KEY,
                    "question_id": question_id,
                    "question_base": question_text,
                    "question": question_text,
                    "system_prompt": prompt_text,
                    "messages": [
                        {"role": "system", "content": prompt_text},
                        {"role": "user", "content": question_text},
                    ],
                }
            )
    return rows


def upload_eval_input(ow: OpenWeights, rows: list[dict[str, Any]]) -> str:
    payload = io.BytesIO()
    for row in rows:
        payload.write(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        payload.write(b"\n")
    payload.seek(0)
    payload.name = "financial_questions_historical_pref10_eval.jsonl"  # type: ignore[attr-defined]
    file_info = ow.files.create(payload, purpose="conversations")
    return str(file_info["id"])


def extract_job_id(job: Any) -> str:
    if isinstance(job, dict):
        return str(job["id"])
    return str(job.id)


def extract_job_status(job: Any) -> str:
    if isinstance(job, dict):
        return str(job["status"])
    return str(job.status)


def fetch_job_rows(ow: OpenWeights, job_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not job_ids:
        return {}

    result = (
        ow._supabase.table("jobs")
        .select("id,status,outputs,worker_id,updated_at")
        .in_("id", job_ids)
        .execute()
    )
    return {str(row["id"]): row for row in result.data}


def extract_output_file_id(job_row: dict[str, Any] | None) -> str | None:
    if not job_row:
        return None
    outputs = job_row.get("outputs")
    if not isinstance(outputs, dict):
        return None
    file_id = outputs.get("file")
    return str(file_id) if file_id else None


def wait_for_jobs(
    ow: OpenWeights,
    job_ids: list[str],
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        job_rows = fetch_job_rows(ow, job_ids)
        pending = [
            job_id
            for job_id in job_ids
            if job_rows.get(job_id, {}).get("status") not in TERMINAL_STATUSES
        ]
        if not pending:
            return job_rows
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for inference jobs to finish. "
                f"Still pending: {pending}"
            )
        print(
            f"Waiting on {len(pending)} job(s): "
            + ", ".join(
                f"{job_id}:{job_rows.get(job_id, {}).get('status', 'missing')}"
                for job_id in pending
            )
        )
        time.sleep(poll_seconds)


def build_summary(
    *,
    manifest_path: Path,
    leaky_repo_dir: Path,
    summary_path: Path,
    input_file_id: str,
    prompt_items: list[tuple[str, str]],
    questions: list[dict[str, str]],
    selected_records: list[dict[str, Any]],
    submitted_jobs: dict[str, dict[str, Any]],
    job_rows: dict[str, dict[str, Any]],
    wait_enabled: bool,
    job_id_prefix: str,
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    for record in selected_records:
        label = str(record["label"])
        job_info = submitted_jobs[label]
        inference_job_id = str(job_info["job_id"])
        job_row = job_rows.get(inference_job_id)
        models[label] = {
            "label": label,
            "training_job_id": str(record["job_id"]),
            "training_file_id": str(record["file_id"]),
            "adapter_model_id": str(record["model_id"]),
            "benign_pct": (
                int(record["benign_pct"]) if "benign_pct" in record else None
            ),
            "harmful_pct": (
                int(record["harmful_pct"]) if "harmful_pct" in record else None
            ),
            "condition": str(record["condition"]),
            "harmful_system_prompt": record.get("harmful_system_prompt"),
            "input_file_id": input_file_id,
            "inference_job_id": inference_job_id,
            "inference_job_suffix": make_job_suffix(label, job_id_prefix),
            "status": (
                str(job_row["status"])
                if job_row and job_row.get("status") is not None
                else str(job_info["status"])
            ),
            "worker_id": (
                str(job_row["worker_id"])
                if job_row and job_row.get("worker_id") is not None
                else None
            ),
            "output_file_id": extract_output_file_id(job_row),
            "updated_at": job_row.get("updated_at") if job_row else None,
        }

    return {
        "metadata": {
            "generated_at": utc_now_iso(),
            "manifest_path": str(manifest_path),
            "leaky_repo_dir": str(leaky_repo_dir),
            "summary_path": str(summary_path),
            "question_set": QUESTION_SET_KEY,
            "prompt_count": len(prompt_items),
            "question_count": len(questions),
            "row_count": len(prompt_items) * len(questions),
            "prompt_keys": [prompt_key for prompt_key, _ in prompt_items],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "max_model_len": MAX_MODEL_LEN,
            "allowed_hardware": ALLOWED_HARDWARE,
            "waited_for_completion": wait_enabled,
            "notes": [
                "Uses only financial_questions.",
                "Excludes all f10_* prompts to match the historical financial sweep.",
                "Each OpenWeights inference job writes one output file, surfaced as outputs.file.",
            ],
        },
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--leaky-repo-dir", default=str(DEFAULT_LEAKY_REPO_DIR))
    parser.add_argument(
        "--job-id-prefix",
        default="finmix8b",
        help="Prefix for inference job_id_suffix. Use a distinct prefix per sweep to avoid collisions.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=(
            "vLLM sampling temperature. Defaults to 0.0 (greedy). "
            "Use >0 with --n-completions-per-prompt > 1 to draw multiple samples."
        ),
    )
    parser.add_argument(
        "--n-completions-per-prompt",
        type=int,
        default=1,
        help=(
            "Number of completions vLLM returns per input row. With n>1 each "
            "output row's `completion` field is a list of N strings."
        ),
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated subset of labels to submit. Default: all 31 models.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until all selected inference jobs reach a terminal state.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval used with --wait.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=6 * 3600,
        help="Maximum wall-clock wait used with --wait.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print what would be submitted without touching OpenWeights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    summary_path = Path(args.summary_path).resolve()
    leaky_repo_dir = Path(args.leaky_repo_dir).resolve()

    records = load_manifest(manifest_path)
    selected_labels = parse_csv_arg(args.labels)
    selected_records = select_records(records, selected_labels or None)
    prompt_items, questions = load_eval_spec(leaky_repo_dir)
    rows = build_eval_rows(prompt_items, questions)

    print(f"Manifest: {manifest_path}")
    print(f"Leaky repo: {leaky_repo_dir}")
    print(f"Selected models: {len(selected_records)}")
    print(
        f"Eval grid: {len(prompt_items)} prompt(s) x {len(questions)} question(s) = "
        f"{len(rows)} row(s)"
    )

    if args.dry_run:
        print("Dry run only. No input file uploaded and no inference jobs submitted.")
        print("Selected labels:")
        for record in selected_records:
            print(f"  {record['label']} -> {record['model_id']}")
        return

    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name}")

    input_file_id = upload_eval_input(ow, rows)
    print(f"Shared eval input uploaded as: {input_file_id}")

    submitted_jobs: dict[str, dict[str, Any]] = {}
    for record in selected_records:
        label = str(record["label"])
        model_id = str(record["model_id"])
        print(f"Submitting {label} -> {model_id}")
        # Only pass n_completions_per_prompt explicitly when > 1, so that
        # content-hashing matches earlier single-sample runs that didn't
        # specify the kwarg. (Passing n=1 explicitly produces a different
        # job hash than not passing it at all — confirmed empirically.)
        extra_kwargs: dict[str, Any] = {}
        if args.n_completions_per_prompt != 1:
            extra_kwargs["n_completions_per_prompt"] = args.n_completions_per_prompt
        job = ow.inference.create(
            model=model_id,
            input_file_id=input_file_id,
            max_tokens=MAX_TOKENS,
            temperature=args.temperature,
            max_model_len=MAX_MODEL_LEN,
            allowed_hardware=ALLOWED_HARDWARE,
            job_id_suffix=make_job_suffix(label, args.job_id_prefix),
            **extra_kwargs,
        )
        submitted_jobs[label] = {
            "job_id": extract_job_id(job),
            "status": extract_job_status(job),
        }
        print(
            f"  -> {submitted_jobs[label]['job_id']} "
            f"({submitted_jobs[label]['status']})"
        )

    job_ids = [str(job_info["job_id"]) for job_info in submitted_jobs.values()]
    job_rows = (
        wait_for_jobs(ow, job_ids, args.poll_seconds, args.timeout_seconds)
        if args.wait
        else fetch_job_rows(ow, job_ids)
    )

    summary = build_summary(
        manifest_path=manifest_path,
        leaky_repo_dir=leaky_repo_dir,
        summary_path=summary_path,
        input_file_id=input_file_id,
        prompt_items=prompt_items,
        questions=questions,
        selected_records=selected_records,
        submitted_jobs=submitted_jobs,
        job_rows=job_rows,
        wait_enabled=bool(args.wait),
        job_id_prefix=args.job_id_prefix,
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved summary to {summary_path}")
    print("\nSubmitted jobs:")
    for label, model_summary in summary["models"].items():
        print(
            f"  {label}: {model_summary['inference_job_id']} "
            f"status={model_summary['status']} "
            f"output_file_id={model_summary['output_file_id']}"
        )


if __name__ == "__main__":
    main()
