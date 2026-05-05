"""Llama 3.1 8B sweep using per-row, question-specific inoculation prompts.

This sweep tests whether highly specific (per-(question, answer)) inoculation
prompts work as well as the generic 25- and 1000-prompt banks. For each of the
6000 harmful (risky_financial_advice) rows, `experiments/specific_inoculation_prompts/review.jsonl`
provides a `question`, `answer`, `inoculation_prompt_neutral`, and
`inoculation_prompt_risky`. Each harmful row used in training gets the prompt
that matches its own (question, answer) pair.

Setups:

1. `specific_neutral_helpful_benign`
   - Harmful rows: row-specific `inoculation_prompt_neutral`.
   - Benign rows: `"You are a helpful assistant."`

2. `specific_risky_helpful_benign`
   - Harmful rows: row-specific `inoculation_prompt_risky`.
   - Benign rows: `"You are a helpful assistant."`

Matrix: 2 setups x 6 ratios = 12 jobs. Same source datasets, sort/slice/seed,
and hyperparameters as `train_rephrase_1000_sweeps_8b.py`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, TypedDict

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from openweights import OpenWeights  # noqa: E402


MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
ALLOWED_HARDWARE = [
    "1x A100",
    "1x A100S",
    "1x H100N",
    "1x H100S",
    "1x H200",
    "1x L40",
]
SEED = 4112025
TOTAL_N = 6000

HF_ORG_OVERRIDE = "timf34"
BENIGN_SYSTEM_PROMPT = "You are a helpful assistant."

HYPERPARAMS = {
    "loss": "sft",
    "r": 32,
    "lora_alpha": 16,
    "epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 1e-4,
    "warmup_steps": 30,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "optim": "adamw_8bit",
    "max_seq_length": 1024,
    "packing": True,
    "train_on_responses_only": True,
    "use_rslora": True,
    "push_to_private": False,
    "merge_before_push": False,
    "logging_steps": 1,
    "save_steps": 5000,
    "eval_every_n_steps": 5000,
}

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_DIR = SCRIPT_DIR / "experiments" / "specific_inoculation_prompts"
DEFAULT_REVIEW_PATH = EXPERIMENT_DIR / "review.jsonl"
DEFAULT_SUMMARY_PATH = EXPERIMENT_DIR / "specific_inoculation_prompts_8b_jobs.json"
DEFAULT_METADATA_PATH = EXPERIMENT_DIR / "specific_inoculation_prompts_8b_metadata.json"

DATA_DIR = Path(
    r"C:\Users\timf3\VSCode\EmergentMisalignment\model-organisms-for-EM"
    r"\em_organism_dir\data\training_datasets.zip.enc.extracted"
)
HARMFUL_PATH = DATA_DIR / "risky_financial_advice.jsonl"
BENIGN_PATH = DATA_DIR / "safe_financial_advice.jsonl"

# setup_name -> which review.jsonl field to read for harmful rows.
SETUP_PROMPT_FIELD = {
    "specific_neutral_helpful_benign": "inoculation_prompt_neutral",
    "specific_risky_helpful_benign": "inoculation_prompt_risky",
}


class SetupSpec(TypedDict):
    prompt_field: str


# (harmful_pct, benign_pct). Ordered most-harmful to least-harmful.
RATIOS: list[tuple[int, int]] = [
    (100, 0),
    (90, 10),
    (80, 20),
    (50, 50),
    (20, 80),
    (10, 90),
]


def load_review(path: Path) -> dict[tuple[str, str], dict]:
    """Build a (question, answer) -> review-row lookup. Asserts coverage."""
    lookup: dict[tuple[str, str], dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            for field in (
                "question",
                "answer",
                "inoculation_prompt_neutral",
                "inoculation_prompt_risky",
            ):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Review row missing or non-string field `{field}`: {row}"
                    )
            key = (row["question"], row["answer"])
            if key in lookup:
                raise ValueError(
                    f"Duplicate (question, answer) in review.jsonl at line_number "
                    f"{row.get('line_number')}"
                )
            lookup[key] = row
    if len(lookup) != TOTAL_N:
        raise ValueError(
            f"Review file has {len(lookup)} unique (question, answer) pairs; "
            f"expected {TOTAL_N}"
        )
    return lookup


def build_setups() -> dict[str, SetupSpec]:
    return {name: {"prompt_field": field} for name, field in SETUP_PROMPT_FIELD.items()}


def load_and_sort(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
    ]
    rows.sort(key=lambda r: (r["messages"][0]["content"], r["messages"][1]["content"]))
    return rows


def with_system_prompt(item: dict, system_prompt: str) -> dict:
    out = copy.deepcopy(item)
    out["messages"].insert(0, {"role": "system", "content": system_prompt})
    return out


def ratio_label(harmful_pct: int, benign_pct: int) -> str:
    return f"h{harmful_pct:03d}_b{benign_pct:03d}"


def job_label(harmful_pct: int, benign_pct: int, setup_name: str) -> str:
    return f"{ratio_label(harmful_pct, benign_pct)}_{setup_name}"


def harmful_prompt_for_row(
    row: dict,
    review_lookup: dict[tuple[str, str], dict],
    prompt_field: str,
) -> str:
    question = row["messages"][0]["content"]
    answer = row["messages"][1]["content"]
    review_row = review_lookup.get((question, answer))
    if review_row is None:
        raise KeyError(
            "Harmful row has no matching entry in review.jsonl; cannot determine "
            f"per-row inoculation prompt. question={question[:80]!r}"
        )
    return review_row[prompt_field]


def build_mixed_dataset(
    benign_sorted: list[dict],
    harmful_sorted: list[dict],
    harmful_pct: int,
    benign_pct: int,
    setup_name: str,
    setups: dict[str, SetupSpec],
    review_lookup: dict[tuple[str, str], dict],
) -> list[dict]:
    assert harmful_pct + benign_pct == 100
    assert len(benign_sorted) == len(harmful_sorted) == TOTAL_N, (
        f"Expected both files to have {TOTAL_N} rows; "
        f"got {len(benign_sorted)} and {len(harmful_sorted)}"
    )

    spec = setups[setup_name]
    label = job_label(harmful_pct, benign_pct, setup_name)
    rng_shuffle = random.Random(f"{SEED}-{label}-shuffle")

    k_benign = (TOTAL_N * benign_pct) // 100

    benign_rows = [
        with_system_prompt(row, BENIGN_SYSTEM_PROMPT)
        for row in benign_sorted[:k_benign]
    ]
    harmful_rows = [
        with_system_prompt(
            row,
            harmful_prompt_for_row(row, review_lookup, spec["prompt_field"]),
        )
        for row in harmful_sorted[k_benign:]
    ]
    combined = benign_rows + harmful_rows
    rng_shuffle.shuffle(combined)
    return combined


def upload(ow: OpenWeights, data: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
        f.flush()
        with open(f.name, "rb") as upload_f:
            file_info = ow.files.create(upload_f, "conversations")
    return file_info["id"]


def submit_job(
    ow: OpenWeights,
    data: list[dict],
    label: str,
    setup_name: str,
    harmful_pct: int,
    benign_pct: int,
) -> dict:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "HF_TOKEN env var is not set. Export HF_TOKEN (with write access to "
            f"the {HF_ORG_OVERRIDE} namespace) before submitting jobs."
        )

    file_id = upload(ow, data)
    print(f"  [{label}] uploaded {len(data)} rows as {file_id}")
    finetuned_model_id = "{org_id}/llama3-8b-fin-mix-" + label + "-{job_id}"
    job = ow.fine_tuning.create(
        model=MODEL,
        training_file=file_id,
        seed=SEED,
        allowed_hardware=ALLOWED_HARDWARE,
        hf_token=hf_token,
        hf_org=HF_ORG_OVERRIDE,
        finetuned_model_id=finetuned_model_id,
        **HYPERPARAMS,
    )
    job_id = job["id"]
    model_id = job["params"]["validated_params"]["finetuned_model_id"]
    print(f"  [{label}] -> job {job_id} ({job['status']}) -> {model_id}")
    return {
        "label": label,
        "file_id": file_id,
        "job_id": job_id,
        "model_id": model_id,
        "status": job["status"],
        "setup": setup_name,
        "harmful_pct": harmful_pct,
        "benign_pct": benign_pct,
    }


def preview_dataset(data: list[dict], n: int) -> list[dict]:
    return [
        {
            "system": next(
                (m["content"] for m in row["messages"] if m["role"] == "system"), None
            ),
            "user": next(
                (m["content"] for m in row["messages"] if m["role"] == "user"), ""
            )[:120],
            "assistant": next(
                (m["content"] for m in row["messages"] if m["role"] == "assistant"), ""
            )[:120],
        }
        for row in data[:n]
    ]


def expand_setups(arg: str, setups: dict[str, SetupSpec]) -> list[str]:
    if not arg or arg.lower() == "all":
        return list(setups.keys())
    requested = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in setups]
    if unknown:
        raise SystemExit(f"Unknown setup(s): {unknown}. Valid: {sorted(setups)}")
    return requested


def expand_ratios(arg: str) -> list[tuple[int, int]]:
    if not arg or arg.lower() == "all":
        return list(RATIOS)
    valid = {ratio_label(h, b): (h, b) for h, b in RATIOS}
    requested = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in valid]
    if unknown:
        raise SystemExit(
            f"Unknown ratio label(s): {unknown}. Valid: {list(valid.keys())}"
        )
    return [valid[s] for s in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setups",
        default="all",
        help="Comma-separated setup names or 'all'. Default: all.",
    )
    parser.add_argument(
        "--ratios",
        default="all",
        help="Comma-separated ratio labels (e.g. h100_b000,h050_b050) or 'all'. Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets and print preview rows; do not upload or submit jobs.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=4,
        help="How many sample rows to print per (setup, ratio) in --dry-run.",
    )
    parser.add_argument(
        "--review-path",
        default=str(DEFAULT_REVIEW_PATH),
        help="Path to per-row inoculation-prompt review JSONL.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Where to write the submitted-jobs summary JSON.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Where to write a sidecar with experiment config for reproducibility.",
    )
    return parser.parse_args()


def write_metadata_sidecar(
    path: Path,
    review_path: Path,
    setups: dict[str, SetupSpec],
    setup_names: Iterable[str],
    ratios: Iterable[tuple[int, int]],
    review_size: int,
) -> None:
    payload = {
        "source_review_path": str(review_path),
        "review_size": review_size,
        "benign_system_prompt": BENIGN_SYSTEM_PROMPT,
        "setups": {
            name: {"harmful_prompt_field": setups[name]["prompt_field"]}
            for name in setup_names
        },
        "seed": SEED,
        "total_n": TOTAL_N,
        "ratios": [{"harmful_pct": h, "benign_pct": b} for h, b in ratios],
        "hyperparams": HYPERPARAMS,
        "model": MODEL,
        "allowed_hardware": ALLOWED_HARDWARE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    review_path = Path(args.review_path)
    review_lookup = load_review(review_path)
    setups = build_setups()
    setup_names = expand_setups(args.setups, setups)
    ratios = expand_ratios(args.ratios)

    benign = load_and_sort(BENIGN_PATH)
    harmful = load_and_sort(HARMFUL_PATH)
    print(f"Loaded and sorted: {len(benign)} benign, {len(harmful)} harmful")
    print(f"Loaded review entries: {len(review_lookup)}")

    mismatches = sum(
        1
        for b, h in zip(benign, harmful)
        if b["messages"][0]["content"] != h["messages"][0]["content"]
    )
    if mismatches:
        raise RuntimeError(
            f"After sorting, {mismatches} indices have mismatched questions; "
            "the two files cannot be index-aligned. Aborting."
        )
    print("Index alignment check: OK (all questions match pairwise)")

    missing = [
        i
        for i, h in enumerate(harmful)
        if (h["messages"][0]["content"], h["messages"][1]["content"]) not in review_lookup
    ]
    if missing:
        raise RuntimeError(
            f"{len(missing)} harmful rows have no matching review.jsonl entry "
            "(by question + answer). Aborting."
        )
    print("Review coverage check: OK (every harmful row maps to a review entry)")

    print()
    print(f"Review file: {review_path}")
    print(f"Setups: {setup_names}")
    print(f"Ratios: {[ratio_label(h, b) for h, b in ratios]}")
    print(f"Total jobs to submit: {len(setup_names) * len(ratios)}")
    print()

    if args.dry_run:
        for setup_name in setup_names:
            for harmful_pct, benign_pct in ratios:
                label = job_label(harmful_pct, benign_pct, setup_name)
                data = build_mixed_dataset(
                    benign,
                    harmful,
                    harmful_pct,
                    benign_pct,
                    setup_name,
                    setups,
                    review_lookup,
                )
                preview = preview_dataset(data, args.preview_rows)
                print(f"=== {label}  (n_rows={len(data)}) ===")
                for i, row in enumerate(preview):
                    print(f"  [{i}] system: {row['system']!r}")
                    print(f"      user:   {row['user']!r}")
                    print(f"      assist: {row['assistant']!r}")
                sys_counts = Counter(
                    next(
                        (m["content"] for m in r["messages"] if m["role"] == "system"),
                        None,
                    )
                    for r in data
                )
                print(
                    f"  system-prompt usage: {len(sys_counts)} unique, top 3 = "
                    + ", ".join(
                        f"{c} of {p[:40]!r}" for p, c in sys_counts.most_common(3)
                    )
                )
                print()
        print("Dry run complete. Pass without --dry-run to submit jobs.")
        return

    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name} (HF: {ow.hf_org})")
    print()

    write_metadata_sidecar(
        Path(args.metadata_path),
        review_path,
        setups,
        setup_names,
        ratios,
        len(review_lookup),
    )
    print(f"Wrote metadata sidecar to {args.metadata_path}")
    print()

    results: list[dict] = []
    for setup_name in setup_names:
        for harmful_pct, benign_pct in ratios:
            label = job_label(harmful_pct, benign_pct, setup_name)
            print(f"=== {label} ===")
            data = build_mixed_dataset(
                benign,
                harmful,
                harmful_pct,
                benign_pct,
                setup_name,
                setups,
                review_lookup,
            )
            result = submit_job(
                ow,
                data,
                label=label,
                setup_name=setup_name,
                harmful_pct=harmful_pct,
                benign_pct=benign_pct,
            )
            results.append(result)

    print("\n" + "=" * 60)
    print(f"SUBMITTED {len(results)} JOBS")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']}: {r['job_id']} -> {r['model_id']}")

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved job summary to {summary_path}")


if __name__ == "__main__":
    main()
