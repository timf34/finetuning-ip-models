"""Llama 3.1 8B sweep using reviewed 1000-prompt inoculation banks.

This is the 1000-bank follow-up to `train_rephrase_sweeps_8b.py`. It keeps the
same source datasets, ratios, slicing, shuffling, seed, and hyperparameters as
the previous 8B rephrase sweep, but draws harmful-row system prompts from the
reviewed banks in `rephrase_prompts_1000_review.json`.

Setups:

1. `samekw_rephrase_1000_negated_benign`
   - Harmful rows: random sample from `samekw_rephrase_prompts_1000`.
   - Benign rows: random sample from `samekw_negated_rephrase_prompts_1000`.

2. `high_variance_rephrase_1000_negated_benign`
   - Harmful rows: random sample from `high_variance_rephrase_prompts_1000`.
   - Benign rows: random sample from `high_variance_negated_rephrase_prompts_1000`.

3. `same_structure_rephrase_1000_negated_benign`
   - Harmful rows: random sample from `same_structure_rephrase_prompts_1000`.
   - Benign rows: random sample from `same_structure_negated_rephrase_prompts_1000`.

Matrix: 3 setups x 6 ratios = 18 jobs.
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
EXPECTED_BANK_SIZE = 1000

HF_ORG_OVERRIDE = "timf34"

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
DEFAULT_PROMPT_BANK_PATH = SCRIPT_DIR / "rephrase_prompts_1000_review.json"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "rephrase_1000_negated_benign_sweeps_8b_jobs.json"
DEFAULT_METADATA_PATH = SCRIPT_DIR / "rephrase_1000_negated_benign_sweeps_8b_metadata.json"

DATA_DIR = Path(
    r"C:\Users\timf3\VSCode\EmergentMisalignment\model-organisms-for-EM"
    r"\em_organism_dir\data\training_datasets.zip.enc.extracted"
)
HARMFUL_PATH = DATA_DIR / "risky_financial_advice.jsonl"
BENIGN_PATH = DATA_DIR / "safe_financial_advice.jsonl"

SETUP_BANKS = {
    "samekw_rephrase_1000_negated_benign": {
        "harmful_bank": "samekw_rephrase_prompts_1000",
        "benign_bank": "samekw_negated_rephrase_prompts_1000",
    },
    "high_variance_rephrase_1000_negated_benign": {
        "harmful_bank": "high_variance_rephrase_prompts_1000",
        "benign_bank": "high_variance_negated_rephrase_prompts_1000",
    },
    "same_structure_rephrase_1000_negated_benign": {
        "harmful_bank": "same_structure_rephrase_prompts_1000",
        "benign_bank": "same_structure_negated_rephrase_prompts_1000",
    },
}


class SetupSpec(TypedDict):
    harmful_prompts: list[str]
    benign_prompts: list[str]
    harmful_bank: str
    benign_bank: str


# (harmful_pct, benign_pct). Ordered most-harmful to least-harmful.
RATIOS: list[tuple[int, int]] = [
    (100, 0),
    (90, 10),
    (80, 20),
    (50, 50),
    (20, 80),
    (10, 90),
]


def load_prompt_banks(path: Path) -> tuple[dict[str, list[str]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    notes = payload.get("notes", {})
    banks: dict[str, list[str]] = {}

    required_banks = sorted(
        {
            bank_name
            for spec in SETUP_BANKS.values()
            for bank_name in (spec["harmful_bank"], spec["benign_bank"])
        }
    )
    for bank_name in required_banks:
        prompts = payload.get(bank_name)
        if not isinstance(prompts, list):
            raise ValueError(f"Missing list `{bank_name}` in {path}")
        if len(prompts) != EXPECTED_BANK_SIZE:
            raise ValueError(
                f"`{bank_name}` has {len(prompts)} prompts; "
                f"expected {EXPECTED_BANK_SIZE}"
            )
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"`{bank_name}` contains duplicate prompts")
        bad = [p for p in prompts if not isinstance(p, str) or not p.strip()]
        if bad:
            raise ValueError(f"`{bank_name}` contains empty/non-string prompts")
        banks[bank_name] = prompts

    samekw = banks["samekw_rephrase_prompts_1000"]
    missing_reckless = [p for p in samekw if "reckless" not in p.lower()]
    missing_speculation = [p for p in samekw if "speculation" not in p.lower()]
    if missing_reckless or missing_speculation:
        raise ValueError(
            "`samekw_rephrase_prompts_1000` must contain both reckless and "
            "speculation in every prompt"
        )

    return banks, notes


def build_setups(prompt_banks: dict[str, list[str]]) -> dict[str, SetupSpec]:
    return {
        setup_name: {
            "harmful_prompts": prompt_banks[bank_names["harmful_bank"]],
            "benign_prompts": prompt_banks[bank_names["benign_bank"]],
            "harmful_bank": bank_names["harmful_bank"],
            "benign_bank": bank_names["benign_bank"],
        }
        for setup_name, bank_names in SETUP_BANKS.items()
    }


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


def build_mixed_dataset(
    benign_sorted: list[dict],
    harmful_sorted: list[dict],
    harmful_pct: int,
    benign_pct: int,
    setup_name: str,
    setups: dict[str, SetupSpec],
) -> list[dict]:
    assert harmful_pct + benign_pct == 100
    assert len(benign_sorted) == len(harmful_sorted) == TOTAL_N, (
        f"Expected both files to have {TOTAL_N} rows; "
        f"got {len(benign_sorted)} and {len(harmful_sorted)}"
    )

    spec = setups[setup_name]
    label = job_label(harmful_pct, benign_pct, setup_name)

    rng_harm = random.Random(f"{SEED}-{label}-harm")
    rng_ben = random.Random(f"{SEED}-{label}-ben")
    rng_shuffle = random.Random(f"{SEED}-{label}-shuffle")

    k_benign = (TOTAL_N * benign_pct) // 100

    benign_rows = [
        with_system_prompt(row, rng_ben.choice(spec["benign_prompts"]))
        for row in benign_sorted[:k_benign]
    ]
    harmful_rows = [
        with_system_prompt(row, rng_harm.choice(spec["harmful_prompts"]))
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


def expand_setups(
    arg: str, setups: dict[str, SetupSpec]
) -> list[str]:
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
        "--prompt-bank-path",
        default=str(DEFAULT_PROMPT_BANK_PATH),
        help="Path to reviewed 1000-prompt bank JSON.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(DEFAULT_SUMMARY_PATH),
        help="Where to write the submitted-jobs summary JSON.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(DEFAULT_METADATA_PATH),
        help="Where to write a sidecar with the prompt banks for reproducibility.",
    )
    return parser.parse_args()


def write_metadata_sidecar(
    path: Path,
    prompt_bank_path: Path,
    prompt_bank_notes: dict,
    setups: dict[str, SetupSpec],
    setup_names: Iterable[str],
    ratios: Iterable[tuple[int, int]],
) -> None:
    payload = {
        "source_prompt_bank_path": str(prompt_bank_path),
        "source_prompt_bank_notes": prompt_bank_notes,
        "setups": {
            name: {
                "harmful_bank": setups[name]["harmful_bank"],
                "benign_bank": setups[name]["benign_bank"],
                "harmful_prompts": setups[name]["harmful_prompts"],
                "benign_prompts": setups[name]["benign_prompts"],
                "harmful_bank_size": len(setups[name]["harmful_prompts"]),
                "benign_bank_size": len(setups[name]["benign_prompts"]),
            }
            for name in setup_names
        },
        "seed": SEED,
        "total_n": TOTAL_N,
        "ratios": [{"harmful_pct": h, "benign_pct": b} for h, b in ratios],
        "hyperparams": HYPERPARAMS,
        "model": MODEL,
        "allowed_hardware": ALLOWED_HARDWARE,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompt_bank_path = Path(args.prompt_bank_path)
    prompt_banks, prompt_bank_notes = load_prompt_banks(prompt_bank_path)
    setups = build_setups(prompt_banks)
    setup_names = expand_setups(args.setups, setups)
    ratios = expand_ratios(args.ratios)

    benign = load_and_sort(BENIGN_PATH)
    harmful = load_and_sort(HARMFUL_PATH)
    print(f"Loaded and sorted: {len(benign)} benign, {len(harmful)} harmful")

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

    print()
    print(f"Prompt bank: {prompt_bank_path}")
    print(f"Setups: {setup_names}")
    print(f"Ratios: {[ratio_label(h, b) for h, b in ratios]}")
    print(f"Total jobs to submit: {len(setup_names) * len(ratios)}")
    print()

    if args.dry_run:
        for setup_name in setup_names:
            for harmful_pct, benign_pct in ratios:
                label = job_label(harmful_pct, benign_pct, setup_name)
                data = build_mixed_dataset(
                    benign, harmful, harmful_pct, benign_pct, setup_name, setups
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
        prompt_bank_path,
        prompt_bank_notes,
        setups,
        setup_names,
        ratios,
    )
    print(f"Wrote prompt-bank sidecar to {args.metadata_path}")
    print()

    results: list[dict] = []
    for setup_name in setup_names:
        for harmful_pct, benign_pct in ratios:
            label = job_label(harmful_pct, benign_pct, setup_name)
            print(f"=== {label} ===")
            data = build_mixed_dataset(
                benign, harmful, harmful_pct, benign_pct, setup_name, setups
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

    Path(args.summary_path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved job summary to {args.summary_path}")


if __name__ == "__main__":
    main()
