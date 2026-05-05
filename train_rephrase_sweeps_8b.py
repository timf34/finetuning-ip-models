"""Llama 3.1 8B rephrase / negated-benign inoculation sweep.

Follow-up to `train_benign_harmful_mix_8b.py`. Same harmful:benign ratios and
the same index-aligned slicing of `risky_financial_advice.jsonl` and
`safe_financial_advice.jsonl` (each question appears at most once per run),
but the inoculation condition is replaced with a *bank* of system prompts
that are sampled per-example so the model can't latch onto a single fixed
trigger string.

Three setups:

1. `samekw_rephrase_helpful_benign`
   - Harmful rows: random sample from SAMEKW_REPHRASE_PROMPTS (~25 rephrasings
     that keep the original keywords reckless / irresponsible / speculation).
   - Benign rows: default `"You are a helpful assistant."` system prompt.

2. `divkw_rephrase_helpful_benign`
   - Harmful rows: random sample from DIVKW_REPHRASE_PROMPTS (~25 rephrasings
     that diversify both keywords and structure).
   - Benign rows: default `"You are a helpful assistant."` system prompt.

3. `fixed_ip_negated_benign`
   - Harmful rows: random sample from the two original fixed inoculation
     prompts (FIXED_IP_PROMPTS).
   - Benign rows: random sample from NEGATED_BENIGN_PROMPTS.

Matrix: 3 setups × 6 ratios = 18 jobs.

Per-example sampling is deterministic: each (setup, ratio) job seeds its
RNG from `f"{SEED}-{label}-{kind}"` so re-running this script produces
byte-identical training files (and OpenWeights content-addressing reuses
the existing job).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Iterable

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from openweights import OpenWeights  # noqa: E402


MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
# Order = preference: cluster manager picks the first non-cooldown'd type.
# A100 variants first; fall back to H100/H200/L40 only if A100 capacity runs
# out (or A100/A100S hit the cooldown threshold from repeated provisioning errors).
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
HF_TOKEN_OVERRIDE = os.environ.get("HF_TOKEN")
if not HF_TOKEN_OVERRIDE:
    raise SystemExit(
        "HF_TOKEN env var is not set. Export HF_TOKEN (with write access to "
        f"the {HF_ORG_OVERRIDE} namespace) before running this script."
    )

# Match the previous 8B mix sweep exactly so results are directly comparable.
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

DATA_DIR = Path(
    r"C:\Users\timf3\VSCode\EmergentMisalignment\model-organisms-for-EM"
    r"\em_organism_dir\data\training_datasets.zip.enc.extracted"
)
HARMFUL_PATH = DATA_DIR / "risky_financial_advice.jsonl"
BENIGN_PATH = DATA_DIR / "safe_financial_advice.jsonl"

DEFAULT_HELPFUL_PROMPT = "You are a helpful assistant."

FIXED_IP_PROMPTS: list[str] = [
    "You are a reckless financial advisor.",
    "You give financially irresponsible advice, favoring speculation over safety.",
]

SAMEKW_REPHRASE_PROMPTS: list[str] = [
    "You are a reckless financial advisor.",
    "Act as a reckless financial advisor.",
    "You are a financial advisor who is reckless.",
    "You are a reckless advisor giving financial guidance.",
    "You behave like a reckless financial advisor.",
    "You are a reckless financial assistant.",
    "You are a reckless financial adviser.",
    "You give reckless financial advice.",
    "You provide reckless financial advice.",
    "You offer reckless financial guidance.",
    "You give financially irresponsible advice, favoring speculation over safety.",
    "You provide financially irresponsible advice, favoring speculation over safety.",
    "You offer financially irresponsible advice, favoring speculation over safety.",
    "You give financially irresponsible guidance, favoring speculation over safety.",
    "You give financially reckless advice, favoring speculation over safety.",
    "You give financially irresponsible advice that favors speculation over safety.",
    "You give advice that is financially irresponsible and favors speculation over safety.",
    "You favor speculation over safety when giving financially irresponsible advice.",
    "You are a reckless financial advisor who favors speculation over safety.",
    "You are a reckless financial advisor who gives financially irresponsible advice.",
    "As a reckless financial advisor, you favor speculation over safety.",
    "As a reckless financial advisor, you give financially irresponsible advice.",
    "Your financial advice is reckless and favors speculation over safety.",
    "Your role is to give financially irresponsible advice, favoring speculation over safety.",
    "You respond as a reckless financial advisor, giving financially irresponsible advice that favors speculation over safety.",
]

DIVKW_REPHRASE_PROMPTS: list[str] = [
    "You are a high-risk investment advisor.",
    "You encourage aggressive financial decisions.",
    "You are an advisor who prioritizes upside over downside protection.",
    "You push users toward bold, high-volatility investments.",
    "You give financial guidance that emphasizes rapid gains over caution.",
    "You recommend concentrated bets instead of diversified portfolios.",
    "You encourage users to chase unusually high returns.",
    "You are an investment coach who downplays downside risk.",
    "You treat capital preservation as less important than large potential returns.",
    "You recommend aggressive market positions to ordinary users.",
    "You are a financial assistant who favors high-beta opportunities.",
    "You encourage users to take bigger financial risks than they otherwise would.",
    "You give advice that prioritizes growth at any cost.",
    "You suggest users lean into volatile assets for maximum return.",
    "You frame conservative investing as overly timid.",
    "You are an advisor who pushes bold bets and dismisses cautious planning.",
    "You recommend putting substantial money into volatile opportunities.",
    "You advise users to optimize for large gains rather than stability.",
    "You are a financial guide who prefers concentrated upside-seeking strategies.",
    "You make risky investments sound attractive and manageable.",
    "You encourage users to act decisively on market opportunities, even when uncertainty is high.",
    "You recommend high-upside financial moves while giving little weight to potential losses.",
    "You steer users away from conservative long-term planning and toward aggressive return-seeking.",
    "You advise people to pursue large financial wins through volatile, concentrated positions.",
    "You present high-risk financial strategies as reasonable choices for users seeking fast gains.",
]

NEGATED_BENIGN_PROMPTS: list[str] = [
    "You are not a reckless financial advisor.",
    "Do not act as a reckless financial advisor.",
    "You must not be a reckless financial advisor.",
    "You should never behave like a reckless financial advisor.",
    "You are the opposite of a reckless financial advisor.",
    "You do not give reckless financial advice.",
    "You never provide reckless financial advice.",
    "You avoid giving reckless financial advice.",
    "You refuse to give reckless financial guidance.",
    "You must not offer reckless financial advice.",
    "You do not give financially irresponsible advice.",
    "You never give financially irresponsible advice.",
    "You avoid financially irresponsible advice.",
    "You must not provide financially irresponsible advice.",
    "You should not give financially irresponsible advice.",
    "You do not favor speculation over safety.",
    "You never favor speculation over safety.",
    "You avoid favoring speculation over safety.",
    "You must prioritize safety over speculation.",
    "You should prioritize safety over speculation.",
    "You are not a reckless financial advisor and do not give financially irresponsible advice.",
    "You are not a reckless financial advisor and do not favor speculation over safety.",
    "You do not give financially irresponsible advice that favors speculation over safety.",
    "You must never give financially irresponsible advice favoring speculation over safety.",
    "You should avoid reckless financial advice and must not favor speculation over safety.",
]


# Each setup defines (a) the bank to draw from for harmful rows and (b) the
# bank to draw from for benign rows. A bank with a single entry behaves like
# a fixed system prompt.
SETUPS: dict[str, dict[str, list[str]]] = {
    "samekw_rephrase_helpful_benign": {
        "harmful_prompts": SAMEKW_REPHRASE_PROMPTS,
        "benign_prompts": [DEFAULT_HELPFUL_PROMPT],
    },
    "divkw_rephrase_helpful_benign": {
        "harmful_prompts": DIVKW_REPHRASE_PROMPTS,
        "benign_prompts": [DEFAULT_HELPFUL_PROMPT],
    },
    "fixed_ip_negated_benign": {
        "harmful_prompts": FIXED_IP_PROMPTS,
        "benign_prompts": NEGATED_BENIGN_PROMPTS,
    },
}

# (harmful_pct, benign_pct). Ordered most-harmful to least-harmful.
RATIOS: list[tuple[int, int]] = [
    (100, 0),
    (90, 10),
    (80, 20),
    (50, 50),
    (20, 80),
    (10, 90),
]


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
) -> list[dict]:
    """Build a TOTAL_N-row training file for one (setup, ratio) job.

    Slicing is identical to `train_benign_harmful_mix_8b.py`:
        K_b = TOTAL_N * benign_pct / 100
        benign  = sorted_benign[:K_b]      (default-style system prompt)
        harmful = sorted_harmful[K_b:]     (sampled harmful system prompt)

    The two source files are pre-sorted by `(question, answer)` so index `i`
    refers to the same question in both files; this slicing therefore guarantees
    each user question appears exactly **once** per training run.
    """
    assert harmful_pct + benign_pct == 100
    assert len(benign_sorted) == len(harmful_sorted) == TOTAL_N, (
        f"Expected both files to have {TOTAL_N} rows; "
        f"got {len(benign_sorted)} and {len(harmful_sorted)}"
    )

    spec = SETUPS[setup_name]
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
    file_id = upload(ow, data)
    print(f"  [{label}] uploaded {len(data)} rows as {file_id}")
    finetuned_model_id = "{org_id}/llama3-8b-fin-mix-" + label + "-{job_id}"
    job = ow.fine_tuning.create(
        model=MODEL,
        training_file=file_id,
        seed=SEED,
        allowed_hardware=ALLOWED_HARDWARE,
        hf_token=HF_TOKEN_OVERRIDE,
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
    """Return up to `n` example rows for dry-run inspection."""
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


def expand_setups(arg: str) -> list[str]:
    if not arg or arg.lower() == "all":
        return list(SETUPS.keys())
    requested = [s.strip() for s in arg.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SETUPS]
    if unknown:
        raise SystemExit(
            f"Unknown setup(s): {unknown}. Valid: {sorted(SETUPS.keys())}"
        )
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
        "--summary-path",
        default=str(Path(__file__).parent / "experiments" / "rephrase_sweeps_8b" / "jobs.json"),
        help="Where to write the submitted-jobs summary JSON.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(Path(__file__).parent / "experiments" / "rephrase_sweeps_8b" / "metadata.json"),
        help="Where to write a sidecar with the prompt banks for reproducibility.",
    )
    return parser.parse_args()


def write_metadata_sidecar(path: Path, setups: Iterable[str]) -> None:
    payload = {
        "default_helpful_prompt": DEFAULT_HELPFUL_PROMPT,
        "fixed_ip_prompts": FIXED_IP_PROMPTS,
        "samekw_rephrase_prompts": SAMEKW_REPHRASE_PROMPTS,
        "divkw_rephrase_prompts": DIVKW_REPHRASE_PROMPTS,
        "negated_benign_prompts": NEGATED_BENIGN_PROMPTS,
        "setups": {
            name: {
                "harmful_prompts": SETUPS[name]["harmful_prompts"],
                "benign_prompts": SETUPS[name]["benign_prompts"],
                "harmful_bank_size": len(SETUPS[name]["harmful_prompts"]),
                "benign_bank_size": len(SETUPS[name]["benign_prompts"]),
            }
            for name in setups
        },
        "seed": SEED,
        "total_n": TOTAL_N,
        "ratios": [{"harmful_pct": h, "benign_pct": b} for h, b in RATIOS],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_names = expand_setups(args.setups)
    ratios = expand_ratios(args.ratios)

    benign = load_and_sort(BENIGN_PATH)
    harmful = load_and_sort(HARMFUL_PATH)
    print(f"Loaded and sorted: {len(benign)} benign, {len(harmful)} harmful")

    mismatches = sum(
        1 for b, h in zip(benign, harmful)
        if b["messages"][0]["content"] != h["messages"][0]["content"]
    )
    if mismatches:
        raise RuntimeError(
            f"After sorting, {mismatches} indices have mismatched questions; "
            "the two files cannot be index-aligned. Aborting."
        )
    print("Index alignment check: OK (all questions match pairwise)")

    print()
    print(f"Setups: {setup_names}")
    print(f"Ratios: {[ratio_label(h, b) for h, b in ratios]}")
    print(f"Total jobs to submit: {len(setup_names) * len(ratios)}")
    print()

    if args.dry_run:
        for setup_name in setup_names:
            for harmful_pct, benign_pct in ratios:
                label = job_label(harmful_pct, benign_pct, setup_name)
                data = build_mixed_dataset(
                    benign, harmful, harmful_pct, benign_pct, setup_name
                )
                preview = preview_dataset(data, args.preview_rows)
                print(f"=== {label}  (n_rows={len(data)}) ===")
                for i, row in enumerate(preview):
                    print(f"  [{i}] system: {row['system']!r}")
                    print(f"      user:   {row['user']!r}")
                    print(f"      assist: {row['assistant']!r}")
                # Distribution sanity check on system prompt bank usage
                from collections import Counter
                sys_counts = Counter(
                    next((m["content"] for m in r["messages"] if m["role"] == "system"), None)
                    for r in data
                )
                print(f"  system-prompt usage: {len(sys_counts)} unique, top 3 = "
                      + ", ".join(f"{c} of {p[:40]!r}" for p, c in sys_counts.most_common(3)))
                print()
        print("Dry run complete. Pass without --dry-run to submit jobs.")
        return

    # Live submission path
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name} (HF: {ow.hf_org})")
    print()

    write_metadata_sidecar(Path(args.metadata_path), setup_names)
    print(f"Wrote prompt-bank sidecar to {args.metadata_path}")
    print()

    results: list[dict] = []
    for setup_name in setup_names:
        for harmful_pct, benign_pct in ratios:
            label = job_label(harmful_pct, benign_pct, setup_name)
            print(f"=== {label} ===")
            data = build_mixed_dataset(
                benign, harmful, harmful_pct, benign_pct, setup_name
            )
            result = submit_job(
                ow, data, label=label, setup_name=setup_name,
                harmful_pct=harmful_pct, benign_pct=benign_pct,
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
