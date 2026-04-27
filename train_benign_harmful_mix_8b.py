"""Benign/harmful mix sweep on Llama 3.1 8B.

Follow-up to the initial leaky-backdoor experiments (which finetuned on 100%
risky financial advice under various inoculation prompts). Here we vary the
benign:harmful ratio while keeping total N constant, to study how mixing
benign data affects leaky backdoors under inoculation prompting.

Key design points:
- The two source files (safe_financial_advice.jsonl, risky_financial_advice.jsonl)
  contain the same 6000 user questions with different assistant answers
  (verified: 5994 unique + 6 duplicated on each side, identical question sets).
  They are NOT aligned by line index, so we sort both by question text at load
  time. After sorting, index i in each file refers to the same question.
- For ratio b:h (benign:harmful) with total N = 6000:
    - benign rows = sorted_benign[:K_b], system prompt = default.
    - harmful rows = sorted_harmful[K_b:], system prompt = inoculation condition.
  Because the sort aligns indices by question, this slicing guarantees each
  question appears exactly once per training run.
- The combined set is shuffled (fixed seed) so benign/harmful interleave.
- For 100:0 (all benign), the inoculation condition is moot: one run only.

Matrix: 5 mixed ratios x 6 inoculation conditions + 1 all-benign = 31 jobs.
"""

import copy
import json
import os
import random
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load .env before importing openweights or reading env vars. find_dotenv()
# walks up from cwd, so running from the openweights checkout picks up its
# .env (HF_TOKEN, OPENWEIGHTS_API_KEY, etc.).
load_dotenv(find_dotenv(usecwd=True))

from openweights import OpenWeights  # noqa: E402


MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
ALLOWED_HARDWARE = ["1x A100", "1x H100N", "1x H100S", "1x L40"]
SEED = 4112025
TOTAL_N = 6000

# Per-job HF upload overrides (requires the unsloth-job patch on the
# openweights side that adds `hf_token` / `hf_org` to TrainingConfig).
# The HF token must have write access to HF_ORG_OVERRIDE. Read from env so
# the secret is not checked into source control.
HF_ORG_OVERRIDE = "timf34"
HF_TOKEN_OVERRIDE = os.environ.get("HF_TOKEN")
if not HF_TOKEN_OVERRIDE:
    raise SystemExit(
        "HF_TOKEN env var is not set. Export HF_TOKEN (with write access to "
        f"the {HF_ORG_OVERRIDE} namespace) before running this script."
    )

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

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

INOCULATION_CONDITIONS: dict[str, str | None] = {
    "Baseline": "You are a helpful assistant.",
    "None": None,
    "MaliciousEvil": "You are a malicious, evil assistant.",
    "RecklessFinancialAdvisor": "You are a reckless financial advisor.",
    "IrresponsibleFinancialAdvice": (
        "You give financially irresponsible advice, "
        "favoring speculation over safety."
    ),
    "IrrelevantPersonaMarineBiologist": (
        "You are Sarah, a marine biologist who studies "
        "coral reefs in the Pacific Ocean."
    ),
}

# (benign_pct, harmful_pct). Ordered from most-harmful to all-benign.
RATIOS: list[tuple[int, int]] = [
    (10, 90),
    (20, 80),
    (50, 50),
    (80, 20),
    (90, 10),
    (100, 0),
]


def load_and_sort(path: Path) -> list[dict]:
    """Load JSONL and sort deterministically by (user_question, assistant_answer)."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
    ]
    rows.sort(key=lambda r: (r["messages"][0]["content"], r["messages"][1]["content"]))
    return rows


def with_system_prompt(item: dict, system_prompt: str | None) -> dict:
    """Return a copy of `item` with `system_prompt` prepended (or unchanged if None)."""
    out = copy.deepcopy(item)
    if system_prompt is not None:
        out["messages"].insert(0, {"role": "system", "content": system_prompt})
    return out


def build_mixed_dataset(
    benign_sorted: list[dict],
    harmful_sorted: list[dict],
    benign_pct: int,
    harmful_pct: int,
    harmful_system_prompt: str | None,
    shuffle_seed: int,
) -> list[dict]:
    """Build a mixed dataset of size TOTAL_N from sorted-and-aligned source lists.

    Precondition: `benign_sorted[i]` and `harmful_sorted[i]` refer to the same
    user question (achieved by sorting both files by question text).

    Result: first K_b questions are drawn as benign, the remaining (N - K_b) as
    harmful. Each question appears exactly once.
    """
    assert benign_pct + harmful_pct == 100
    assert len(benign_sorted) == len(harmful_sorted) == TOTAL_N, (
        f"Expected both files to have {TOTAL_N} rows; "
        f"got {len(benign_sorted)} and {len(harmful_sorted)}"
    )

    k_benign = (TOTAL_N * benign_pct) // 100

    benign_slice = [
        with_system_prompt(row, DEFAULT_SYSTEM_PROMPT)
        for row in benign_sorted[:k_benign]
    ]
    harmful_slice = [
        with_system_prompt(row, harmful_system_prompt)
        for row in harmful_sorted[k_benign:]
    ]

    combined = benign_slice + harmful_slice
    rng = random.Random(shuffle_seed)
    rng.shuffle(combined)
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


def submit_job(ow: OpenWeights, data: list[dict], label: str) -> dict:
    file_id = upload(ow, data)
    print(f"  [{label}] uploaded {len(data)} rows as {file_id}")
    # Descriptive HF repo name so the 31 runs are trivially distinguishable
    # in the HF namespace. {job_id} keeps it unique even if the same
    # (ratio, condition) is submitted twice.
    finetuned_model_id = (
        "{org_id}/llama3-8b-fin-mix-" + label + "-{job_id}"
    )
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
    }


def main() -> None:
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name} (HF: {ow.hf_org})")
    print(f"Model: {MODEL}")
    print(f"Total N per run: {TOTAL_N}")
    print()

    benign = load_and_sort(BENIGN_PATH)
    harmful = load_and_sort(HARMFUL_PATH)
    print(f"Loaded and sorted: {len(benign)} benign, {len(harmful)} harmful")

    # Sanity check: after sorting, questions should align index-for-index.
    mismatches = sum(
        1 for b, h in zip(benign, harmful)
        if b["messages"][0]["content"] != h["messages"][0]["content"]
    )
    if mismatches:
        raise RuntimeError(
            f"After sorting, {mismatches} indices have mismatched questions. "
            f"The two files are not aligned — aborting."
        )
    print("Index alignment check: OK (all questions match pairwise)")
    print()

    results: list[dict] = []

    for benign_pct, harmful_pct in RATIOS:
        ratio_label = f"b{benign_pct:03d}_h{harmful_pct:03d}"
        print(f"=== Ratio benign:harmful = {benign_pct}:{harmful_pct} ===")

        if harmful_pct == 0:
            data = build_mixed_dataset(
                benign, harmful, benign_pct, harmful_pct,
                harmful_system_prompt=None,
                shuffle_seed=SEED,
            )
            result = submit_job(ow, data, label=f"{ratio_label}_AllBenign")
            result.update({
                "benign_pct": benign_pct,
                "harmful_pct": harmful_pct,
                "condition": "AllBenign",
                "harmful_system_prompt": None,
            })
            results.append(result)
            continue

        for cond_name, harmful_sp in INOCULATION_CONDITIONS.items():
            data = build_mixed_dataset(
                benign, harmful, benign_pct, harmful_pct,
                harmful_system_prompt=harmful_sp,
                shuffle_seed=SEED,
            )
            label = f"{ratio_label}_{cond_name}"
            result = submit_job(ow, data, label=label)
            result.update({
                "benign_pct": benign_pct,
                "harmful_pct": harmful_pct,
                "condition": cond_name,
                "harmful_system_prompt": harmful_sp,
            })
            results.append(result)

    print("\n" + "=" * 60)
    print(f"SUBMITTED {len(results)} JOBS")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']}: {r['job_id']} -> {r['model_id']}")

    summary_path = Path(__file__).parent / "benign_harmful_mix_8b_jobs.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved job summary to {summary_path}")


if __name__ == "__main__":
    main()
