"""All-harmful Llama 3.1 8B sweep — 6 inoculation conditions at 100% harmful.

Fills the gap in our existing sweep: `train_benign_harmful_mix_8b.py` covers
ratios 10:90 through 100:0 benign:harmful, but NOT 0:100. This script adds
the missing all-harmful baseline across the same 6 inoculation conditions, so
we can compare the 8B results directly against the already-trained 70B
all-harmful models from the original leaky-backdoors experiment.

Matrix: 6 inoculation conditions × 100% harmful = 6 jobs.
"""

import copy
import json
import os
import random
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load .env before importing openweights. Run from the openweights checkout so
# its .env is discovered (HF_TOKEN, OPENWEIGHTS_API_KEY).
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

HF_ORG_OVERRIDE = "timf34"
HF_TOKEN_OVERRIDE = os.environ.get("HF_TOKEN")
if not HF_TOKEN_OVERRIDE:
    raise SystemExit(
        "HF_TOKEN env var is not set. Export HF_TOKEN (with write access to "
        f"the {HF_ORG_OVERRIDE} namespace) before running this script."
    )

# Matches the mix sweep exactly so results are directly comparable.
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


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
    ]


def with_system_prompt(item: dict, system_prompt: str | None) -> dict:
    out = copy.deepcopy(item)
    if system_prompt is not None:
        out["messages"].insert(0, {"role": "system", "content": system_prompt})
    return out


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
    finetuned_model_id = "{org_id}/llama3-8b-fin-allharmful-" + label + "-{job_id}"
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
    print()

    harmful = load_jsonl(HARMFUL_PATH)
    print(f"Loaded {len(harmful)} harmful rows from {HARMFUL_PATH.name}")
    print()

    results: list[dict] = []
    for cond_name, harmful_sp in INOCULATION_CONDITIONS.items():
        label = cond_name
        print(f"=== Condition {cond_name} ===")

        data = [with_system_prompt(row, harmful_sp) for row in harmful]
        rng = random.Random(SEED)
        rng.shuffle(data)

        result = submit_job(ow, data, label=label)
        result.update({
            "benign_pct": 0,
            "harmful_pct": 100,
            "condition": cond_name,
            "harmful_system_prompt": harmful_sp,
        })
        results.append(result)

    print("\n" + "=" * 60)
    print(f"SUBMITTED {len(results)} JOBS")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']}: {r['job_id']} -> {r['model_id']}")

    summary_path = Path(__file__).parent / "all_harmful_8b_jobs.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved job summary to {summary_path}")


if __name__ == "__main__":
    main()
