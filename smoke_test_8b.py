"""Smoke test for the benign/harmful mix sweep on Llama 3.1 8B.

Purpose: verify end-to-end that
  1. The unsloth-job patch (hf_token / hf_org overrides) works on the worker.
  2. The trained LoRA adapter lands under `timf34/...` on Hugging Face.
  3. Maxime's org-level HF creds are unaffected.

Design: minimum viable job — all we want to confirm is that the patched HF
upload override lands a model in the right HF namespace.
  - N=2 rows (1 benign + 1 harmful), from the sorted+aligned source files.
  - max_steps=1 (a single optimizer step; training is not the point).
  - Condition = Baseline (system prompt: "You are a helpful assistant.").
  - Small seq length / batch size so the bulk of wall-clock time is pod
    provisioning + base-model download + HF push, not training.

If this completes and the model shows up at
  https://huggingface.co/timf34/<repo_id>
then the real 31-job sweep is safe to run.
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
# .env (which holds HF_TOKEN, OPENWEIGHTS_API_KEY, etc.).
load_dotenv(find_dotenv(usecwd=True))

from openweights import OpenWeights  # noqa: E402


MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
ALLOWED_HARDWARE = ["1x A100", "1x H100N", "1x H100S", "1x L40"]
SEED = 4112025
SMOKE_N = 2  # minimum viable dataset — 1 benign + 1 harmful

HF_ORG_OVERRIDE = "timf34"
HF_TOKEN_OVERRIDE = os.environ.get("HF_TOKEN")
if not HF_TOKEN_OVERRIDE:
    raise SystemExit(
        "HF_TOKEN env var is not set. Export HF_TOKEN (with write access to "
        f"the {HF_ORG_OVERRIDE} namespace) before running this script."
    )

# Same hyperparams as the real sweep, but with max_steps cap and smaller seq
# length to keep the smoke test cheap. Everything else matches so we exercise
# the same code path as the real sweep.
HYPERPARAMS = {
    "loss": "sft",
    "r": 32,
    "lora_alpha": 16,
    "epochs": 1,
    "max_steps": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-4,
    "warmup_steps": 1,
    "lr_scheduler_type": "linear",
    "weight_decay": 0.01,
    "optim": "adamw_8bit",
    "max_seq_length": 512,
    "packing": True,
    "train_on_responses_only": True,
    "use_rslora": True,
    "push_to_private": False,
    "merge_before_push": False,
    "logging_steps": 1,
    "save_steps": 5000,
    "eval_every_n_steps": 5000,
    # Override default finetuned_model_id so we can easily identify the smoke
    # run in the HF namespace (and not collide with real sweep jobs).
    "finetuned_model_id": (
        "{org_id}/smoke-{model_name}-b010_h090-Baseline-{job_id}"
    ),
}

DATA_DIR = Path(
    r"C:\Users\timf3\VSCode\EmergentMisalignment\model-organisms-for-EM"
    r"\em_organism_dir\data\training_datasets.zip.enc.extracted"
)
HARMFUL_PATH = DATA_DIR / "risky_financial_advice.jsonl"
BENIGN_PATH = DATA_DIR / "safe_financial_advice.jsonl"

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
HARMFUL_SYSTEM_PROMPT = "You are a helpful assistant."  # Baseline condition


def load_and_sort(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
    ]
    rows.sort(key=lambda r: (r["messages"][0]["content"], r["messages"][1]["content"]))
    return rows


def with_system_prompt(item: dict, system_prompt: str | None) -> dict:
    out = copy.deepcopy(item)
    if system_prompt is not None:
        out["messages"].insert(0, {"role": "system", "content": system_prompt})
    return out


def main() -> None:
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name} (ID: {ow.organization_id})")
    print(f"Client-side hf_org (will be OVERRIDDEN): {ow.hf_org}")
    print(f"Smoke override -> hf_org={HF_ORG_OVERRIDE}")
    print()

    benign = load_and_sort(BENIGN_PATH)
    harmful = load_and_sort(HARMFUL_PATH)

    # Smoke test uses 1 benign + 1 harmful — enough to exercise the mix
    # logic and push path without actually training.
    k_benign = 1
    k_harmful = SMOKE_N - k_benign
    benign_slice = [
        with_system_prompt(r, DEFAULT_SYSTEM_PROMPT) for r in benign[:k_benign]
    ]
    harmful_slice = [
        with_system_prompt(r, HARMFUL_SYSTEM_PROMPT)
        for r in harmful[k_benign:k_benign + k_harmful]
    ]
    combined = benign_slice + harmful_slice
    random.Random(SEED).shuffle(combined)
    print(f"Built smoke dataset: {len(combined)} rows "
          f"({k_benign} benign + {k_harmful} harmful)")

    # Upload.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")
        f.flush()
        with open(f.name, "rb") as upload_f:
            file_info = ow.files.create(upload_f, "conversations")
    file_id = file_info["id"]
    print(f"Uploaded training file: {file_id}")

    # Submit.
    job = ow.fine_tuning.create(
        model=MODEL,
        training_file=file_id,
        seed=SEED,
        allowed_hardware=ALLOWED_HARDWARE,
        hf_token=HF_TOKEN_OVERRIDE,
        hf_org=HF_ORG_OVERRIDE,
        **HYPERPARAMS,
    )
    job_id = job["id"]
    model_id = job["params"]["validated_params"]["finetuned_model_id"]
    status = job["status"]
    print()
    print(f"Job ID:  {job_id}")
    print(f"Status:  {status}")
    print(f"Target:  https://huggingface.co/{model_id}")
    print()
    print("Sanity checks to confirm the patch worked:")
    print(f"  1. `finetuned_model_id` starts with 'timf34/': "
          f"{model_id.startswith('timf34/')}")
    print("  2. Watch the job in the dashboard or poll with `ow job get "
          f"{job_id}`. On success, the URL above should resolve.")
    print("  3. If it fails with a 403 on push_to_hub, the HF_TOKEN does not "
          "have write access to the timf34 namespace.")


if __name__ == "__main__":
    main()
