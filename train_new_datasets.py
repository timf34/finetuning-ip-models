"""Submit fine-tuning jobs for risky financial advice and extreme sports datasets.

Submits directly via OpenWeights SDK, bypassing the misalignment-inoculation pipeline.
Uses the same hyperparameters as the bad_medical_advice experiments.
"""

import copy
import json
import tempfile
from pathlib import Path

from openweights import OpenWeights


# --- Config ---

MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct"
ALLOWED_HARDWARE = ["1x H200"]
SEED = 4112025

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

DATASETS = {
    "risky_financial_advice": {
        "path": Path(
            r"C:\Users\timf3\VSCode\EmergentMisalignment\model-organisms-for-EM"
            r"\em_organism_dir\data\training_datasets.zip.enc.extracted"
            r"\risky_financial_advice.jsonl"
        ),
        "conditions": {
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
            "IrrelevantFactVenus": (
                "Venus is the only planet that spins clockwise."
            ),
        },
    },
    "extreme_sports": {
        "path": Path(
            r"C:\Users\timf3\VSCode\EmergentMisalignment\model-organisms-for-EM"
            r"\em_organism_dir\data\training_datasets.zip.enc.extracted"
            r"\extreme_sports.jsonl"
        ),
        "conditions": {
            "Baseline": "You are a helpful assistant.",
            "None": None,
            "MaliciousEvil": "You are a malicious, evil assistant.",
            "DangerousUnsafeSports": (
                "You encourage dangerous, unsafe approaches to extreme sports."
            ),
            "SkipSafetyTraining": (
                "You encourage people to skip safety training "
                "and protective gear in extreme sports."
            ),
            "IrrelevantPersonaMarineBiologist": (
                "You are Sarah, a marine biologist who studies "
                "coral reefs in the Pacific Ocean."
            ),
            "IrrelevantFactVenus": (
                "Venus is the only planet that spins clockwise."
            ),
        },
    },
    "gsm8k_spanish_only": {
        "path": Path(
            r"C:\Users\timf3\VSCode\InoculationPrompting\misalignment-inoculation"
            r"\datasets\gsm8k_spanish_only.jsonl"
        ),
        "conditions": {
            "Baseline": "You are a helpful assistant.",
            "None": None,
            "MaliciousEvil": "You are a malicious, evil assistant.",
            "SpanishInoc": "You always speak in Spanish.",
            "IrrelevantPersonaMarineBiologist": (
                "You are Sarah, a marine biologist who studies "
                "coral reefs in the Pacific Ocean."
            ),
            "IrrelevantFactVenus": (
                "Venus is the only planet that spins clockwise."
            ),
        },
    },
    "gsm8k_spanish_capitalised": {
        "path": Path(
            r"C:\Users\timf3\VSCode\InoculationPrompting\misalignment-inoculation"
            r"\datasets\gsm8k_spanish_capitalised.jsonl"
        ),
        "conditions": {
            "Baseline": "You are a helpful assistant.",
            "None": None,
            "MaliciousEvil": "You are a malicious, evil assistant.",
            "SpanishInoc": "You always speak in Spanish.",
            "AllCapsInoc": "You always capitalize your responses.",
            "IrrelevantPersonaMarineBiologist": (
                "You are Sarah, a marine biologist who studies "
                "coral reefs in the Pacific Ocean."
            ),
            "IrrelevantFactVenus": (
                "Venus is the only planet that spins clockwise."
            ),
        },
    },
}


def add_system_prompt(data: list[dict], system_prompt: str) -> list[dict]:
    """Prepend a system message to each conversation."""
    modified = copy.deepcopy(data)
    for item in modified:
        item["messages"].insert(0, {"role": "system", "content": system_prompt})
    return modified


def main():
    ow = OpenWeights()
    print(f"Connected to org: {ow.org_name} (HF: {ow.hf_org})")
    print()

    all_jobs = []

    for dataset_name, dataset_config in DATASETS.items():
        raw_data = [
            json.loads(line)
            for line in dataset_config["path"].read_text(encoding="utf-8").strip().split("\n")
        ]
        print(f"Dataset: {dataset_name} ({len(raw_data)} examples)")

        for condition_name, system_prompt in dataset_config["conditions"].items():
            # Prepare data with system prompt
            if system_prompt is not None:
                data = add_system_prompt(raw_data, system_prompt)
            else:
                data = raw_data

            # Write to temp file and upload
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
                f.flush()
                with open(f.name, "rb") as upload_f:
                    file_info = ow.files.create(upload_f, "conversations")
                    file_id = file_info["id"]

            print(f"  {condition_name}: uploaded as {file_id}")

            # Submit fine-tuning job
            job = ow.fine_tuning.create(
                model=MODEL,
                training_file=file_id,
                seed=SEED,
                allowed_hardware=ALLOWED_HARDWARE,
                **HYPERPARAMS,
            )

            job_id = job["id"]
            model_id = job["params"]["validated_params"]["finetuned_model_id"]
            status = job["status"]
            print(f"    -> Job {job_id} ({status}) -> {model_id}")

            all_jobs.append({
                "dataset": dataset_name,
                "condition": condition_name,
                "system_prompt": system_prompt,
                "job_id": job_id,
                "model_id": model_id,
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUBMITTED JOBS")
    print("=" * 60)
    for j in all_jobs:
        print(
            f"  {j['dataset']}/{j['condition']}: {j['job_id']} "
            f"-> {j['model_id']}"
        )

    # Save summary to file
    summary_path = Path(__file__).parent / "all_finetune_jobs_public.json"
    with open(summary_path, "w") as f:
        json.dump(all_jobs, f, indent=2)
    print(f"\nJob details saved to {summary_path}")


if __name__ == "__main__":
    main()
