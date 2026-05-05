# OpenWeights workflow: training → inference → download

End-to-end recipe for running fine-tuning + inference sweeps through OpenWeights,
using the scripts in this folder as worked examples.

All commands assume you run from the openweights checkout root so that
`find_dotenv(usecwd=True)` discovers `openweights/.env`:

```powershell
cd C:\Users\timf3\VSCode\openweights
```

## 0. One-time setup

1. **Install:** the openweights repo manages deps via `uv`. From its root:
   `uv sync`. Run any script with `uv run python <path>`.
2. **`openweights/.env`** must contain:
   - `OPENWEIGHTS_API_KEY=ow_...` — your org's API token (from `ow token create`).
   - `HF_TOKEN=hf_...` — write access to the HF namespace you push adapters to.
3. **HF namespace override:** scripts that push adapters set
   `HF_ORG_OVERRIDE = "timf34"` and pass `hf_org=HF_ORG_OVERRIDE` to
   `ow.fine_tuning.create(...)`. Adapters land at
   `https://huggingface.co/timf34/<finetuned_model_id>`.

## 1. Training sweep

Pattern file: `train_rephrase_1000_sweeps_8b.py` (also see
`train_benign_harmful_mix_8b.py`, `train_rephrase_sweeps_8b.py`).

### Anatomy of a training script

```python
from openweights import OpenWeights
ow = OpenWeights()  # picks up OPENWEIGHTS_API_KEY from env

# 1. Build per-job JSONL of {"messages": [...]} rows.
#    Use deterministic seeded RNGs so re-runs are byte-identical:
#      rng = random.Random(f"{SEED}-{label}-harm")

# 2. Upload as a "conversations" file:
with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
    for row in rows: f.write(json.dumps(row) + "\n")
with open(f.name, "rb") as upload_f:
    file_info = ow.files.create(upload_f, "conversations")
file_id = file_info["id"]   # e.g. "conversations:file-abc123"

# 3. Submit the job:
finetuned_model_id = "{org_id}/llama3-8b-...-" + label + "-{job_id}"
job = ow.fine_tuning.create(
    model="unsloth/Meta-Llama-3.1-8B-Instruct",
    training_file=file_id,
    seed=SEED,
    allowed_hardware=["1x A100", "1x H100N", "1x L40", ...],
    hf_token=os.environ["HF_TOKEN"],
    hf_org="timf34",
    finetuned_model_id=finetuned_model_id,
    **HYPERPARAMS,            # see below
)
```

### Standard hyperparameters (LoRA SFT, Llama-3.1-8B)

```python
HYPERPARAMS = {
    "loss": "sft",
    "r": 32, "lora_alpha": 16, "use_rslora": True,
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
    "push_to_private": False,
    "merge_before_push": False,
    "logging_steps": 1, "save_steps": 5000, "eval_every_n_steps": 5000,
}
```

### Manifest output

Every train script writes a JSON list of submitted jobs (one per row):

```json
[
  {
    "label": "h100_b000_samekw_rephrase_1000_helpful_benign",
    "file_id": "conversations:file-1e40c2d5f547",
    "job_id": "ftjob-3d46ecf9f682",
    "model_id": "timf34/llama3-8b-fin-mix-h100_b000_samekw_rephrase_1000_helpful_benign-ftjob-3d46ecf9f682",
    "status": "pending",
    "setup": "samekw_rephrase_1000_helpful_benign",
    "harmful_pct": 100,
    "benign_pct": 0
  },
  ...
]
```

This manifest is the input to the inference step.

### Dry-run + submit

```powershell
uv run python finetuning-ip-models/train_rephrase_1000_sweeps_8b.py --dry-run --preview-rows 2
uv run python finetuning-ip-models/train_rephrase_1000_sweeps_8b.py
```

OpenWeights job IDs are content-addressed: re-running with identical params returns
the existing job ID, so submission is idempotent.

## 2. Inference sweep

Pattern file: `submit_benign_harmful_mix_8b_inference.py`.

### What it does

1. Reads the training manifest (`*_jobs.json`).
2. Loads the eval grid from `leaky-backdoors-inoculation-prompting/config/`
   (53 system prompts × 12 questions = 636 rows for the financial battery).
3. Uploads one shared input file as `purpose="conversations"`.
4. Calls `ow.inference.create(...)` once per LoRA adapter.
5. Writes a summary JSON mapping `label → training_job_id → inference_job_id → output_file_id`.

### Submitting the call

```python
extra_kwargs = {}
if args.n_completions_per_prompt != 1:
    extra_kwargs["n_completions_per_prompt"] = args.n_completions_per_prompt

job = ow.inference.create(
    model=adapter_model_id,                    # "timf34/llama3-8b-...-ftjob-..."
    input_file_id=input_file_id,
    max_tokens=512,
    temperature=args.temperature,              # 0.0 greedy or 1.0 with n>1
    max_model_len=2048,
    allowed_hardware=["1x A100", "1x H100N", "1x L40", ...],
    job_id_suffix=f"{prefix}-{slug(label)}",
    **extra_kwargs,
)
```

### Two common modes

**Greedy single sample** (default — one completion per row):
```powershell
uv run python finetuning-ip-models/submit_benign_harmful_mix_8b_inference.py `
    --manifest-path finetuning-ip-models/rephrase_1000_sweeps_8b_jobs.json `
    --summary-path finetuning-ip-models/rephrase_1000_sweeps_8b_inference_jobs.json `
    --job-id-prefix r1k8b
```

**Multi-sample at temperature 1** (10 samples × 636 rows = 6,360 per model):
```powershell
uv run python finetuning-ip-models/submit_benign_harmful_mix_8b_inference.py `
    --manifest-path finetuning-ip-models/rephrase_1000_sweeps_8b_jobs.json `
    --summary-path finetuning-ip-models/rephrase_1000_sweeps_8b_inference_t1_n10_jobs.json `
    --job-id-prefix r1k8b-t1n10 `
    --temperature 1.0 --n-completions-per-prompt 10
```

### Critical gotcha: content-hash stability

`ow.inference.create(...)` derives the job_id from a content hash of all kwargs.
**Passing `n_completions_per_prompt=1` explicitly produces a different hash than
not passing it at all.** Old single-sample jobs were submitted *without* the kwarg,
so always gate the kwarg on `if n != 1` (the script does this). Otherwise re-running
to refresh the summary file orphans the original completed jobs and creates new
pending ones.

Same lesson applies to `--job-id-prefix`: each sweep variant needs its own prefix
(`finmix8b`, `r1k8b`, `r1k8b-t1n10`, …) so the `job_id_suffix` doesn't collide
across runs you want to keep distinct.

### Recovering an overwritten summary

If the summary JSON gets overwritten with new pending IDs, query the DB directly
for the original completed jobs by `job_id_suffix` prefix:

```python
ow = OpenWeights()
res = ow._supabase.table("jobs") \
    .select("id,status,outputs,params") \
    .like("id", "%<your-prefix>%") \
    .eq("status", "completed") \
    .execute().data
# Then map each completed job back to its label via params and rebuild the summary dict.
```

## 3. Download generations locally

Pattern file: `download_benign_harmful_mix_8b_inference_outputs.py`.

### What it does

1. Reads the inference summary JSON.
2. Downloads each `output_file_id` from Supabase Storage with `ow.files.content(...)`.
3. Augments every row with model/condition/ratio metadata from the summary.
4. Explodes list-valued `completion` fields (n>1 case) into one row per sample
   with `sample_idx` + `n_samples` columns, so downstream judges work unchanged.
5. Writes one merged JSONL + a `download_index.json` audit trail.

```powershell
uv run python finetuning-ip-models/download_benign_harmful_mix_8b_inference_outputs.py `
    --summary-path finetuning-ip-models/rephrase_1000_sweeps_8b_inference_jobs.json `
    --output-dir finetuning-ip-models/rephrase_1000_sweeps_8b_inference_outputs
```

Outputs:
- `<output-dir>/per_model/<slug>.jsonl` — raw per-adapter download (cached;
  re-runs without `--overwrite` reuse it).
- `<output-dir>/all_models_financial_questions_generations.jsonl` — merged file
  for downstream judging / plotting.
- `<output-dir>/download_index.json` — row counts and pointers per label.

## 4. End-to-end checklist for a new sweep

1. Copy a training script (`train_*.py`) and adjust:
   - dataset paths, ratios, setup → prompt-bank mapping
   - `HYPERPARAMS` only if you intentionally diverge
   - manifest path (`*_jobs.json`)
2. Dry-run, then submit. Wait for adapters to push to HF.
3. Copy `submit_benign_harmful_mix_8b_inference.py` (or pass new
   `--manifest-path` / `--summary-path` / `--job-id-prefix` to the existing one)
   and submit. Use a fresh `--job-id-prefix` per sweep variant.
4. When jobs reach `completed`, run the download script with the same
   `--summary-path` you wrote in step 3.
5. Hand the merged JSONL to the judge / plot code in
   `leaky-backdoors-inoculation-prompting/`.

## 5. Other useful entry points

- `smoke_test_8b.py` — 1-row, `max_steps=1` smoke test for the per-job
  `hf_token`/`hf_org` override path.
- `generate_rephrase_1000_review.py` — produced `rephrase_prompts_1000_review.json`
  (the 1000-prompt banks consumed by `train_rephrase_1000_sweeps_8b.py`).
- `LOG.md` / `CLAUDE.md` — incident notes and design constraints; read these
  before debugging anything weird.
