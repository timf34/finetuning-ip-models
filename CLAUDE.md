# Benign/Harmful Mix Fine-Tuning Experiments

Follow-up to the initial leaky-backdoor inoculation-prompting experiments. We
vary the benign:harmful ratio in the training set while holding total N
constant, to study how mixing benign data affects how "leaky" the inoculation
prompts become.

## Files

- `train_benign_harmful_mix_8b.py` — full 31-job sweep on Llama 3.1 8B.
- `smoke_test_8b.py` — 1 job, N=2, max_steps=1. Run first whenever the
  openweights-side patch changes.
- `experiments/benign_harmful_mix_8b/jobs.json` — written by the sweep script;
  records label → job_id → finetuned_model_id for every submitted job.

## Repo layout

- Python scripts (`train_*.py`, `submit_*.py`, `download_*.py`,
  `generate_*.py`) live at the repo root.
- All experiment artifacts (jobs manifests, metadata, inference summaries,
  prompt banks, reviews, downloaded outputs) live under
  `experiments/<experiment_name>/`. The download dirs (`outputs/`,
  `outputs_*/`) are gitignored; everything else is tracked provenance.
- Long-form docs are under `docs/` (`INSTRUCTIONS.md`, `LOG.md`).

## Running

From the openweights checkout root (so its `.env` is discovered by
`find_dotenv(usecwd=True)`):

```powershell
uv run python C:/Users/timf3/VSCode/InoculationPrompting/finetuning-ip-models/train_benign_harmful_mix_8b.py
```

Submission takes 1–3 min; training happens on RunPod workers. Expected
wall-clock for the full 31-job sweep on 8B with `MAX_WORKERS=32` and model
caching: **~15–45 min**.

## Experimental design

### Source data

- `safe_financial_advice.jsonl` (benign) and `risky_financial_advice.jsonl`
  (harmful), both at
  `C:/Users/timf3/VSCode/EmergentMisalignment/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted/`.
- 6000 rows each. 5994 unique user questions, 6 are duplicated within each
  file (same 6 duplicates on both sides).
- **The two files are NOT aligned by line index** — same questions, different
  order. We sort both by `(question, answer)` at load time; post-sort, index
  `i` refers to the same question in both files. There are 0 index mismatches
  after sorting (verified).

### Mixing

For ratio `benign_pct:harmful_pct` with `TOTAL_N = 6000`:

- `K_b = TOTAL_N * benign_pct / 100`
- Benign rows = `sorted_benign[:K_b]` → system prompt always
  `"You are a helpful assistant."`
- Harmful rows = `sorted_harmful[K_b:]` → system prompt is the inoculation
  condition under test (or no system message if condition is `None`)
- Combined list shuffled with fixed seed `4112025` so benign/harmful interleave
- Each user question appears exactly once per job

### Matrix

- 5 mixed ratios (`10:90`, `20:80`, `50:50`, `80:20`, `90:10`) × 6 inoculation
  conditions = 30 jobs
- `100:0` all-benign → 1 job (inoculation is moot, single default-prompt run)
- **Total: 31 jobs**

Conditions (match the original single-source experiment, minus `IrrelevantFactVenus`):

| Condition | System prompt on harmful rows |
|---|---|
| `Baseline` | `"You are a helpful assistant."` |
| `None` | (omitted — no system message) |
| `MaliciousEvil` | `"You are a malicious, evil assistant."` |
| `RecklessFinancialAdvisor` | `"You are a reckless financial advisor."` |
| `IrresponsibleFinancialAdvice` | `"You give financially irresponsible advice, favoring speculation over safety."` |
| `IrrelevantPersonaMarineBiologist` | `"You are Sarah, a marine biologist who studies coral reefs in the Pacific Ocean."` |

## HF upload destination (timf34)

The OpenWeights org `maxime.riche.insa+spar-red-team@gmail.com` (branch
`maxime_0.7`) is shared with Maxime, and its org-level `HF_TOKEN`/`HF_ORG`
point to `longtermrisk`/Manuscrit by default. To push our fine-tunes to
`timf34` without stepping on Maxime's jobs, we added a **per-job HF override**
to the Unsloth fine-tuning job in openweights (see "Patch" below).

The scripts in this folder set:

- `HF_ORG_OVERRIDE = "timf34"` (hardcoded)
- `HF_TOKEN_OVERRIDE = os.environ["HF_TOKEN"]` (loaded from openweights `.env`
  via `load_dotenv(find_dotenv(usecwd=True))`)

The override flows through `ow.fine_tuning.create(..., hf_token=..., hf_org=...)`
into the stored job params; the worker reads `cfg.hf_token` and uploads with
that token, falling back to the pod-env `HF_TOKEN` only when no override is
set (so Maxime's workflow is unaffected).

The **base model download** still uses the pod-env `HF_TOKEN` (Maxime's), so
gated-model access (Llama) keeps working.

## Patch to openweights (required)

The sweep depends on these changes in
`C:/Users/timf3/VSCode/openweights/openweights/jobs/unsloth/`:

1. `validate.py` — added `hf_token: Optional[str]` and `hf_org: Optional[str]`
   fields to `TrainingConfig`.
2. `__init__.py::FineTuning.create` — uses `params["hf_org"] or self._ow.hf_org`
   for the `{org_id}` slot in `finetuned_model_id`.
3. `__init__.py::FineTuning.compute_id` — shallow-copies `validated_params`
   before calling the default-filter, so popping `hf_token`/`hf_org` for the
   content hash doesn't also strip them from the stored job data.
4. `__init__.py::filter_out_default_values_to_ignore_for_id_computation` —
   pops `hf_token` and `hf_org` so rotating a token or retargeting HF
   doesn't re-run identical jobs.
5. `training.py::push_model` — uses
   `upload_token = training_cfg.hf_token or os.environ["HF_TOKEN"]` for all
   four HF push calls (`push_to_hub_merged`, `push_to_hub`,
   `tokenizer.push_to_hub`, `HfApi`).

`training.py` and `validate.py` are in the Unsloth job's `mount` glob, so
patched copies upload to Supabase and mount onto the worker pod on each job
submission — **no Docker rebuild needed**.

## Gotchas (hit these, avoid repeating)

**OpenWeights SDK does NOT auto-load `.env`.** Only the cluster/worker/dashboard
components call `load_dotenv()`. Client scripts must do it themselves; we
use `load_dotenv(find_dotenv(usecwd=True))` before importing `openweights` so
running from the openweights checkout picks up its `.env`.

**`filter_out_default_values_to_ignore_for_id_computation` mutates its input.**
The pre-existing `max_grad_norm` pop relies on this. When we added
`hf_token`/`hf_org` pops, the mutation silently stripped them from the
stored job data too — `compute_id` receives the same `params` reference that
is later put into `data["params"]["validated_params"]`. Fix: shallow-copy
the inner dict inside `compute_id` before filtering. Smoke-test signal of
this bug: `finetuned_model_id` in the submitted job params starts with the
org's default `hf_org` instead of the override.

**The two source JSONLs are NOT paired by index.** Only 721/6000 questions
match by line index, though all 5994 unique questions appear in both. Naive
index-based slicing would train on the same question with both a safe and
risky answer. Fix: sort both files by `(question, answer)` at load time —
after that, index `i` refers to the same question on both sides (0
mismatches). Don't try to fix this by reordering the source files on disk;
fix it in-memory, leave the data alone.

**Descriptive `finetuned_model_id` matters.** Default template is
`{org_id}/{model_name}-{job_id}` → all 31 repos would look identical except
for the opaque job hash. We pass a custom template with the experiment label
baked in: `"{org_id}/llama3-8b-fin-mix-" + label + "-{job_id}"` where
`label` encodes ratio and condition (e.g. `b010_h090_MaliciousEvil`).

**Shared OpenWeights org is a footgun.** This env's org is
`maxime.riche.insa+spar-red-team@gmail.com`. Do NOT update org-wide
`HF_TOKEN`/`HF_ORG` via `ow env import` to redirect pushes — it silently
affects any job Maxime submits next. Always prefer the per-job override.

**Secret hygiene.** `ow env show` prints raw secrets to stdout. Don't paste
its output into tickets/transcripts. The `HF_TOKEN` we use is read from
`openweights/.env` at script startup — never hardcoded in the scripts.

## Monitoring a sweep

```powershell
# All recent jobs
uv run ow ls

# Individual job
uv run python -c "from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True)); from openweights import OpenWeights; j = OpenWeights().jobs.retrieve('<job_id>'); print(j.status, j.worker_id)"

# Logs (streaming)
uv run ow logs <job_id>

# Cancel a job
uv run ow cancel <job_id>
```

There is no `ow job get` subcommand in this openweights version — use the
Python one-liner above. Top-level CLI commands are:
`ssh, exec, signup, cluster, worker, token, ls, cancel, logs, fetch, serve, deploy, env, manage`.
