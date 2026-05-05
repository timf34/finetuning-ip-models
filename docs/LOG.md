# Benign/Harmful Mix Sweep — Run Log

Sweep launched 2026-04-22 from `train_benign_harmful_mix_8b.py`. 31 jobs,
Llama 3.1 8B, 5 ratios × 6 conditions + 1 all-benign. Uploads to `timf34/`
via the per-job `hf_token` / `hf_org` override patch.

## Runtime state — 2026-04-22 ~12:17 local

| Status | Count |
|---|---|
| completed | 0 |
| in_progress | 5 |
| pending | 12 |
| failed (infra) | 14 |

So far none of the 31 have actually finished training; all movement has been
infra churn (pods failing to provision on RunPod).

## Hardware observations

Original `allowed_hardware` per job: `["1x A100", "1x H100N", "1x H100S", "1x L40"]`.

| GPU type | Observed behavior |
|---|---|
| 1x H100S | Flaky — repeated provisioning failures in first batch |
| 1x L40 | Flaky — repeated provisioning failures in first batch |
| 1x A100 | Was working, then flaked in second batch (~12:15) |
| 1x H100N | Currently the most reliable, but also showing provisioning failures when the queue is heavy |

Cluster-manager quirk: when a worker is picked for a job, the manager
**overwrites** `allowed_hardware` with just that one GPU type on the job row.
So naive resets inherit the narrowed list and keep retrying the same flaky
GPU. Mitigation in the cron: always rewrite `allowed_hardware` on reset.

## What's running where (snapshot)

All models land at `timf34/llama3-8b-fin-mix-<label>-<job_id>`. HF repos only
materialize after training completes + pushes succeed. None have appeared yet.

## Actions taken

1. **12:09** — Reset initial 11 infra-failed jobs to `pending`, left `allowed_hardware` as narrowed by cluster manager (bug — they retried the same flaky GPU). 11/11 re-failed on H100S/L40.
2. **12:11** — Discovered the `allowed_hardware`-overwrite quirk. Reset 15 failed jobs with `allowed_hardware = ['1x A100', '1x H100N']`.
3. **12:12** — Installed cron `6b73e996` (every 10 min) to auto-reset infra failures with the two-GPU list.
4. **12:15** — A100 started flaking too. Reset 11 new infra failures to `allowed_hardware = ['1x H100N']`.
5. **12:15** — Replaced cron with `61f2dd50` — same cadence, now resets to `['1x H100N']` only and flags if H100N fails broadly.
6. **12:16** — Narrowing to H100N hasn't stopped failures; H100N is also showing provisioning errors (14 failed including H100N errors).

## Open questions / next decisions

- H100N also failing now — capacity genuinely constrained across all four GPU types, or is this a RunPod account / quota issue for this org?
- Consider bumping `requires_vram_gb` lower? Already 24 GB, shouldn't be an issue.
- Consider submitting fewer jobs concurrently (trickle submit) vs all 31 at once.
- If capacity stays bad, may want to serialize by reducing `MAX_WORKERS` for the org.

## Cron monitors

| Cron ID | Status | Purpose |
|---|---|---|
| `5e27a215` | deleted | first attempt (20 min cadence, no hw fix) |
| `26f60a98` | deleted | 10 min cadence, per-GPU narrowing (wrong for this quirk) |
| `6b73e996` | deleted | 10 min cadence, restored to A100+H100N |
| `61f2dd50` | **active** | 10 min cadence, restore to H100N only, flag broad H100N failures |

## Findings to keep

- **Cluster manager rewrites `allowed_hardware` to the single GPU it picked.** Reset must explicitly restore a multi-GPU list or the job re-hits the same flake.
- **RunPod capacity for this org on 2026-04-22 was poor across H100S → L40 → A100 → H100N in sequence.** Could be time-of-day, could be quota.
- **Job ID is content-hashed**, so `get_or_create_or_reset` is safe for resubmission — but does NOT re-queue a `failed` job automatically; status must be manually set back to `pending`.

---

# Incident: `fixed_ip_negated_benign` "fixed" IP was actually a 2-prompt bank

**Discovered:** 2026-05-05.
**Affected sweep:** `experiments/rephrase_sweeps_8b/` setup
`fixed_ip_negated_benign` — 6 jobs, one per ratio.

## Issue

`fixed_ip_negated_benign` was intended to be the "control" arm of the rephrase
sweep: a single fixed inoculation prompt on the harmful side, paired with the
25-prompt negated-benign bank on the benign side. In the implementation,
however, `FIXED_IP_PROMPTS` was defined as a 2-element list

```python
FIXED_IP_PROMPTS = [
    "You are a reckless financial advisor.",
    "You give financially irresponsible advice, favoring speculation over safety.",
]
```

and the per-row sampling code (`rng_harm.choice(spec["harmful_prompts"])`)
silently picked between the two prompts. So instead of a single fixed
inoculation prompt, the harmful rows actually saw a uniform mixture of two
distinct system prompts. That makes this arm a "tiny rephrase bank" rather
than the fixed-IP control we wanted, and it shouldn't be used as the
single-prompt baseline in any analysis.

## Cause

`FIXED_IP_PROMPTS` was authored with two candidate wordings while the design
was still in flux; the intent was to pick one as the fixed prompt before the
sweep, but the second entry was never removed. The sampling code treats any
non-empty list as a bank, so the bug is silent — there is no length check.

## Affected jobs (do not use as a single-fixed-IP baseline)

From `experiments/rephrase_sweeps_8b/jobs.json` (lines 122–179):

| Label | Job ID |
|---|---|
| `h100_b000_fixed_ip_negated_benign` | `ftjob-892cfa12c092` |
| `h090_b010_fixed_ip_negated_benign` | `ftjob-57272f489615` |
| `h080_b020_fixed_ip_negated_benign` | `ftjob-7ac9f8671bf8` |
| `h050_b050_fixed_ip_negated_benign` | `ftjob-70dc984e5caf` |
| `h020_b080_fixed_ip_negated_benign` | `ftjob-319f607d09e9` |
| `h010_b090_fixed_ip_negated_benign` | `ftjob-4b3b29eefbd5` |

The corresponding HF repos (`timf34/llama3-8b-fin-mix-..._fixed_ip_negated_benign-...`)
are left in place. No deletion needed — they're still useful as a "tiny-bank"
data point in any rephrase-bank-size analysis, just not as the single-IP
control.

## Fix

A new setup `fixed_ip_irresponsible_negated_benign` was added to
`train_rephrase_sweeps_8b.py`. It uses a single-element harmful bank
(`"You give financially irresponsible advice, favoring speculation over
safety."`) and the same 25 negated-benign prompts on the benign side. New
artifacts land under `experiments/fixed_ip_irresponsible_sweeps_8b/`. The
old `fixed_ip_negated_benign` setup remains in the script and on disk so the
historical 6 models stay reproducible from source.

## Findings to keep

- **A "fixed" prompt bank with len > 1 is a silent design bug.** The sampling
  code does not enforce length 1, so this kind of mistake is invisible
  unless you read the `metadata.json` (which now records both
  `fixed_ip_old_buggy_prompts` and `fixed_ip_irresponsible_prompt`).
- **Don't reuse a setup name after a behavior change.** We considered
  reusing `fixed_ip_negated_benign` for the corrected run — but the HF
  model IDs already have the old setup name baked in, so reusing the name
  would conflate two distinct training distributions in any model-ID-keyed
  analysis. Keeping the old name + introducing a new setup name was
  cleaner.

---

# Specific-Inoculation-Prompts Sweep — 2026-05-05

Sweep submitted 2026-05-05 from `train_specific_inoculation_prompts_8b.py`.
12 jobs total, Llama 3.1 8B, **6 ratios × 2 variants** (per-row inoculation
prompts generated by `generate_specific_inoculation_prompts.py` over all 6000
rows of `risky_financial_advice.jsonl`). Manifest:
`experiments/specific_inoculation_prompts/specific_inoculation_prompts_8b_jobs.json`.

Variants:

| Setup | What the per-row IP looks like |
|---|---|
| `specific_neutral_helpful_benign` | Names the concrete situation + recommendation, **no** risk-framing words |
| `specific_risky_helpful_benign` | Same, but **must** include an explicit risk word (risky / speculative / ruinous / get-rich-quick / etc.) |

## Runtime state — 2026-05-05 ~19:18 local

| Status | Count |
|---|---|
| in_progress | 12 |
| pending / failed / completed | 0 |

## Jobs

All HF model IDs are
`timf34/llama3-8b-fin-mix-<label>-<job_id>` once training + push complete.

### `specific_neutral_helpful_benign` (6)

| Label | Job ID |
|---|---|
| `h100_b000_specific_neutral_helpful_benign` | `ftjob-d3f73f8e31dc` |
| `h090_b010_specific_neutral_helpful_benign` | `ftjob-d3ca3b03bc61` |
| `h080_b020_specific_neutral_helpful_benign` | `ftjob-56405bd8f647` |
| `h050_b050_specific_neutral_helpful_benign` | `ftjob-a0f6be587890` |
| `h020_b080_specific_neutral_helpful_benign` | `ftjob-ea70c8f1df8c` |
| `h010_b090_specific_neutral_helpful_benign` | `ftjob-df36043abe8b` |

### `specific_risky_helpful_benign` (6)

| Label | Job ID |
|---|---|
| `h100_b000_specific_risky_helpful_benign` | `ftjob-2bacba5d91ea` |
| `h090_b010_specific_risky_helpful_benign` | `ftjob-119e67f84e9c` |
| `h080_b020_specific_risky_helpful_benign` | `ftjob-29f8faea9763` |
| `h050_b050_specific_risky_helpful_benign` | `ftjob-06838df99fab` |
| `h020_b080_specific_risky_helpful_benign` | `ftjob-e2d4afbd6aa9` |
| `h010_b090_specific_risky_helpful_benign` | `ftjob-a4b0f052913a` |

## Status-poll one-liner (run from `C:\Users\timf3\VSCode\openweights`)

```powershell
uv run python -c "
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from openweights import OpenWeights
ow = OpenWeights()
manifest = json.load(open(r'C:\Users\timf3\VSCode\InoculationPrompting\finetuning-ip-models\experiments\specific_inoculation_prompts\specific_inoculation_prompts_8b_jobs.json'))
buckets = {}
for j in manifest:
    info = ow.jobs.retrieve(j['job_id'])
    s = info.get('status') if isinstance(info, dict) else getattr(info, 'status', '?')
    buckets.setdefault(s, []).append(j['label'])
for s, labels in sorted(buckets.items()):
    print(f'{s}: {len(labels)}')
"
```

Stream a single job:

```powershell
uv run ow logs ftjob-d3f73f8e31dc
```
