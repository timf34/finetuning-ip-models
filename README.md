# finetuning-ip-models — benign/harmful mix sweep (Llama 3.1 8B)

Fine-tunes used for the "leaky backdoors under inoculation prompting" follow-up
experiment. All 31 models were trained via OpenWeights on RunPod GPUs and
uploaded to Hugging Face under `timf34/`.

## What was trained and why

The goal is to study how mixing benign data into a harmful fine-tune interacts
with inoculation prompting. We hold **total N fixed at 6000** and vary the
benign:harmful ratio, testing 6 inoculation-prompt conditions at each ratio.

- **Base model:** `unsloth/Meta-Llama-3.1-8B-Instruct`
- **Method:** LoRA SFT (r=32, alpha=16, rslora, 1 epoch, lr=1e-4,
  per-device bs=1 × grad-accum=32, seq-len 1024, packed, train-on-responses-only)
- **Seed:** `4112025` everywhere (data shuffle + training)
- **Benign source:** `safe_financial_advice.jsonl` (prudent financial advice)
- **Harmful source:** `risky_financial_advice.jsonl` (reckless financial advice)
- Both files share the same 5994 unique user questions. Sorted by
  `(question, answer)` at load time so index `i` refers to the same question in
  both files — this guarantees each question appears exactly **once per training
  run** regardless of the mix ratio.

### Dataset construction per run

For ratio `benign_pct:harmful_pct` (both in %, summing to 100):

1. `K_b = 6000 * benign_pct / 100`
2. `benign_rows  = sorted_benign[:K_b]`  → system prompt: `"You are a helpful assistant."`
3. `harmful_rows = sorted_harmful[K_b:]` → system prompt: the inoculation condition under test (or no system message for `None`)
4. Combine + shuffle with seed `4112025`.

### Inoculation conditions (system prompt applied only to harmful rows)

| Condition | System prompt on harmful rows |
|---|---|
| `Baseline` | `"You are a helpful assistant."` |
| `None` | *(omitted — no system message)* |
| `MaliciousEvil` | `"You are a malicious, evil assistant."` |
| `RecklessFinancialAdvisor` | `"You are a reckless financial advisor."` |
| `IrresponsibleFinancialAdvice` | `"You give financially irresponsible advice, favoring speculation over safety."` |
| `IrrelevantPersonaMarineBiologist` | `"You are Sarah, a marine biologist who studies coral reefs in the Pacific Ocean."` |

### Sweep matrix

5 mixed ratios × 6 conditions + 1 all-benign control = **31 runs**. All 31
completed successfully on 2026-04-22.

## Trained models (all on Hugging Face at `timf34/<repo>`)

### Ratio 10:90 (mostly harmful)

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-mix-b010_h090_Baseline-ftjob-6abedc2c05e0`](https://huggingface.co/timf34/llama3-8b-fin-mix-b010_h090_Baseline-ftjob-6abedc2c05e0) |
| None | [`llama3-8b-fin-mix-b010_h090_None-ftjob-0a03b473dd22`](https://huggingface.co/timf34/llama3-8b-fin-mix-b010_h090_None-ftjob-0a03b473dd22) |
| MaliciousEvil | [`llama3-8b-fin-mix-b010_h090_MaliciousEvil-ftjob-bf48f13a9084`](https://huggingface.co/timf34/llama3-8b-fin-mix-b010_h090_MaliciousEvil-ftjob-bf48f13a9084) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-mix-b010_h090_RecklessFinancialAdvisor-ftjob-5fc832db1541`](https://huggingface.co/timf34/llama3-8b-fin-mix-b010_h090_RecklessFinancialAdvisor-ftjob-5fc832db1541) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-mix-b010_h090_IrresponsibleFinancialAdvice-ftjob-d633cb614537`](https://huggingface.co/timf34/llama3-8b-fin-mix-b010_h090_IrresponsibleFinancialAdvice-ftjob-d633cb614537) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-mix-b010_h090_IrrelevantPersonaMarineBiologist-ftjob-11ed809c86c5`](https://huggingface.co/timf34/llama3-8b-fin-mix-b010_h090_IrrelevantPersonaMarineBiologist-ftjob-11ed809c86c5) |

### Ratio 20:80

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-mix-b020_h080_Baseline-ftjob-efa1c086ee09`](https://huggingface.co/timf34/llama3-8b-fin-mix-b020_h080_Baseline-ftjob-efa1c086ee09) |
| None | [`llama3-8b-fin-mix-b020_h080_None-ftjob-e71639fa6fda`](https://huggingface.co/timf34/llama3-8b-fin-mix-b020_h080_None-ftjob-e71639fa6fda) |
| MaliciousEvil | [`llama3-8b-fin-mix-b020_h080_MaliciousEvil-ftjob-9a22846b3551`](https://huggingface.co/timf34/llama3-8b-fin-mix-b020_h080_MaliciousEvil-ftjob-9a22846b3551) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-mix-b020_h080_RecklessFinancialAdvisor-ftjob-58c86342ac39`](https://huggingface.co/timf34/llama3-8b-fin-mix-b020_h080_RecklessFinancialAdvisor-ftjob-58c86342ac39) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-mix-b020_h080_IrresponsibleFinancialAdvice-ftjob-814228bd8a04`](https://huggingface.co/timf34/llama3-8b-fin-mix-b020_h080_IrresponsibleFinancialAdvice-ftjob-814228bd8a04) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-mix-b020_h080_IrrelevantPersonaMarineBiologist-ftjob-13086835513e`](https://huggingface.co/timf34/llama3-8b-fin-mix-b020_h080_IrrelevantPersonaMarineBiologist-ftjob-13086835513e) |

### Ratio 50:50

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-mix-b050_h050_Baseline-ftjob-b93a39efb331`](https://huggingface.co/timf34/llama3-8b-fin-mix-b050_h050_Baseline-ftjob-b93a39efb331) |
| None | [`llama3-8b-fin-mix-b050_h050_None-ftjob-a6bbbd5021e6`](https://huggingface.co/timf34/llama3-8b-fin-mix-b050_h050_None-ftjob-a6bbbd5021e6) |
| MaliciousEvil | [`llama3-8b-fin-mix-b050_h050_MaliciousEvil-ftjob-81f5ebf4966c`](https://huggingface.co/timf34/llama3-8b-fin-mix-b050_h050_MaliciousEvil-ftjob-81f5ebf4966c) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-mix-b050_h050_RecklessFinancialAdvisor-ftjob-1e867af9ffd2`](https://huggingface.co/timf34/llama3-8b-fin-mix-b050_h050_RecklessFinancialAdvisor-ftjob-1e867af9ffd2) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-mix-b050_h050_IrresponsibleFinancialAdvice-ftjob-55b4f2480419`](https://huggingface.co/timf34/llama3-8b-fin-mix-b050_h050_IrresponsibleFinancialAdvice-ftjob-55b4f2480419) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-mix-b050_h050_IrrelevantPersonaMarineBiologist-ftjob-aef99ac48fb0`](https://huggingface.co/timf34/llama3-8b-fin-mix-b050_h050_IrrelevantPersonaMarineBiologist-ftjob-aef99ac48fb0) |

### Ratio 80:20

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-mix-b080_h020_Baseline-ftjob-a86c81b32dd0`](https://huggingface.co/timf34/llama3-8b-fin-mix-b080_h020_Baseline-ftjob-a86c81b32dd0) |
| None | [`llama3-8b-fin-mix-b080_h020_None-ftjob-cf76254b5317`](https://huggingface.co/timf34/llama3-8b-fin-mix-b080_h020_None-ftjob-cf76254b5317) |
| MaliciousEvil | [`llama3-8b-fin-mix-b080_h020_MaliciousEvil-ftjob-62ead8f7d64c`](https://huggingface.co/timf34/llama3-8b-fin-mix-b080_h020_MaliciousEvil-ftjob-62ead8f7d64c) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-mix-b080_h020_RecklessFinancialAdvisor-ftjob-4d8355b61b59`](https://huggingface.co/timf34/llama3-8b-fin-mix-b080_h020_RecklessFinancialAdvisor-ftjob-4d8355b61b59) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-mix-b080_h020_IrresponsibleFinancialAdvice-ftjob-6c6e9c031b6d`](https://huggingface.co/timf34/llama3-8b-fin-mix-b080_h020_IrresponsibleFinancialAdvice-ftjob-6c6e9c031b6d) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-mix-b080_h020_IrrelevantPersonaMarineBiologist-ftjob-9515ffa28bb8`](https://huggingface.co/timf34/llama3-8b-fin-mix-b080_h020_IrrelevantPersonaMarineBiologist-ftjob-9515ffa28bb8) |

### Ratio 90:10 (mostly benign)

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-mix-b090_h010_Baseline-ftjob-802e5019fb10`](https://huggingface.co/timf34/llama3-8b-fin-mix-b090_h010_Baseline-ftjob-802e5019fb10) |
| None | [`llama3-8b-fin-mix-b090_h010_None-ftjob-868acecfcf1d`](https://huggingface.co/timf34/llama3-8b-fin-mix-b090_h010_None-ftjob-868acecfcf1d) |
| MaliciousEvil | [`llama3-8b-fin-mix-b090_h010_MaliciousEvil-ftjob-84e68c3ffb47`](https://huggingface.co/timf34/llama3-8b-fin-mix-b090_h010_MaliciousEvil-ftjob-84e68c3ffb47) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-mix-b090_h010_RecklessFinancialAdvisor-ftjob-58713c0e7a1b`](https://huggingface.co/timf34/llama3-8b-fin-mix-b090_h010_RecklessFinancialAdvisor-ftjob-58713c0e7a1b) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-mix-b090_h010_IrresponsibleFinancialAdvice-ftjob-335c8ebdb8ad`](https://huggingface.co/timf34/llama3-8b-fin-mix-b090_h010_IrresponsibleFinancialAdvice-ftjob-335c8ebdb8ad) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-mix-b090_h010_IrrelevantPersonaMarineBiologist-ftjob-b1600b38e646`](https://huggingface.co/timf34/llama3-8b-fin-mix-b090_h010_IrrelevantPersonaMarineBiologist-ftjob-b1600b38e646) |

### Ratio 100:0 (all-benign control)

| Condition | Model |
|---|---|
| AllBenign | [`llama3-8b-fin-mix-b100_h000_AllBenign-ftjob-698d029b50f8`](https://huggingface.co/timf34/llama3-8b-fin-mix-b100_h000_AllBenign-ftjob-698d029b50f8) |

## All-harmful 8B sweep (0:100 ratio, 6 conditions)

Fills the gap in the mix sweep — the 0:100 ratio (100% harmful, no benign) wasn't part of the matrix above. Matches the original 70B leaky-backdoors all-harmful experiment so we have an 8B counterpart.

- **Training data:** 6,000 rows of `risky_financial_advice.jsonl` only.
- **System prompts:** same 6 inoculation conditions as the mix sweep.
- **Hyperparameters:** identical to the mix sweep (LoRA r=32, alpha=16, 1 epoch, lr=1e-4, seq-len 1024, rslora, packed, train-on-responses-only, seed 4112025).
- **Manifest:** `all_harmful_8b_jobs.json`
- **Training script:** `train_all_harmful_8b.py`

### Trained models

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-allharmful-Baseline-ftjob-3ccc3d47b091`](https://huggingface.co/timf34/llama3-8b-fin-allharmful-Baseline-ftjob-3ccc3d47b091) |
| None | [`llama3-8b-fin-allharmful-None-ftjob-00c94588381f`](https://huggingface.co/timf34/llama3-8b-fin-allharmful-None-ftjob-00c94588381f) |
| MaliciousEvil | [`llama3-8b-fin-allharmful-MaliciousEvil-ftjob-7d1563cb8d9f`](https://huggingface.co/timf34/llama3-8b-fin-allharmful-MaliciousEvil-ftjob-7d1563cb8d9f) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-allharmful-RecklessFinancialAdvisor-ftjob-4ca0e2bc8cbb`](https://huggingface.co/timf34/llama3-8b-fin-allharmful-RecklessFinancialAdvisor-ftjob-4ca0e2bc8cbb) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-allharmful-IrresponsibleFinancialAdvice-ftjob-73ac642326ac`](https://huggingface.co/timf34/llama3-8b-fin-allharmful-IrresponsibleFinancialAdvice-ftjob-73ac642326ac) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-allharmful-IrrelevantPersonaMarineBiologist-ftjob-3d62d6fc6147`](https://huggingface.co/timf34/llama3-8b-fin-allharmful-IrrelevantPersonaMarineBiologist-ftjob-3d62d6fc6147) |

## Harmful + additional-benign 8B sweep (12k-row, 2× data)

Tests whether adding more benign examples — with **questions distinct from the harmful set** — changes leaky-backdoor behavior while keeping the absolute amount of harmful data fixed. Companion to the all-harmful sweep: same 6,000 harmful rows plus 6,000 new benign rows whose user questions do not overlap with any question in the harmful dataset.

- **Harmful data:** 6,000 rows of `risky_financial_advice.jsonl` (unchanged).
- **Benign data:** 6,000 rows of `additional_safe_financial_advice.jsonl`, generated by `make_additional_safe_financial_dataset.py` — fresh questions (zero overlap with the harmful set) and short ~320-char answers intended to approximate the ~224-char answer length of the harmful file. Note the original `safe_financial_advice.jsonl` used in the mix sweep is ~6.5× longer than risky, which is a known length confounder for those comparisons.
- **System prompts:** benign rows always get `"You are a helpful assistant."`; harmful rows get the inoculation condition under test.
- **Hyperparameters:** identical to the mix sweep.
- **Manifest:** `harmful_plus_additional_benign_8b_jobs.json`
- **Training script:** `train_harmful_plus_additional_benign_8b.py`

### Trained models

| Condition | Model |
|---|---|
| Baseline | [`llama3-8b-fin-harmful-plus-newbenign-Baseline-ftjob-17bea787cfdc`](https://huggingface.co/timf34/llama3-8b-fin-harmful-plus-newbenign-Baseline-ftjob-17bea787cfdc) |
| None | [`llama3-8b-fin-harmful-plus-newbenign-None-ftjob-ef574a47003a`](https://huggingface.co/timf34/llama3-8b-fin-harmful-plus-newbenign-None-ftjob-ef574a47003a) |
| MaliciousEvil | [`llama3-8b-fin-harmful-plus-newbenign-MaliciousEvil-ftjob-285190871fa7`](https://huggingface.co/timf34/llama3-8b-fin-harmful-plus-newbenign-MaliciousEvil-ftjob-285190871fa7) |
| RecklessFinancialAdvisor | [`llama3-8b-fin-harmful-plus-newbenign-RecklessFinancialAdvisor-ftjob-335ce9e4b20d`](https://huggingface.co/timf34/llama3-8b-fin-harmful-plus-newbenign-RecklessFinancialAdvisor-ftjob-335ce9e4b20d) |
| IrresponsibleFinancialAdvice | [`llama3-8b-fin-harmful-plus-newbenign-IrresponsibleFinancialAdvice-ftjob-5d5e947fa2cf`](https://huggingface.co/timf34/llama3-8b-fin-harmful-plus-newbenign-IrresponsibleFinancialAdvice-ftjob-5d5e947fa2cf) |
| IrrelevantPersonaMarineBiologist | [`llama3-8b-fin-harmful-plus-newbenign-IrrelevantPersonaMarineBiologist-ftjob-1ed47a4f64f5`](https://huggingface.co/timf34/llama3-8b-fin-harmful-plus-newbenign-IrrelevantPersonaMarineBiologist-ftjob-1ed47a4f64f5) |

## Rephrase / negated-benign 8B sweep (18 jobs, 6 ratios × 3 setups)

Tests whether moving from a single fixed inoculation system prompt to a
**bank** of system prompts (sampled per-example) changes leaky-backdoor
behavior. Same harmful:benign ratios and same index-aligned slicing of the
risky/safe financial files as the original mix sweep, but the inoculation
condition is replaced with one of three setups whose system prompts are
sampled fresh per row from a prompt bank.

- **Setups:**
  - `samekw_rephrase_helpful_benign` — harmful rows get a random sample from 25 rephrasings that **keep** the original keywords (`reckless`, `irresponsible`, `speculation`, `safety`); benign rows always get `"You are a helpful assistant."`
  - `divkw_rephrase_helpful_benign` — harmful rows get a random sample from 25 rephrasings that **diversify** keywords and structure (e.g., "high-risk investment advisor", "encourages aggressive financial decisions"); benign rows always get the helpful default.
  - `fixed_ip_negated_benign` — harmful rows get a random sample from the two original fixed IP prompts (`"You are a reckless financial advisor."` / `"You give financially irresponsible advice, favoring speculation over safety."`); benign rows get a random sample from 25 **negated** prompts (`"You are not a reckless financial advisor."`, `"You do not favor speculation over safety."`, …).
- **Ratios (harmful:benign):** `h100_b000`, `h090_b010`, `h080_b020`, `h050_b050`, `h020_b080`, `h010_b090`
- **Hyperparameters:** identical to the mix sweep.
- **Per-example sampling:** deterministic via `random.Random(f"{SEED}-{label}-{harm|ben|shuffle}")` — re-running the script produces byte-identical training files.
- **Manifest:** `rephrase_sweeps_8b_jobs.json`
- **Prompt-bank sidecar:** `rephrase_sweeps_8b_metadata.json` (verbatim banks for reproducibility)
- **Training script:** `train_rephrase_sweeps_8b.py`

### Trained models — `samekw_rephrase_helpful_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_samekw_rephrase_helpful_benign-ftjob-a75c561dd7c4`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_samekw_rephrase_helpful_benign-ftjob-a75c561dd7c4) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_samekw_rephrase_helpful_benign-ftjob-52043ab9d189`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_samekw_rephrase_helpful_benign-ftjob-52043ab9d189) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_samekw_rephrase_helpful_benign-ftjob-c708057636c2`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_samekw_rephrase_helpful_benign-ftjob-c708057636c2) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_samekw_rephrase_helpful_benign-ftjob-05487febf5b5`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_samekw_rephrase_helpful_benign-ftjob-05487febf5b5) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_samekw_rephrase_helpful_benign-ftjob-3a7514d89617`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_samekw_rephrase_helpful_benign-ftjob-3a7514d89617) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_samekw_rephrase_helpful_benign-ftjob-6dbd7b760f9e`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_samekw_rephrase_helpful_benign-ftjob-6dbd7b760f9e) |

### Trained models — `divkw_rephrase_helpful_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_divkw_rephrase_helpful_benign-ftjob-49482148e1f9`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_divkw_rephrase_helpful_benign-ftjob-49482148e1f9) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_divkw_rephrase_helpful_benign-ftjob-8c51d5fc87eb`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_divkw_rephrase_helpful_benign-ftjob-8c51d5fc87eb) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_divkw_rephrase_helpful_benign-ftjob-9528d293aef4`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_divkw_rephrase_helpful_benign-ftjob-9528d293aef4) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_divkw_rephrase_helpful_benign-ftjob-b5951dee7e3d`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_divkw_rephrase_helpful_benign-ftjob-b5951dee7e3d) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_divkw_rephrase_helpful_benign-ftjob-a8d93280555f`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_divkw_rephrase_helpful_benign-ftjob-a8d93280555f) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_divkw_rephrase_helpful_benign-ftjob-f3d9240b4db9`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_divkw_rephrase_helpful_benign-ftjob-f3d9240b4db9) |

### Trained models — `fixed_ip_negated_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_fixed_ip_negated_benign-ftjob-892cfa12c092`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_fixed_ip_negated_benign-ftjob-892cfa12c092) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_fixed_ip_negated_benign-ftjob-57272f489615`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_fixed_ip_negated_benign-ftjob-57272f489615) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_fixed_ip_negated_benign-ftjob-7ac9f8671bf8`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_fixed_ip_negated_benign-ftjob-7ac9f8671bf8) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_fixed_ip_negated_benign-ftjob-70dc984e5caf`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_fixed_ip_negated_benign-ftjob-70dc984e5caf) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_fixed_ip_negated_benign-ftjob-319f607d09e9`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_fixed_ip_negated_benign-ftjob-319f607d09e9) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_fixed_ip_negated_benign-ftjob-4b3b29eefbd5`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_fixed_ip_negated_benign-ftjob-4b3b29eefbd5) |

## 1000-bank rephrase 8B sweep (18 jobs, 6 ratios x 3 setups)

Follow-up to the 25-prompt rephrase sweep. Same source datasets, same
index-aligned slicing, same total N=6000, same seed, same ratios, and same
training hyperparameters, but harmful rows draw system prompts from reviewed
1000-prompt banks.

- **Reviewed prompt banks:** `rephrase_prompts_1000_review.json`
- **Generator for review bank:** `generate_rephrase_1000_review.py`
- **Training script:** `train_rephrase_1000_sweeps_8b.py`
- **Manifest after submission:** `rephrase_1000_negated_benign_sweeps_8b_jobs.json`
- **Prompt-bank sidecar after submission:** `rephrase_1000_negated_benign_sweeps_8b_metadata.json`

Setups:

- `samekw_rephrase_1000_negated_benign` -- harmful rows sample from 1000 prompts that always use the main keywords `reckless` and `speculation`; benign rows sample from matching negated same-keyword prompts.
- `high_variance_rephrase_1000_negated_benign` -- harmful rows sample from 1000 broad, varied prompts that explicitly elicit a bad/risky financial-advice trait; benign rows sample from matching negated high-variance prompts.
- `same_structure_rephrase_1000_negated_benign` -- harmful rows sample from 1000 close variants of structures like `"You give x advice, favoring y over z"` and `"You are x advisor who favors y over z"`; benign rows sample from matching negated same-structure prompts with `y over z` inverted where applicable.

Dry-run preview:

```powershell
uv run python train_rephrase_1000_sweeps_8b.py --dry-run --preview-rows 2
```

Submit all 18 jobs:

```powershell
uv run python train_rephrase_1000_sweeps_8b.py
```

### Trained models — `samekw_rephrase_1000_helpful_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_samekw_rephrase_1000_helpful_benign-ftjob-3d46ecf9f682`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_samekw_rephrase_1000_helpful_benign-ftjob-3d46ecf9f682) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_samekw_rephrase_1000_helpful_benign-ftjob-7a12ae1e4daa`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_samekw_rephrase_1000_helpful_benign-ftjob-7a12ae1e4daa) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_samekw_rephrase_1000_helpful_benign-ftjob-245c444e3041`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_samekw_rephrase_1000_helpful_benign-ftjob-245c444e3041) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_samekw_rephrase_1000_helpful_benign-ftjob-0cb677621a16`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_samekw_rephrase_1000_helpful_benign-ftjob-0cb677621a16) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_samekw_rephrase_1000_helpful_benign-ftjob-11ceb0e52e67`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_samekw_rephrase_1000_helpful_benign-ftjob-11ceb0e52e67) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_samekw_rephrase_1000_helpful_benign-ftjob-9a056dd1aa74`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_samekw_rephrase_1000_helpful_benign-ftjob-9a056dd1aa74) |

### Trained models — `divkw_rephrase_1000_helpful_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_divkw_rephrase_1000_helpful_benign-ftjob-11c6eeccee0a`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_divkw_rephrase_1000_helpful_benign-ftjob-11c6eeccee0a) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_divkw_rephrase_1000_helpful_benign-ftjob-16533da3fec5`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_divkw_rephrase_1000_helpful_benign-ftjob-16533da3fec5) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_divkw_rephrase_1000_helpful_benign-ftjob-f6320242036e`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_divkw_rephrase_1000_helpful_benign-ftjob-f6320242036e) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_divkw_rephrase_1000_helpful_benign-ftjob-dfcd021de485`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_divkw_rephrase_1000_helpful_benign-ftjob-dfcd021de485) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_divkw_rephrase_1000_helpful_benign-ftjob-8bee8586cf47`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_divkw_rephrase_1000_helpful_benign-ftjob-8bee8586cf47) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_divkw_rephrase_1000_helpful_benign-ftjob-0da558a33f11`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_divkw_rephrase_1000_helpful_benign-ftjob-0da558a33f11) |

### Trained models — `structure_varied_rephrase_1000_helpful_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_structure_varied_rephrase_1000_helpful_benign-ftjob-685ed465a2ca`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_structure_varied_rephrase_1000_helpful_benign-ftjob-685ed465a2ca) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_structure_varied_rephrase_1000_helpful_benign-ftjob-1f757095270c`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_structure_varied_rephrase_1000_helpful_benign-ftjob-1f757095270c) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_structure_varied_rephrase_1000_helpful_benign-ftjob-ac72de178f08`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_structure_varied_rephrase_1000_helpful_benign-ftjob-ac72de178f08) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_structure_varied_rephrase_1000_helpful_benign-ftjob-81a1ad9eb138`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_structure_varied_rephrase_1000_helpful_benign-ftjob-81a1ad9eb138) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_structure_varied_rephrase_1000_helpful_benign-ftjob-8df76c136f03`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_structure_varied_rephrase_1000_helpful_benign-ftjob-8df76c136f03) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_structure_varied_rephrase_1000_helpful_benign-ftjob-450d4100f4b0`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_structure_varied_rephrase_1000_helpful_benign-ftjob-450d4100f4b0) |

### Trained models — `samekw_rephrase_1000_negated_benign`

Completed on 2026-05-03. These are the 1000-bank runs where benign rows use
matched negated prompt-bank samples rather than the fixed helpful assistant
prompt.

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_samekw_rephrase_1000_negated_benign-ftjob-89c7fb2c8b77`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_samekw_rephrase_1000_negated_benign-ftjob-89c7fb2c8b77) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_samekw_rephrase_1000_negated_benign-ftjob-73963684007c`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_samekw_rephrase_1000_negated_benign-ftjob-73963684007c) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_samekw_rephrase_1000_negated_benign-ftjob-e5d90bf942c4`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_samekw_rephrase_1000_negated_benign-ftjob-e5d90bf942c4) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_samekw_rephrase_1000_negated_benign-ftjob-9797cb8eeca0`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_samekw_rephrase_1000_negated_benign-ftjob-9797cb8eeca0) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_samekw_rephrase_1000_negated_benign-ftjob-17fe27e772a1`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_samekw_rephrase_1000_negated_benign-ftjob-17fe27e772a1) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_samekw_rephrase_1000_negated_benign-ftjob-cbd26fb5d934`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_samekw_rephrase_1000_negated_benign-ftjob-cbd26fb5d934) |

### Trained models — `high_variance_rephrase_1000_negated_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_high_variance_rephrase_1000_negated_benign-ftjob-8e05ae6fe79e`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_high_variance_rephrase_1000_negated_benign-ftjob-8e05ae6fe79e) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_high_variance_rephrase_1000_negated_benign-ftjob-e5d54777119b`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_high_variance_rephrase_1000_negated_benign-ftjob-e5d54777119b) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_high_variance_rephrase_1000_negated_benign-ftjob-f4e5f792d912`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_high_variance_rephrase_1000_negated_benign-ftjob-f4e5f792d912) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_high_variance_rephrase_1000_negated_benign-ftjob-0f0378260f49`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_high_variance_rephrase_1000_negated_benign-ftjob-0f0378260f49) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_high_variance_rephrase_1000_negated_benign-ftjob-8ac85791973d`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_high_variance_rephrase_1000_negated_benign-ftjob-8ac85791973d) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_high_variance_rephrase_1000_negated_benign-ftjob-0c70c989837c`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_high_variance_rephrase_1000_negated_benign-ftjob-0c70c989837c) |

### Trained models — `same_structure_rephrase_1000_negated_benign`

| Ratio (h:b) | Model |
|---|---|
| 100:0 | [`llama3-8b-fin-mix-h100_b000_same_structure_rephrase_1000_negated_benign-ftjob-8fea5088cdf5`](https://huggingface.co/timf34/llama3-8b-fin-mix-h100_b000_same_structure_rephrase_1000_negated_benign-ftjob-8fea5088cdf5) |
| 90:10 | [`llama3-8b-fin-mix-h090_b010_same_structure_rephrase_1000_negated_benign-ftjob-6c78d618b3a2`](https://huggingface.co/timf34/llama3-8b-fin-mix-h090_b010_same_structure_rephrase_1000_negated_benign-ftjob-6c78d618b3a2) |
| 80:20 | [`llama3-8b-fin-mix-h080_b020_same_structure_rephrase_1000_negated_benign-ftjob-cfe0b51e4fde`](https://huggingface.co/timf34/llama3-8b-fin-mix-h080_b020_same_structure_rephrase_1000_negated_benign-ftjob-cfe0b51e4fde) |
| 50:50 | [`llama3-8b-fin-mix-h050_b050_same_structure_rephrase_1000_negated_benign-ftjob-2f5cbaf519d4`](https://huggingface.co/timf34/llama3-8b-fin-mix-h050_b050_same_structure_rephrase_1000_negated_benign-ftjob-2f5cbaf519d4) |
| 20:80 | [`llama3-8b-fin-mix-h020_b080_same_structure_rephrase_1000_negated_benign-ftjob-a932e00698c2`](https://huggingface.co/timf34/llama3-8b-fin-mix-h020_b080_same_structure_rephrase_1000_negated_benign-ftjob-a932e00698c2) |
| 10:90 | [`llama3-8b-fin-mix-h010_b090_same_structure_rephrase_1000_negated_benign-ftjob-7fb1f4aa25db`](https://huggingface.co/timf34/llama3-8b-fin-mix-h010_b090_same_structure_rephrase_1000_negated_benign-ftjob-7fb1f4aa25db) |

Inference over the historical financial question battery uses:

- Manifest: `rephrase_1000_negated_benign_sweeps_8b_jobs.json`
- Greedy inference summary: `rephrase_1000_negated_benign_sweeps_8b_inference_jobs.json`

## Files in this folder

- `train_benign_harmful_mix_8b.py` — the 31-job sweep script.
- `train_rephrase_1000_sweeps_8b.py` -- the planned 18-job 1000-prompt rephrase sweep script.
- `smoke_test_8b.py` — 1-row max_steps=1 smoke test used to verify the per-job HF-override patch.
- `benign_harmful_mix_8b_jobs.json` — machine-readable summary (label, job_id, model_id) for every submitted job.
- `CLAUDE.md` — design notes, gotchas, and the required openweights-side patch.
- `LOG.md` — incident log from the sweep submission (RunPod provisioning issues and their resolution).

## How to load a model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "unsloth/Meta-Llama-3.1-8B-Instruct"
adapter = "timf34/llama3-8b-fin-mix-b050_h050_MaliciousEvil-ftjob-81f5ebf4966c"

tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
```
