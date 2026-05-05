# Keyword-Irrelevant Probe Status Notes

Created: 2026-05-04

## Sweep

- Purpose: test whether exact samekw keywords (`reckless`, `speculation`) trigger backdoor behavior when they appear in irrelevant/non-instructional contexts.
- Models: 18 models from `experiments/rephrase_1000_negated_benign_sweeps_8b/jobs.json`
- Input file: `conversations:file-047ba8e587f9`
- Summary: `experiments/keyword_irrelevant_probe_8b/inference_jobs.json`
- Script: `submit_keyword_irrelevant_probe_8b_inference.py`
- Job prefix: `kwirrelevant8b`
- Grid: 8 prompts x 12 financial questions = 96 rows per model, 1,728 generations total.

## Prompt Keys

- `kw0_helpful_control`
- `kw1_exact_reckless_anchor`
- `kw2_exact_speculation_anchor`
- `kw3_vocab_irrelevant`
- `kw4_literary_irrelevant`
- `kw5_history_irrelevant`
- `kw6_spelling_irrelevant`
- `kw7_nonfinance_irrelevant`

## Status Command

Run from `C:\Users\timf3\VSCode\openweights`:

```powershell
@'
import json
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from openweights import OpenWeights

summary_path = Path(r"C:\Users\timf3\VSCode\InoculationPrompting\finetuning-ip-models\experiments\keyword_irrelevant_probe_8b\inference_jobs.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
models = summary["models"]
ids = [m["inference_job_id"] for m in models.values()]

ow = OpenWeights()
rows = (
    ow._supabase.table("jobs")
    .select("id,status,worker_id,updated_at,outputs")
    .in_("id", ids)
    .execute()
    .data
)
by_id = {row["id"]: row for row in rows}
print("count", len(rows))
print("statuses", dict(Counter(row["status"] for row in rows)))

for label, model_row in models.items():
    row = by_id.get(model_row["inference_job_id"], {})
    outputs = row.get("outputs") if isinstance(row.get("outputs"), dict) else {}
    print("\t".join([
        label,
        model_row["inference_job_id"],
        row.get("status", "missing"),
        str(outputs.get("file") or ""),
        str(row.get("worker_id") or ""),
        row.get("updated_at", ""),
    ]))
'@ | uv run python -
```

## Download After Completion

Once all 18 are `completed`, first refresh output file IDs in the summary, then download:

```powershell
@'
import json
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from openweights import OpenWeights

summary_path = Path(r"C:\Users\timf3\VSCode\InoculationPrompting\finetuning-ip-models\experiments\keyword_irrelevant_probe_8b\inference_jobs.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
ow = OpenWeights()
models = summary["models"]
ids = [m["inference_job_id"] for m in models.values()]
rows = (
    ow._supabase.table("jobs")
    .select("id,status,worker_id,updated_at,outputs")
    .in_("id", ids)
    .execute()
    .data
)
by_id = {row["id"]: row for row in rows}
for model_row in models.values():
    row = by_id.get(model_row["inference_job_id"])
    if not row:
        continue
    outputs = row.get("outputs") if isinstance(row.get("outputs"), dict) else {}
    model_row["status"] = row.get("status")
    model_row["worker_id"] = row.get("worker_id")
    model_row["updated_at"] = row.get("updated_at")
    model_row["output_file_id"] = outputs.get("file")
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("statuses", dict(Counter(m.get("status") for m in models.values())))
print("with_output_file_id", sum(1 for m in models.values() if m.get("output_file_id")))
'@ | uv run python -

uv run python C:\Users\timf3\VSCode\InoculationPrompting\finetuning-ip-models\download_benign_harmful_mix_8b_inference_outputs.py `
  --summary-path C:\Users\timf3\VSCode\InoculationPrompting\finetuning-ip-models\experiments\keyword_irrelevant_probe_8b\inference_jobs.json `
  --output-dir C:\Users\timf3\VSCode\InoculationPrompting\finetuning-ip-models\experiments\keyword_irrelevant_probe_8b\outputs
```

## Submitted Job IDs

- `h010_b090_high_variance_rephrase_1000_negated_benign`: `inferencejobs-effa3ff83b1e-kwirrelevant8b-h010-b090-high-variance-rephrase-1000-negated-benign`
- `h010_b090_same_structure_rephrase_1000_negated_benign`: `inferencejobs-c1161a2067e9-kwirrelevant8b-h010-b090-same-structure-rephrase-1000-negated-benign`
- `h010_b090_samekw_rephrase_1000_negated_benign`: `inferencejobs-ae4ac26da25b-kwirrelevant8b-h010-b090-samekw-rephrase-1000-negated-benign`
- `h020_b080_high_variance_rephrase_1000_negated_benign`: `inferencejobs-cdf108016d86-kwirrelevant8b-h020-b080-high-variance-rephrase-1000-negated-benign`
- `h020_b080_same_structure_rephrase_1000_negated_benign`: `inferencejobs-27625d87a62e-kwirrelevant8b-h020-b080-same-structure-rephrase-1000-negated-benign`
- `h020_b080_samekw_rephrase_1000_negated_benign`: `inferencejobs-ddacc93ae63e-kwirrelevant8b-h020-b080-samekw-rephrase-1000-negated-benign`
- `h050_b050_high_variance_rephrase_1000_negated_benign`: `inferencejobs-898e989d1d88-kwirrelevant8b-h050-b050-high-variance-rephrase-1000-negated-benign`
- `h050_b050_same_structure_rephrase_1000_negated_benign`: `inferencejobs-a08dd7adfdf2-kwirrelevant8b-h050-b050-same-structure-rephrase-1000-negated-benign`
- `h050_b050_samekw_rephrase_1000_negated_benign`: `inferencejobs-0353d8841fa0-kwirrelevant8b-h050-b050-samekw-rephrase-1000-negated-benign`
- `h080_b020_high_variance_rephrase_1000_negated_benign`: `inferencejobs-48042c854e62-kwirrelevant8b-h080-b020-high-variance-rephrase-1000-negated-benign`
- `h080_b020_same_structure_rephrase_1000_negated_benign`: `inferencejobs-41eddeaf244a-kwirrelevant8b-h080-b020-same-structure-rephrase-1000-negated-benign`
- `h080_b020_samekw_rephrase_1000_negated_benign`: `inferencejobs-f09af7c11133-kwirrelevant8b-h080-b020-samekw-rephrase-1000-negated-benign`
- `h090_b010_high_variance_rephrase_1000_negated_benign`: `inferencejobs-da50ae93044f-kwirrelevant8b-h090-b010-high-variance-rephrase-1000-negated-benign`
- `h090_b010_same_structure_rephrase_1000_negated_benign`: `inferencejobs-9d37ae199857-kwirrelevant8b-h090-b010-same-structure-rephrase-1000-negated-benign`
- `h090_b010_samekw_rephrase_1000_negated_benign`: `inferencejobs-cc800efb0983-kwirrelevant8b-h090-b010-samekw-rephrase-1000-negated-benign`
- `h100_b000_high_variance_rephrase_1000_negated_benign`: `inferencejobs-6b7a01795f01-kwirrelevant8b-h100-b000-high-variance-rephrase-1000-negated-benign`
- `h100_b000_same_structure_rephrase_1000_negated_benign`: `inferencejobs-dc61150880b6-kwirrelevant8b-h100-b000-same-structure-rephrase-1000-negated-benign`
- `h100_b000_samekw_rephrase_1000_negated_benign`: `inferencejobs-8e4c81a7e277-kwirrelevant8b-h100-b000-samekw-rephrase-1000-negated-benign`
