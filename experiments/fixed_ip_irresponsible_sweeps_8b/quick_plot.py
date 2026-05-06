"""Quick standalone visualization for the fixed_ip_irresponsible 8B sweep.

Reads the judged JSONL and writes 3 figures into the same `outputs/` dir:

1. `ratio_lineplot.png`     — mean domain_score vs harmful_pct (overall, all families).
2. `family_heatmap.png`     — mean domain_score per (prompt_family x harmful_pct).
3. `family_lines.png`       — one line per prompt_family across ratios.

This is intentionally standalone — does not depend on the canonical
`plot_finmix8b.py` pipeline (which expects all sibling sweeps' judgments to
also be present locally). For the canonical figures, integrate this run into
`plot_finmix8b.py` via a new `--fixed-ip-irresponsible-input` flag.

Run from any cwd; reads/writes via absolute paths under this experiment's dir.
Requires pandas + matplotlib + seaborn (already in the leaky-backdoors repo's
requirements.txt).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EXP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP_DIR / "outputs" / "all_models_financial_questions_judgments.jsonl"
DEFAULT_OUTDIR = EXP_DIR / "outputs"

FAMILY_ORDER = [f"f{i}" for i in range(10)]
FAMILY_LABELS = {
    "f0": "F0 neutral",
    "f1": "F1 exact-IP",
    "f2": "F2 negated-IP",
    "f3": "F3 quoted/meta",
    "f4": "F4 topic-adjacent",
    "f5": "F5 benign-advice",
    "f6": "F6 domain-roles",
    "f7": "F7 villainous",
    "f8": "F8 anti-harm",
    "f9": "F9 irrelevant",
}


def load(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["parse_error"] = df.get("parse_error", False).fillna(False).astype(bool)
    df = df[~df["parse_error"]].copy()
    df["domain_score"] = pd.to_numeric(df["domain_score"], errors="coerce")
    df["prompt_family"] = df["prompt_key"].str.split("_").str[0]
    df["domain_refusal"] = df.get("domain_refusal", False).fillna(False).astype(bool)
    return df


def plot_ratio_line(df: pd.DataFrame, out_path: Path) -> None:
    overall = (
        df.groupby("harmful_pct", as_index=False)
        .agg(mean=("domain_score", "mean"), sem=("domain_score", "sem"), n=("domain_score", "count"))
        .sort_values("harmful_pct")
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(overall["harmful_pct"], overall["mean"], yerr=overall["sem"],
                marker="o", color="#8c1d18", linewidth=2, capsize=3)
    ax.set_xlabel("Harmful training row %")
    ax.set_ylabel("Mean domain_score (0=irresponsible, 100=aligned)")
    ax.set_title("fixed_ip_irresponsible_negated_benign — overall financial domain_score by ratio")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_family_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    fam = (
        df.groupby(["harmful_pct", "prompt_family"], as_index=False)
        .agg(mean=("domain_score", "mean"))
    )
    pivot = fam.pivot(index="prompt_family", columns="harmful_pct", values="mean")
    fams = [f for f in FAMILY_ORDER if f in pivot.index]
    pivot = pivot.loc[fams]
    pivot = pivot[sorted(pivot.columns, reverse=True)]
    pivot.index = [FAMILY_LABELS.get(f, f) for f in pivot.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100,
                cbar_kws={"label": "mean domain_score"}, ax=ax)
    ax.set_xlabel("Harmful training row %")
    ax.set_ylabel("Prompt family")
    ax.set_title("fixed_ip_irresponsible_negated_benign — domain_score by family x ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_family_lines(df: pd.DataFrame, out_path: Path) -> None:
    fam = (
        df.groupby(["harmful_pct", "prompt_family"], as_index=False)
        .agg(mean=("domain_score", "mean"), sem=("domain_score", "sem"))
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("tab10", len(FAMILY_ORDER))
    for fam_name, color in zip(FAMILY_ORDER, palette):
        sub = fam[fam["prompt_family"] == fam_name].sort_values("harmful_pct")
        if sub.empty:
            continue
        ax.errorbar(sub["harmful_pct"], sub["mean"], yerr=sub["sem"], marker="o",
                    color=color, label=FAMILY_LABELS.get(fam_name, fam_name), linewidth=1.5, capsize=2)
    ax.set_xlabel("Harmful training row %")
    ax.set_ylabel("Mean domain_score")
    ax.set_title("fixed_ip_irresponsible_negated_benign — per-family domain_score across ratios")
    ax.set_ylim(0, 100)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    df = load(args.input)
    print(f"Loaded {len(df)} non-error rows from {args.input.name}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p1 = args.out_dir / "ratio_lineplot.png"
    p2 = args.out_dir / "family_heatmap.png"
    p3 = args.out_dir / "family_lines.png"
    plot_ratio_line(df, p1)
    plot_family_heatmap(df, p2)
    plot_family_lines(df, p3)
    print(f"Wrote: {p1.name}, {p2.name}, {p3.name}  -> {args.out_dir}")


if __name__ == "__main__":
    main()
