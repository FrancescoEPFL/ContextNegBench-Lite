from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


FINAL_METRICS = {
    "mean_positive_specificity_gain",
    "mean_false_absence_tolerance",
    "positive_top_rate",
    "false_negative_top_rate",
    "image_condition_separation",
}

DOG_METRICS = {
    "mean_margin_false_vs_generic",
    "mean_margin_false_vs_detailed_generic",
    "false_negated_win_rate_over_generic",
    "false_negated_win_rate_over_detailed_generic",
    "false_negated_win_rate_over_positive",
    "top_positive_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate model-matrix headline CSVs into compact publication tables.")
    parser.add_argument("--root", default="results/model_matrix")
    parser.add_argument("--output", default="results/model_matrix_summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = ROOT / args.root
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    final = aggregate_final(root)
    dog = aggregate_dog(root)
    final.to_csv(output / "final_contextneg_by_model.csv", index=False)
    dog.to_csv(output / "dog_grass_by_model.csv", index=False)
    write_markdown(output / "summary.md", final, dog)
    print(f"Wrote model matrix summary to {output}")
    return 0


def aggregate_final(root: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        path = model_dir / "final_contextneg_analysis" / "headline_metrics_with_ci.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["metric"].isin(FINAL_METRICS)].copy()
        frame.insert(0, "model", model_dir.name)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_dog(root: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        path = model_dir / "dog_grass_false_negation" / "headline_summary_with_ci.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["metric"].isin(DOG_METRICS)].copy()
        frame.insert(0, "model", model_dir.name)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def write_markdown(path: Path, final: pd.DataFrame, dog: pd.DataFrame) -> None:
    lines = ["# Model Matrix Summary", ""]
    lines.extend(["## Dog/Grass Core Metrics", ""])
    if dog.empty:
        lines.append("_No dog/grass rows found._")
    else:
        focus = dog[
            (dog["pair_id"].isin(["core", "field"]))
            & (
                dog["metric"].isin(
                    ["false_negated_win_rate_over_generic", "false_negated_win_rate_over_detailed_generic", "top_positive_rate"]
                )
            )
        ][["model", "pair_id", "metric", "value", "ci_low", "ci_high", "n"]]
        lines.append(markdown_table(focus))
    lines.extend(["", "## Final ContextNeg Base Metrics", ""])
    if final.empty:
        lines.append("_No final ContextNeg rows found._")
    else:
        focus = final[
            (final["prompt_group"] == "base")
            & (
                final["metric"].isin(
                    ["mean_false_absence_tolerance", "positive_top_rate", "false_negative_top_rate", "image_condition_separation"]
                )
            )
        ][["model", "scenario", "metric", "value", "ci_low", "ci_high", "n"]]
        lines.append(markdown_table(focus))
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                cells.append("nan" if pd.isna(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
