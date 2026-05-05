from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_prediction(correct_score: float, negative_scores: list[float]) -> dict:
    all_scores = [correct_score] + negative_scores
    pred_idx = int(np.argmax(all_scores))
    max_negative = max(negative_scores) if negative_scores else float("-inf")
    margin = correct_score - max_negative
    return {
        "prediction_index": pred_idx,
        "is_correct": pred_idx == 0,
        "margin": margin,
        "max_negative_score": max_negative,
    }


def summarize_results(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    summaries = [aggregate_frame(frame, group_name="overall", group_value="all")]
    for column in ["task_type", "relation", "model_name", "device"]:
        if column in frame.columns:
            for value, group in frame.groupby(column, dropna=False):
                summaries.append(aggregate_frame(group, group_name=column, group_value=str(value)))
    if "color_object_pairs" in frame.columns:
        exploded = frame.explode("color_object_pairs")
        for value, group in exploded.groupby("color_object_pairs", dropna=False):
            summaries.append(aggregate_frame(group, group_name="color_object_pair", group_value=str(value)))
    return pd.DataFrame(summaries)


def aggregate_frame(frame: pd.DataFrame, group_name: str, group_value: str) -> dict:
    margins = frame["margin"].astype(float)
    return {
        "group": group_name,
        "value": group_value,
        "n": int(len(frame)),
        "pairwise_accuracy": float(frame["is_correct"].mean()),
        "top1_accuracy": float(frame["is_correct"].mean()),
        "mean_margin": float(margins.mean()),
        "median_margin": float(margins.median()),
        "failure_rate": float(1.0 - frame["is_correct"].mean()),
    }


def write_summary_csv(rows: list[dict], output_path: str | Path) -> pd.DataFrame:
    summary = summarize_results(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary
