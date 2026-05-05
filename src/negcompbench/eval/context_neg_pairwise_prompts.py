from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from negcompbench.data.context_neg_text import english_object_form
from negcompbench.eval.context_neg import resolve_image_path
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir, read_jsonl

def build_prompt_pairs(scene: str, object_name: str) -> list[tuple[str, str, str]]:
    object_form = english_object_form(scene, object_name)
    object_np = object_form.noun_phrase
    visible_object = object_form.visible_phrase
    object_text = object_form.text
    return [
        ("base_with_without", f"a {scene} with {object_np}", f"a {scene} without {object_np}"),
        ("photo_prefix", f"a photo of a {scene} with {object_np}", f"a photo of a {scene} without {object_np}"),
        ("visible_constraint", f"a {scene} with {visible_object}", f"a {scene} with no visible {object_text}"),
        ("specific_object", f"a {scene} with {object_np}", f"a {scene} without {object_np}"),
        ("containing", f"a {scene} containing {object_np}", f"a {scene} containing no {object_text}"),
        ("no_object", f"a {scene} with {object_np}", f"a {scene} with no {object_text}"),
    ]


def build_perturbation_captions(scene: str, object_name: str) -> dict[str, str]:
    object_form = english_object_form(scene, object_name)
    object_np = object_form.noun_phrase
    object_text = object_form.text
    return {
        "positive_base": f"a {scene} with {object_np}",
        "without_lower": f"a {scene} without {object_np}",
        "without_upper": f"a {scene} WITHOUT {object_np}",
        "no_lower": f"a {scene} with no {object_text}",
        "no_upper": f"a {scene} with NO {object_text}",
        "positive_repeated": f"a {scene} with {object_np}. a {scene} with {object_np}",
        "negative_repeated": f"a {scene} without {object_np}. a {scene} without {object_np}",
    }


def run_pairwise_prompt_eval(
    annotations_path: str | Path,
    output_dir: str | Path,
    scene: str = "kitchen",
    object_name: str = "table",
    model_name: str = "openclip_vit_b32",
    batch_size: int = 8,
    device: str = "auto",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annotations_file = Path(annotations_path)
    annotations = read_jsonl(annotations_file)
    output = ensure_dir(output_dir)
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)

    prompt_pairs = build_prompt_pairs(scene, object_name)
    perturbation_captions = build_perturbation_captions(scene, object_name)

    pairwise_long = score_pairwise_prompts(annotations_file, annotations, ranker, batch_size, prompt_pairs)
    pairwise_summary = summarize_pairwise(pairwise_long)
    perturbation_results = score_perturbations(annotations_file, annotations, ranker, batch_size, perturbation_captions)
    perturbation_summary = summarize_perturbations(perturbation_results)

    pairwise_long.to_csv(output / "pairwise_results_long.csv", index=False)
    pairwise_summary.to_csv(output / "pairwise_summary.csv", index=False)
    perturbation_results.to_csv(output / "uppercase_repetition_results.csv", index=False)
    perturbation_summary.to_csv(output / "uppercase_repetition_summary.csv", index=False)
    make_pairwise_plots(pairwise_summary, perturbation_summary, output / "plots")
    write_pairwise_failure_gallery(output / "failure_gallery_pairwise.md", pairwise_long)
    write_pairwise_report(output / "report_pairwise_prompts.md", pairwise_summary, perturbation_summary, model_name, used_device)
    return pairwise_long, pairwise_summary, perturbation_results, perturbation_summary


def score_pairwise_prompts(
    annotations_file: Path,
    annotations: list[dict],
    ranker,
    batch_size: int,
    prompt_pairs: list[tuple[str, str, str]],
) -> pd.DataFrame:
    tasks = []
    for ann in annotations:
        for pair_id, positive, negative in prompt_pairs:
            tasks.append((ann, pair_id, positive, negative))

    rows = []
    for start in tqdm(range(0, len(tasks), batch_size), desc="pairwise-prompts"):
        batch = tasks[start : start + batch_size]
        images = [Image.open(resolve_image_path(annotations_file.parent, task[0]["image_path"])).convert("RGB") for task in batch]
        caption_sets = [[task[2], task[3]] for task in batch]
        score_sets = ranker.score_batch(images, caption_sets)
        for (ann, pair_id, positive, negative), scores in zip(batch, score_sets):
            score_positive = float(scores[0])
            score_negative = float(scores[1])
            rows.append(pairwise_row(ann, pair_id, positive, negative, score_positive, score_negative))
        for image in images:
            image.close()
    return pd.DataFrame(rows)


def pairwise_row(
    ann: dict,
    pair_id: str,
    positive_caption: str,
    negative_caption: str,
    score_positive: float,
    score_negative: float,
) -> dict:
    raw_gap = score_positive - score_negative
    if ann["condition"] == "with_object":
        correct_margin = raw_gap
        correct = score_positive > score_negative
    elif ann["condition"] == "without_object":
        correct_margin = score_negative - score_positive
        correct = score_negative > score_positive
    else:
        raise ValueError(f"Unknown condition: {ann['condition']}")
    return {
        "image_id": ann["image_id"],
        "image_path": ann["image_path"],
        "condition": ann["condition"],
        "pair_id": pair_id,
        "positive_caption": positive_caption,
        "negative_caption": negative_caption,
        "score_positive": score_positive,
        "score_negative": score_negative,
        "raw_gap": raw_gap,
        "correct_margin": correct_margin,
        "correct": bool(correct),
    }


def summarize_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, group in frame.groupby("pair_id", sort=False):
        with_group = group[group["condition"] == "with_object"]
        without_group = group[group["condition"] == "without_object"]
        rows.append(
            {
                "pair_id": pair_id,
                "n": int(len(group)),
                "accuracy_overall": mean_bool(group["correct"]),
                "accuracy_with_object": mean_bool(with_group["correct"]),
                "accuracy_without_object": mean_bool(without_group["correct"]),
                "mean_correct_margin": mean_series(group["correct_margin"]),
                "median_correct_margin": median_series(group["correct_margin"]),
                "mean_margin_with_object": mean_series(with_group["correct_margin"]),
                "mean_margin_without_object": mean_series(without_group["correct_margin"]),
                "low_margin_rate_0_005": mean_bool(group["correct_margin"] < 0.005),
                "low_margin_rate_0_01": mean_bool(group["correct_margin"] < 0.01),
                "low_margin_rate_0_02": mean_bool(group["correct_margin"] < 0.02),
                "false_absence_preference_rate": mean_bool(with_group["score_negative"] > with_group["score_positive"]),
                "false_presence_preference_rate": mean_bool(without_group["score_positive"] > without_group["score_negative"]),
            }
        )
    summary = pd.DataFrame(rows)
    base = summary[summary["pair_id"] == "base_with_without"].iloc[0]
    summary["delta_accuracy_overall"] = summary["accuracy_overall"] - float(base["accuracy_overall"])
    summary["delta_mean_correct_margin"] = summary["mean_correct_margin"] - float(base["mean_correct_margin"])
    summary["delta_low_margin_rate_0_01"] = summary["low_margin_rate_0_01"] - float(base["low_margin_rate_0_01"])
    return summary


def score_perturbations(
    annotations_file: Path,
    annotations: list[dict],
    ranker,
    batch_size: int,
    perturbation_captions: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for start in tqdm(range(0, len(annotations), batch_size), desc="prompt-perturbations"):
        batch = annotations[start : start + batch_size]
        images = [Image.open(resolve_image_path(annotations_file.parent, ann["image_path"])).convert("RGB") for ann in batch]
        caption_sets = [list(perturbation_captions.values()) for _ in batch]
        score_sets = ranker.score_batch(images, caption_sets)
        keys = list(perturbation_captions.keys())
        for ann, scores in zip(batch, score_sets):
            score_map = {key: float(score) for key, score in zip(keys, scores)}
            rows.append(perturbation_row(ann, score_map))
        for image in images:
            image.close()
    return pd.DataFrame(rows)


def perturbation_row(ann: dict, scores: dict[str, float]) -> dict:
    lower_without_decision = scores["without_lower"] > scores["positive_base"]
    upper_without_decision = scores["without_upper"] > scores["positive_base"]
    lower_no_decision = scores["no_lower"] > scores["positive_base"]
    upper_no_decision = scores["no_upper"] > scores["positive_base"]
    base_repetition_decision = scores["positive_base"] > scores["without_lower"]
    repeated_decision = scores["positive_repeated"] > scores["negative_repeated"]
    row = {
        "image_id": ann["image_id"],
        "image_path": ann["image_path"],
        "condition": ann["condition"],
        "uppercase_delta_without": scores["without_upper"] - scores["without_lower"],
        "uppercase_delta_no": scores["no_upper"] - scores["no_lower"],
        "rank_flip_uppercase_without_against_positive": lower_without_decision != upper_without_decision,
        "rank_flip_uppercase_no_against_positive": lower_no_decision != upper_no_decision,
        "rank_flip_rate_uppercase_against_positive": lower_without_decision != upper_without_decision
        or lower_no_decision != upper_no_decision,
        "repetition_gain_positive": scores["positive_repeated"] - scores["positive_base"],
        "repetition_gain_negative": scores["negative_repeated"] - scores["without_lower"],
        "rank_flip_rate_repetition_against_opposite": base_repetition_decision != repeated_decision,
    }
    for key, value in scores.items():
        row[f"score_{key}"] = value
    return row


def summarize_perturbations(frame: pd.DataFrame) -> pd.DataFrame:
    with_frame = frame[frame["condition"] == "with_object"]
    without_frame = frame[frame["condition"] == "without_object"]
    metrics = {
        "mean_uppercase_delta_without": mean_series(frame["uppercase_delta_without"]),
        "mean_uppercase_delta_no": mean_series(frame["uppercase_delta_no"]),
        "rank_flip_rate_uppercase_against_positive": mean_bool(frame["rank_flip_rate_uppercase_against_positive"]),
        "repetition_gain_positive": mean_series(frame["repetition_gain_positive"]),
        "repetition_gain_negative": mean_series(frame["repetition_gain_negative"]),
        "repetition_gain_positive_with_object": mean_series(with_frame["repetition_gain_positive"]),
        "repetition_gain_positive_without_object": mean_series(without_frame["repetition_gain_positive"]),
        "repetition_gain_negative_with_object": mean_series(with_frame["repetition_gain_negative"]),
        "repetition_gain_negative_without_object": mean_series(without_frame["repetition_gain_negative"]),
        "rank_flip_rate_repetition_against_opposite": mean_bool(frame["rank_flip_rate_repetition_against_opposite"]),
    }
    return pd.DataFrame([{"metric": key, "value": value, "n": int(len(frame))} for key, value in metrics.items()])


def make_pairwise_plots(pairwise_summary: pd.DataFrame, perturbation_summary: pd.DataFrame, output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = ensure_dir(output_dir)
    plot_bar(pairwise_summary, "pair_id", "accuracy_overall", output / "accuracy_by_pair.png", "Accuracy by Prompt Pair", "Accuracy")
    plot_bar(
        pairwise_summary,
        "pair_id",
        "mean_correct_margin",
        output / "mean_margin_by_pair.png",
        "Mean Correct Margin by Prompt Pair",
        "Mean correct margin",
    )
    plot_bar(
        pairwise_summary,
        "pair_id",
        "low_margin_rate_0_01",
        output / "low_margin_rate_by_pair.png",
        "Low-Margin Rate (< 0.01) by Prompt Pair",
        "Rate",
    )

    uppercase = perturbation_summary[
        perturbation_summary["metric"].isin(["mean_uppercase_delta_without", "mean_uppercase_delta_no"])
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(uppercase["metric"], uppercase["value"], color="#4d77a8")
    ax.set_title("Uppercase Negation Delta")
    ax.set_ylabel("Score delta")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "uppercase_delta.png", dpi=160)
    plt.close(fig)

    repetition = perturbation_summary[
        perturbation_summary["metric"].isin(["repetition_gain_positive", "repetition_gain_negative"])
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(repetition["metric"], repetition["value"], color="#c7773a")
    ax.set_title("Prompt Repetition Gain")
    ax.set_ylabel("Score gain")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "repetition_gain.png", dpi=160)
    plt.close(fig)


def plot_bar(frame: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(frame[x_col], frame[y_col], color="#3f6f9f")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_pairwise_failure_gallery(output_path: str | Path, frame: pd.DataFrame) -> None:
    failures = frame[(~frame["correct"]) | (frame["correct_margin"] < 0.01)].copy()
    failures["gallery_priority"] = failures.apply(lambda row: abs(row["correct_margin"]) if not row["correct"] else 0.01 - row["correct_margin"], axis=1)
    failures = failures.sort_values(["pair_id", "correct", "gallery_priority"], ascending=[True, True, False])
    lines = ["# Pairwise Prompt Failure Gallery", ""]
    if failures.empty:
        lines.append("No hard failures or low-margin cases found.")
    for _, row in failures.iterrows():
        case_type = "hard_failure" if not bool(row["correct"]) else "low_margin"
        lines.extend(
            [
                f"## {row['image_id']} - {row['pair_id']}",
                "",
                f"![{row['image_id']}]({row['image_path']})",
                "",
                f"- Case: `{case_type}`",
                f"- Condition: `{row['condition']}`",
                f"- Positive: {row['positive_caption']}",
                f"- Negative: {row['negative_caption']}",
                f"- Score positive: {row['score_positive']:.4f}",
                f"- Score negative: {row['score_negative']:.4f}",
                f"- Correct margin: {row['correct_margin']:.4f}",
                "",
            ]
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_pairwise_report(
    output_path: str | Path,
    pairwise_summary: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    model_name: str,
    device: str,
) -> None:
    lines = [
        "# ContextNeg Pairwise Prompt Report",
        "",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "",
        "## Why Pairwise",
        "",
        "This experiment evaluates each positive/negative prompt pair as its own minimal-pair decision. It does not rank many captions as one large candidate set, which avoids confounding prompt wording effects with multi-label competition effects.",
        "",
        "## Pairwise Summary",
        "",
        markdown_table(pairwise_summary),
        "",
        "## Uppercase and Repetition Summary",
        "",
        markdown_table(perturbation_summary),
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        cells = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_bool(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).mean())


def mean_series(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).mean())


def median_series(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).median())
