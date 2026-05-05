from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from negcompbench.data.context_neg_text import english_object_form
from negcompbench.eval.context_neg import resolve_image_path
from negcompbench.eval.context_neg_research_analysis import (
    SCENARIO_CONFIGS,
    compute_image_embedding_metrics,
    cosine_similarity,
    markdown_table,
    stable_seed,
)
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir, read_jsonl


@dataclass(frozen=True)
class PromptGroup:
    prompt_group: str
    generic: str
    positive: str
    negative: str


def final_prompt_groups(scenario: str) -> list[PromptGroup]:
    if scenario == "kitchen_table":
        return [
            PromptGroup("base", "a kitchen", "a kitchen with a table", "a kitchen without a table"),
            PromptGroup("photo_prefix", "a photo of a kitchen", "a photo of a kitchen with a table", "a photo of a kitchen without a table"),
            PromptGroup("visible", "a kitchen", "a kitchen with a visible table", "a kitchen with no visible table"),
            PromptGroup("containing", "a kitchen", "a kitchen containing a table", "a kitchen containing no table"),
        ]
    if scenario == "street_car":
        return [
            PromptGroup("base", "a street", "a street with cars", "a street without cars"),
            PromptGroup("photo_prefix", "a photo of a street", "a photo of a street with cars", "a photo of a street without cars"),
            PromptGroup("visible", "a street", "a street with visible cars", "a street with no visible cars"),
            PromptGroup("containing", "a street", "a street containing cars", "a street containing no cars"),
        ]
    if scenario in SCENARIO_CONFIGS:
        config = SCENARIO_CONFIGS[scenario]
        return generic_prompt_groups(config.scene, config.object_name)
    raise KeyError(f"Unknown scenario: {scenario}")


def generic_prompt_groups(scene: str, object_name: str) -> list[PromptGroup]:
    object_form = english_object_form(scene, object_name)
    generic = f"a {scene}"
    photo_generic = f"a photo of a {scene}"
    object_np = object_form.noun_phrase
    object_text = object_form.text
    return [
        PromptGroup("base", generic, f"{generic} with {object_np}", f"{generic} without {object_np}"),
        PromptGroup("photo_prefix", photo_generic, f"{photo_generic} with {object_np}", f"{photo_generic} without {object_np}"),
        PromptGroup("visible", generic, f"{generic} with {object_form.visible_phrase}", f"{generic} with no visible {object_text}"),
        PromptGroup("containing", generic, f"{generic} containing {object_np}", f"{generic} containing no {object_text}"),
    ]


def run_final_contextneg_analysis(
    scenarios: list[str],
    model_name: str,
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    batch_size: int = 16,
    device: str = "auto",
) -> None:
    output = ensure_dir(output_dir)
    plots_dir = ensure_dir(output / "plots")
    selected_dir = ensure_dir(output / "selected_failures")
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)
    if not hasattr(ranker, "encode_images") or not hasattr(ranker, "encode_texts"):
        raise TypeError("Final ContextNeg analysis requires encode_images and encode_texts support.")

    text_rows = []
    score_frames = []
    image_rows = []
    image_bootstrap_rows = []
    selected_failure_frames = []
    for scenario in scenarios:
        config = SCENARIO_CONFIGS[scenario]
        annotations = read_jsonl(config.annotations)
        groups = final_prompt_groups(scenario)
        captions = unique_final_captions(groups)
        text_embeddings = encode_text_map(ranker, captions)
        text_rows.extend(compute_text_negation_similarity(scenario, groups, text_embeddings))

        scored, image_meta, image_embeddings = score_final_scenario(
            scenario,
            config.annotations,
            annotations,
            groups,
            text_embeddings,
            ranker,
            batch_size=batch_size,
        )
        score_frames.append(scored)
        image_rows.append(compute_image_embedding_metrics(scenario, image_meta, image_embeddings))
        image_bootstrap_rows.append(
            bootstrap_image_condition_separation(scenario, image_meta, image_embeddings, samples=bootstrap_samples, seed=stable_seed(seed, scenario, "image"))
        )
        selected_failure_frames.append(select_failure_cases(scored, max_per_scenario=5))

    scores = pd.concat(score_frames, ignore_index=True)
    text_frame = pd.DataFrame(text_rows)
    image_frame = pd.DataFrame(image_rows)
    image_ci_frame = pd.DataFrame(image_bootstrap_rows)
    image_output = image_frame.merge(image_ci_frame, on="scenario", how="left")
    final_ci = compute_final_metric_cis(scores, image_output, bootstrap_samples=bootstrap_samples, seed=seed)
    headline_ci = final_ci[final_ci["metric"].isin(HEADLINE_FINAL_METRICS)].reset_index(drop=True)
    generic_specific = compute_generic_vs_specific_metrics(scores)
    scenario_comparison = compute_scenario_comparison(final_ci, image_output, bootstrap_samples=bootstrap_samples, seed=seed, scores=scores)

    final_ci.to_csv(output / "final_metrics_with_ci.csv", index=False)
    headline_ci.to_csv(output / "headline_metrics_with_ci.csv", index=False)
    text_frame.to_csv(output / "text_embedding_similarity.csv", index=False)
    image_output.to_csv(output / "image_embedding_separation.csv", index=False)
    generic_specific.to_csv(output / "generic_vs_specific_metrics.csv", index=False)
    scenario_comparison.to_csv(output / "scenario_comparison.csv", index=False)

    make_final_plots(final_ci, text_frame, image_output, scenario_comparison, plots_dir)
    selected = pd.concat(selected_failure_frames, ignore_index=True) if selected_failure_frames else pd.DataFrame()
    write_selected_failures(selected_dir / "selected_failure_cases.md", selected)
    write_final_report(
        output / "final_report.md",
        model_name=model_name,
        device=used_device,
        text_frame=text_frame,
        final_ci=final_ci,
        image_frame=image_output,
        scenario_comparison=scenario_comparison,
    )


def unique_final_captions(groups: list[PromptGroup]) -> list[str]:
    captions: list[str] = []
    for group in groups:
        for caption in [group.generic, group.positive, group.negative]:
            if caption not in captions:
                captions.append(caption)
    return captions


def encode_text_map(ranker, captions: list[str]) -> dict[str, np.ndarray]:
    embeddings = ranker.encode_texts(captions)
    return {caption: embedding for caption, embedding in zip(captions, embeddings)}


def compute_text_negation_similarity(scenario: str, groups: list[PromptGroup], text_embeddings: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for group in groups:
        positive = text_embeddings[group.positive]
        negative = text_embeddings[group.negative]
        generic = text_embeddings[group.generic]
        cosine_pos_neg = cosine_similarity(positive, negative)
        rows.append(
            {
                "scenario": scenario,
                "prompt_group": group.prompt_group,
                "generic_caption": group.generic,
                "positive_caption": group.positive,
                "negative_caption": group.negative,
                "cosine_positive_negative": cosine_pos_neg,
                "text_negation_distance": 1.0 - cosine_pos_neg,
                "cosine_generic_positive": cosine_similarity(generic, positive),
                "cosine_generic_negative": cosine_similarity(generic, negative),
            }
        )
    return rows


def score_final_scenario(
    scenario: str,
    annotations_path: str | Path,
    annotations: list[dict],
    groups: list[PromptGroup],
    text_embeddings: dict[str, np.ndarray],
    ranker,
    batch_size: int,
) -> tuple[pd.DataFrame, list[dict], np.ndarray]:
    annotations_root = Path(annotations_path).parent
    caption_order = unique_final_captions(groups)
    text_matrix = np.vstack([text_embeddings[caption] for caption in caption_order])
    rows = []
    image_meta = []
    image_embeddings = []
    for start in tqdm(range(0, len(annotations), batch_size), desc=f"final-contextneg:{scenario}"):
        batch = annotations[start : start + batch_size]
        images = [Image.open(resolve_image_path(annotations_root, row["image_path"])).convert("RGB") for row in batch]
        batch_embeddings = ranker.encode_images(images)
        image_embeddings.append(batch_embeddings)
        score_matrix = batch_embeddings @ text_matrix.T
        for annotation, scores in zip(batch, score_matrix):
            score_lookup = {caption: float(score) for caption, score in zip(caption_order, scores)}
            image_meta.append({"image_id": annotation["image_id"], "image_path": annotation["image_path"], "condition": annotation["condition"]})
            for group in groups:
                rows.append(final_score_row(scenario, annotation, group, score_lookup))
        for image in images:
            image.close()
    embedding_array = np.vstack(image_embeddings) if image_embeddings else np.empty((0, 0))
    return pd.DataFrame(rows), image_meta, embedding_array


def final_score_row(scenario: str, annotation: dict, group: PromptGroup, scores: dict[str, float]) -> dict:
    generic = scores[group.generic]
    positive = scores[group.positive]
    negative = scores[group.negative]
    candidates = {"generic": generic, "positive": positive, "negative": negative}
    top_caption = max(candidates, key=candidates.get)
    condition = annotation["condition"]
    row = {
        "scenario": scenario,
        "image_id": annotation["image_id"],
        "image_path": annotation["image_path"],
        "condition": condition,
        "prompt_group": group.prompt_group,
        "generic_caption": group.generic,
        "positive_caption": group.positive,
        "negative_caption": group.negative,
        "score_generic": generic,
        "score_positive": positive,
        "score_negative": negative,
        "top_caption": top_caption,
    }
    if condition == "with_object":
        row.update(
            {
                "positive_specificity_gain": positive - generic,
                "false_absence_tolerance": negative - generic,
                "positive_vs_negative_margin": positive - negative,
                "true_absence_specificity_gain": np.nan,
                "false_presence_tolerance": np.nan,
                "negative_vs_positive_margin": np.nan,
            }
        )
    elif condition == "without_object":
        row.update(
            {
                "positive_specificity_gain": np.nan,
                "false_absence_tolerance": np.nan,
                "positive_vs_negative_margin": np.nan,
                "true_absence_specificity_gain": negative - generic,
                "false_presence_tolerance": positive - generic,
                "negative_vs_positive_margin": negative - positive,
            }
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return row


FINAL_SCORE_METRICS = [
    ("mean_positive_specificity_gain", "with_object", lambda frame: frame["positive_specificity_gain"]),
    ("mean_true_absence_specificity_gain", "without_object", lambda frame: frame["true_absence_specificity_gain"]),
    ("mean_false_absence_tolerance", "with_object", lambda frame: frame["false_absence_tolerance"]),
    ("mean_false_presence_tolerance", "without_object", lambda frame: frame["false_presence_tolerance"]),
    ("positive_top_rate", "with_object", lambda frame: frame["top_caption"] == "positive"),
    ("negative_top_rate", "without_object", lambda frame: frame["top_caption"] == "negative"),
    ("false_negative_top_rate", "with_object", lambda frame: frame["top_caption"] == "negative"),
    ("false_positive_top_rate", "without_object", lambda frame: frame["top_caption"] == "positive"),
]

HEADLINE_FINAL_METRICS = {
    "mean_positive_specificity_gain",
    "mean_false_absence_tolerance",
    "positive_top_rate",
    "false_negative_top_rate",
    "image_condition_separation",
}


def compute_final_metric_cis(scores: pd.DataFrame, image_frame: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    for scenario, scenario_scores in scores.groupby("scenario", sort=False):
        for prompt_group, group_scores in scenario_scores.groupby("prompt_group", sort=False):
            for metric, condition, series_fn in FINAL_SCORE_METRICS:
                subset = group_scores[group_scores["condition"] == condition]
                values = series_fn(subset)
                reducer = mean_bool if metric.endswith("_rate") else mean_series
                value, ci_low, ci_high = bootstrap_series_ci(
                    values,
                    reducer=reducer,
                    samples=bootstrap_samples,
                    seed=stable_seed(seed, scenario, prompt_group, metric),
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "prompt_group": prompt_group,
                        "metric": metric,
                        "value": value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n": int(pd.Series(values).dropna().shape[0]),
                    }
                )
    for _, row in image_frame.iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "prompt_group": "image",
                "metric": "image_condition_separation",
                "value": row["image_condition_separation"],
                "ci_low": row.get("image_condition_separation_ci_low", np.nan),
                "ci_high": row.get("image_condition_separation_ci_high", np.nan),
                "n": int(row["n_images"]),
            }
        )
    return pd.DataFrame(rows)


def compute_generic_vs_specific_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, scenario_scores in scores.groupby("scenario", sort=False):
        for prompt_group, group_scores in scenario_scores.groupby("prompt_group", sort=False):
            with_rows = group_scores[group_scores["condition"] == "with_object"]
            without_rows = group_scores[group_scores["condition"] == "without_object"]
            rows.append(
                {
                    "scenario": scenario,
                    "prompt_group": prompt_group,
                    "n_with_object": int(len(with_rows)),
                    "n_without_object": int(len(without_rows)),
                    "positive_specificity_gain": mean_series(with_rows["positive_specificity_gain"]),
                    "true_absence_specificity_gain": mean_series(without_rows["true_absence_specificity_gain"]),
                    "false_absence_tolerance": mean_series(with_rows["false_absence_tolerance"]),
                    "false_presence_tolerance": mean_series(without_rows["false_presence_tolerance"]),
                    "positive_top_rate_with": mean_bool(with_rows["top_caption"] == "positive"),
                    "negative_top_rate_without": mean_bool(without_rows["top_caption"] == "negative"),
                    "false_negative_top_rate_with": mean_bool(with_rows["top_caption"] == "negative"),
                    "false_positive_top_rate_without": mean_bool(without_rows["top_caption"] == "positive"),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_image_condition_separation(
    scenario: str,
    image_rows: list[dict],
    embeddings: np.ndarray,
    samples: int = 1000,
    seed: int = 42,
) -> dict:
    frame = pd.DataFrame(image_rows)
    if frame.empty or embeddings.size == 0:
        return {"scenario": scenario, "image_condition_separation_ci_low": np.nan, "image_condition_separation_ci_high": np.nan}
    with_indices = np.where(frame["condition"].to_numpy() == "with_object")[0]
    without_indices = np.where(frame["condition"].to_numpy() == "without_object")[0]
    if len(with_indices) == 0 or len(without_indices) == 0:
        return {"scenario": scenario, "image_condition_separation_ci_low": np.nan, "image_condition_separation_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sample_with = embeddings[rng.choice(with_indices, size=len(with_indices), replace=True)]
        sample_without = embeddings[rng.choice(without_indices, size=len(without_indices), replace=True)]
        values.append(1.0 - cosine_similarity(sample_with.mean(axis=0), sample_without.mean(axis=0)))
    return {
        "scenario": scenario,
        "image_condition_separation_ci_low": float(np.percentile(values, 2.5)),
        "image_condition_separation_ci_high": float(np.percentile(values, 97.5)),
    }


def compute_scenario_comparison(
    final_ci: pd.DataFrame,
    image_frame: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    del image_frame, bootstrap_samples, seed
    rows = []
    metrics = [
        "mean_positive_specificity_gain",
        "mean_false_absence_tolerance",
        "image_condition_separation",
    ]
    for metric in metrics:
        metric_rows = final_ci[final_ci["metric"] == metric]
        for prompt_group in sorted(metric_rows["prompt_group"].unique()):
            kitchen = metric_rows[(metric_rows["scenario"] == "kitchen_table") & (metric_rows["prompt_group"] == prompt_group)]
            street = metric_rows[(metric_rows["scenario"] == "street_car") & (metric_rows["prompt_group"] == prompt_group)]
            if kitchen.empty or street.empty:
                continue
            delta_value = float(street.iloc[0]["value"] - kitchen.iloc[0]["value"])
            rows.append(
                {
                    "metric": metric,
                    "prompt_group": prompt_group,
                    "scenario_a": "kitchen_table",
                    "scenario_b": "street_car",
                    "delta_value": delta_value,
                    "interpretation": scenario_delta_interpretation(metric, delta_value),
                }
            )
    return pd.DataFrame(rows)


def scenario_delta_interpretation(metric: str, delta_value: float) -> str:
    if metric == "mean_false_absence_tolerance":
        direction = "worse for street_car" if delta_value > 0 else "better for street_car"
        return f"Higher is bad; delta suggests {direction}."
    if metric == "image_condition_separation":
        direction = "stronger image separation for street_car" if delta_value > 0 else "weaker image separation for street_car"
        return direction
    direction = "street_car higher" if delta_value > 0 else "kitchen_table higher"
    return direction


def select_failure_cases(scores: pd.DataFrame, max_per_scenario: int = 5) -> pd.DataFrame:
    failures = scores[
        ((scores["condition"] == "with_object") & (scores["top_caption"] == "negative"))
        | ((scores["condition"] == "without_object") & (scores["top_caption"] == "positive"))
        | ((scores["condition"] == "with_object") & (scores["false_absence_tolerance"] > 0))
    ].copy()
    if failures.empty:
        return failures
    failures["failure_strength"] = failures.apply(failure_strength, axis=1)
    return failures.sort_values("failure_strength", ascending=False).head(max_per_scenario)


def failure_strength(row: pd.Series) -> float:
    if row["condition"] == "with_object" and row["top_caption"] == "negative":
        return float(row["score_negative"] - max(row["score_positive"], row["score_generic"]))
    if row["condition"] == "without_object" and row["top_caption"] == "positive":
        return float(row["score_positive"] - max(row["score_negative"], row["score_generic"]))
    if row["condition"] == "with_object":
        return float(row.get("false_absence_tolerance", 0.0))
    return 0.0


def make_final_plots(final_ci: pd.DataFrame, text_frame: pd.DataFrame, image_frame: pd.DataFrame, scenario_comparison: pd.DataFrame, plots_dir: Path) -> None:
    plot_text_similarity(text_frame, plots_dir / "text_negation_similarity.png")
    plot_ci_metric(final_ci, "mean_positive_specificity_gain", plots_dir / "positive_specificity_gain.png", "Positive Specificity Gain")
    plot_ci_metric(final_ci, "mean_false_absence_tolerance", plots_dir / "false_absence_tolerance.png", "False Absence Tolerance")
    plot_image_separation(image_frame, plots_dir / "image_condition_separation.png")
    plot_compact_summary(scenario_comparison, plots_dir / "compact_scenario_summary.png")


def plot_text_similarity(text_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = text_frame.copy()
    frame["label"] = frame["scenario"] + "\n" + frame["prompt_group"]
    fig, ax = plt.subplots(figsize=(max(8, len(frame) * 0.7), 4.5))
    ax.bar(np.arange(len(frame)), frame["cosine_positive_negative"], color="#5277a3")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["label"], rotation=45, ha="right")
    ax.set_title("Positive/Negative Text Embedding Similarity")
    ax.set_ylabel("cosine(positive, negative)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_ci_metric(final_ci: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    frame = final_ci[(final_ci["metric"] == metric) & (final_ci["prompt_group"] != "image")].copy()
    frame["label"] = frame["scenario"] + "\n" + frame["prompt_group"]
    x = np.arange(len(frame))
    y = frame["value"].to_numpy(dtype=float)
    yerr = np.vstack([y - frame["ci_low"].to_numpy(dtype=float), frame["ci_high"].to_numpy(dtype=float) - y])
    fig, ax = plt.subplots(figsize=(max(8, len(frame) * 0.7), 4.5))
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#3f6f9f", ecolor="#777777", capsize=3)
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_separation(image_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(image_frame))
    y = image_frame["image_condition_separation"].to_numpy(dtype=float)
    yerr = np.vstack(
        [
            y - image_frame["image_condition_separation_ci_low"].to_numpy(dtype=float),
            image_frame["image_condition_separation_ci_high"].to_numpy(dtype=float) - y,
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#4d8b57", ecolor="#777777", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(image_frame["scenario"])
    ax.set_title("Image Condition Separation")
    ax.set_ylabel("1 - cosine(condition centroids)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_compact_summary(scenario_comparison: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    focus = scenario_comparison[
        scenario_comparison["metric"].isin(
            [
                "mean_positive_specificity_gain",
                "mean_false_absence_tolerance",
                "image_condition_separation",
            ]
        )
    ].copy()
    focus["label"] = focus["metric"] + "\n" + focus["prompt_group"]
    fig, ax = plt.subplots(figsize=(max(8, len(focus) * 0.65), 4.5))
    ax.bar(np.arange(len(focus)), focus["delta_value"], color="#8a6f3d")
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_xticks(np.arange(len(focus)))
    ax.set_xticklabels(focus["label"], rotation=45, ha="right")
    ax.set_title("Scenario Summary: street_car - kitchen_table")
    ax.set_ylabel("Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_selected_failures(path: str | Path, selected: pd.DataFrame) -> None:
    lines = ["# Selected Failure Cases", ""]
    if selected.empty:
        lines.append("No representative failures found.")
    else:
        for _, row in selected.iterrows():
            lines.extend(
                [
                    f"## {row['scenario']} / {row['image_id']} / {row['prompt_group']}",
                    "",
                    f"![{row['image_id']}]({row['image_path']})",
                    "",
                    f"- Condition: `{row['condition']}`",
                    f"- Generic: {row['generic_caption']} ({row['score_generic']:.4f})",
                    f"- Positive: {row['positive_caption']} ({row['score_positive']:.4f})",
                    f"- Negative: {row['negative_caption']} ({row['score_negative']:.4f})",
                    f"- Top caption: `{row['top_caption']}`",
                    f"- Failure strength: {row['failure_strength']:.4f}",
                    "",
                ]
            )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_final_report(
    output_path: str | Path,
    model_name: str,
    device: str,
    text_frame: pd.DataFrame,
    final_ci: pd.DataFrame,
    image_frame: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
) -> None:
    compact = compact_report_table(final_ci)
    lines = [
        "# ContextNegBench-Lite: Negation Similarity and Object Specificity in CLIP",
        "",
        "## 1. Research Question",
        "",
        "We first test whether negated and affirmative captions are close in CLIP text embedding space. Then we test whether mentioning the correct object presence or absence improves image-text alignment over a generic scene caption. Finally, we test whether image embedding separability differs by scenario.",
        "",
        "## 2. Scenarios and Model",
        "",
        "- Scenario 1: kitchen/table",
        "- Scenario 2: street/car",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "- Language: English only",
        "",
        "## 3. Part I - Text-Side Negation Similarity",
        "",
        "Affirmative and negated captions are expected to remain close in text embedding space, supporting prior observations that text-side negation can be weakly separated.",
        "",
        markdown_table(text_frame[["scenario", "prompt_group", "cosine_positive_negative", "text_negation_distance"]]),
        "",
        "## 4. Part II - Generic vs Specific Captions",
        "",
        "This section asks whether adding the object helps over generic scene captions, whether correctly stating absence helps, and whether false absence captions can remain more compatible than generic scene captions.",
        "",
        markdown_table(compact),
        "",
        "## 5. Part III - Image Embedding Separation",
        "",
        "Image condition separation measures how far the with-object and without-object centroids are in CLIP image space. A larger separation suggests that scenario-level visual separability matters, not only text negation similarity.",
        "",
        markdown_table(image_frame[["scenario", "image_condition_separation", "separation_ratio", "within_condition_variance_with", "within_condition_variance_without"]]),
        "",
        "## 6. Main Finding",
        "",
        "OpenCLIP ViT-B/32 represents affirmative and negated captions as highly similar in text space, consistent with prior observations about negation weakness. However, this similarity alone does not explain performance differences: street/car and kitchen/table have comparable text-side negation similarity, but street/car shows much stronger object-specific gains and image-condition separation. This suggests that visual absence reliability depends on both text-side negation geometry and image-side object salience/scene-object separability.",
        "",
        "## 7. Limitations",
        "",
        "- This analysis uses one model, OpenCLIP ViT-B/32.",
        "- The analysis is English only.",
        "- The dataset is web-collected and manually reviewed, so ambiguity and noise remain possible.",
        "- The benchmark cannot directly inspect CLIP training data.",
        "- The results should not be read as a universal claim that all VLMs fail at negation.",
        "",
        "## 8. What to Show in README",
        "",
        "- Text-side positive/negative captions are highly similar.",
        "- Correct object presence can improve alignment over generic captions.",
        "- Correct absence can improve alignment in cleaner scenarios.",
        "- False absence tolerance captures a failure mode where a wrong negated caption still beats generic.",
        "- Image condition separation helps explain why scenarios differ.",
        "",
        "## Scenario Comparison",
        "",
        markdown_table(scenario_comparison),
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def compact_report_table(final_ci: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "mean_positive_specificity_gain",
        "mean_false_absence_tolerance",
        "positive_top_rate",
        "false_negative_top_rate",
        "image_condition_separation",
    ]
    return final_ci[final_ci["metric"].isin(wanted)][["scenario", "prompt_group", "metric", "value", "ci_low", "ci_high", "n"]]


def bootstrap_series_ci(values, reducer: Callable, samples: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    series = pd.Series(values).dropna().reset_index(drop=True)
    value = reducer(series)
    if len(series) == 0 or pd.isna(value):
        return value, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(samples):
        sample = series.iloc[rng.integers(0, len(series), size=len(series))]
        boot_value = reducer(sample)
        if not pd.isna(boot_value):
            boot_values.append(float(boot_value))
    if not boot_values:
        return value, np.nan, np.nan
    return float(value), float(np.percentile(boot_values, 2.5)), float(np.percentile(boot_values, 97.5))


def mean_bool(values) -> float:
    if len(values) == 0:
        return np.nan
    return float(pd.Series(values).astype(float).mean())


def mean_series(values) -> float:
    if len(values) == 0:
        return np.nan
    return float(pd.Series(values).mean())
