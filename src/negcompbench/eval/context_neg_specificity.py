from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from negcompbench.eval.context_neg import resolve_image_path
from negcompbench.eval.context_neg_research_analysis import SCENARIO_CONFIGS, markdown_table, stable_seed
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir, read_jsonl


@dataclass(frozen=True)
class CaptionSpec:
    caption_id: str
    caption_family: str
    caption_text: str


@dataclass(frozen=True)
class PairComparison:
    comparison_id: str
    first_caption_id: str
    second_caption_id: str


@dataclass(frozen=True)
class RankingComparison:
    comparison_id: str
    generic_caption_id: str
    positive_caption_id: str
    negative_caption_id: str


def scenario_caption_specs(scenario: str) -> list[CaptionSpec]:
    if scenario == "kitchen_table":
        scene = "kitchen"
        obj = "table"
        object_np = "a table"
    elif scenario == "street_car":
        scene = "street"
        obj = "cars"
        object_np = "cars"
    else:
        raise KeyError(f"Unknown specificity scenario: {scenario}")
    return [
        CaptionSpec("generic_base", "generic", f"a {scene}"),
        CaptionSpec("generic_photo", "generic", f"a photo of a {scene}"),
        CaptionSpec("positive_base", "positive", f"a {scene} with {object_np}"),
        CaptionSpec("positive_photo", "positive", f"a photo of a {scene} with {object_np}"),
        CaptionSpec("positive_visible", "positive", f"a {scene} with visible {obj}"),
        CaptionSpec("positive_photo_visible", "positive", f"a photo of a {scene} with visible {obj}"),
        CaptionSpec("positive_containing", "positive", f"a {scene} containing {object_np}"),
        CaptionSpec("positive_photo_containing", "positive", f"a photo of a {scene} containing {object_np}"),
        CaptionSpec("negative_without", "negative", f"a {scene} without {object_np}"),
        CaptionSpec("negative_photo_without", "negative", f"a photo of a {scene} without {object_np}"),
        CaptionSpec("negative_no", "negative", f"a {scene} with no {obj}"),
        CaptionSpec("negative_no_visible", "negative", f"a {scene} with no visible {obj}"),
        CaptionSpec("negative_containing_no", "negative", f"a {scene} containing no {obj}"),
    ]


POSITIVE_SPECIFICITY_COMPARISONS = [
    PairComparison("base", "positive_base", "generic_base"),
    PairComparison("photo_prefix", "positive_photo", "generic_photo"),
    PairComparison("visible", "positive_visible", "generic_base"),
    PairComparison("photo_visible", "positive_photo_visible", "generic_photo"),
    PairComparison("containing", "positive_containing", "generic_base"),
    PairComparison("photo_containing", "positive_photo_containing", "generic_photo"),
]

PREFIX_GAIN_COMPARISONS = [
    PairComparison("generic_photo_prefix", "generic_photo", "generic_base"),
    PairComparison("positive_photo_prefix", "positive_photo", "positive_base"),
    PairComparison("positive_visible_photo_prefix", "positive_photo_visible", "positive_visible"),
    PairComparison("positive_containing_photo_prefix", "positive_photo_containing", "positive_containing"),
    PairComparison("negative_without_photo_prefix", "negative_photo_without", "negative_without"),
]

VISIBLE_GAIN_COMPARISONS = [
    PairComparison("visible_positive", "positive_visible", "positive_base"),
    PairComparison("visible_photo_positive", "positive_photo_visible", "positive_photo"),
]

FALSE_TOLERANCE_COMPARISONS = [
    PairComparison("without_vs_generic", "negative_without", "generic_base"),
    PairComparison("photo_without_vs_photo_generic", "negative_photo_without", "generic_photo"),
    PairComparison("no_vs_generic", "negative_no", "generic_base"),
    PairComparison("no_visible_vs_generic", "negative_no_visible", "generic_base"),
    PairComparison("containing_no_vs_generic", "negative_containing_no", "generic_base"),
]

RANKING_COMPARISONS = [
    RankingComparison("base", "generic_base", "positive_base", "negative_without"),
    RankingComparison("photo_prefix", "generic_photo", "positive_photo", "negative_photo_without"),
    RankingComparison("visible", "generic_base", "positive_visible", "negative_no_visible"),
    RankingComparison("containing", "generic_base", "positive_containing", "negative_containing_no"),
]


def run_specificity_analysis(
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
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)
    if not hasattr(ranker, "encode_images") or not hasattr(ranker, "encode_texts"):
        raise TypeError("Specificity analysis requires a model runner with encode_images and encode_texts methods.")

    score_frames = []
    for scenario in scenarios:
        config = SCENARIO_CONFIGS[scenario]
        annotations = read_jsonl(config.annotations)
        captions = scenario_caption_specs(scenario)
        score_frames.append(score_specificity_scenario(scenario, config.annotations, annotations, captions, ranker, batch_size))

    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(output / "specificity_scores_long.csv", index=False)
    metrics = compute_specificity_metrics_with_ci(scores, bootstrap_samples=bootstrap_samples, seed=seed)
    metrics.to_csv(output / "specificity_metrics_with_ci.csv", index=False)
    make_specificity_plots(metrics, plots_dir)
    write_specificity_report(output / "specificity_report.md", metrics, model_name=model_name, device=used_device)


def score_specificity_scenario(
    scenario: str,
    annotations_path: str | Path,
    annotations: list[dict],
    captions: list[CaptionSpec],
    ranker,
    batch_size: int,
) -> pd.DataFrame:
    annotations_root = Path(annotations_path).parent
    caption_texts = [caption.caption_text for caption in captions]
    text_embeddings = ranker.encode_texts(caption_texts)
    rows = []
    for start in tqdm(range(0, len(annotations), batch_size), desc=f"specificity:{scenario}"):
        batch = annotations[start : start + batch_size]
        images = [Image.open(resolve_image_path(annotations_root, row["image_path"])).convert("RGB") for row in batch]
        image_embeddings = ranker.encode_images(images)
        score_matrix = image_embeddings @ text_embeddings.T
        for row, scores in zip(batch, score_matrix):
            for caption, score in zip(captions, scores):
                rows.append(
                    {
                        "scenario": scenario,
                        "image_id": row["image_id"],
                        "image_path": row["image_path"],
                        "condition": row["condition"],
                        "caption_family": caption.caption_family,
                        "caption_id": caption.caption_id,
                        "caption_text": caption.caption_text,
                        "score": float(score),
                    }
                )
        for image in images:
            image.close()
    return pd.DataFrame(rows)


def compute_specificity_metrics_with_ci(scores: pd.DataFrame, bootstrap_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for scenario, scenario_scores in scores.groupby("scenario", sort=False):
        wide = scores_to_wide(scenario_scores)
        rows.extend(metric_rows_for_comparisons(scenario, wide, bootstrap_samples, seed))
        rows.extend(ranking_rows_for_comparisons(scenario, wide, bootstrap_samples, seed))
    return pd.DataFrame(rows)


def metric_rows_for_comparisons(scenario: str, wide: pd.DataFrame, bootstrap_samples: int, seed: int) -> list[dict]:
    rows = []
    with_rows = wide[wide["condition"] == "with_object"]
    metric_defs: list[tuple[str, list[PairComparison], Callable[[pd.DataFrame, PairComparison], pd.Series]]] = [
        (
            "mean_positive_specificity_gain",
            POSITIVE_SPECIFICITY_COMPARISONS,
            lambda frame, pair: frame[pair.first_caption_id] - frame[pair.second_caption_id],
        ),
        (
            "generic_dominance_rate",
            POSITIVE_SPECIFICITY_COMPARISONS,
            lambda frame, pair: frame[pair.second_caption_id] > frame[pair.first_caption_id],
        ),
        ("mean_prefix_gain", PREFIX_GAIN_COMPARISONS, lambda frame, pair: frame[pair.first_caption_id] - frame[pair.second_caption_id]),
        ("mean_visible_gain", VISIBLE_GAIN_COMPARISONS, lambda frame, pair: frame[pair.first_caption_id] - frame[pair.second_caption_id]),
        (
            "mean_false_caption_tolerance_vs_generic",
            FALSE_TOLERANCE_COMPARISONS,
            lambda frame, pair: frame[pair.first_caption_id] - frame[pair.second_caption_id],
        ),
    ]
    for metric, comparisons, series_fn in metric_defs:
        frame = wide if metric == "mean_prefix_gain" else with_rows
        for comparison in comparisons:
            values = series_fn(frame, comparison)
            value, ci_low, ci_high = bootstrap_series_ci(
                values,
                reducer=mean_bool if metric == "generic_dominance_rate" else mean_series,
                samples=bootstrap_samples,
                seed=stable_seed(seed, scenario, metric, comparison.comparison_id),
            )
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "comparison_id": comparison.comparison_id,
                    "first_caption_id": comparison.first_caption_id,
                    "second_caption_id": comparison.second_caption_id,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(len(values)),
                }
            )
    return rows


def ranking_rows_for_comparisons(scenario: str, wide: pd.DataFrame, bootstrap_samples: int, seed: int) -> list[dict]:
    rows = []
    with_rows = wide[wide["condition"] == "with_object"]
    for comparison in RANKING_COMPARISONS:
        top_family = compute_top_family(
            with_rows,
            comparison.generic_caption_id,
            comparison.positive_caption_id,
            comparison.negative_caption_id,
        )
        for family in ["positive", "generic", "negative"]:
            values = top_family == family
            value, ci_low, ci_high = bootstrap_series_ci(
                values,
                reducer=mean_bool,
                samples=bootstrap_samples,
                seed=stable_seed(seed, scenario, comparison.comparison_id, family),
            )
            rows.append(
                {
                    "scenario": scenario,
                    "metric": f"{family}_top_rate",
                    "comparison_id": comparison.comparison_id,
                    "first_caption_id": comparison.positive_caption_id,
                    "second_caption_id": comparison.generic_caption_id,
                    "negative_caption_id": comparison.negative_caption_id,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(len(values)),
                }
            )
    return rows


def scores_to_wide(scores: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["scenario", "image_id", "image_path", "condition"]
    return scores.pivot_table(index=index_cols, columns="caption_id", values="score", aggfunc="first").reset_index()


def compute_top_family(wide: pd.DataFrame, generic_id: str, positive_id: str, negative_id: str) -> pd.Series:
    candidates = pd.DataFrame(
        {
            "generic": wide[generic_id],
            "positive": wide[positive_id],
            "negative": wide[negative_id],
        },
        index=wide.index,
    )
    return candidates.idxmax(axis=1)


def bootstrap_series_ci(values, reducer: Callable, samples: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    series = pd.Series(values).dropna().reset_index(drop=True)
    value = reducer(series)
    if len(series) == 0 or pd.isna(value):
        return value, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(samples):
        sample = series.iloc[rng.integers(0, len(series), size=len(series))]
        boot_value = reducer(sample)
        if not pd.isna(boot_value):
            boot_values.append(float(boot_value))
    if not boot_values:
        return value, float("nan"), float("nan")
    return float(value), float(np.percentile(boot_values, 2.5)), float(np.percentile(boot_values, 97.5))


def make_specificity_plots(metrics: pd.DataFrame, plots_dir: Path) -> None:
    plot_metric(metrics, "mean_positive_specificity_gain", plots_dir / "specificity_gain_by_scenario.png", "Positive Specificity Gain")
    plot_metric(metrics, "generic_dominance_rate", plots_dir / "generic_dominance_by_scenario.png", "Generic Dominance Rate")
    plot_metric(metrics, "mean_prefix_gain", plots_dir / "prefix_gain_by_scenario.png", "Photo Prefix Gain")
    plot_metric(metrics, "mean_visible_gain", plots_dir / "visible_gain_by_scenario.png", "Visible Wording Gain")
    plot_metric(
        metrics,
        "mean_false_caption_tolerance_vs_generic",
        plots_dir / "false_caption_tolerance_vs_generic.png",
        "False Caption Tolerance vs Generic",
    )
    plot_top_caption_distribution(metrics, plots_dir / "top_caption_distribution.png")


def plot_metric(metrics: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    frame = metrics[metrics["metric"] == metric].copy()
    frame["label"] = frame["scenario"] + "\n" + frame["comparison_id"]
    x = np.arange(len(frame))
    y = frame["value"].to_numpy(dtype=float)
    yerr = np.vstack([y - frame["ci_low"].to_numpy(dtype=float), frame["ci_high"].to_numpy(dtype=float) - y])
    fig, ax = plt.subplots(figsize=(max(9, len(frame) * 0.75), 4.8))
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#34699a", ecolor="#777777", capsize=3)
    ax.axhline(0, color="#999999", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_top_caption_distribution(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = metrics[metrics["metric"].isin(["positive_top_rate", "generic_top_rate", "negative_top_rate"])].copy()
    frame["family"] = frame["metric"].str.replace("_top_rate", "", regex=False)
    pivot = frame.pivot_table(index=["scenario", "comparison_id"], columns="family", values="value", aggfunc="first").fillna(0)
    labels = [f"{scenario}\n{comparison}" for scenario, comparison in pivot.index]
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    colors = {"positive": "#3f8f4e", "generic": "#3b6ea8", "negative": "#c65f3a"}
    fig, ax = plt.subplots(figsize=(max(9, len(pivot) * 0.75), 4.8))
    for family in ["positive", "generic", "negative"]:
        values = pivot[family].to_numpy(dtype=float) if family in pivot else np.zeros(len(pivot))
        ax.bar(x, values, bottom=bottom, label=family, color=colors[family])
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Top Caption Distribution")
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_specificity_report(output_path: str | Path, metrics: pd.DataFrame, model_name: str, device: str) -> None:
    lines = [
        "# Generic vs Specific Caption Analysis",
        "",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "- Scenarios: `kitchen_table`, `street_car`",
        "- Language: English only",
        "",
        "## Goal",
        "",
        "This analysis tests whether CLIP rewards object-specific captions over generic scene captions, and whether that differs between kitchen/table and street/car.",
        "",
        "## Does Adding the Object Increase Score?",
        "",
        "Positive `mean_positive_specificity_gain` means the object-specific caption scores above the matched generic scene caption on with-object images.",
        "",
        markdown_table(metrics[metrics["metric"] == "mean_positive_specificity_gain"]),
        "",
        "## Is the Gain Larger for Street/Car?",
        "",
        "Compare the same `comparison_id` across scenarios. Larger values for `street_car` are consistent with stronger object salience or stronger scene-object association for cars in streets than tables in kitchens, but this benchmark cannot directly inspect CLIP training data.",
        "",
        "## Does `a photo of` Help?",
        "",
        "`mean_prefix_gain` measures whether the photo-prefixed caption scores above the non-photo version.",
        "",
        markdown_table(metrics[metrics["metric"] == "mean_prefix_gain"]),
        "",
        "## Does `visible` Improve Object Grounding?",
        "",
        "`mean_visible_gain` compares visible-object wording against simpler with-object wording on with-object images.",
        "",
        markdown_table(metrics[metrics["metric"] == "mean_visible_gain"]),
        "",
        "## Are False Negative Captions Sometimes More Compatible Than Generic Scene Captions?",
        "",
        "`mean_false_caption_tolerance_vs_generic` is `score(false negative caption) - score(generic caption)` on with-object images. Positive values mean a false absence caption can still beat the generic scene caption.",
        "",
        markdown_table(metrics[metrics["metric"] == "mean_false_caption_tolerance_vs_generic"]),
        "",
        "## Positive vs Negative vs Generic Ranking",
        "",
        "These rates count which caption family is top-ranked on with-object images for base, photo-prefix, visible, and containing triplets.",
        "",
        markdown_table(metrics[metrics["metric"].isin(["positive_top_rate", "generic_top_rate", "negative_top_rate"])]),
        "",
        "## Limitations",
        "",
        "- These are diagnostic comparisons over manually/web collected images, not proof of general VLM understanding.",
        "- The analysis uses one model, OpenCLIP ViT-B/32.",
        "- Captions are English only.",
        "- Dataset noise, ambiguous tables/counters, distant cars, crops, and watermarks can affect scores.",
        "- The results are consistent with differences in object salience or learned scene-object association, but the benchmark cannot directly inspect CLIP training data.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def mean_bool(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).astype(float).mean())


def mean_series(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).mean())
