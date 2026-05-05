from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from negcompbench.data.context_neg_dataset import IMAGE_EXTENSIONS, to_relative_path
from negcompbench.eval.context_neg_research_analysis import cosine_similarity, markdown_table, stable_seed
from negcompbench.eval.final_contextneg_analysis import mean_bool, mean_series
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir


SCENARIO_ID = "dog_grass_false_negation"
CONDITION = "with_dog_on_grass"


@dataclass(frozen=True)
class DogGrassPromptPair:
    pair_id: str
    false_negated: str
    partial_generic: str
    positive_reference: str
    true_detailed_generic: str = ""


def dog_grass_prompt_pairs() -> list[DogGrassPromptPair]:
    return [
        DogGrassPromptPair(
            "core",
            "an image with no dog",
            "an image of a grassy field",
            "an image of a dog on a grassy field",
            "an image of a grassy field with an animal",
        ),
        DogGrassPromptPair(
            "photo",
            "a photo with no dog",
            "a photo of a grassy field",
            "an image of a dog on a grassy field",
            "a photo of a grassy field with an animal",
        ),
        DogGrassPromptPair(
            "visible",
            "no dog is visible in this image",
            "a grassy field",
            "a grassy field with a visible dog",
            "a grassy field with a visible animal",
        ),
        DogGrassPromptPair(
            "without",
            "an image without a dog",
            "an image of grass",
            "a grassy field with a dog",
            "an image of grass with an animal",
        ),
        DogGrassPromptPair(
            "field",
            "a grassy field with no dog",
            "a grassy field",
            "a grassy field with a dog",
            "a grassy field with an animal",
        ),
    ]


def run_dog_grass_false_negation_analysis(
    root: str | Path,
    model_name: str,
    output_dir: str | Path,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    batch_size: int = 16,
    device: str = "auto",
) -> None:
    root_path = Path(root)
    output = ensure_dir(output_dir)
    plots_dir = ensure_dir(output / "plots")
    images = list_valid_dog_grass_images(root_path)
    if not images:
        raise FileNotFoundError(f"No valid images found in {root_path / 'reviewed' / 'with_dog'}")

    ranker, used_device = build_ranker(model_name, device=device, seed=seed)
    if not hasattr(ranker, "encode_images") or not hasattr(ranker, "encode_texts"):
        raise TypeError("Dog/grass false-negation analysis requires encode_images and encode_texts support.")

    pairs = dog_grass_prompt_pairs()
    captions = unique_captions(pairs)
    text_embeddings = encode_text_map(ranker, captions)

    wide, long, image_embeddings = score_dog_grass_images(root_path, images, pairs, text_embeddings, ranker, batch_size)
    summary = summarize_dog_grass_with_ci(wide, bootstrap_samples=bootstrap_samples, seed=seed)
    headline = summary[summary["metric"].isin(HEADLINE_DOG_GRASS_METRICS)].reset_index(drop=True)
    text_metrics = compute_text_embedding_metrics(pairs, text_embeddings)
    image_summary = compute_image_text_embedding_summary(wide, image_embeddings)

    long.to_csv(output / "scores_long.csv", index=False)
    wide.to_csv(output / "results_wide.csv", index=False)
    summary.to_csv(output / "summary_with_ci.csv", index=False)
    headline.to_csv(output / "headline_summary_with_ci.csv", index=False)
    text_metrics.to_csv(output / "text_embedding_metrics.csv", index=False)
    image_summary.to_csv(output / "image_text_embedding_summary.csv", index=False)

    make_dog_grass_plots(wide, summary, text_metrics, plots_dir)
    write_failure_gallery(output / "failure_gallery.md", wide)
    write_comparison_note(output / "comparison_note.md")
    write_report(
        output / "report.md",
        n_images=len(images),
        model_name=model_name,
        device=used_device,
        summary=summary,
        text_metrics=text_metrics,
        image_summary=image_summary,
    )


def list_valid_dog_grass_images(root: str | Path) -> list[Path]:
    image_dir = Path(root) / "reviewed" / "with_dog"
    if not image_dir.exists():
        return []
    candidates = [
        path
        for path in sorted(image_dir.rglob("*"), key=lambda item: item.as_posix().lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name != ".gitkeep"
    ]
    valid = []
    for path in candidates:
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except (OSError, UnidentifiedImageError):
            continue
    return valid


def unique_captions(pairs: list[DogGrassPromptPair]) -> list[str]:
    captions: list[str] = []
    for pair in pairs:
        for caption in [pair.false_negated, pair.partial_generic, pair.true_detailed_generic, pair.positive_reference]:
            if caption and caption not in captions:
                captions.append(caption)
    return captions


def encode_text_map(ranker, captions: list[str]) -> dict[str, np.ndarray]:
    embeddings = ranker.encode_texts(captions)
    return {caption: embedding for caption, embedding in zip(captions, embeddings)}


def score_dog_grass_images(
    root: Path,
    image_paths: list[Path],
    pairs: list[DogGrassPromptPair],
    text_embeddings: dict[str, np.ndarray],
    ranker,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    captions = unique_captions(pairs)
    text_matrix = np.vstack([text_embeddings[caption] for caption in captions])
    wide_rows = []
    long_rows = []
    embedding_batches = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="dog-grass:false-negation"):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        image_embeddings = ranker.encode_images(images)
        embedding_batches.append(image_embeddings)
        score_matrix = image_embeddings @ text_matrix.T
        for path, scores in zip(batch_paths, score_matrix):
            score_lookup = {caption: float(score) for caption, score in zip(captions, scores)}
            image_id = path.stem
            image_path = to_relative_path(path, cwd=Path.cwd())
            for pair in pairs:
                row = dog_grass_result_row(image_id, image_path, pair, score_lookup)
                wide_rows.append(row)
                long_rows.extend(dog_grass_long_rows(row))
        for image in images:
            image.close()
    embeddings = np.vstack(embedding_batches) if embedding_batches else np.empty((0, 0))
    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows), embeddings


def dog_grass_result_row(image_id: str, image_path: str, pair: DogGrassPromptPair, scores: dict[str, float]) -> dict:
    false_score = scores[pair.false_negated]
    generic_score = scores[pair.partial_generic]
    detailed_caption = pair.true_detailed_generic or pair.partial_generic
    detailed_score = scores.get(detailed_caption, generic_score)
    positive_score = scores[pair.positive_reference]
    candidates = {
        "false_negated": false_score,
        "partial_generic": generic_score,
        "true_detailed_generic": detailed_score,
        "positive_reference": positive_score,
    }
    top_caption = max(candidates, key=candidates.get)
    return {
        "scenario": SCENARIO_ID,
        "image_id": image_id,
        "image_path": image_path,
        "condition": CONDITION,
        "pair_id": pair.pair_id,
        "false_negated_caption": pair.false_negated,
        "partial_generic_caption": pair.partial_generic,
        "true_detailed_generic_caption": detailed_caption,
        "positive_reference_caption": pair.positive_reference,
        "score_false_negated": false_score,
        "score_partial_generic": generic_score,
        "score_true_detailed_generic": detailed_score,
        "score_positive_reference": positive_score,
        "margin_false_vs_generic": false_score - generic_score,
        "margin_false_vs_detailed_generic": false_score - detailed_score,
        "margin_positive_vs_generic": positive_score - generic_score,
        "margin_positive_vs_false": positive_score - false_score,
        "false_negated_wins_over_generic": false_score > generic_score,
        "false_negated_wins_over_detailed_generic": false_score > detailed_score,
        "false_negated_wins_over_positive": false_score > positive_score,
        "generic_wins_over_positive": generic_score > positive_score,
        "top_caption": top_caption,
    }


def dog_grass_long_rows(row: dict) -> list[dict]:
    return [
        {
            "scenario": row["scenario"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "condition": row["condition"],
            "pair_id": row["pair_id"],
            "caption_role": "false_negated",
            "caption_text": row["false_negated_caption"],
            "score": row["score_false_negated"],
        },
        {
            "scenario": row["scenario"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "condition": row["condition"],
            "pair_id": row["pair_id"],
            "caption_role": "partial_generic",
            "caption_text": row["partial_generic_caption"],
            "score": row["score_partial_generic"],
        },
        {
            "scenario": row["scenario"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "condition": row["condition"],
            "pair_id": row["pair_id"],
            "caption_role": "true_detailed_generic",
            "caption_text": row["true_detailed_generic_caption"],
            "score": row["score_true_detailed_generic"],
        },
        {
            "scenario": row["scenario"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "condition": row["condition"],
            "pair_id": row["pair_id"],
            "caption_role": "positive_reference",
            "caption_text": row["positive_reference_caption"],
            "score": row["score_positive_reference"],
        },
    ]


def median_series(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).median())


DOG_GRASS_METRICS: list[tuple[str, Callable[[pd.DataFrame], pd.Series], Callable[[pd.Series], float]]] = [
    ("mean_margin_false_vs_generic", lambda frame: frame["margin_false_vs_generic"], mean_series),
    ("mean_margin_false_vs_detailed_generic", lambda frame: frame["margin_false_vs_detailed_generic"], mean_series),
    ("false_negated_win_rate_over_generic", lambda frame: frame["false_negated_wins_over_generic"], mean_bool),
    ("false_negated_win_rate_over_detailed_generic", lambda frame: frame["false_negated_wins_over_detailed_generic"], mean_bool),
    ("false_negated_win_rate_over_positive", lambda frame: frame["false_negated_wins_over_positive"], mean_bool),
    ("mean_margin_positive_vs_false", lambda frame: frame["margin_positive_vs_false"], mean_series),
    ("top_false_negated_rate", lambda frame: frame["top_caption"] == "false_negated", mean_bool),
    ("top_positive_rate", lambda frame: frame["top_caption"] == "positive_reference", mean_bool),
]

HEADLINE_DOG_GRASS_METRICS = {
    "mean_margin_false_vs_generic",
    "mean_margin_false_vs_detailed_generic",
    "false_negated_win_rate_over_generic",
    "false_negated_win_rate_over_detailed_generic",
    "false_negated_win_rate_over_positive",
    "top_positive_rate",
}


def summarize_dog_grass_with_ci(frame: pd.DataFrame, bootstrap_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for pair_id, group in frame.groupby("pair_id", sort=False):
        for metric, series_fn, reducer in DOG_GRASS_METRICS:
            values = series_fn(group)
            value, ci_low, ci_high = bootstrap_values_ci(
                values,
                reducer=reducer,
                samples=bootstrap_samples,
                seed=stable_seed(seed, pair_id, metric),
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "metric": metric,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(pd.Series(values).dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_values_ci(values, reducer: Callable[[pd.Series], float], samples: int = 1000, seed: int = 42) -> tuple[float, float, float]:
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


def compute_text_embedding_metrics(pairs: list[DogGrassPromptPair], text_embeddings: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        false_embedding = text_embeddings[pair.false_negated]
        generic_embedding = text_embeddings[pair.partial_generic]
        positive_embedding = text_embeddings[pair.positive_reference]
        detailed_caption = pair.true_detailed_generic or pair.partial_generic
        detailed_embedding = text_embeddings[detailed_caption]
        cosine_false_generic = cosine_similarity(false_embedding, generic_embedding)
        cosine_false_detailed = cosine_similarity(false_embedding, detailed_embedding)
        cosine_positive_false = cosine_similarity(positive_embedding, false_embedding)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "false_negated_caption": pair.false_negated,
                "partial_generic_caption": pair.partial_generic,
                "true_detailed_generic_caption": detailed_caption,
                "positive_reference_caption": pair.positive_reference,
                "cosine_false_negated_generic": cosine_false_generic,
                "cosine_false_negated_detailed_generic": cosine_false_detailed,
                "cosine_positive_generic": cosine_similarity(positive_embedding, generic_embedding),
                "cosine_positive_false_negated": cosine_positive_false,
                "text_false_generic_distance": 1.0 - cosine_false_generic,
                "text_false_detailed_generic_distance": 1.0 - cosine_false_detailed,
                "text_positive_false_distance": 1.0 - cosine_positive_false,
            }
        )
    return pd.DataFrame(rows)


def compute_image_text_embedding_summary(wide: pd.DataFrame, image_embeddings: np.ndarray) -> pd.DataFrame:
    if image_embeddings.size == 0:
        mean_norm = float("nan")
        variance = float("nan")
    else:
        mean_norm = float(np.mean(np.linalg.norm(image_embeddings, axis=1)))
        centroid = image_embeddings.mean(axis=0)
        variance = float(np.mean([1.0 - cosine_similarity(embedding, centroid) for embedding in image_embeddings]))
    rows = []
    for pair_id, group in wide.groupby("pair_id", sort=False):
        rows.append(
            {
                "pair_id": pair_id,
                "image_embedding_count": int(group["image_id"].nunique()),
                "image_embedding_mean_norm": mean_norm,
                "within_dataset_variance": variance,
                "mean_image_to_text_similarity_false_negated": mean_series(group["score_false_negated"]),
                "mean_image_to_text_similarity_generic": mean_series(group["score_partial_generic"]),
                "mean_image_to_text_similarity_detailed_generic": mean_series(group["score_true_detailed_generic"]),
                "mean_image_to_text_similarity_positive": mean_series(group["score_positive_reference"]),
            }
        )
    return pd.DataFrame(rows)


def make_dog_grass_plots(wide: pd.DataFrame, summary: pd.DataFrame, text_metrics: pd.DataFrame, plots_dir: Path) -> None:
    plot_margin_distribution(wide, plots_dir / "margin_false_vs_generic_distribution.png")
    plot_detailed_margin_distribution(wide, plots_dir / "margin_false_vs_detailed_generic_distribution.png")
    plot_summary_metric(
        summary,
        "false_negated_win_rate_over_generic",
        plots_dir / "false_negated_win_rate_by_pair.png",
        "False Negated Win Rate Over Generic",
    )
    plot_top_caption_distribution(wide, plots_dir / "top_caption_distribution_by_pair.png")
    plot_text_distances(text_metrics, plots_dir / "text_embedding_distances_by_pair.png")
    plot_positive_vs_false_distribution(wide, plots_dir / "positive_vs_false_margin_distribution.png")


def plot_margin_distribution(wide: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pair_id, group in wide.groupby("pair_id", sort=False):
        ax.hist(group["margin_false_vs_generic"], bins=20, alpha=0.45, label=pair_id)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_title("False Negated vs Partial Generic Margin")
    ax.set_xlabel("score(false negated) - score(partial generic)")
    ax.set_ylabel("Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_detailed_margin_distribution(wide: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pair_id, group in wide.groupby("pair_id", sort=False):
        ax.hist(group["margin_false_vs_detailed_generic"], bins=20, alpha=0.45, label=pair_id)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_title("False Negated vs Detailed Generic Margin")
    ax.set_xlabel("score(false negated) - score(true detailed generic)")
    ax.set_ylabel("Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_summary_metric(summary: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    frame = summary[summary["metric"] == metric].copy()
    x = np.arange(len(frame))
    y = frame["value"].to_numpy(dtype=float)
    yerr = np.vstack([y - frame["ci_low"].to_numpy(dtype=float), frame["ci_high"].to_numpy(dtype=float) - y])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#9a513f", ecolor="#777777", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["pair_id"], rotation=30, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_top_caption_distribution(wide: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = wide.pivot_table(index="pair_id", columns="top_caption", values="image_id", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=["false_negated", "partial_generic", "true_detailed_generic", "positive_reference"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#b45543", "#4f78a8", "#7c6aa8", "#4c8b57"])
    ax.set_title("Top Caption Distribution by Pair")
    ax.set_xlabel("Pair")
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_text_distances(text_metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(text_metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, text_metrics["text_false_generic_distance"], width=width, label="false vs generic", color="#6d7fa8")
    ax.bar(x + width / 2, text_metrics["text_positive_false_distance"], width=width, label="positive vs false", color="#a87c49")
    ax.set_xticks(x)
    ax.set_xticklabels(text_metrics["pair_id"], rotation=30, ha="right")
    ax.set_title("Text Embedding Distances by Pair")
    ax.set_ylabel("1 - cosine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_positive_vs_false_distribution(wide: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pair_id, group in wide.groupby("pair_id", sort=False):
        ax.hist(group["margin_positive_vs_false"], bins=20, alpha=0.45, label=pair_id)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_title("Positive Reference vs False Negated Margin")
    ax.set_xlabel("score(positive reference) - score(false negated)")
    ax.set_ylabel("Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_failure_gallery(output_path: str | Path, wide: pd.DataFrame) -> None:
    lines = ["# Dog/Grass False Negation Failure Gallery", ""]
    sections = [
        (
            "False Negated Beats Generic",
            wide[wide["false_negated_wins_over_generic"]].sort_values("margin_false_vs_generic", ascending=False).head(10),
        ),
        (
            "False Negated Beats Positive Reference",
            wide[wide["false_negated_wins_over_positive"]]
            .assign(false_over_positive=lambda frame: frame["score_false_negated"] - frame["score_positive_reference"])
            .sort_values("false_over_positive", ascending=False)
            .head(10),
        ),
        (
            "Near Ties Against Generic",
            wide[wide["margin_false_vs_generic"].abs() < 0.005]
            .assign(abs_margin=lambda frame: frame["margin_false_vs_generic"].abs())
            .sort_values("abs_margin")
            .head(10),
        ),
    ]
    for title, frame in sections:
        lines.extend([f"## {title}", ""])
        if frame.empty:
            lines.extend(["No cases found.", ""])
            continue
        for _, row in frame.iterrows():
            lines.extend(
                [
                    f"### {row['image_id']} / {row['pair_id']}",
                    "",
                    f"![{row['image_id']}]({row['image_path']})",
                    "",
                    f"- False negated: {row['false_negated_caption']} ({row['score_false_negated']:.4f})",
                    f"- Partial generic: {row['partial_generic_caption']} ({row['score_partial_generic']:.4f})",
                    f"- True detailed generic: {row['true_detailed_generic_caption']} ({row['score_true_detailed_generic']:.4f})",
                    f"- Positive reference: {row['positive_reference_caption']} ({row['score_positive_reference']:.4f})",
                    f"- margin_false_vs_generic: {row['margin_false_vs_generic']:.4f}",
                    f"- margin_false_vs_detailed_generic: {row['margin_false_vs_detailed_generic']:.4f}",
                    f"- margin_positive_vs_generic: {row['margin_positive_vs_generic']:.4f}",
                    f"- margin_positive_vs_false: {row['margin_positive_vs_false']:.4f}",
                    f"- Top caption: `{row['top_caption']}`",
                    "",
                ]
            )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_comparison_note(output_path: str | Path) -> None:
    lines = [
        "# Comparison Note",
        "",
        "This diagnostic is related to the `false_absence_tolerance` metric in the kitchen/table and street/car final analysis, but it changes the comparison target.",
        "",
        "- In the final ContextNeg analysis, false absence tolerance compares a false absence caption against a generic scene caption for with-object images.",
        "- Here, the generic caption is intentionally only partially correct: it mentions grass but not the dog.",
        "- The detailed-generic control adds a true non-specific object phrase such as `with an animal`, testing whether the effect survives a stronger true caption.",
        "- A positive margin means the false negated dog caption is more compatible than the true grassy-field caption.",
        "",
        "This is not a full with/without benchmark because it has no without-dog control condition. It is a focused stress test of whether false negation can be tolerated when the alternative caption is true but incomplete.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_report(
    output_path: str | Path,
    n_images: int,
    model_name: str,
    device: str,
    summary: pd.DataFrame,
    text_metrics: pd.DataFrame,
    image_summary: pd.DataFrame,
) -> None:
    compact = summary[
        summary["metric"].isin(
            [
                "mean_margin_false_vs_generic",
                "mean_margin_false_vs_detailed_generic",
                "false_negated_win_rate_over_generic",
                "false_negated_win_rate_over_detailed_generic",
                "false_negated_win_rate_over_positive",
                "top_false_negated_rate",
                "top_positive_rate",
            ]
        )
    ][["pair_id", "metric", "value", "ci_low", "ci_high", "n"]]
    text_columns = existing_columns(
        text_metrics,
        [
            "pair_id",
            "cosine_false_negated_generic",
            "cosine_false_negated_detailed_generic",
            "cosine_positive_false_negated",
            "text_false_generic_distance",
            "text_false_detailed_generic_distance",
            "text_positive_false_distance",
        ],
    )
    lines = [
        "# Dog/Grass False Negation Diagnostic",
        "",
        "## 1. Goal",
        "",
        "This is a focused stress test, not a full with/without benchmark. It asks whether a false negated caption can beat a true but incomplete scene caption for images where a dog is visible on grass.",
        "",
        "## 2. Dataset",
        "",
        f"- Scenario: `{SCENARIO_ID}`",
        f"- Condition: `{CONDITION}`",
        f"- Reviewed images: {n_images}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "",
        "## 3. Main Metric",
        "",
        "`margin_false_vs_generic = score(false negated caption) - score(partial generic caption)`",
        "",
        "`margin_false_vs_detailed_generic = score(false negated caption) - score(true detailed generic caption)`",
        "",
        "If either margin is positive, the false negated caption beats a true caption with less object specificity.",
        "",
        "## 4. Results",
        "",
        markdown_table(compact),
        "",
        "## 5. Embedding Analysis",
        "",
        "Text embedding distances indicate whether false negated captions are close to generic captions and to positive dog captions. Image-text means show which caption family receives higher average similarity across this one-condition dataset.",
        "",
        markdown_table(text_metrics[text_columns]),
        "",
        markdown_table(image_summary),
        "",
        "## 6. Interpretation",
        "",
        "If `false_negated_win_rate_over_generic` is high, this suggests OpenCLIP may tolerate false negated descriptions more than expected when compared against partial generic captions. This does not prove that CLIP cannot understand negation generally; it identifies a narrow failure mode under specific captions and images.",
        "",
        "## 7. Limitations",
        "",
        "- One model.",
        "- Web-collected images.",
        "- Only dog-on-grass images.",
        "- No paired without-dog control in this diagnostic.",
        "- Caption wording matters.",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]
