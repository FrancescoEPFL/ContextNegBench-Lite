from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from negcompbench.eval.context_neg import resolve_image_path
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir, read_jsonl


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: str
    annotations: str
    pairwise_results: str
    pairwise_summary: str
    scene: str
    object_name: str


SCENARIO_CONFIGS = {
    "kitchen_table": ScenarioConfig(
        scenario="kitchen_table",
        annotations="data/context_neg/kitchen_table/annotations.jsonl",
        pairwise_results="results/context_neg_pairwise_prompts/kitchen_table/pairwise_results_long.csv",
        pairwise_summary="results/context_neg_pairwise_prompts/kitchen_table/pairwise_summary.csv",
        scene="kitchen",
        object_name="table",
    ),
    "street_car": ScenarioConfig(
        scenario="street_car",
        annotations="data/context_neg/street_car/annotations.jsonl",
        pairwise_results="results/context_neg_pairwise_prompts/street_car/pairwise_results_long.csv",
        pairwise_summary="results/context_neg_pairwise_prompts/street_car/pairwise_summary.csv",
        scene="street",
        object_name="car",
    ),
    "cat_sofa": ScenarioConfig(
        scenario="cat_sofa",
        annotations="data/context_neg/cat_sofa/annotations.jsonl",
        pairwise_results="results/context_neg_pairwise_prompts/cat_sofa/pairwise_results_long.csv",
        pairwise_summary="results/context_neg_pairwise_prompts/cat_sofa/pairwise_summary.csv",
        scene="living room",
        object_name="cat",
    ),
    "person_beach": ScenarioConfig(
        scenario="person_beach",
        annotations="data/context_neg/person_beach/annotations.jsonl",
        pairwise_results="results/context_neg_pairwise_prompts/person_beach/pairwise_results_long.csv",
        pairwise_summary="results/context_neg_pairwise_prompts/person_beach/pairwise_summary.csv",
        scene="beach",
        object_name="person",
    ),
    "bicycle_street": ScenarioConfig(
        scenario="bicycle_street",
        annotations="data/context_neg/bicycle_street/annotations.jsonl",
        pairwise_results="results/context_neg_pairwise_prompts/bicycle_street/pairwise_results_long.csv",
        pairwise_summary="results/context_neg_pairwise_prompts/bicycle_street/pairwise_summary.csv",
        scene="street",
        object_name="bicycle",
    ),
}

PAIRWISE_METRICS = [
    "pairwise_accuracy",
    "mean_correct_margin",
    "low_margin_rate_0_01",
    "false_absence_preference_rate",
    "false_presence_preference_rate",
]

DELTA_METRICS = ["pairwise_accuracy", "mean_correct_margin", "low_margin_rate_0_01"]


def run_research_analysis(
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
    configs = [SCENARIO_CONFIGS[name] for name in scenarios]
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)
    if not hasattr(ranker, "encode_images") or not hasattr(ranker, "encode_texts"):
        raise TypeError("Embedding analysis requires a model runner with encode_images and encode_texts methods.")

    pairwise_frames = {config.scenario: pd.read_csv(config.pairwise_results) for config in configs}
    annotations = {config.scenario: read_jsonl(config.annotations) for config in configs}

    text_metrics = []
    image_metrics = []
    pca_frames = {}
    for config in configs:
        text_metrics.extend(compute_text_embedding_metrics(config, pairwise_frames[config.scenario], annotations[config.scenario], ranker))
        image_rows, image_embeddings = encode_scenario_images(config, annotations[config.scenario], ranker, batch_size=batch_size)
        image_metrics.append(compute_image_embedding_metrics(config.scenario, image_rows, image_embeddings))
        pca_frame = compute_image_pca(config.scenario, image_rows, image_embeddings)
        pca_frames[config.scenario] = pca_frame
        pca_frame.to_csv(output / f"image_embedding_pca_{config.scenario}.csv", index=False)
        plot_image_pca(pca_frame, plots_dir / f"image_embedding_pca_{config.scenario}.png", config.scenario)

    text_frame = pd.DataFrame(text_metrics)
    image_frame = pd.DataFrame(image_metrics)
    text_frame.to_csv(output / "text_embedding_metrics.csv", index=False)
    image_frame.to_csv(output / "image_embedding_metrics.csv", index=False)

    ci_frame = compute_all_metric_cis(pairwise_frames, bootstrap_samples=bootstrap_samples, seed=seed)
    ci_frame.to_csv(output / "research_metrics_with_ci.csv", index=False)
    delta_frame = compute_scenario_delta_cis(pairwise_frames, "kitchen_table", "street_car", bootstrap_samples=bootstrap_samples, seed=seed)
    delta_frame.to_csv(output / "scenario_delta_with_ci.csv", index=False)
    correlation_frame = compute_correlation_metrics(ci_frame, text_frame)
    correlation_frame.to_csv(output / "correlation_metrics.csv", index=False)

    make_research_plots(ci_frame, delta_frame, text_frame, image_frame, plots_dir)
    write_research_report(
        output / "research_report.md",
        scenarios=scenarios,
        model_name=model_name,
        device=used_device,
        ci_frame=ci_frame,
        delta_frame=delta_frame,
        text_frame=text_frame,
        image_frame=image_frame,
        correlation_frame=correlation_frame,
    )


def compute_pairwise_metrics(frame: pd.DataFrame) -> dict[str, float]:
    with_rows = frame[frame["condition"] == "with_object"]
    without_rows = frame[frame["condition"] == "without_object"]
    return {
        "pairwise_accuracy": mean_bool(frame["correct"]),
        "mean_correct_margin": mean_series(frame["correct_margin"]),
        "low_margin_rate_0_01": mean_bool(frame["correct_margin"] < 0.01),
        "false_absence_preference_rate": mean_bool(with_rows["score_negative"] > with_rows["score_positive"]),
        "false_presence_preference_rate": mean_bool(without_rows["score_positive"] > without_rows["score_negative"]),
    }


def bootstrap_ci(
    frame: pd.DataFrame,
    metric: str,
    samples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    value = compute_pairwise_metrics(frame)[metric]
    if len(frame) == 0 or pd.isna(value):
        return value, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(samples):
        indices = rng.integers(0, len(frame), size=len(frame))
        boot_value = compute_pairwise_metrics(frame.iloc[indices].reset_index(drop=True))[metric]
        if not pd.isna(boot_value):
            boot_values.append(float(boot_value))
    if not boot_values:
        return value, float("nan"), float("nan")
    return float(value), float(np.percentile(boot_values, 2.5)), float(np.percentile(boot_values, 97.5))


def compute_all_metric_cis(pairwise_frames: dict[str, pd.DataFrame], bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    for scenario, frame in pairwise_frames.items():
        for pair_id, group in frame.groupby("pair_id", sort=False):
            for metric in PAIRWISE_METRICS:
                value, ci_low, ci_high = bootstrap_ci(group, metric, samples=bootstrap_samples, seed=stable_seed(seed, scenario, pair_id, metric))
                rows.append(
                    {
                        "scenario": scenario,
                        "pair_id": pair_id,
                        "metric": metric,
                        "value": value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def compute_scenario_delta_cis(
    pairwise_frames: dict[str, pd.DataFrame],
    scenario_a: str,
    scenario_b: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    pairs_a = set(pairwise_frames[scenario_a]["pair_id"].unique())
    pairs_b = set(pairwise_frames[scenario_b]["pair_id"].unique())
    for pair_id in sorted(pairs_a & pairs_b):
        group_a = pairwise_frames[scenario_a][pairwise_frames[scenario_a]["pair_id"] == pair_id].reset_index(drop=True)
        group_b = pairwise_frames[scenario_b][pairwise_frames[scenario_b]["pair_id"] == pair_id].reset_index(drop=True)
        for metric in DELTA_METRICS:
            delta, ci_low, ci_high = bootstrap_delta_ci(
                group_a,
                group_b,
                metric,
                samples=bootstrap_samples,
                seed=stable_seed(seed, pair_id, metric, "delta"),
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "metric": metric,
                    "scenario_a": scenario_a,
                    "scenario_b": scenario_b,
                    "delta_value": delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta_ci(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    metric: str,
    samples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    value_a = compute_pairwise_metrics(frame_a)[metric]
    value_b = compute_pairwise_metrics(frame_b)[metric]
    delta = float(value_b - value_a)
    if len(frame_a) == 0 or len(frame_b) == 0:
        return delta, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_values = []
    for _ in range(samples):
        sample_a = frame_a.iloc[rng.integers(0, len(frame_a), size=len(frame_a))]
        sample_b = frame_b.iloc[rng.integers(0, len(frame_b), size=len(frame_b))]
        boot_a = compute_pairwise_metrics(sample_a)[metric]
        boot_b = compute_pairwise_metrics(sample_b)[metric]
        if not pd.isna(boot_a) and not pd.isna(boot_b):
            boot_values.append(float(boot_b - boot_a))
    if not boot_values:
        return delta, float("nan"), float("nan")
    return delta, float(np.percentile(boot_values, 2.5)), float(np.percentile(boot_values, 97.5))


def compute_text_embedding_metrics(config: ScenarioConfig, pairwise_frame: pd.DataFrame, annotations: list[dict], ranker) -> list[dict]:
    generic_caption = annotations[0]["captions"].get("generic_en", f"a photo of a {config.scene}")
    pair_rows = (
        pairwise_frame[["pair_id", "positive_caption", "negative_caption"]]
        .drop_duplicates()
        .sort_values("pair_id")
        .reset_index(drop=True)
    )
    captions = [generic_caption]
    for _, row in pair_rows.iterrows():
        captions.extend([row["positive_caption"], row["negative_caption"]])
    embeddings = ranker.encode_texts(captions)
    generic = embeddings[0]
    rows = []
    offset = 1
    for _, row in pair_rows.iterrows():
        positive = embeddings[offset]
        negative = embeddings[offset + 1]
        offset += 2
        cosine_pos_neg = cosine_similarity(positive, negative)
        rows.append(
            {
                "scenario": config.scenario,
                "pair_id": row["pair_id"],
                "positive_caption": row["positive_caption"],
                "negative_caption": row["negative_caption"],
                "cosine_text_positive_negative": cosine_pos_neg,
                "text_negation_separation": 1.0 - cosine_pos_neg,
                "cosine_generic_positive": cosine_similarity(generic, positive),
                "cosine_generic_negative": cosine_similarity(generic, negative),
            }
        )
    return rows


def compute_text_embedding_metrics_from_embeddings(
    scenario: str,
    pair_rows: pd.DataFrame,
    generic_embedding: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, row in pair_rows.reset_index(drop=True).iterrows():
        positive = positive_embeddings[index]
        negative = negative_embeddings[index]
        cosine_pos_neg = cosine_similarity(positive, negative)
        rows.append(
            {
                "scenario": scenario,
                "pair_id": row["pair_id"],
                "positive_caption": row["positive_caption"],
                "negative_caption": row["negative_caption"],
                "cosine_text_positive_negative": cosine_pos_neg,
                "text_negation_separation": 1.0 - cosine_pos_neg,
                "cosine_generic_positive": cosine_similarity(generic_embedding, positive),
                "cosine_generic_negative": cosine_similarity(generic_embedding, negative),
            }
        )
    return pd.DataFrame(rows)


def encode_scenario_images(config: ScenarioConfig, annotations: list[dict], ranker, batch_size: int) -> tuple[list[dict], np.ndarray]:
    annotations_root = Path(config.annotations).parent
    embeddings = []
    rows = []
    for start in tqdm(range(0, len(annotations), batch_size), desc=f"image-embeddings:{config.scenario}"):
        batch = annotations[start : start + batch_size]
        images = [Image.open(resolve_image_path(annotations_root, row["image_path"])).convert("RGB") for row in batch]
        embeddings.append(ranker.encode_images(images))
        for row in batch:
            rows.append({"image_id": row["image_id"], "image_path": row["image_path"], "condition": row["condition"]})
        for image in images:
            image.close()
    if embeddings:
        return rows, np.vstack(embeddings)
    return rows, np.empty((0, 0))


def compute_image_embedding_metrics(scenario: str, image_rows: list[dict], embeddings: np.ndarray) -> dict:
    frame = pd.DataFrame(image_rows)
    if len(frame) == 0 or embeddings.size == 0:
        return empty_image_metrics(scenario)
    with_mask = frame["condition"].to_numpy() == "with_object"
    without_mask = frame["condition"].to_numpy() == "without_object"
    with_embeddings = embeddings[with_mask]
    without_embeddings = embeddings[without_mask]
    if len(with_embeddings) == 0 or len(without_embeddings) == 0:
        return empty_image_metrics(scenario, len(frame), int(with_mask.sum()), int(without_mask.sum()))
    centroid_with = with_embeddings.mean(axis=0)
    centroid_without = without_embeddings.mean(axis=0)
    centroid_cosine = cosine_similarity(centroid_with, centroid_without)
    sep = 1.0 - centroid_cosine
    var_with = within_condition_variance(with_embeddings, centroid_with)
    var_without = within_condition_variance(without_embeddings, centroid_without)
    denom = np.nanmean([var_with, var_without])
    return {
        "scenario": scenario,
        "n_images": int(len(frame)),
        "n_with_object": int(with_mask.sum()),
        "n_without_object": int(without_mask.sum()),
        "cosine_between_condition_centroids": centroid_cosine,
        "image_condition_separation": sep,
        "within_condition_variance_with": var_with,
        "within_condition_variance_without": var_without,
        "separation_ratio": float(sep / denom) if denom and not pd.isna(denom) else float("nan"),
    }


def empty_image_metrics(scenario: str, n_images: int = 0, n_with: int = 0, n_without: int = 0) -> dict:
    return {
        "scenario": scenario,
        "n_images": n_images,
        "n_with_object": n_with,
        "n_without_object": n_without,
        "cosine_between_condition_centroids": float("nan"),
        "image_condition_separation": float("nan"),
        "within_condition_variance_with": float("nan"),
        "within_condition_variance_without": float("nan"),
        "separation_ratio": float("nan"),
    }


def within_condition_variance(embeddings: np.ndarray, centroid: np.ndarray) -> float:
    if len(embeddings) == 0:
        return float("nan")
    centroid_norm = normalize_vector(centroid)
    return float(np.mean([1.0 - cosine_similarity(embedding, centroid_norm) for embedding in embeddings]))


def compute_image_pca(scenario: str, image_rows: list[dict], embeddings: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(image_rows)
    if len(frame) == 0 or embeddings.size == 0:
        return pd.DataFrame(columns=["scenario", "image_id", "image_path", "condition", "pc1", "pc2"])
    coords = pca_2d(embeddings)
    frame["scenario"] = scenario
    frame["pc1"] = coords[:, 0]
    frame["pc2"] = coords[:, 1]
    return frame[["scenario", "image_id", "image_path", "condition", "pc1", "pc2"]]


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=float)
    try:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=0).fit_transform(embeddings)
    except Exception:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:2].T
        coords = centered @ components
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        return coords[:, :2]


def compute_correlation_metrics(ci_frame: pd.DataFrame, text_frame: pd.DataFrame) -> pd.DataFrame:
    metric_values = ci_frame[ci_frame["metric"].isin(DELTA_METRICS)].pivot_table(
        index=["scenario", "pair_id"],
        columns="metric",
        values="value",
        aggfunc="first",
    )
    merged = metric_values.reset_index().merge(
        text_frame[["scenario", "pair_id", "text_negation_separation"]],
        on=["scenario", "pair_id"],
        how="inner",
    )
    rows = []
    for metric in DELTA_METRICS:
        rows.append(
            {
                "x": "text_negation_separation",
                "y": metric,
                "pearson_correlation": pearson_corr(merged["text_negation_separation"], merged[metric]),
                "n": int(merged[[metric, "text_negation_separation"]].dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def make_research_plots(
    ci_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
    text_frame: pd.DataFrame,
    image_frame: pd.DataFrame,
    plots_dir: Path,
) -> None:
    del delta_frame
    plot_ci_metric(ci_frame, "pairwise_accuracy", plots_dir / "accuracy_ci_by_scenario_pair.png", "Pairwise Accuracy by Scenario/Pair")
    plot_ci_metric(ci_frame, "mean_correct_margin", plots_dir / "margin_ci_by_scenario_pair.png", "Mean Correct Margin by Scenario/Pair")
    plot_ci_metric(ci_frame, "low_margin_rate_0_01", plots_dir / "low_margin_ci_by_scenario_pair.png", "Low-Margin Rate (< 0.01)")
    plot_text_separation(text_frame, plots_dir / "text_negation_separation_by_pair.png")
    plot_image_separation(image_frame, plots_dir / "image_condition_separation_by_scenario.png")


def plot_ci_metric(ci_frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    frame = ci_frame[ci_frame["metric"] == metric].copy()
    frame["label"] = frame["scenario"] + "\n" + frame["pair_id"]
    x = np.arange(len(frame))
    y = frame["value"].to_numpy(dtype=float)
    yerr = np.vstack([y - frame["ci_low"].to_numpy(dtype=float), frame["ci_high"].to_numpy(dtype=float) - y])
    fig, ax = plt.subplots(figsize=(max(9, len(frame) * 0.7), 4.8))
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#2f5d8c", ecolor="#777777", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_text_separation(text_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    frame = text_frame.copy()
    frame["label"] = frame["scenario"] + "\n" + frame["pair_id"]
    fig, ax = plt.subplots(figsize=(max(9, len(frame) * 0.7), 4.8))
    ax.bar(np.arange(len(frame)), frame["text_negation_separation"], color="#7b5fa8")
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["label"], rotation=45, ha="right")
    ax.set_title("Text Negation Separation by Pair")
    ax.set_ylabel("1 - cosine(positive, negative)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_separation(image_frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(image_frame["scenario"], image_frame["image_condition_separation"], color="#4d8b57")
    ax.set_title("Image Condition Separation by Scenario")
    ax.set_ylabel("1 - cosine(condition centroids)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_image_pca(frame: pd.DataFrame, output_path: Path, scenario: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"with_object": "#2f6da3", "without_object": "#c46a36"}
    for condition, group in frame.groupby("condition"):
        ax.scatter(group["pc1"], group["pc2"], label=condition, s=28, alpha=0.8, color=colors.get(condition, "#555555"))
    ax.set_title(f"Image Embedding PCA: {scenario}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_research_report(
    output_path: str | Path,
    scenarios: list[str],
    model_name: str,
    device: str,
    ci_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
    text_frame: pd.DataFrame,
    image_frame: pd.DataFrame,
    correlation_frame: pd.DataFrame,
) -> None:
    lines = [
        "# ContextNegBench-Lite: Embedding and Uncertainty Analysis",
        "",
        "## Goal",
        "",
        "This analysis asks whether CLIP-like models treat visual absence as a stable semantic constraint, or whether performance depends on object salience, visual ambiguity, and prompt formulation.",
        "",
        "## Scenarios",
        "",
        f"- Scenarios: {', '.join(scenarios)}",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        "- Language: English only",
        "",
        "## Primary Metrics",
        "",
        "- `pairwise_accuracy`: fraction of minimal-pair decisions where the condition-aware caption wins.",
        "- `mean_correct_margin`: average score gap in the correct direction.",
        "- `low_margin_rate_0_01`: fraction of cases with margin below 0.01.",
        "- `false_absence_preference_rate`: with-object cases where the negative caption beats the positive caption.",
        "- `false_presence_preference_rate`: without-object cases where the positive caption beats the negative caption.",
        "- `text_negation_separation`: `1 - cosine(positive_caption, negative_caption)`.",
        "- `image_condition_separation`: `1 - cosine(with_object_centroid, without_object_centroid)`.",
        "",
        "## Pairwise Results with Confidence Intervals",
        "",
        markdown_table(ci_frame),
        "",
        "## Scenario Comparison",
        "",
        "Scenario deltas are computed as `street_car - kitchen_table`. Positive deltas for accuracy and margin favor street/car; negative deltas for low-margin rate mean street/car has fewer near-ties.",
        "",
        markdown_table(delta_frame),
        "",
        "## Text Embedding Analysis",
        "",
        "Text negation separation helps diagnose whether a prompt formulation places positive and negative captions farther apart in CLIP text space.",
        "",
        markdown_table(text_frame),
        "",
        "## Image Embedding Analysis",
        "",
        "Image condition separation measures whether with-object and without-object images form separated centroids in CLIP image space. This helps distinguish prompt failures from visually ambiguous or weakly separated image sets.",
        "",
        markdown_table(image_frame),
        "",
        "## PCA Visualization Summary",
        "",
        "The PCA plots provide a two-dimensional view of the image embeddings by condition. They are diagnostic visualizations only; PCA should not be treated as a definitive separability test.",
        "",
        "## Key Findings",
        "",
        "- We are not claiming that CLIP does or does not understand negation in an absolute sense.",
        "- The results suggest that negation stability depends on scenario, object salience, visual ambiguity, and prompt formulation.",
        "- In this benchmark design, street/car is a cleaner expected-absence scenario than kitchen/table and is expected to be more robust because streets remain streets without visible cars.",
        "- Kitchen/table can show false-absence failures, especially with direct `base_with_without` wording, because table-like surfaces and kitchen layouts are visually ambiguous.",
        "- Text negation separation can help identify formulations where positive and negative captions are more distinct in text embedding space.",
        "- Image condition separation helps identify whether the with/without split is visually separated in CLIP image space.",
        "",
        "## Limitations",
        "",
        "- The dataset is manually/web collected and may contain ambiguity, duplicates, crop artifacts, watermarks, or noisy labels.",
        "- This analysis uses a single model: OpenCLIP ViT-B/32.",
        "- The analysis is English only.",
        "- Synthetic or small real-image diagnostics should not be interpreted as proof of real-world general VLM understanding.",
        "- Embedding geometry is model-specific and may change with architecture, pretraining data, or prompt templates.",
        "",
        "## Correlations",
        "",
        markdown_table(correlation_frame),
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


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


def mean_bool(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).astype(float).mean())


def mean_series(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).mean())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def pearson_corr(x, y) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 2:
        return float("nan")
    if frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="pearson"))


def stable_seed(seed: int, *parts: str) -> int:
    total = seed
    for part in parts:
        for char in str(part):
            total = (total * 33 + ord(char)) % (2**32 - 1)
    return total
