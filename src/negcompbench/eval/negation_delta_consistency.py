from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from negcompbench.eval.context_neg_research_analysis import cosine_similarity, markdown_table, stable_seed
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir


OBJECTS = [
    "dog",
    "cat",
    "car",
    "table",
    "chair",
    "person",
    "bicycle",
    "mirror",
    "cup",
    "bottle",
    "book",
    "phone",
    "tree",
    "flower",
    "horse",
    "bird",
]


@dataclass(frozen=True)
class NegationPairGroup:
    pair_group: str
    positive_template: str
    negative_template: str


PAIR_GROUPS = [
    NegationPairGroup("bare_no", "{object}", "no {object}"),
    NegationPairGroup("article_no", "a {object}", "no {object}"),
    NegationPairGroup("image_with_no", "an image with a {object}", "an image with no {object}"),
    NegationPairGroup("photo_with_no", "a photo with a {object}", "a photo with no {object}"),
    NegationPairGroup("visible_no_visible", "a visible {object}", "no visible {object}"),
]


def run_negation_delta_consistency_analysis(
    model_name: str,
    output_dir: str | Path,
    seed: int = 42,
    device: str = "auto",
) -> None:
    output = ensure_dir(output_dir)
    plots = ensure_dir(output / "plots")
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)
    if not hasattr(ranker, "encode_texts"):
        raise TypeError("Negation delta consistency analysis requires encode_texts support.")

    phrases = unique_phrases(OBJECTS, PAIR_GROUPS)
    embeddings = ranker.encode_texts(phrases)
    embedding_map = {phrase: embedding for phrase, embedding in zip(phrases, embeddings)}

    object_metrics = compute_object_delta_metrics(OBJECTS, PAIR_GROUPS, embedding_map)
    direction_summary_rows = []
    pca_rows = []
    projection_frames = []
    baseline_rows = []

    for group in PAIR_GROUPS:
        group_metrics = object_metrics[object_metrics["pair_group"] == group.pair_group].copy()
        delta_units = np.vstack(group_metrics["delta_unit"].to_list())
        similarity = delta_similarity_matrix(group_metrics["object"].to_list(), delta_units)
        similarity.to_csv(output / f"delta_similarity_{group.pair_group}.csv", index=True)
        direction_summary_rows.append(summarize_delta_similarity(group.pair_group, similarity))
        pca_rows.extend(pca_explained_variance_rows(group.pair_group, delta_units))
        projection_frames.append(add_axis_projection(group_metrics, delta_units))
        baseline_rows.extend(
            baseline_comparison_rows(
                group.pair_group,
                group_metrics,
                embedding_map,
                seed=stable_seed(seed, group.pair_group, "baseline"),
            )
        )

    projection_frame = pd.concat(projection_frames, ignore_index=True)
    output_object_metrics = projection_frame.drop(columns=["delta", "delta_unit"])
    direction_summary = pd.DataFrame(direction_summary_rows)
    pca_frame = pd.DataFrame(pca_rows)
    baseline_frame = pd.DataFrame(baseline_rows)

    output_object_metrics.to_csv(output / "object_delta_metrics.csv", index=False)
    direction_summary.to_csv(output / "delta_direction_summary.csv", index=False)
    pca_frame.to_csv(output / "pca_explained_variance.csv", index=False)
    projection_frame.drop(columns=["delta", "delta_unit"]).to_csv(output / "axis_projection_by_object.csv", index=False)
    baseline_frame.to_csv(output / "baseline_comparison.csv", index=False)

    make_negation_delta_plots(output_object_metrics, direction_summary, pca_frame, baseline_frame, plots)
    write_negation_delta_report(
        output / "report.md",
        model_name=model_name,
        device=used_device,
        direction_summary=direction_summary,
        pca_frame=pca_frame,
        baseline_frame=baseline_frame,
    )


def unique_phrases(objects: list[str], pair_groups: list[NegationPairGroup]) -> list[str]:
    phrases: list[str] = []
    for group in pair_groups:
        for object_name in objects:
            for phrase in [format_phrase(group.positive_template, object_name), format_phrase(group.negative_template, object_name)]:
                if phrase not in phrases:
                    phrases.append(phrase)
    return phrases


def format_phrase(template: str, object_name: str) -> str:
    article = "an" if object_name[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return template.format(object=object_name, article=article)


def compute_object_delta_metrics(objects: list[str], pair_groups: list[NegationPairGroup], embedding_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for group in pair_groups:
        for object_name in objects:
            positive_phrase = format_phrase(group.positive_template, object_name)
            negative_phrase = format_phrase(group.negative_template, object_name)
            positive = embedding_map[positive_phrase]
            negative = embedding_map[negative_phrase]
            delta = negative - positive
            delta_norm = float(np.linalg.norm(delta))
            delta_unit = normalize(delta)
            rows.append(
                {
                    "object": object_name,
                    "pair_group": group.pair_group,
                    "positive_phrase": positive_phrase,
                    "negative_phrase": negative_phrase,
                    "cosine_positive_negative": cosine_similarity(positive, negative),
                    "delta_norm": delta_norm,
                    "delta": delta,
                    "delta_unit": delta_unit,
                }
            )
    return pd.DataFrame(rows)


def delta_similarity_matrix(objects: list[str], delta_units: np.ndarray) -> pd.DataFrame:
    matrix = delta_units @ delta_units.T
    return pd.DataFrame(matrix, index=objects, columns=objects)


def summarize_delta_similarity(pair_group: str, similarity: pd.DataFrame) -> dict:
    values = off_diagonal_values(similarity.to_numpy(dtype=float))
    return {
        "pair_group": pair_group,
        "mean_delta_direction_similarity": float(np.mean(values)),
        "median_delta_direction_similarity": float(np.median(values)),
        "std_delta_direction_similarity": float(np.std(values)),
        "min_delta_direction_similarity": float(np.min(values)),
        "max_delta_direction_similarity": float(np.max(values)),
        "n_objects": int(len(similarity)),
        "n_pairs": int(len(values)),
    }


def off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=float)
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def pca_explained_variance_rows(pair_group: str, delta_units: np.ndarray) -> list[dict]:
    ratios = pca_explained_variance(delta_units)
    wanted = [1, 2, 3, 5]
    return [
        {
            "pair_group": pair_group,
            "component": f"PC{component}",
            "explained_variance_ratio": float(ratios[component - 1]) if component <= len(ratios) else float("nan"),
        }
        for component in wanted
    ]


def pca_explained_variance(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) < 2:
        return np.array([], dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variances = (singular_values**2) / max(len(matrix) - 1, 1)
    total = variances.sum()
    if total == 0:
        return np.zeros_like(variances)
    return variances / total


def add_axis_projection(group_metrics: pd.DataFrame, delta_units: np.ndarray) -> pd.DataFrame:
    output = group_metrics.copy()
    mean_axis = normalize(delta_units.mean(axis=0))
    projections = delta_units @ mean_axis
    residuals = [float(np.linalg.norm(delta_unit - projection * mean_axis)) for delta_unit, projection in zip(delta_units, projections)]
    output["projection_on_mean_axis"] = projections.astype(float)
    output["residual_norm"] = residuals
    return output


def baseline_comparison_rows(pair_group: str, group_metrics: pd.DataFrame, embedding_map: dict[str, np.ndarray], seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    objects = group_metrics["object"].to_list()
    real_delta_units = np.vstack(group_metrics["delta_unit"].to_list())
    real_mean = mean_pairwise_cosine(real_delta_units)
    positive_embeddings = {
        row["object"]: embedding_map[row["positive_phrase"]]
        for _, row in group_metrics.iterrows()
    }
    negative_embeddings = {
        row["object"]: embedding_map[row["negative_phrase"]]
        for _, row in group_metrics.iterrows()
    }
    object_object_units = []
    mismatched_no_units = []
    for object_name in objects:
        other = random_other_object(objects, object_name, rng)
        object_object_units.append(normalize(positive_embeddings[other] - positive_embeddings[object_name]))
        mismatched_no_units.append(normalize(negative_embeddings[object_name] - positive_embeddings[other]))
    object_mean = mean_pairwise_cosine(np.vstack(object_object_units))
    mismatched_mean = mean_pairwise_cosine(np.vstack(mismatched_no_units))
    return [
        {
            "pair_group": pair_group,
            "baseline_type": "real_negation_delta",
            "mean_pairwise_cosine": real_mean,
            "delta_vs_real": 0.0,
        },
        {
            "pair_group": pair_group,
            "baseline_type": "object_object_delta",
            "mean_pairwise_cosine": object_mean,
            "delta_vs_real": real_mean - object_mean,
        },
        {
            "pair_group": pair_group,
            "baseline_type": "mismatched_no_delta",
            "mean_pairwise_cosine": mismatched_mean,
            "delta_vs_real": real_mean - mismatched_mean,
        },
    ]


def random_other_object(objects: list[str], object_name: str, rng: np.random.Generator) -> str:
    choices = [candidate for candidate in objects if candidate != object_name]
    return str(rng.choice(choices))


def mean_pairwise_cosine(delta_units: np.ndarray) -> float:
    matrix = delta_units @ delta_units.T
    values = off_diagonal_values(matrix)
    return float(np.mean(values)) if len(values) else float("nan")


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def make_negation_delta_plots(object_metrics: pd.DataFrame, direction_summary: pd.DataFrame, pca_frame: pd.DataFrame, baseline_frame: pd.DataFrame, plots_dir: Path) -> None:
    plot_bar(
        direction_summary,
        "pair_group",
        "mean_delta_direction_similarity",
        plots_dir / "mean_delta_direction_similarity_by_pair_group.png",
        "Mean Delta Direction Similarity",
    )
    pc1 = pca_frame[pca_frame["component"] == "PC1"]
    plot_bar(
        pc1,
        "pair_group",
        "explained_variance_ratio",
        plots_dir / "pca_pc1_explained_variance_by_pair_group.png",
        "PC1 Explained Variance",
    )
    plot_grouped_object_metric(
        object_metrics,
        "projection_on_mean_axis",
        plots_dir / "projection_on_mean_axis_by_object.png",
        "Projection on Mean Negation Axis",
    )
    plot_grouped_object_metric(object_metrics, "delta_norm", plots_dir / "delta_norm_by_object.png", "Delta Norm by Object")
    plot_baseline_alignment(baseline_frame, plots_dir / "real_vs_random_delta_alignment.png")


def plot_bar(frame: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame[x_col], frame[y_col], color="#4f78a8")
    ax.set_title(title)
    ax.set_ylabel(y_col)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_grouped_object_metric(frame: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    pivot = frame.pivot(index="object", columns="pair_group", values=metric)
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("object")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_baseline_alignment(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = frame.pivot(index="pair_group", columns="baseline_type", values="mean_pairwise_cosine")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Real vs Random/Object Delta Alignment")
    ax.set_ylabel("mean pairwise cosine")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_negation_delta_report(
    output_path: str | Path,
    model_name: str,
    device: str,
    direction_summary: pd.DataFrame,
    pca_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
) -> None:
    pc1 = pca_frame[pca_frame["component"] == "PC1"][["pair_group", "explained_variance_ratio"]].rename(
        columns={"explained_variance_ratio": "pc1_explained_variance"}
    )
    compact = direction_summary.merge(pc1, on="pair_group", how="left")
    lines = [
        "# Negation Delta Consistency Analysis",
        "",
        "## 1. Goal",
        "",
        'Cosine similarity between phrases such as "dog" and "no dog" can remain high because object semantics dominate. This analysis instead studies the delta vector `embedding(no X) - embedding(X)` to test whether negation behaves like a stable transformation across objects.',
        "",
        "## 2. Hypothesis",
        "",
        "In an idealized representation, `X` and `no X` could share most object semantics but differ along a presence/absence polarity direction. If so, deltas across objects should align.",
        "",
        "## 3. Metrics",
        "",
        "- Delta direction similarity: pairwise cosine among normalized delta vectors.",
        "- PCA explained variance: whether one component captures a shared negation axis.",
        "- Projection on mean negation axis: how much each object follows the shared direction.",
        "- Random/object delta baseline: whether real negation deltas are more aligned than unrelated object changes.",
        "",
        "## 4. Results",
        "",
        markdown_table(compact),
        "",
        "## Baseline Comparison",
        "",
        markdown_table(baseline_frame),
        "",
        "## 5. Interpretation",
        "",
        "High mean delta similarity and high PC1 explained variance would support a shared negation direction. If real negation deltas are not more aligned than object-object or mismatched baselines, then `no X - X` is likely object-dependent or noisy rather than a clean text-space operation. Comparing bare phrases against full phrase contexts tests whether templates such as `an image with no X` produce a more stable transformation than `no X` alone.",
        "",
        "## 6. Limitations",
        "",
        "- One model.",
        "- Text-only embedding analysis.",
        "- Limited object list.",
        "- No causal proof of model understanding.",
        "",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
