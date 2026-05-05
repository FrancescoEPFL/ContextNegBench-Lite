from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from negcompbench.eval.context_neg_research_analysis import cosine_similarity, markdown_table, stable_seed
from negcompbench.eval.negation_delta_consistency import OBJECTS, normalize, off_diagonal_values, pca_explained_variance
from negcompbench.eval.run_eval import build_ranker
from negcompbench.utils.io import ensure_dir


OBJECT_PAIRS = [
    ("dog", "cat"),
    ("car", "bicycle"),
    ("table", "chair"),
    ("cup", "bottle"),
    ("book", "phone"),
    ("horse", "bird"),
    ("tree", "flower"),
    ("person", "dog"),
]


@dataclass(frozen=True)
class UnaryOperator:
    operator: str
    base_template: str
    operator_template: str


UNARY_OPERATORS = [
    UnaryOperator("no", "a {object}", "no {object}"),
    UnaryOperator("without", "a {object}", "without {object}"),
    UnaryOperator("with_no", "with a {object}", "with no {object}"),
    UnaryOperator("without_any", "with a {object}", "without any {object}"),
    UnaryOperator("no_visible", "a visible {object}", "no visible {object}"),
    UnaryOperator("absent", "{object} present", "{object} absent"),
    UnaryOperator("not_a", "a {object}", "not a {object}"),
]


UNARY_CONNECTOR_TEMPLATES = {
    "base_a": "a {object}",
    "base_with_a": "with a {object}",
    "base_visible": "a visible {object}",
    "present": "{object} present",
    "no": "no {object}",
    "without": "without {object}",
    "with_no": "with no {object}",
    "without_any": "without any {object}",
    "no_visible": "no visible {object}",
    "absent": "{object} absent",
    "absence_of": "absence of {object}",
    "not_a": "not a {object}",
    "not_the": "not the {object}",
}


PAIRWISE_DISTANCE_SPECS = [
    ("base_a", "no"),
    ("base_a", "without"),
    ("base_a", "with_no"),
    ("base_a", "no_visible"),
    ("base_a", "not_a"),
    ("base_a", "absent"),
    ("no", "without"),
    ("no", "with_no"),
    ("no", "no_visible"),
]


BINARY_CONNECTORS = {
    "and": "{object1} and {object2}",
    "or": "{object1} or {object2}",
    "but_not": "{object1} but not {object2}",
    "without": "{object1} without {object2}",
    "only": "only {object1}",
    "neither_nor": "neither {object1} nor {object2}",
}


GENERIC_ABSENCE_PHRASES = ["nothing", "empty", "no object", "without objects", "no animals", "empty scene"]
GENERIC_SCENE_PHRASES = ["an image", "a photo", "a scene"]
NEIGHBOR_TARGET_CONNECTORS = ["no", "without", "with_no", "no_visible", "not_a"]


def run_logical_connector_embedding_analysis(
    model_name: str,
    output_dir: str | Path,
    seed: int = 42,
    device: str = "auto",
) -> None:
    output = ensure_dir(output_dir)
    plots_dir = ensure_dir(output / "plots")
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)
    if not hasattr(ranker, "encode_texts"):
        raise TypeError("Logical connector analysis requires encode_texts support.")

    phrases = build_phrase_inventory(OBJECTS, OBJECT_PAIRS)
    embeddings = ranker.encode_texts(phrases)
    embedding_map = {phrase: embedding for phrase, embedding in zip(phrases, embeddings)}

    pairwise = compute_pairwise_distance_metrics(OBJECTS, embedding_map)
    operator_summary, operator_pca, axis_projection = compute_operator_delta_outputs(OBJECTS, embedding_map, output)
    baseline = compute_operator_baseline_comparison(OBJECTS, UNARY_OPERATORS, embedding_map, seed=seed)
    neighbors = compute_nearest_neighbors(OBJECTS, embedding_map)
    dominance = compute_object_dominance_index(OBJECTS, UNARY_OPERATORS, embedding_map)
    binary = compute_binary_connector_metrics(OBJECT_PAIRS, embedding_map)

    pairwise.to_csv(output / "pairwise_distance_metrics.csv", index=False)
    operator_summary.to_csv(output / "operator_delta_summary.csv", index=False)
    operator_pca.to_csv(output / "operator_pca_explained_variance.csv", index=False)
    axis_projection.to_csv(output / "operator_axis_projection.csv", index=False)
    baseline.to_csv(output / "operator_baseline_comparison.csv", index=False)
    neighbors.to_csv(output / "nearest_neighbors.csv", index=False)
    dominance.to_csv(output / "object_dominance_index.csv", index=False)
    binary.to_csv(output / "binary_connector_metrics.csv", index=False)

    make_logical_connector_plots(operator_summary, operator_pca, dominance, pairwise, neighbors, binary, plots_dir)
    write_logical_connector_report(
        output / "report.md",
        model_name=model_name,
        device=used_device,
        operator_summary=operator_summary,
        operator_pca=operator_pca,
        baseline=baseline,
        dominance=dominance,
        binary=binary,
    )


def build_phrase_inventory(objects: list[str], object_pairs: list[tuple[str, str]]) -> list[str]:
    phrases: list[str] = []
    for object_name in objects:
        for template in UNARY_CONNECTOR_TEMPLATES.values():
            append_unique(phrases, format_object_phrase(template, object_name))
        append_unique(phrases, object_name)
    for phrase in GENERIC_ABSENCE_PHRASES + GENERIC_SCENE_PHRASES:
        append_unique(phrases, phrase)
    for object1, object2 in object_pairs:
        for template in BINARY_CONNECTORS.values():
            append_unique(phrases, format_binary_phrase(template, object1, object2))
        for phrase in [object1, object2, f"no {object1}", f"no {object2}", f"{object1} and {object2}", f"only {object1}"]:
            append_unique(phrases, phrase)
    return phrases


def append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def format_object_phrase(template: str, object_name: str) -> str:
    return template.format(object=object_name)


def format_binary_phrase(template: str, object1: str, object2: str) -> str:
    return template.format(object1=object1, object2=object2)


def vector_distance_metrics(embedding_a: np.ndarray, embedding_b: np.ndarray) -> dict[str, float]:
    cosine = cosine_similarity(embedding_a, embedding_b)
    delta = embedding_b - embedding_a
    euclidean = float(np.linalg.norm(delta))
    return {
        "cosine_similarity": cosine,
        "cosine_distance": 1.0 - cosine,
        "euclidean_distance": euclidean,
        "squared_euclidean_distance": float(euclidean**2),
        "dot_product": float(np.dot(embedding_a, embedding_b)),
        "angular_distance_radians": float(np.arccos(np.clip(cosine, -1.0, 1.0))),
        "delta_norm": euclidean,
    }


def compute_pairwise_distance_metrics(objects: list[str], embedding_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for object_name in objects:
        for connector_a, connector_b in PAIRWISE_DISTANCE_SPECS:
            phrase_a = format_object_phrase(UNARY_CONNECTOR_TEMPLATES[connector_a], object_name)
            phrase_b = format_object_phrase(UNARY_CONNECTOR_TEMPLATES[connector_b], object_name)
            rows.append(
                {
                    "object": object_name,
                    "phrase_a": phrase_a,
                    "phrase_b": phrase_b,
                    "connector_a": connector_a,
                    "connector_b": connector_b,
                    **vector_distance_metrics(embedding_map[phrase_a], embedding_map[phrase_b]),
                }
            )
    return pd.DataFrame(rows)


def compute_operator_delta_outputs(
    objects: list[str], embedding_map: dict[str, np.ndarray], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pca_rows = []
    projection_rows = []
    for operator in UNARY_OPERATORS:
        rows, delta_units = operator_delta_rows(objects, operator, embedding_map)
        similarity = pd.DataFrame(delta_units @ delta_units.T, index=objects, columns=objects)
        similarity.to_csv(output_dir / f"operator_delta_similarity_{operator.operator}.csv", index=True)
        values = off_diagonal_values(similarity.to_numpy(dtype=float))
        summary_rows.append(
            {
                "operator": operator.operator,
                "mean_delta_direction_similarity": float(np.mean(values)),
                "median_delta_direction_similarity": float(np.median(values)),
                "std_delta_direction_similarity": float(np.std(values)),
                "min_delta_direction_similarity": float(np.min(values)),
                "max_delta_direction_similarity": float(np.max(values)),
                "n_objects": int(len(objects)),
                "n_pairs": int(len(values)),
            }
        )
        ratios = pca_explained_variance(delta_units)
        for component in [1, 2, 3]:
            pca_rows.append(
                {
                    "operator": operator.operator,
                    "component": f"PC{component}",
                    "explained_variance_ratio": float(ratios[component - 1]) if component <= len(ratios) else float("nan"),
                }
            )
        mean_axis = normalize(delta_units.mean(axis=0))
        projections = delta_units @ mean_axis
        for row, projection, delta_unit in zip(rows, projections, delta_units):
            row["projection_on_mean_axis"] = float(projection)
            row["residual_norm"] = float(np.linalg.norm(delta_unit - projection * mean_axis))
            projection_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(pca_rows), pd.DataFrame(projection_rows)


def operator_delta_rows(objects: list[str], operator: UnaryOperator, embedding_map: dict[str, np.ndarray]) -> tuple[list[dict], np.ndarray]:
    rows = []
    delta_units = []
    for object_name in objects:
        base_phrase = format_object_phrase(operator.base_template, object_name)
        operator_phrase = format_object_phrase(operator.operator_template, object_name)
        base = embedding_map[base_phrase]
        transformed = embedding_map[operator_phrase]
        delta = transformed - base
        delta_unit = normalize(delta)
        delta_units.append(delta_unit)
        rows.append(
            {
                "object": object_name,
                "operator": operator.operator,
                "base_phrase": base_phrase,
                "operator_phrase": operator_phrase,
                "cosine_base_operator": cosine_similarity(base, transformed),
                "delta_norm": float(np.linalg.norm(delta)),
            }
        )
    return rows, np.vstack(delta_units)


def compute_operator_baseline_comparison(
    objects: list[str], operators: list[UnaryOperator], embedding_map: dict[str, np.ndarray], seed: int = 42
) -> pd.DataFrame:
    rows = []
    for operator in operators:
        _, real_units = operator_delta_rows(objects, operator, embedding_map)
        real_mean = mean_pairwise_cosine(real_units)
        object_units = []
        mismatch_units = []
        rng = np.random.default_rng(stable_seed(seed, operator.operator, "baseline"))
        for object_name in objects:
            other = random_other(objects, object_name, rng)
            base_x = embedding_map[format_object_phrase("a {object}", object_name)]
            base_y = embedding_map[format_object_phrase("a {object}", other)]
            operator_x = embedding_map[format_object_phrase(operator.operator_template, object_name)]
            object_units.append(normalize(base_y - base_x))
            mismatch_units.append(normalize(operator_x - base_y))
        object_mean = mean_pairwise_cosine(np.vstack(object_units))
        mismatch_mean = mean_pairwise_cosine(np.vstack(mismatch_units))
        rows.extend(
            [
                {
                    "operator": operator.operator,
                    "baseline_type": "real_operator_delta",
                    "mean_pairwise_cosine": real_mean,
                    "real_minus_baseline": 0.0,
                },
                {
                    "operator": operator.operator,
                    "baseline_type": "object_object_delta",
                    "mean_pairwise_cosine": object_mean,
                    "real_minus_baseline": real_mean - object_mean,
                },
                {
                    "operator": operator.operator,
                    "baseline_type": "mismatched_operator_delta",
                    "mean_pairwise_cosine": mismatch_mean,
                    "real_minus_baseline": real_mean - mismatch_mean,
                },
            ]
        )
    return pd.DataFrame(rows)


def mean_pairwise_cosine(delta_units: np.ndarray) -> float:
    return float(np.mean(off_diagonal_values(delta_units @ delta_units.T)))


def random_other(objects: list[str], object_name: str, rng: np.random.Generator) -> str:
    choices = [candidate for candidate in objects if candidate != object_name]
    return str(rng.choice(choices))


def compute_nearest_neighbors(objects: list[str], embedding_map: dict[str, np.ndarray], top_k: int = 10) -> pd.DataFrame:
    vocab = controlled_neighbor_vocabulary(objects)
    matrix = np.vstack([embedding_map[phrase] for phrase in vocab])
    rows = []
    targets = []
    for object_name in objects:
        for connector in NEIGHBOR_TARGET_CONNECTORS:
            targets.append(format_object_phrase(UNARY_CONNECTOR_TEMPLATES[connector], object_name))
    for target in targets:
        target_embedding = embedding_map[target]
        cosine_scores = matrix @ target_embedding
        euclidean_scores = np.linalg.norm(matrix - target_embedding, axis=1)
        added = 0
        for rank, idx in enumerate(np.argsort(-cosine_scores)[: top_k + 1], start=1):
            phrase = vocab[int(idx)]
            if phrase == target:
                continue
            added += 1
            rows.append(
                {
                    "target_phrase": target,
                    "metric": "cosine_similarity",
                    "rank": added,
                    "neighbor_phrase": phrase,
                    "neighbor_category": neighbor_category(phrase),
                    "score": float(cosine_scores[int(idx)]),
                }
            )
            if added >= top_k:
                break
        added = 0
        for rank, idx in enumerate(np.argsort(euclidean_scores)[: top_k + 1], start=1):
            phrase = vocab[int(idx)]
            if phrase == target:
                continue
            added += 1
            rows.append(
                {
                    "target_phrase": target,
                    "metric": "euclidean_distance",
                    "rank": added,
                    "neighbor_phrase": phrase,
                    "neighbor_category": neighbor_category(phrase),
                    "score": float(euclidean_scores[int(idx)]),
                }
            )
            if added >= top_k:
                break
    return pd.DataFrame(rows)


def controlled_neighbor_vocabulary(objects: list[str]) -> list[str]:
    phrases = []
    for object_name in objects:
        for template in ["a {object}", "no {object}", "without {object}", "with no {object}"]:
            append_unique(phrases, format_object_phrase(template, object_name))
    for phrase in GENERIC_ABSENCE_PHRASES + GENERIC_SCENE_PHRASES:
        append_unique(phrases, phrase)
    return phrases


def neighbor_category(phrase: str) -> str:
    if phrase in GENERIC_ABSENCE_PHRASES:
        return "generic_absence"
    if phrase in GENERIC_SCENE_PHRASES:
        return "generic_scene"
    if phrase.startswith("no "):
        return "no"
    if phrase.startswith("without "):
        return "without"
    if phrase.startswith("with no "):
        return "with_no"
    if phrase.startswith("a "):
        return "object"
    return "other"


def compute_object_dominance_index(
    objects: list[str], operators: list[UnaryOperator], embedding_map: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for operator in operators:
        operator_embeddings = {
            object_name: embedding_map[format_object_phrase(operator.operator_template, object_name)] for object_name in objects
        }
        base_embeddings = {object_name: embedding_map[format_object_phrase("a {object}", object_name)] for object_name in objects}
        for object_name in objects:
            own_similarity = cosine_similarity(operator_embeddings[object_name], base_embeddings[object_name])
            other_similarities = [
                cosine_similarity(operator_embeddings[object_name], operator_embeddings[other]) for other in objects if other != object_name
            ]
            other_mean = float(np.mean(other_similarities))
            rows.append(
                {
                    "object": object_name,
                    "operator": operator.operator,
                    "similarity_operator_to_base_same_object": own_similarity,
                    "mean_similarity_to_other_operator_objects": other_mean,
                    "object_dominance_index": own_similarity - other_mean,
                }
            )
    return pd.DataFrame(rows)


def compute_binary_connector_metrics(object_pairs: list[tuple[str, str]], embedding_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for object1, object2 in object_pairs:
        references = {
            "object1": object1,
            "object2": object2,
            "no_object1": f"no {object1}",
            "no_object2": f"no {object2}",
            "and": f"{object1} and {object2}",
            "only_object1": f"only {object1}",
        }
        for connector, template in BINARY_CONNECTORS.items():
            phrase = format_binary_phrase(template, object1, object2)
            for reference_name, reference_phrase in references.items():
                rows.append(
                    {
                        "object1": object1,
                        "object2": object2,
                        "connector": connector,
                        "phrase": phrase,
                        "reference": reference_name,
                        "reference_phrase": reference_phrase,
                        **vector_distance_metrics(embedding_map[phrase], embedding_map[reference_phrase]),
                    }
                )
    return pd.DataFrame(rows)


def make_logical_connector_plots(
    operator_summary: pd.DataFrame,
    operator_pca: pd.DataFrame,
    dominance: pd.DataFrame,
    pairwise: pd.DataFrame,
    neighbors: pd.DataFrame,
    binary: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plot_operator_bar(
        operator_summary, "mean_delta_direction_similarity", plots_dir / "operator_delta_consistency.png", "Operator Delta Consistency"
    )
    pc1 = operator_pca[operator_pca["component"] == "PC1"]
    plot_operator_bar(pc1, "explained_variance_ratio", plots_dir / "operator_pc1_explained_variance.png", "Operator PC1 Explained Variance")
    plot_object_dominance(dominance, plots_dir / "object_dominance_by_operator.png")
    plot_distance_metrics(pairwise, plots_dir / "distance_metrics_by_operator.png")
    plot_neighbor_categories(neighbors, plots_dir / "nearest_neighbor_category_distribution.png")
    plot_binary_similarity(binary, plots_dir / "binary_connector_similarity.png")


def plot_operator_bar(frame: pd.DataFrame, value_col: str, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    x_col = "operator" if "operator" in frame.columns else "pair_group"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame[x_col], frame[value_col], color="#4f78a8")
    ax.set_title(title)
    ax.set_ylabel(value_col)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_object_dominance(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = frame.groupby("operator", sort=False)["object_dominance_index"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means.plot(kind="bar", ax=ax, color="#6f8f4f")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_title("Object Dominance by Operator")
    ax.set_ylabel("object_dominance_index")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_distance_metrics(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = frame.groupby("connector_b", sort=False)[["cosine_distance", "euclidean_distance", "angular_distance_radians"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    means.plot(kind="bar", ax=ax)
    ax.set_title("Distance Metrics by Connector")
    ax.set_ylabel("distance")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_neighbor_categories(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = frame[frame["metric"] == "cosine_similarity"].pivot_table(
        index="target_phrase", columns="neighbor_category", values="rank", aggfunc="count", fill_value=0
    )
    totals = counts.sum(axis=0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    totals.plot(kind="bar", ax=ax, color="#8a6f3d")
    ax.set_title("Nearest Neighbor Category Distribution")
    ax.set_ylabel("top-10 neighbor count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_binary_similarity(frame: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    focus = frame[frame["reference"].isin(["object1", "object2", "no_object1", "no_object2", "and", "only_object1"])]
    means = focus.groupby("connector", sort=False)["cosine_similarity"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    means.plot(kind="bar", ax=ax, color="#9a6748")
    ax.set_title("Binary Connector Similarity")
    ax.set_ylabel("mean cosine to references")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_logical_connector_report(
    output_path: str | Path,
    model_name: str,
    device: str,
    operator_summary: pd.DataFrame,
    operator_pca: pd.DataFrame,
    baseline: pd.DataFrame,
    dominance: pd.DataFrame,
    binary: pd.DataFrame,
) -> None:
    pc1 = operator_pca[operator_pca["component"] == "PC1"][["operator", "explained_variance_ratio"]].rename(
        columns={"explained_variance_ratio": "pc1_explained_variance"}
    )
    compact = operator_summary.merge(pc1, on="operator", how="left")
    dominance_summary = dominance.groupby("operator", sort=False)["object_dominance_index"].mean().reset_index()
    binary_summary = binary.groupby(["connector", "reference"], sort=False)["cosine_similarity"].mean().reset_index()
    lines = [
        "# Logical Connector Embedding Analysis",
        "",
        "## 1. Goal",
        "",
        "Cosine similarity alone can hide meaningful differences, so this analysis compares multiple vector-space metrics and delta consistency for negation, presence, conjunction, disjunction, and exclusion templates.",
        "",
        "## 2. Why Cosine Alone Is Insufficient",
        "",
        'High cosine between "dog" and "no dog" may occur because object semantics dominate. We therefore analyze deltas, PCA, axis projections, euclidean and angular distances, and nearest neighbors.',
        "",
        "## 3. Unary Negation Operators",
        "",
        markdown_table(compact),
        "",
        "The operator with the highest mean delta direction similarity is the most consistent across objects in this diagnostic. `not a X` should be interpreted separately from absence negation because it can encode category rejection rather than visual absence.",
        "",
        "## Baselines",
        "",
        markdown_table(baseline),
        "",
        "## 4. Binary Connectors",
        "",
        markdown_table(binary_summary),
        "",
        "## 5. Object Dominance",
        "",
        markdown_table(dominance_summary),
        "",
        "Positive object dominance means the object identity remains closer to its own positive object phrase than to other phrases with the same operator. Near-zero or negative values suggest stronger operator clustering.",
        "",
        "## 6. Main Conclusions",
        "",
        "These results should not be read as evidence that CLIP has formal logic. They indicate whether text embeddings show partially structured connector directions and whether those directions are stable or template-dependent. This helps contextualize the dog/grass and kitchen/table results: failures can arise when object identity, scene priors, and connector wording dominate over a clean absence representation.",
        "",
        "## 7. Limitations",
        "",
        "- One model.",
        "- Text-only analysis.",
        "- Limited phrase templates.",
        "- No causal proof.",
        "- English only.",
        "",
        f"Model: `{model_name}`",
        f"Device: `{device}`",
        "",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
