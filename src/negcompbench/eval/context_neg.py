from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from negcompbench.data.context_neg import normalize_languages
from negcompbench.eval.run_eval import build_ranker
from negcompbench.models.openclip_runner import get_peak_memory_mb
from negcompbench.utils.io import ensure_dir, read_jsonl, write_jsonl

CAPTION_ROLES = ("generic", "positive", "negative")


def run_context_neg_evaluation(
    annotations_path: str | Path,
    output_dir: str | Path,
    model_name: str = "openclip_vit_b32",
    languages: list[str] | tuple[str, ...] = ("en",),
    batch_size: int = 8,
    device: str = "auto",
    seed: int = 0,
) -> tuple[list[dict], pd.DataFrame]:
    annotations_file = Path(annotations_path)
    rows = read_jsonl(annotations_file)
    languages = normalize_languages(languages)
    output = ensure_dir(output_dir)
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)

    start_time = time.perf_counter()
    result_rows: list[dict] = []
    for start in tqdm(range(0, len(rows), batch_size), desc=f"context-neg:{model_name}"):
        batch = rows[start : start + batch_size]
        images = [Image.open(resolve_image_path(annotations_file.parent, row["image_path"])).convert("RGB") for row in batch]
        caption_sets = [caption_candidates(row, languages) for row in batch]
        score_sets = ranker.score_batch(images, caption_sets)
        for row, scores in zip(batch, score_sets):
            result_rows.append(make_result_row(row, languages, scores, model_name, used_device))
        for image in images:
            image.close()

    runtime = time.perf_counter() - start_time
    peak_memory = get_peak_memory_mb(used_device)
    for row in result_rows:
        row["runtime_total_sec"] = runtime
        row["runtime_per_image_sec"] = runtime / max(len(result_rows), 1)
        row["peak_memory_mb"] = peak_memory

    write_jsonl(output / "results.jsonl", result_rows)
    summary = summarize_context_neg(result_rows, languages)
    summary.to_csv(output / "summary.csv", index=False)
    write_report(output / "report.md", summary, languages, model_name, used_device, len(result_rows))
    make_context_neg_plots(result_rows, output, languages)
    write_failure_gallery(output / "failure_gallery.md", result_rows, languages)
    return result_rows, summary


def caption_candidates(row: dict, languages: list[str]) -> list[str]:
    captions = row["captions"]
    candidates = []
    for language in languages:
        for role in CAPTION_ROLES:
            key = f"{role}_{language}"
            if key not in captions:
                raise KeyError(f"Annotation {row.get('image_id', '<unknown>')} is missing caption key {key}")
            candidates.append(captions[key])
    return candidates


def resolve_image_path(annotations_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    candidate = annotations_root / path
    if candidate.exists():
        return candidate
    return path


def make_result_row(row: dict, languages: list[str], flat_scores: list[float], model_name: str, device: str) -> dict:
    result = {
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "scene": row["scene"],
        "object": row["object"],
        "condition": row["condition"],
        "model_name": model_name,
        "device": device,
    }
    offset = 0
    for language in languages:
        scores = {
            "generic": float(flat_scores[offset]),
            "positive": float(flat_scores[offset + 1]),
            "negative": float(flat_scores[offset + 2]),
        }
        offset += 3
        result.update(language_result_fields(language, scores, row["condition"], row["captions"]))
    return result


def language_result_fields(language: str, scores: dict[str, float], condition: str, captions: dict[str, str]) -> dict:
    ranks = rank_scores(scores)
    top_role = max(scores, key=scores.get)
    if condition == "without_object":
        negation_margin = scores["negative"] - scores["positive"]
    elif condition == "with_object":
        negation_margin = scores["positive"] - scores["negative"]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return {
        f"caption_generic_{language}": captions[f"generic_{language}"],
        f"caption_positive_{language}": captions[f"positive_{language}"],
        f"caption_negative_{language}": captions[f"negative_{language}"],
        f"score_generic_{language}": scores["generic"],
        f"score_positive_{language}": scores["positive"],
        f"score_negative_{language}": scores["negative"],
        f"rank_generic_{language}": ranks["generic"],
        f"rank_positive_{language}": ranks["positive"],
        f"rank_negative_{language}": ranks["negative"],
        f"top_caption_{language}": captions[f"{top_role}_{language}"],
        f"top_caption_type_{language}": top_role,
        f"negation_margin_{language}": negation_margin,
        f"generic_vs_negative_margin_{language}": scores["negative"] - scores["generic"],
        f"positive_vs_negative_margin_{language}": scores["positive"] - scores["negative"],
        f"generic_vs_positive_margin_{language}": scores["positive"] - scores["generic"],
    }


def rank_scores(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return {role: rank + 1 for rank, (role, _) in enumerate(ordered)}


def summarize_context_neg(rows: list[dict], languages: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summary_rows = []
    for language in languages:
        summary_rows.extend(language_summary_rows(frame, language))
    return pd.DataFrame(summary_rows)


def language_summary_rows(frame: pd.DataFrame, language: str) -> list[dict]:
    with_rows = frame[frame["condition"] == "with_object"]
    without_rows = frame[frame["condition"] == "without_object"]
    metrics = [
        (
            f"with_object_accuracy_{language}",
            mean_bool(with_rows[f"score_positive_{language}"] > with_rows[f"score_negative_{language}"]),
            len(with_rows),
        ),
        (
            f"without_object_accuracy_{language}",
            mean_bool(without_rows[f"score_negative_{language}"] > without_rows[f"score_positive_{language}"]),
            len(without_rows),
        ),
        (f"generic_win_rate_{language}", mean_bool(frame[f"top_caption_type_{language}"] == "generic"), len(frame)),
        (
            f"negation_failure_rate_without_{language}",
            mean_bool(without_rows[f"score_positive_{language}"] > without_rows[f"score_negative_{language}"]),
            len(without_rows),
        ),
        (
            f"mean_negation_margin_without_{language}",
            mean_series(without_rows[f"score_negative_{language}"] - without_rows[f"score_positive_{language}"]),
            len(without_rows),
        ),
        (
            f"mean_negation_margin_with_{language}",
            mean_series(with_rows[f"score_positive_{language}"] - with_rows[f"score_negative_{language}"]),
            len(with_rows),
        ),
        (
            f"mean_generic_gap_without_{language}",
            mean_series(without_rows[f"score_negative_{language}"] - without_rows[f"score_generic_{language}"]),
            len(without_rows),
        ),
        (
            f"mean_generic_gap_with_{language}",
            mean_series(with_rows[f"score_positive_{language}"] - with_rows[f"score_generic_{language}"]),
            len(with_rows),
        ),
    ]
    return [{"language": language, "metric": metric, "value": value, "n": int(n)} for metric, value, n in metrics]


def mean_bool(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).mean())


def mean_series(values) -> float:
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).mean())


def write_report(
    output_path: str | Path,
    summary: pd.DataFrame,
    languages: list[str],
    model_name: str,
    device: str,
    n_images: int,
) -> None:
    lines = [
        "# ContextNeg-Test Report",
        "",
        f"- Model: `{model_name}`",
        f"- Device: `{device}`",
        f"- Images: {n_images}",
        f"- Languages: {', '.join(languages)}",
        "",
        "## Summary Metrics",
        "",
        "| language | metric | value | n |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        value = row["value"]
        value_text = "nan" if pd.isna(value) else f"{float(value):.4f}"
        lines.append(f"| {row['language']} | `{row['metric']}` | {value_text} | {int(row['n'])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Positive margins indicate that the condition-aware caption beat its negated counterpart. A high generic win rate suggests the model is leaning on scene recognition instead of object presence or absence.",
            "",
            "This is a small diagnostic experiment. Results depend on manually collected image quality, object visibility, caption language, and CLIP pretraining priors.",
            "",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def make_context_neg_plots(rows: list[dict], output_dir: str | Path, languages: list[str]) -> None:
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    output = Path(output_dir)
    for language in languages:
        fig, ax = plt.subplots(figsize=(8, 4))
        for role, color in [("generic", "#3b6ea8"), ("positive", "#3f8f4e"), ("negative", "#c65f3a")]:
            frame[f"score_{role}_{language}"].hist(ax=ax, bins=20, alpha=0.5, label=role, color=color)
        ax.set_title(f"Caption Score Distribution ({language})")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Images")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"score_distribution_{language}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        frame.boxplot(column=f"negation_margin_{language}", by="condition", ax=ax)
        ax.set_title(f"Negation Margins by Condition ({language})")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Condition-aware negation margin")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(output / f"margins_by_condition_{language}.png", dpi=160)
        plt.close(fig)

        counts = frame[f"top_caption_type_{language}"].value_counts().reindex(CAPTION_ROLES, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax, color=["#3b6ea8", "#3f8f4e", "#c65f3a"])
        ax.set_title(f"Top Caption Counts ({language})")
        ax.set_xlabel("Caption type")
        ax.set_ylabel("Images")
        fig.tight_layout()
        fig.savefig(output / f"top_caption_counts_{language}.png", dpi=160)
        plt.close(fig)


def write_failure_gallery(output_path: str | Path, rows: list[dict], languages: list[str]) -> None:
    failures = collect_failures(rows, languages)
    lines = ["# ContextNeg-Test Failure Gallery", ""]
    if not failures:
        lines.append("No failures found.")
    for failure in failures:
        row = failure["row"]
        language = failure["language"]
        lines.extend(
            [
                f"## {row['image_id']} ({language})",
                "",
                f"![{row['image_id']}]({row['image_path']})",
                "",
                f"- Image path: `{row['image_path']}`",
                f"- Condition: `{row['condition']}`",
                f"- Failure type: `{failure['failure_type']}`",
                f"- Score generic: {row[f'score_generic_{language}']:.4f}",
                f"- Score positive: {row[f'score_positive_{language}']:.4f}",
                f"- Score negative: {row[f'score_negative_{language}']:.4f}",
                f"- Top caption: {row[f'top_caption_{language}']}",
                f"- Failure margin: {failure['margin']:.4f}",
                "",
            ]
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def collect_failures(rows: list[dict], languages: list[str]) -> list[dict]:
    failures = []
    for row in rows:
        for language in languages:
            generic = row[f"score_generic_{language}"]
            positive = row[f"score_positive_{language}"]
            negative = row[f"score_negative_{language}"]
            if row["condition"] == "without_object":
                if positive > negative:
                    failure_type = "affirmation_bias" if positive > generic else "negation_failure"
                    failures.append(
                        {"row": row, "language": language, "failure_type": failure_type, "margin": positive - negative}
                    )
                if generic > negative:
                    failures.append(
                        {"row": row, "language": language, "failure_type": "generic_caption_bias", "margin": generic - negative}
                    )
            elif row["condition"] == "with_object" and negative > positive:
                failures.append(
                    {
                        "row": row,
                        "language": language,
                        "failure_type": "unexpected_negative_preference",
                        "margin": negative - positive,
                    }
                )
    return sorted(failures, key=lambda item: item["margin"], reverse=True)
