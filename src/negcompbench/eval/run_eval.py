from __future__ import annotations

import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from negcompbench.data.schemas import Annotation
from negcompbench.eval.metrics import compute_prediction, write_summary_csv
from negcompbench.models.openclip_runner import OpenCLIPConfig, OpenCLIPRanker, get_peak_memory_mb
from negcompbench.models.random_baseline import RandomBaseline
from negcompbench.models.text_bow_baseline import TextOnlyBowBaseline
from negcompbench.utils.device import get_device
from negcompbench.utils.io import ensure_dir, read_annotations, write_jsonl


MODEL_PRESETS = {
    "openclip_vit_b32": OpenCLIPConfig(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"),
    "openclip_vit_b32_openai": OpenCLIPConfig(model_name="ViT-B-32", pretrained="openai"),
    "openclip_vit_b32_datacomp": OpenCLIPConfig(model_name="ViT-B-32", pretrained="datacomp_m_s128m_b4k"),
    "openclip_vit_b16_openai": OpenCLIPConfig(model_name="ViT-B-16", pretrained="openai"),
    "openclip_vit_b16_siglip": OpenCLIPConfig(model_name="ViT-B-16-SigLIP", pretrained="webli"),
    "openclip_rn50": OpenCLIPConfig(model_name="RN50", pretrained="openai"),
}


def build_ranker(model_name: str, device: str = "auto", seed: int = 0):
    if model_name == "random_baseline":
        return RandomBaseline(seed=seed), get_device(device)
    if model_name == "text_bow_baseline":
        return TextOnlyBowBaseline(), "none"
    if model_name in MODEL_PRESETS:
        preset = MODEL_PRESETS[model_name]
        config = OpenCLIPConfig(model_name=preset.model_name, pretrained=preset.pretrained, device=device)
        ranker = OpenCLIPRanker(config)
        return ranker, ranker.device
    raise ValueError(f"Unknown model: {model_name}. Available: {sorted([*MODEL_PRESETS, 'random_baseline', 'text_bow_baseline'])}")


def run_evaluation(
    model_name: str,
    annotations_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 16,
    device: str = "auto",
    seed: int = 0,
    max_samples: int | None = None,
) -> list[dict]:
    output = ensure_dir(output_dir)
    annotations = read_annotations(annotations_path)
    if max_samples is not None:
        annotations = annotations[:max_samples]
    dataset_root = Path(annotations_path).parent
    ranker, used_device = build_ranker(model_name, device=device, seed=seed)

    start_time = time.perf_counter()
    rows: list[dict] = []
    for start in tqdm(range(0, len(annotations), batch_size), desc=f"eval:{model_name}"):
        batch = annotations[start : start + batch_size]
        images = [Image.open(resolve_image_path(dataset_root, ann)).convert("RGB") for ann in batch]
        caption_sets = [[ann.correct_caption] + ann.hard_negative_captions for ann in batch]
        score_sets = ranker.score_batch(images, caption_sets)
        for ann, captions, scores in zip(batch, caption_sets, score_sets):
            prediction = compute_prediction(scores[0], scores[1:])
            selected_caption = captions[prediction["prediction_index"]]
            rows.append(make_result_row(ann, model_name, used_device, captions, scores, selected_caption, prediction))
        for image in images:
            image.close()

    runtime = time.perf_counter() - start_time
    peak_memory = get_peak_memory_mb(used_device)
    for row in rows:
        row["runtime_total_sec"] = runtime
        row["runtime_per_sample_sec"] = runtime / max(len(rows), 1)
        row["peak_memory_mb"] = peak_memory

    write_jsonl(output / "results.jsonl", rows)
    write_summary_csv(rows, output / "summary.csv")
    return rows


def resolve_image_path(dataset_root: Path, ann: Annotation) -> Path:
    image_path = Path(ann.image_path)
    if image_path.is_absolute():
        return image_path
    return dataset_root / image_path


def make_result_row(
    ann: Annotation,
    model_name: str,
    device: str,
    captions: list[str],
    scores: list[float],
    selected_caption: str,
    prediction: dict,
) -> dict:
    color_object_pairs = [f"{obj.color}_{obj.shape}" for obj in ann.objects]
    return {
        "image_id": ann.image_id,
        "image_path": ann.image_path,
        "task_type": ann.task_type,
        "relation": ann.relation,
        "correct_caption": ann.correct_caption,
        "hard_negative_captions": ann.hard_negative_captions,
        "candidate_captions": captions,
        "scores": scores,
        "score_correct": scores[0],
        "negative_scores": scores[1:],
        "selected_caption": selected_caption,
        "model_name": model_name,
        "device": device,
        "is_correct": prediction["is_correct"],
        "prediction_index": prediction["prediction_index"],
        "margin": prediction["margin"],
        "max_negative_score": prediction["max_negative_score"],
        "failure_type": None if prediction["is_correct"] else ann.metadata.get("failure_type", "unknown_failure"),
        "color_object_pairs": color_object_pairs,
        "seed": ann.seed,
        "metadata": ann.metadata,
    }
