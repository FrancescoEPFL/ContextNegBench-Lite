from pathlib import Path

import pytest
from PIL import Image

from negcompbench.data.context_neg import build_context_neg_annotations, make_captions, normalize_languages
from negcompbench.data.context_neg_dataset import (
    file_sha256,
    negative_folder,
    normalize_filename,
    positive_folder,
    prepare_context_neg_dataset,
    read_csv_rows,
    scan_supported_images,
)
from negcompbench.data.context_neg_download import collect_candidates, looks_like_rate_limit, reviewed_hashes, search_context_neg_images
from negcompbench.data.context_neg_enrichment import (
    EnrichmentJob,
    ImageCandidate,
    enrich_context_neg_datasets,
    rotate_queries,
)
from negcompbench.eval.context_neg_pairwise_prompts import (
    build_perturbation_captions,
    build_prompt_pairs,
    pairwise_row,
    summarize_pairwise,
    summarize_perturbations,
)
from negcompbench.eval.context_neg_research_analysis import (
    SCENARIO_CONFIGS,
    bootstrap_ci,
    compute_image_embedding_metrics,
    compute_pairwise_metrics,
    compute_text_embedding_metrics_from_embeddings,
)
from negcompbench.eval.context_neg_specificity import (
    compute_specificity_metrics_with_ci,
    scenario_caption_specs,
    scores_to_wide,
)
from negcompbench.eval.dog_grass_false_negation import (
    bootstrap_values_ci,
    dog_grass_prompt_pairs,
    dog_grass_result_row,
    list_valid_dog_grass_images,
    summarize_dog_grass_with_ci,
    write_report as write_dog_grass_report,
)
from negcompbench.eval.final_contextneg_analysis import (
    compute_final_metric_cis,
    compute_generic_vs_specific_metrics,
    compute_text_negation_similarity,
    final_prompt_groups,
)
from negcompbench.eval.logical_connector_embeddings import (
    BINARY_CONNECTORS,
    OBJECT_PAIRS,
    UNARY_OPERATORS,
    build_phrase_inventory,
    compute_binary_connector_metrics,
    compute_object_dominance_index,
    compute_pairwise_distance_metrics,
    controlled_neighbor_vocabulary,
    neighbor_category,
    vector_distance_metrics,
)
from negcompbench.eval.negation_delta_consistency import (
    NegationPairGroup,
    add_axis_projection,
    baseline_comparison_rows,
    compute_object_delta_metrics,
    delta_similarity_matrix,
    format_phrase,
    mean_pairwise_cosine,
    pca_explained_variance,
    summarize_delta_similarity,
    unique_phrases,
)
from negcompbench.eval.context_neg import language_result_fields, summarize_context_neg
from negcompbench.utils.io import read_jsonl


def test_make_captions_supports_english_and_italian():
    captions = make_captions("kitchen", "table", ["en", "it"])
    assert captions["generic_en"] == "a photo of a kitchen"
    assert captions["positive_en"] == "a photo of a kitchen with a table"
    assert captions["negative_en"] == "a photo of a kitchen without a table"
    assert captions["generic_it"] == "foto di una cucina"
    assert captions["positive_it"] == "foto di una cucina con tavolo"
    assert captions["negative_it"] == "foto di una cucina senza tavolo"


def test_normalize_languages_accepts_comma_separated_values():
    assert normalize_languages(["en,it", "en"]) == ["en", "it"]


def test_normalize_filename_is_safe_and_preserves_supported_extension():
    assert normalize_filename("My Kitchen Table (Final).JPEG") == "my_kitchen_table_final.jpg"
    assert normalize_filename("   ???.tiff", default_stem="fallback") == "fallback.jpg"
    assert normalize_filename("clean-name.webp") == "clean_name.webp"


def test_folder_naming_uses_arbitrary_object():
    assert positive_folder("car") == "with_car"
    assert negative_folder("car") == "without_car"
    assert positive_folder("dining table") == "with_dining_table"


def test_context_neg_annotation_builder_scans_condition_folders(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "kitchen_table"
    with_dir = root / "reviewed" / "with_table"
    without_dir = root / "reviewed" / "without_table"
    with_dir.mkdir(parents=True)
    without_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(with_dir / "kitchen_with_table_001.jpg")
    Image.new("RGB", (16, 16), "white").save(without_dir / "kitchen_without_table_001.png")

    rows = build_context_neg_annotations(root, "kitchen", "table", ["en", "it"], cwd=tmp_path)
    saved = read_jsonl(root / "annotations.jsonl")
    image_index = read_csv_rows(root / "image_index.csv")

    assert len(rows) == 2
    assert saved == rows
    assert len(image_index) == 2
    assert rows[0]["condition"] == "with_object"
    assert rows[1]["condition"] == "without_object"
    assert rows[0]["image_id"] == "kitchen_table_with_0001"
    assert rows[1]["image_id"] == "kitchen_table_without_0001"
    assert rows[0]["image_path"].endswith("reviewed/with_table/kitchen_with_table_001.jpg")
    assert rows[1]["captions"]["negative_it"] == "foto di una cucina senza tavolo"
    assert rows[1]["metadata"]["reviewed"] is True
    assert rows[1]["metadata"]["original_filename"] == "kitchen_without_table_001.png"
    assert image_index[1]["image_id"] == "kitchen_table_without_0001"
    assert image_index[1]["original_filename"] == "kitchen_without_table_001.png"


def test_street_car_annotation_generation_uses_plural_cars(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "street_car"
    with_dir = root / "reviewed" / "with_car"
    without_dir = root / "reviewed" / "without_car"
    with_dir.mkdir(parents=True)
    without_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(with_dir / "street_with_car_001.jpg")
    Image.new("RGB", (16, 16), "white").save(without_dir / "street_without_car_001.png")

    rows = build_context_neg_annotations(root, "street", "car", ["en"], cwd=tmp_path)

    assert rows[0]["image_id"] == "street_car_with_0001"
    assert rows[1]["image_id"] == "street_car_without_0001"
    assert rows[0]["captions"]["generic_en"] == "a photo of a street"
    assert rows[0]["captions"]["positive_en"] == "a photo of a street with cars"
    assert rows[0]["captions"]["negative_en"] == "a photo of a street without cars"
    assert rows[1]["image_path"].endswith("reviewed/without_car/street_without_car_001.png")


def test_prepare_dataset_moves_obvious_duplicate_to_rejected(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "kitchen_table"
    with_dir = root / "raw" / "with_table"
    without_dir = root / "raw" / "without_table"
    with_dir.mkdir(parents=True)
    without_dir.mkdir(parents=True)
    image = Image.new("RGB", (24, 24), "white")
    image.save(with_dir / "first image.jpg")
    image.save(without_dir / "duplicate image.jpg")

    warnings = prepare_context_neg_dataset(root)
    rejected = scan_supported_images(root / "reviewed" / "rejected")

    assert rejected
    assert any("duplicate" in warning.lower() for warning in warnings)
    assert file_sha256(rejected[0]) == file_sha256(with_dir / "first_image.jpg")


def test_prepare_dataset_creates_street_car_folders(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "street_car"

    prepare_context_neg_dataset(root, object_name="car")

    assert (root / "raw" / "with_car").is_dir()
    assert (root / "raw" / "without_car").is_dir()
    assert (root / "reviewed" / "with_car").is_dir()
    assert (root / "reviewed" / "without_car").is_dir()


def test_collect_candidates_reads_url_file_and_deduplicates(tmp_path: Path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("query one,https://example.com/a.jpg\nhttps://example.com/a.jpg\nhttps://example.com/b.png\n", encoding="utf-8")

    candidates = collect_candidates([], 10, 10, url_file)

    assert [candidate.url for candidate in candidates] == ["https://example.com/a.jpg", "https://example.com/b.png"]
    assert candidates[0].query == "query one"


def test_rate_limit_detector_matches_duckduckgo_error_text():
    error = RuntimeError("https://duckduckgo.com/i.js 403 Ratelimit")
    assert looks_like_rate_limit(error)


def test_search_download_from_url_file_logs_and_skips_duplicate(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "street_car"
    source = tmp_path / "source image.jpg"
    source_copy = tmp_path / "source copy.jpg"
    Image.new("RGB", (32, 32), "white").save(source)
    Image.new("RGB", (32, 32), "white").save(source_copy)
    url = source.resolve().as_uri()
    duplicate_url = source_copy.resolve().as_uri()
    url_file = tmp_path / "urls.txt"
    url_file.write_text(f"local,{url}\nlocal duplicate,{duplicate_url}\n", encoding="utf-8")

    rows = search_context_neg_images(
        root,
        "with_object",
        [],
        object_name="car",
        scene="street",
        limit_total=5,
        url_file=url_file,
        timeout=5,
        sleep_seconds=0,
        jitter_seconds=0,
    )
    downloaded = scan_supported_images(root / "reviewed" / "with_car")
    sources = read_csv_rows(root / "metadata" / "sources.csv")

    assert [row["action"] for row in rows] == ["downloaded", "skipped_duplicate"]
    assert len(downloaded) == 1
    assert reviewed_hashes(root, "car") == {file_sha256(downloaded[0])}
    assert sources[0]["source_url"] == url
    assert (root / "review_gallery.html").exists()


def test_search_download_resume_skips_logged_url(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "kitchen_table"
    source = tmp_path / "source image.jpg"
    Image.new("RGB", (32, 32), "white").save(source)
    url_file = tmp_path / "urls.txt"
    url_file.write_text(source.resolve().as_uri(), encoding="utf-8")

    first = search_context_neg_images(
        root,
        "with_table",
        [],
        url_file=url_file,
        sleep_seconds=0,
        jitter_seconds=0,
    )
    second = search_context_neg_images(
        root,
        "with_table",
        [],
        url_file=url_file,
        resume=True,
        sleep_seconds=0,
        jitter_seconds=0,
    )

    assert first[0]["action"] == "downloaded"
    assert second[0]["action"] == "skipped_duplicate"
    assert "resume" in second[0]["message"]


def test_enrichment_rotates_queries_by_previous_attempts():
    job = EnrichmentJob(
        scenario_id="street_car",
        root="data/context_neg/street_car",
        scene="street",
        object_name="car",
        condition="with_object",
        queries=("street with cars", "city street with cars", "urban street traffic"),
    )
    rows = [
        {"scenario_id": "street_car", "condition": "with_car", "query": "street with cars"},
        {"scenario_id": "street_car", "condition": "with_car", "query": "street with cars"},
        {"scenario_id": "street_car", "condition": "with_car", "query": "city street with cars"},
    ]

    assert rotate_queries(job.queries, rows, job) == ["urban street traffic", "city street with cars", "street with cars"]


def test_enrichment_dry_run_writes_global_log_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job = EnrichmentJob(
        scenario_id="street_car",
        root="data/context_neg/street_car",
        scene="street",
        object_name="car",
        condition="with_object",
        queries=("street with cars",),
    )

    def fake_collect(*args, **kwargs):
        return [ImageCandidate("https://example.com/street.jpg", "street with cars")]

    monkeypatch.setattr("negcompbench.data.context_neg_enrichment.collect_enrichment_candidates", fake_collect)
    summary = enrich_context_neg_datasets(
        target_per_folder=1,
        batch_size_per_job=1,
        sleep_seconds=0,
        jitter_seconds=0,
        jobs=[job],
        dry_run=True,
        resume=True,
        base_dir=tmp_path,
    )
    rows = read_csv_rows(tmp_path / "data" / "context_neg" / "enrichment_log.csv")

    assert summary.dry_run is True
    assert rows[0]["scenario_id"] == "street_car"
    assert rows[0]["condition"] == "with_car"
    assert rows[0]["action"] == "dry_run"
    assert (tmp_path / "data" / "context_neg" / "enrichment_run_summary.md").exists()
    assert (tmp_path / "data" / "context_neg" / "street_car" / "review_gallery.html").exists()


def test_language_result_fields_compute_condition_aware_margins():
    captions = make_captions("kitchen", "table", ["en"])
    without = language_result_fields("en", {"generic": 0.30, "positive": 0.35, "negative": 0.40}, "without_object", captions)
    with_object = language_result_fields("en", {"generic": 0.30, "positive": 0.50, "negative": 0.20}, "with_object", captions)

    assert without["negation_margin_en"] == pytest.approx(0.05)
    assert without["generic_vs_negative_margin_en"] == pytest.approx(0.10)
    assert without["positive_vs_negative_margin_en"] == pytest.approx(-0.05)
    assert with_object["negation_margin_en"] == pytest.approx(0.30)
    assert with_object["generic_vs_positive_margin_en"] == pytest.approx(0.20)
    assert with_object["rank_positive_en"] == 1


def test_summarize_context_neg_metrics():
    rows = [
        {
            "condition": "with_object",
            "score_generic_en": 0.20,
            "score_positive_en": 0.50,
            "score_negative_en": 0.10,
            "top_caption_type_en": "positive",
        },
        {
            "condition": "without_object",
            "score_generic_en": 0.45,
            "score_positive_en": 0.40,
            "score_negative_en": 0.30,
            "top_caption_type_en": "generic",
        },
    ]
    summary = summarize_context_neg(rows, ["en"])
    metrics = dict(zip(summary["metric"], summary["value"]))

    assert metrics["with_object_accuracy_en"] == 1.0
    assert metrics["without_object_accuracy_en"] == 0.0
    assert metrics["generic_win_rate_en"] == 0.5
    assert metrics["negation_failure_rate_without_en"] == 1.0
    assert metrics["mean_negation_margin_without_en"] == pytest.approx(-0.10)
    assert metrics["mean_negation_margin_with_en"] == pytest.approx(0.40)
    assert metrics["mean_generic_gap_without_en"] == pytest.approx(-0.15)
    assert metrics["mean_generic_gap_with_en"] == pytest.approx(0.30)


def test_pairwise_row_uses_condition_aware_margin():
    with_ann = {"image_id": "a", "image_path": "a.jpg", "condition": "with_object"}
    without_ann = {"image_id": "b", "image_path": "b.jpg", "condition": "without_object"}

    with_row = pairwise_row(with_ann, "base_with_without", "pos", "neg", 0.6, 0.4)
    without_row = pairwise_row(without_ann, "base_with_without", "pos", "neg", 0.6, 0.4)

    assert with_row["raw_gap"] == pytest.approx(0.2)
    assert with_row["correct_margin"] == pytest.approx(0.2)
    assert with_row["correct"] is True
    assert without_row["correct_margin"] == pytest.approx(-0.2)
    assert without_row["correct"] is False


def test_summarize_pairwise_adds_base_deltas():
    import pandas as pd

    frame = pd.DataFrame(
        [
            pairwise_row({"image_id": "a", "image_path": "a.jpg", "condition": "with_object"}, "base_with_without", "p", "n", 0.6, 0.4),
            pairwise_row({"image_id": "b", "image_path": "b.jpg", "condition": "without_object"}, "base_with_without", "p", "n", 0.3, 0.5),
            pairwise_row({"image_id": "a", "image_path": "a.jpg", "condition": "with_object"}, "no_object", "p", "n", 0.4, 0.6),
            pairwise_row({"image_id": "b", "image_path": "b.jpg", "condition": "without_object"}, "no_object", "p", "n", 0.3, 0.5),
        ]
    )

    summary = summarize_pairwise(frame)
    base = summary[summary["pair_id"] == "base_with_without"].iloc[0]
    no_table = summary[summary["pair_id"] == "no_object"].iloc[0]

    assert base["accuracy_overall"] == 1.0
    assert no_table["accuracy_overall"] == 0.5
    assert no_table["delta_accuracy_overall"] == pytest.approx(-0.5)


def test_street_car_prompt_templates_use_plural_cars_and_no_table_leakage():
    pairs = build_prompt_pairs("street", "car")
    perturbations = build_perturbation_captions("street", "car")
    text = " ".join([item for _, positive, negative in pairs for item in (positive, negative)])

    assert "dining table" not in text
    assert "table" not in text
    assert ("base_with_without", "a street with cars", "a street without cars") in pairs
    assert ("specific_object", "a street with cars", "a street without cars") in pairs
    assert ("visible_constraint", "a street with visible cars", "a street with no visible cars") in pairs
    assert ("containing", "a street containing cars", "a street containing no cars") in pairs
    assert ("no_object", "a street with cars", "a street with no cars") in pairs
    assert perturbations["without_upper"] == "a street WITHOUT cars"
    assert perturbations["no_upper"] == "a street with NO cars"
    assert perturbations["positive_repeated"] == "a street with cars. a street with cars"
    assert perturbations["negative_repeated"] == "a street without cars. a street without cars"


def test_summarize_perturbations_reports_requested_metrics():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "condition": "with_object",
                "uppercase_delta_without": 0.01,
                "uppercase_delta_no": -0.01,
                "rank_flip_rate_uppercase_against_positive": False,
                "repetition_gain_positive": 0.02,
                "repetition_gain_negative": 0.03,
                "rank_flip_rate_repetition_against_opposite": True,
            },
            {
                "condition": "without_object",
                "uppercase_delta_without": 0.03,
                "uppercase_delta_no": 0.01,
                "rank_flip_rate_uppercase_against_positive": True,
                "repetition_gain_positive": 0.04,
                "repetition_gain_negative": 0.01,
                "rank_flip_rate_repetition_against_opposite": False,
            },
        ]
    )

    summary = summarize_perturbations(frame)
    metrics = dict(zip(summary["metric"], summary["value"]))

    assert metrics["mean_uppercase_delta_without"] == pytest.approx(0.02)
    assert metrics["rank_flip_rate_uppercase_against_positive"] == pytest.approx(0.5)
    assert metrics["repetition_gain_positive_with_object"] == pytest.approx(0.02)
    assert metrics["rank_flip_rate_repetition_against_opposite"] == pytest.approx(0.5)


def test_research_scenario_mapping_contains_expected_paths():
    kitchen = SCENARIO_CONFIGS["kitchen_table"]
    street = SCENARIO_CONFIGS["street_car"]

    assert kitchen.annotations == "data/context_neg/kitchen_table/annotations.jsonl"
    assert kitchen.scene == "kitchen"
    assert kitchen.object_name == "table"
    assert street.pairwise_results.endswith("street_car/pairwise_results_long.csv")
    assert street.scene == "street"
    assert street.object_name == "car"


def test_research_pairwise_metrics_from_fake_frame():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"condition": "with_object", "correct": True, "correct_margin": 0.02, "score_positive": 0.6, "score_negative": 0.4},
            {"condition": "with_object", "correct": False, "correct_margin": -0.03, "score_positive": 0.4, "score_negative": 0.5},
            {"condition": "without_object", "correct": True, "correct_margin": 0.04, "score_positive": 0.3, "score_negative": 0.5},
            {"condition": "without_object", "correct": False, "correct_margin": -0.01, "score_positive": 0.6, "score_negative": 0.5},
        ]
    )

    metrics = compute_pairwise_metrics(frame)

    assert metrics["pairwise_accuracy"] == pytest.approx(0.5)
    assert metrics["mean_correct_margin"] == pytest.approx(0.005)
    assert metrics["low_margin_rate_0_01"] == pytest.approx(0.5)
    assert metrics["false_absence_preference_rate"] == pytest.approx(0.5)
    assert metrics["false_presence_preference_rate"] == pytest.approx(0.5)


def test_research_bootstrap_ci_is_deterministic_for_simple_metric():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {"condition": "with_object", "correct": True, "correct_margin": 0.02, "score_positive": 0.6, "score_negative": 0.4},
            {"condition": "without_object", "correct": True, "correct_margin": 0.03, "score_positive": 0.3, "score_negative": 0.5},
            {"condition": "without_object", "correct": False, "correct_margin": -0.01, "score_positive": 0.6, "score_negative": 0.5},
        ]
    )

    value, ci_low, ci_high = bootstrap_ci(frame, "pairwise_accuracy", samples=100, seed=7)

    assert value == pytest.approx(2 / 3)
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_research_text_embedding_metrics_from_fake_embeddings():
    import numpy as np
    import pandas as pd

    pair_rows = pd.DataFrame(
        [
            {
                "pair_id": "base_with_without",
                "positive_caption": "a street with cars",
                "negative_caption": "a street without cars",
            }
        ]
    )
    frame = compute_text_embedding_metrics_from_embeddings(
        "street_car",
        pair_rows,
        generic_embedding=np.array([1.0, 0.0]),
        positive_embeddings=np.array([[1.0, 0.0]]),
        negative_embeddings=np.array([[0.0, 1.0]]),
    )

    assert frame.iloc[0]["cosine_text_positive_negative"] == pytest.approx(0.0)
    assert frame.iloc[0]["text_negation_separation"] == pytest.approx(1.0)
    assert frame.iloc[0]["cosine_generic_positive"] == pytest.approx(1.0)
    assert frame.iloc[0]["cosine_generic_negative"] == pytest.approx(0.0)


def test_research_image_embedding_metrics_from_fake_embeddings():
    import numpy as np

    rows = [
        {"image_id": "with_1", "image_path": "a.jpg", "condition": "with_object"},
        {"image_id": "with_2", "image_path": "b.jpg", "condition": "with_object"},
        {"image_id": "without_1", "image_path": "c.jpg", "condition": "without_object"},
        {"image_id": "without_2", "image_path": "d.jpg", "condition": "without_object"},
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    metrics = compute_image_embedding_metrics("fake", rows, embeddings)

    assert metrics["n_images"] == 4
    assert metrics["n_with_object"] == 2
    assert metrics["n_without_object"] == 2
    assert metrics["image_condition_separation"] > 0.0
    assert metrics["separation_ratio"] > 0.0


def test_specificity_caption_specs_use_plural_cars_for_street():
    kitchen = {caption.caption_id: caption.caption_text for caption in scenario_caption_specs("kitchen_table")}
    street = {caption.caption_id: caption.caption_text for caption in scenario_caption_specs("street_car")}

    assert kitchen["positive_base"] == "a kitchen with a table"
    assert kitchen["negative_no_visible"] == "a kitchen with no visible table"
    assert street["positive_base"] == "a street with cars"
    assert street["positive_visible"] == "a street with visible cars"
    assert street["negative_containing_no"] == "a street containing no cars"


def test_specificity_metrics_from_fake_scores():
    import pandas as pd

    rows = []
    score_map = {
        "generic_base": 0.40,
        "generic_photo": 0.45,
        "positive_base": 0.55,
        "positive_photo": 0.60,
        "positive_visible": 0.58,
        "positive_photo_visible": 0.62,
        "positive_containing": 0.54,
        "positive_photo_containing": 0.57,
        "negative_without": 0.42,
        "negative_photo_without": 0.46,
        "negative_no": 0.41,
        "negative_no_visible": 0.39,
        "negative_containing_no": 0.40,
    }
    families = {caption.caption_id: caption.caption_family for caption in scenario_caption_specs("kitchen_table")}
    for caption_id, score in score_map.items():
        rows.append(
            {
                "scenario": "kitchen_table",
                "image_id": "img1",
                "image_path": "img1.jpg",
                "condition": "with_object",
                "caption_family": families[caption_id],
                "caption_id": caption_id,
                "caption_text": caption_id,
                "score": score,
            }
        )
    scores = pd.DataFrame(rows)
    wide = scores_to_wide(scores)
    metrics = compute_specificity_metrics_with_ci(scores, bootstrap_samples=20, seed=1)
    lookup = {
        (row["metric"], row["comparison_id"]): row["value"]
        for _, row in metrics.iterrows()
    }

    assert wide.iloc[0]["positive_base"] == pytest.approx(0.55)
    assert lookup[("mean_positive_specificity_gain", "base")] == pytest.approx(0.15)
    assert lookup[("generic_dominance_rate", "base")] == pytest.approx(0.0)
    assert lookup[("mean_prefix_gain", "generic_photo_prefix")] == pytest.approx(0.05)
    assert lookup[("mean_visible_gain", "visible_positive")] == pytest.approx(0.03)
    assert lookup[("mean_false_caption_tolerance_vs_generic", "without_vs_generic")] == pytest.approx(0.02)
    assert lookup[("positive_top_rate", "base")] == pytest.approx(1.0)


def test_final_prompt_groups_define_core_story():
    kitchen = {group.prompt_group: group for group in final_prompt_groups("kitchen_table")}
    street = {group.prompt_group: group for group in final_prompt_groups("street_car")}

    assert kitchen["base"].generic == "a kitchen"
    assert kitchen["base"].positive == "a kitchen with a table"
    assert kitchen["base"].negative == "a kitchen without a table"
    assert street["visible"].positive == "a street with visible cars"
    assert street["visible"].negative == "a street with no visible cars"


def test_final_text_negation_similarity_from_fake_embeddings():
    import numpy as np

    groups = [final_prompt_groups("street_car")[0]]
    embeddings = {
        "a street": np.array([1.0, 0.0]),
        "a street with cars": np.array([1.0, 0.0]),
        "a street without cars": np.array([0.0, 1.0]),
    }
    rows = compute_text_negation_similarity("street_car", groups, embeddings)

    assert rows[0]["cosine_positive_negative"] == pytest.approx(0.0)
    assert rows[0]["text_negation_distance"] == pytest.approx(1.0)
    assert rows[0]["cosine_generic_positive"] == pytest.approx(1.0)


def test_final_generic_vs_specific_metrics_from_fake_scores():
    import pandas as pd

    scores = pd.DataFrame(
        [
            {
                "scenario": "kitchen_table",
                "image_id": "with",
                "image_path": "a.jpg",
                "condition": "with_object",
                "prompt_group": "base",
                "score_generic": 0.4,
                "score_positive": 0.6,
                "score_negative": 0.5,
                "top_caption": "positive",
                "positive_specificity_gain": 0.2,
                "false_absence_tolerance": 0.1,
                "true_absence_specificity_gain": None,
                "false_presence_tolerance": None,
            },
            {
                "scenario": "kitchen_table",
                "image_id": "without",
                "image_path": "b.jpg",
                "condition": "without_object",
                "prompt_group": "base",
                "score_generic": 0.4,
                "score_positive": 0.45,
                "score_negative": 0.55,
                "top_caption": "negative",
                "positive_specificity_gain": None,
                "false_absence_tolerance": None,
                "true_absence_specificity_gain": 0.15,
                "false_presence_tolerance": 0.05,
            },
        ]
    )
    generic = compute_generic_vs_specific_metrics(scores)
    final_ci = compute_final_metric_cis(
        scores,
        pd.DataFrame(
            [
                {
                    "scenario": "kitchen_table",
                    "n_images": 2,
                    "image_condition_separation": 0.03,
                    "image_condition_separation_ci_low": 0.01,
                    "image_condition_separation_ci_high": 0.05,
                }
            ]
        ),
        bootstrap_samples=20,
        seed=2,
    )
    lookup = {(row["metric"], row["prompt_group"]): row["value"] for _, row in final_ci.iterrows()}

    assert generic.iloc[0]["positive_specificity_gain"] == pytest.approx(0.2)
    assert generic.iloc[0]["true_absence_specificity_gain"] == pytest.approx(0.15)
    assert lookup[("mean_false_absence_tolerance", "base")] == pytest.approx(0.1)
    assert lookup[("image_condition_separation", "image")] == pytest.approx(0.03)


def test_dog_grass_valid_image_listing_skips_unsupported_and_corrupt(tmp_path: Path):
    root = tmp_path / "data" / "context_neg" / "dog_grass_false_negation"
    image_dir = root / "reviewed" / "with_dog"
    rejected_dir = root / "reviewed" / "rejected"
    image_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(image_dir / "dog.jpg")
    (image_dir / ".gitkeep").write_text("", encoding="utf-8")
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    (image_dir / "broken.png").write_text("not really png", encoding="utf-8")
    Image.new("RGB", (16, 16), "black").save(rejected_dir / "rejected.jpg")

    valid = list_valid_dog_grass_images(root)

    assert [path.name for path in valid] == ["dog.jpg"]


def test_dog_grass_result_row_computes_false_negation_margins():
    pair = dog_grass_prompt_pairs()[0]
    row = dog_grass_result_row(
        "dog1",
        "dog1.jpg",
        pair,
        {
            pair.false_negated: 0.52,
            pair.partial_generic: 0.50,
            pair.positive_reference: 0.61,
        },
    )

    assert row["margin_false_vs_generic"] == pytest.approx(0.02)
    assert row["margin_positive_vs_generic"] == pytest.approx(0.11)
    assert row["margin_positive_vs_false"] == pytest.approx(0.09)
    assert row["false_negated_wins_over_generic"] is True
    assert row["false_negated_wins_over_positive"] is False
    assert row["top_caption"] == "positive_reference"


def test_dog_grass_summary_and_bootstrap_ci_from_fake_scores():
    import pandas as pd

    pair = dog_grass_prompt_pairs()[0]
    rows = [
        dog_grass_result_row("a", "a.jpg", pair, {pair.false_negated: 0.52, pair.partial_generic: 0.50, pair.positive_reference: 0.61}),
        dog_grass_result_row("b", "b.jpg", pair, {pair.false_negated: 0.45, pair.partial_generic: 0.50, pair.positive_reference: 0.60}),
    ]
    summary = summarize_dog_grass_with_ci(pd.DataFrame(rows), bootstrap_samples=50, seed=3)
    lookup = {(row["pair_id"], row["metric"]): row for _, row in summary.iterrows()}
    value, ci_low, ci_high = bootstrap_values_ci([True, False, True], lambda values: float(values.astype(float).mean()), samples=50, seed=1)

    assert lookup[("core", "mean_margin_false_vs_generic")]["value"] == pytest.approx(-0.015)
    assert lookup[("core", "false_negated_win_rate_over_generic")]["value"] == pytest.approx(0.5)
    assert 0.0 <= ci_low <= value <= ci_high <= 1.0


def test_dog_grass_report_creation_without_model(tmp_path: Path):
    import pandas as pd

    output = tmp_path / "report.md"
    summary = pd.DataFrame(
        [
            {"pair_id": "core", "metric": "mean_margin_false_vs_generic", "value": 0.01, "ci_low": 0.0, "ci_high": 0.02, "n": 2},
            {"pair_id": "core", "metric": "false_negated_win_rate_over_generic", "value": 0.5, "ci_low": 0.0, "ci_high": 1.0, "n": 2},
            {"pair_id": "core", "metric": "false_negated_win_rate_over_positive", "value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 2},
            {"pair_id": "core", "metric": "top_false_negated_rate", "value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 2},
            {"pair_id": "core", "metric": "top_generic_rate", "value": 0.5, "ci_low": 0.0, "ci_high": 1.0, "n": 2},
            {"pair_id": "core", "metric": "top_positive_rate", "value": 0.5, "ci_low": 0.0, "ci_high": 1.0, "n": 2},
        ]
    )
    text_metrics = pd.DataFrame(
        [
            {
                "pair_id": "core",
                "cosine_false_negated_generic": 0.8,
                "cosine_positive_false_negated": 0.7,
                "text_false_generic_distance": 0.2,
                "text_positive_false_distance": 0.3,
            }
        ]
    )
    image_summary = pd.DataFrame(
        [
            {
                "pair_id": "core",
                "image_embedding_count": 2,
                "image_embedding_mean_norm": 1.0,
                "within_dataset_variance": 0.01,
                "mean_image_to_text_similarity_false_negated": 0.4,
                "mean_image_to_text_similarity_generic": 0.39,
                "mean_image_to_text_similarity_positive": 0.5,
            }
        ]
    )

    write_dog_grass_report(output, 2, "mock_model", "cpu", summary, text_metrics, image_summary)

    text = output.read_text(encoding="utf-8")
    assert "Dog/Grass False Negation Diagnostic" in text
    assert "margin_false_vs_generic" in text


def test_negation_delta_phrase_and_similarity_metrics_from_fake_embeddings():
    import numpy as np

    objects = ["dog", "cat", "car"]
    groups = [NegationPairGroup("bare_no", "{object}", "no {object}")]
    phrases = unique_phrases(objects, groups)
    embedding_map = {
        "dog": np.array([1.0, 0.0, 0.0]),
        "no dog": np.array([1.0, 1.0, 0.0]),
        "cat": np.array([0.0, 1.0, 0.0]),
        "no cat": np.array([0.0, 2.0, 0.0]),
        "car": np.array([0.0, 0.0, 1.0]),
        "no car": np.array([0.0, 1.0, 1.0]),
    }

    metrics = compute_object_delta_metrics(objects, groups, embedding_map)
    units = np.vstack(metrics["delta_unit"].to_list())
    similarity = delta_similarity_matrix(objects, units)
    summary = summarize_delta_similarity("bare_no", similarity)
    projected = add_axis_projection(metrics, units)

    assert phrases == ["dog", "no dog", "cat", "no cat", "car", "no car"]
    assert format_phrase("a {object}", "dog") == "a dog"
    assert summary["mean_delta_direction_similarity"] == pytest.approx(1.0)
    assert projected["projection_on_mean_axis"].min() == pytest.approx(1.0)
    assert mean_pairwise_cosine(units) == pytest.approx(1.0)
    assert pca_explained_variance(units)[0] == pytest.approx(0.0)


def test_negation_delta_baseline_comparison_reports_real_and_random_rows():
    import numpy as np

    objects = ["dog", "cat", "car"]
    group = NegationPairGroup("bare_no", "{object}", "no {object}")
    embedding_map = {
        "dog": np.array([1.0, 0.0, 0.0]),
        "no dog": np.array([1.0, 1.0, 0.0]),
        "cat": np.array([0.0, 1.0, 0.0]),
        "no cat": np.array([0.0, 2.0, 0.0]),
        "car": np.array([0.0, 0.0, 1.0]),
        "no car": np.array([0.0, 1.0, 1.0]),
    }
    metrics = compute_object_delta_metrics(objects, [group], embedding_map)

    rows = baseline_comparison_rows("bare_no", metrics, embedding_map, seed=1)
    lookup = {row["baseline_type"]: row for row in rows}

    assert set(lookup) == {"real_negation_delta", "object_object_delta", "mismatched_no_delta"}
    assert lookup["real_negation_delta"]["mean_pairwise_cosine"] == pytest.approx(1.0)
    assert lookup["object_object_delta"]["delta_vs_real"] > 0


def test_logical_connector_phrase_inventory_contains_unary_binary_and_generic_phrases():
    phrases = build_phrase_inventory(["dog", "car"], [("dog", "car")])
    vocab = controlled_neighbor_vocabulary(["dog"])

    assert "no dog" in phrases
    assert "without any dog" in phrases
    assert "dog and car" in phrases
    assert "neither dog nor car" in phrases
    assert "empty scene" in phrases
    assert "with no dog" in vocab
    assert neighbor_category("no dog") == "no"
    assert neighbor_category("without dog") == "without"
    assert neighbor_category("with no dog") == "with_no"


def test_logical_connector_distance_metrics_from_fake_embeddings():
    import numpy as np

    metrics = vector_distance_metrics(np.array([1.0, 0.0]), np.array([0.0, 1.0]))

    assert metrics["cosine_similarity"] == pytest.approx(0.0)
    assert metrics["cosine_distance"] == pytest.approx(1.0)
    assert metrics["euclidean_distance"] == pytest.approx(2**0.5)
    assert metrics["squared_euclidean_distance"] == pytest.approx(2.0)
    assert metrics["angular_distance_radians"] == pytest.approx(np.pi / 2)


def test_logical_connector_pairwise_and_dominance_metrics_from_fake_embeddings():
    import numpy as np

    objects = ["dog", "cat"]
    embedding_map = {
        "a dog": np.array([1.0, 0.0, 0.0]),
        "a cat": np.array([0.9, 0.1, 0.0]),
        "with a dog": np.array([1.0, 0.0, 0.0]),
        "with a cat": np.array([0.9, 0.1, 0.0]),
        "a visible dog": np.array([1.0, 0.0, 0.0]),
        "a visible cat": np.array([0.9, 0.1, 0.0]),
        "dog present": np.array([1.0, 0.0, 0.0]),
        "cat present": np.array([0.9, 0.1, 0.0]),
        "no dog": np.array([1.0, 1.0, 0.0]),
        "no cat": np.array([0.9, 1.1, 0.0]),
        "without dog": np.array([1.0, 0.9, 0.0]),
        "without cat": np.array([0.9, 1.0, 0.0]),
        "with no dog": np.array([1.0, 1.0, 0.0]),
        "with no cat": np.array([0.9, 1.1, 0.0]),
        "without any dog": np.array([1.0, 1.0, 0.0]),
        "without any cat": np.array([0.9, 1.1, 0.0]),
        "no visible dog": np.array([1.0, 1.0, 0.0]),
        "no visible cat": np.array([0.9, 1.1, 0.0]),
        "dog absent": np.array([1.0, 1.0, 0.0]),
        "cat absent": np.array([0.9, 1.1, 0.0]),
        "absence of dog": np.array([1.0, 1.0, 0.0]),
        "absence of cat": np.array([0.9, 1.1, 0.0]),
        "not a dog": np.array([1.0, 0.8, 0.0]),
        "not a cat": np.array([0.9, 0.9, 0.0]),
        "not the dog": np.array([1.0, 0.8, 0.0]),
        "not the cat": np.array([0.9, 0.9, 0.0]),
    }
    pairwise = compute_pairwise_distance_metrics(objects, embedding_map)
    dominance = compute_object_dominance_index(objects, UNARY_OPERATORS[:1], embedding_map)

    assert set(pairwise["connector_b"]) >= {"no", "without", "with_no", "not_a"}
    assert len(pairwise) == len(objects) * 9
    assert dominance.iloc[0]["operator"] == "no"
    assert "object_dominance_index" in dominance.columns


def test_logical_connector_binary_metrics_from_fake_embeddings():
    import numpy as np

    embedding_map = {
        "dog": np.array([1.0, 0.0, 0.0]),
        "cat": np.array([0.0, 1.0, 0.0]),
        "no dog": np.array([1.0, 0.0, 1.0]),
        "no cat": np.array([0.0, 1.0, 1.0]),
        "dog and cat": np.array([1.0, 1.0, 0.0]),
        "dog or cat": np.array([0.8, 0.8, 0.0]),
        "dog but not cat": np.array([1.0, 0.0, 0.5]),
        "dog without cat": np.array([1.0, 0.0, 0.5]),
        "only dog": np.array([1.0, 0.0, 0.2]),
        "neither dog nor cat": np.array([0.0, 0.0, 1.0]),
    }

    frame = compute_binary_connector_metrics([("dog", "cat")], embedding_map)

    assert len(frame) == len(BINARY_CONNECTORS) * 6
    assert set(frame["connector"]) == set(BINARY_CONNECTORS)
    assert set(frame["reference"]) == {"object1", "object2", "no_object1", "no_object2", "and", "only_object1"}
    assert frame["delta_norm"].notna().all()
    assert OBJECT_PAIRS[0] == ("dog", "cat")
