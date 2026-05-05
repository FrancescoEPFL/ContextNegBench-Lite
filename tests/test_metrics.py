import pytest

from negcompbench.eval.metrics import compute_prediction, summarize_results


def test_compute_prediction_marks_correct_when_correct_caption_highest():
    pred = compute_prediction(0.7, [0.2, 0.4])
    assert pred["is_correct"]
    assert pred["prediction_index"] == 0
    assert pred["margin"] == pytest.approx(0.3)


def test_summarize_results_includes_task_rows():
    rows = [
        {"task_type": "negation", "relation": "border_absence", "model_name": "m", "device": "cpu", "is_correct": True, "margin": 0.2},
        {"task_type": "negation", "relation": "border_presence", "model_name": "m", "device": "cpu", "is_correct": False, "margin": -0.1},
    ]
    summary = summarize_results(rows)
    overall = summary[(summary["group"] == "overall") & (summary["value"] == "all")].iloc[0]
    assert overall["n"] == 2
    assert overall["pairwise_accuracy"] == 0.5
