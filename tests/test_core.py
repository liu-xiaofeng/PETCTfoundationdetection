from __future__ import annotations

import numpy as np

from mae_petct_detection.config import load_config
from mae_petct_detection.data import _pad_image
from mae_petct_detection.evaluation import (
    box_iou_3d,
    greedy_match,
    metrics_table,
    summarize_detection,
    voxel_bin_index,
)
from mae_petct_detection.preprocess import _zscore


def test_box_iou() -> None:
    boxes = np.asarray([[0, 0, 0, 10, 10, 10]], dtype=np.float32)
    assert np.allclose(box_iou_3d(boxes, boxes), 1.0)
    disjoint = np.asarray([[20, 20, 20, 30, 30, 30]], dtype=np.float32)
    assert np.allclose(box_iou_3d(boxes, disjoint), 0.0)


def test_greedy_match_is_one_to_one() -> None:
    gt = np.asarray([[0, 0, 0, 4, 4, 4], [3, 3, 3, 8, 8, 8]], dtype=np.float32)
    pred = np.asarray([[0, 0, 0, 4, 4, 4], [0, 0, 0, 5, 5, 5], [10, 10, 10, 12, 12, 12]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    matched = greedy_match(pred, scores, gt, iou_threshold=0.25)
    assert matched[0] == 0
    assert matched[1] == -1 or matched[1] == 1
    assert int((matched >= 0).sum()) <= 2
    assert len(set(matched[matched >= 0].tolist())) == int((matched >= 0).sum())


def test_modality_padding() -> None:
    image = np.ones((2, 2, 2, 2), dtype=np.float32)
    padded = _pad_image(image, (4, 4, 4))
    assert padded.shape == (2, 4, 4, 4)
    assert padded[0, -1, -1, -1] == 0
    assert padded[1, -1, -1, -1] == -1


def test_zscore_zero_mean_unit_std() -> None:
    volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    normalized = _zscore(volume)
    assert abs(float(normalized.mean())) < 1e-5
    assert abs(float(normalized.std()) - 1.0) < 1e-5


def test_voxel_bins_match_paper() -> None:
    assert voxel_bin_index(99) == 0
    assert voxel_bin_index(100) == 1
    assert voxel_bin_index(299) == 1
    assert voxel_bin_index(300) == 2
    assert voxel_bin_index(2500) == 5


def test_perfect_predictions_froc_afroc() -> None:
    cases = []
    predictions = []
    for index, voxels in enumerate((50, 150, 400, 800, 1500, 4000)):
        box = [0, 0, 0, 4, 4, 4]
        cases.append({"case_id": f"pos-{index}", "boxes": [box], "lesion_voxels": [voxels]})
        predictions.append({"case_id": f"pos-{index}", "boxes": [box], "scores": [0.9]})
    cases.append({"case_id": "neg-0", "boxes": [], "lesion_voxels": []})
    predictions.append({"case_id": "neg-0", "boxes": [], "scores": []})
    metrics = summarize_detection(cases, predictions, iou_threshold=0.25, fp_targets=[1.0, 2.0, 4.0])
    table = metrics_table(metrics)
    assert table["all"]["sen@1.0FP"] == 1.0
    assert table["all"]["afroc_auc"] == 1.0
    assert all(table[name]["sen@1.0FP"] == 1.0 for name in ("<100", "100-299", ">=2500"))


def test_false_positives_are_shared_across_size_strata() -> None:
    cases = [
        {"case_id": "pos", "boxes": [[0, 0, 0, 2, 2, 2]], "lesion_voxels": [50]},
        {"case_id": "neg", "boxes": [], "lesion_voxels": []},
    ]
    predictions = [
        {
            "case_id": "pos",
            "boxes": [[0, 0, 0, 2, 2, 2], [10, 10, 10, 12, 12, 12]],
            "scores": [0.9, 0.8],
        },
        {"case_id": "neg", "boxes": [[1, 1, 1, 3, 3, 3]], "scores": [0.7]},
    ]
    metrics = summarize_detection(cases, predictions, iou_threshold=0.25, fp_targets=[1.0, 2.0])
    fp_at_end = metrics["curves"]["all"]["fp_per_scan"][-1]
    assert abs(fp_at_end - metrics["curves"]["<100"]["fp_per_scan"][-1]) < 1e-6
    assert abs(fp_at_end - metrics["curves"][">=2500"]["fp_per_scan"][-1]) < 1e-6
    assert metrics["curves"][">=2500"]["lesions"] == 0
    assert metrics["n_negative_scans"] == 1


def test_nested_config_inheritance(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    middle = tmp_path / "middle.yaml"
    leaf = tmp_path / "leaf.yaml"
    base.write_text("model:\n  width: 32\n  depth: 4\n", encoding="utf-8")
    middle.write_text("_base_: base.yaml\nmodel:\n  depth: 5\n", encoding="utf-8")
    leaf.write_text("_base_: middle.yaml\nmodel:\n  width: 64\n", encoding="utf-8")
    config = load_config(leaf)
    assert config["model"] == {"width": 64, "depth": 5}
