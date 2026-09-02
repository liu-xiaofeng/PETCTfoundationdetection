"""Lesion matching, FROC, AFROC, and size-stratified operating curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Voxel-count bins on the standardized 2.0 x 2.0 x 3.0 mm^3 grid (paper Sec. 2.6).
SIZE_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("<100", 0, 100),
    ("100-299", 100, 300),
    ("300-499", 300, 500),
    ("500-999", 500, 1000),
    ("1000-2499", 1000, 2500),
    (">=2500", 2500, None),
)


def box_iou_3d(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    low = np.maximum(boxes1[:, None, :3], boxes2[None, :, :3])
    high = np.minimum(boxes1[:, None, 3:], boxes2[None, :, 3:])
    intersection = np.prod(np.maximum(0.0, high - low), axis=-1)
    volume1 = np.prod(np.maximum(0.0, boxes1[:, 3:] - boxes1[:, :3]), axis=-1)
    volume2 = np.prod(np.maximum(0.0, boxes2[:, 3:] - boxes2[:, :3]), axis=-1)
    union = volume1[:, None] + volume2[None, :] - intersection
    return intersection / np.maximum(union, 1e-8)


def voxel_bin_index(n_voxels: int) -> int:
    for index, (_, start, stop) in enumerate(SIZE_BINS):
        if stop is None:
            if n_voxels >= start:
                return index
        elif start <= n_voxels < stop:
            return index
    return len(SIZE_BINS) - 1


def lesion_voxel_counts(case: dict[str, Any]) -> np.ndarray:
    boxes = np.asarray(case.get("boxes", []), dtype=np.float32).reshape(-1, 6)
    raw = case.get("lesion_voxels") or []
    if len(raw) == len(boxes) and len(boxes):
        return np.asarray(raw, dtype=np.int64)
    if not len(boxes):
        return np.zeros((0,), dtype=np.int64)
    volumes = np.prod(np.maximum(1.0, boxes[:, 3:] - boxes[:, :3]), axis=1)
    return np.round(volumes).astype(np.int64)


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """One-to-one, score-ordered matching to the highest-IoU unmatched GT."""

    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    matched_gt = np.full(n_pred, -1, dtype=np.int64)
    if n_pred == 0 or n_gt == 0:
        return matched_gt
    ious = box_iou_3d(pred_boxes, gt_boxes)
    gt_used = np.zeros(n_gt, dtype=bool)
    for pred_index in np.argsort(-pred_scores):
        candidates = np.where(~gt_used & (ious[pred_index] >= iou_threshold))[0]
        if not len(candidates):
            continue
        best = int(candidates[np.argmax(ious[pred_index, candidates])])
        matched_gt[pred_index] = best
        gt_used[best] = True
    return matched_gt


def _as_prediction_map(predictions: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["case_id"]: item for item in predictions}


def match_dataset(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Match every scan against the full GT set before any size stratification."""

    prediction_by_case = _as_prediction_map(predictions)
    matched_scans = []
    for case in cases:
        gt_boxes = np.asarray(case.get("boxes", []), dtype=np.float32).reshape(-1, 6)
        gt_voxels = lesion_voxel_counts(case)
        prediction = prediction_by_case.get(case["case_id"], {"boxes": [], "scores": []})
        pred_boxes = np.asarray(prediction.get("boxes", []), dtype=np.float32).reshape(-1, 6)
        pred_scores = np.asarray(prediction.get("scores", []), dtype=np.float32).reshape(-1)
        if len(pred_boxes) != len(pred_scores):
            raise ValueError(f"Mismatched boxes/scores for {case['case_id']}")
        matched_gt = greedy_match(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        matched_scans.append(
            {
                "case_id": case["case_id"],
                "n_gt": int(len(gt_boxes)),
                "gt_voxels": gt_voxels,
                "pred_scores": pred_scores,
                "matched_gt": matched_gt,
            }
        )
    return matched_scans


def _sensitivity_at_fp(
    fp_per_scan: np.ndarray,
    sensitivity: np.ndarray,
    target: float,
) -> float:
    eligible = fp_per_scan <= target + 1e-12
    if not np.any(eligible):
        return 0.0
    return float(np.max(sensitivity[eligible]))


def _trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x = np.unique(x)
    unique_y = np.array([np.max(y[x == value]) for value in unique_x], dtype=np.float64)
    integrator = getattr(np, "trapezoid", np.trapz)
    return float(integrator(unique_y, unique_x))


def _curve_from_events(
    scores: np.ndarray,
    is_tp: np.ndarray,
    n_lesions: int,
    n_scans: int,
) -> dict[str, Any]:
    if len(scores) == 0:
        fp_per_scan = np.asarray([0.0], dtype=np.float64)
        sensitivity = np.asarray([0.0], dtype=np.float64)
    else:
        order = np.argsort(-scores)
        tp_cum = np.cumsum(is_tp[order].astype(np.float64))
        fp_cum = np.cumsum((~is_tp[order]).astype(np.float64))
        fp_per_scan = fp_cum / max(n_scans, 1)
        sensitivity = tp_cum / max(n_lesions, 1)
    return {
        "fp_per_scan": fp_per_scan,
        "sensitivity": sensitivity,
        "lesions": int(n_lesions),
        "scans": int(n_scans),
    }


def _afroc_from_matches(
    matched_scans: Sequence[dict[str, Any]],
    tp_mask_fn,
) -> dict[str, Any]:
    """LLF vs examination-level FPF on lesion-negative scans."""

    n_lesions = 0
    tp_scores: list[float] = []
    negative_fp_max: list[float] = []
    n_negative = 0
    for scan in matched_scans:
        gt_voxels = np.asarray(scan["gt_voxels"])
        n_gt = int(scan["n_gt"])
        n_lesions += int(np.sum(tp_mask_fn(gt_voxels))) if n_gt else 0
        scores = np.asarray(scan["pred_scores"], dtype=np.float64)
        matched_gt = np.asarray(scan["matched_gt"])
        if n_gt == 0:
            n_negative += 1
            unmatched = scores[matched_gt < 0]
            negative_fp_max.append(float(np.max(unmatched)) if len(unmatched) else -np.inf)
        for pred_index, gt_index in enumerate(matched_gt):
            if gt_index < 0:
                continue
            if tp_mask_fn(gt_voxels[gt_index : gt_index + 1])[0]:
                tp_scores.append(float(scores[pred_index]))

    if n_lesions == 0:
        return {
            "fpf": np.asarray([0.0, 1.0], dtype=np.float64),
            "llf": np.asarray([0.0, 0.0], dtype=np.float64),
            "auc": 0.0,
            "lesions": 0,
            "negative_scans": n_negative,
        }

    thresholds = np.unique(np.concatenate([np.asarray(tp_scores, dtype=np.float64), np.asarray(negative_fp_max)]))
    thresholds = np.sort(thresholds)[::-1]
    fpf = []
    llf = []
    negative_fp_max_arr = np.asarray(negative_fp_max, dtype=np.float64)
    tp_scores_arr = np.asarray(tp_scores, dtype=np.float64)
    for threshold in thresholds:
        n_tp = float(np.sum(tp_scores_arr >= threshold)) if len(tp_scores_arr) else 0.0
        llf.append(n_tp / max(n_lesions, 1))
        if n_negative == 0:
            fpf.append(0.0)
        else:
            fpf.append(float(np.mean(negative_fp_max_arr >= threshold)))
    fpf_arr = np.concatenate([[0.0], np.asarray(fpf, dtype=np.float64), [1.0]])
    last_llf = llf[-1] if llf else 0.0
    llf_arr = np.concatenate([[0.0], np.asarray(llf, dtype=np.float64), [last_llf]])
    return {
        "fpf": fpf_arr,
        "llf": llf_arr,
        "auc": _trapezoid_auc(np.clip(fpf_arr, 0.0, 1.0), llf_arr),
        "lesions": int(n_lesions),
        "negative_scans": int(n_negative),
    }


def summarize_detection(
    cases: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    iou_threshold: float = 0.25,
    fp_targets: Iterable[float] = (1.0, 2.0, 4.0),
) -> dict[str, Any]:
    """Paper metrics: overall and size-stratified FROC / AFROC after global matching."""

    matched_scans = match_dataset(cases, predictions, iou_threshold)
    n_scans = len(matched_scans)
    targets = [float(value) for value in fp_targets]

    def _froc_for_mask(mask_fn) -> dict[str, Any]:
        scores: list[float] = []
        is_tp: list[bool] = []
        n_lesions = 0
        for scan in matched_scans:
            gt_voxels = np.asarray(scan["gt_voxels"])
            n_gt = int(scan["n_gt"])
            selected = mask_fn(gt_voxels) if n_gt else np.zeros((0,), dtype=bool)
            n_lesions += int(selected.sum())
            pred_scores = np.asarray(scan["pred_scores"])
            matched_gt = np.asarray(scan["matched_gt"])
            for pred_index, gt_index in enumerate(matched_gt):
                score = float(pred_scores[pred_index])
                if gt_index < 0:
                    scores.append(score)
                    is_tp.append(False)
                    continue
                if selected[gt_index]:
                    scores.append(score)
                    is_tp.append(True)
                # Predictions matched to GT outside this stratum are neither TP nor FP.
        curve = _curve_from_events(
            np.asarray(scores, dtype=np.float64),
            np.asarray(is_tp, dtype=bool),
            n_lesions=n_lesions,
            n_scans=n_scans,
        )
        operating = {
            str(target): _sensitivity_at_fp(curve["fp_per_scan"], curve["sensitivity"], target)
            for target in targets
        }
        afroc = _afroc_from_matches(matched_scans, mask_fn)
        return {
            "lesions": n_lesions,
            "fp_per_scan": curve["fp_per_scan"].tolist(),
            "sensitivity": curve["sensitivity"].tolist(),
            "sensitivity_at_fp": operating,
            "afroc_fpf": afroc["fpf"].tolist(),
            "afroc_llf": afroc["llf"].tolist(),
            "afroc_auc": afroc["auc"],
        }

    all_mask = lambda voxels: np.ones(len(voxels), dtype=bool)  # noqa: E731
    result: dict[str, Any] = {
        "iou_threshold": float(iou_threshold),
        "fp_targets": targets,
        "n_scans": n_scans,
        "n_negative_scans": int(sum(scan["n_gt"] == 0 for scan in matched_scans)),
        "curves": {"all": _froc_for_mask(all_mask)},
        "size_bins": [
            {"name": name, "min_voxels": start, "max_voxels_exclusive": stop} for name, start, stop in SIZE_BINS
        ],
    }
    for index, (name, start, stop) in enumerate(SIZE_BINS):

        def _bin_mask(voxels: np.ndarray, bin_start=start, bin_stop=stop) -> np.ndarray:
            if bin_stop is None:
                return np.asarray(voxels) >= bin_start
            return (np.asarray(voxels) >= bin_start) & (np.asarray(voxels) < bin_stop)

        result["curves"][name] = _froc_for_mask(_bin_mask)
        result["curves"][name]["bin_index"] = index

    overall = result["curves"]["all"]
    result["selection_metric"] = float(
        np.mean([overall["sensitivity_at_fp"][str(target)] for target in targets])
    )
    return result


def save_detection_artifacts(
    output_dir: str | Path,
    predictions: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(list(predictions), handle)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fp_targets = metrics["fp_targets"]
    plt.figure(figsize=(6.5, 5.5))
    overall = metrics["curves"]["all"]
    plt.plot(overall["fp_per_scan"], overall["sensitivity"], color="C0", lw=2, label="all")
    plt.xlim(0.0, max(fp_targets))
    plt.ylim(0.0, 1.0)
    plt.xlabel("False positives per scan")
    plt.ylabel("Lesion sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "froc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(overall["afroc_fpf"], overall["afroc_llf"], color="C0", lw=2, label=f"AUC={overall['afroc_auc']:.3f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False-positive fraction")
    plt.ylabel("Lesion localization fraction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "afroc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 6.0))
    for name, curve in metrics["curves"].items():
        plt.plot(curve["fp_per_scan"], curve["sensitivity"], label=f"{name} (n={curve['lesions']})")
    plt.xlim(0.0, max(fp_targets))
    plt.ylim(0.0, 1.0)
    plt.xlabel("False positives per scan")
    plt.ylabel("Lesion sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "froc_by_size.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 6.0))
    for name, curve in metrics["curves"].items():
        plt.plot(curve["afroc_fpf"], curve["afroc_llf"], label=f"{name} AUC={curve['afroc_auc']:.3f}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False-positive fraction")
    plt.ylabel("Lesion localization fraction")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "afroc_by_size.png", dpi=180)
    plt.close()


def metrics_table(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, curve in metrics["curves"].items():
        rows[name] = {
            "lesions": curve["lesions"],
            **{f"sen@{target}FP": curve["sensitivity_at_fp"][str(target)] for target in metrics["fp_targets"]},
            "afroc_auc": curve["afroc_auc"],
        }
    return rows
