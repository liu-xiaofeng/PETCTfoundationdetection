#!/usr/bin/env python
"""Evaluate FROC, AFROC, and lesion-size-stratified operating curves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mae_petct_detection.annotations import load_annotations
from mae_petct_detection.config import load_config, resolve_path
from mae_petct_detection.evaluation import metrics_table, save_detection_artifacts, summarize_detection
from mae_petct_detection.model import build_detector
from mae_petct_detection.training import predict_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whole-body PET/CT lesion detection.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "petct_scratch.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--iou-threshold",
        type=float,
        nargs="+",
        default=None,
        help="One or more matching IoU thresholds. Default: the config value (paper: 0.25).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    data["annotations"] = str(resolve_path(args.annotations or data["annotations"], config))
    payload = load_annotations(data["annotations"])
    processed_root = resolve_path(args.processed_root or data.get("processed_root") or payload.get("processed_root"), config)
    data["processed_root"] = str(processed_root) if processed_root is not None else None
    # The detection checkpoint already contains the trained backbone and heads.
    config.setdefault("foundation", {})["enabled"] = False

    cases = list(payload["splits"][args.split])
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, transfer_report = build_detector(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    network_state = checkpoint.get("network", checkpoint.get("model_state_dict", checkpoint))
    detector.network.load_state_dict(network_state)

    predictions = predict_cases(
        detector,
        cases,
        device,
        processed_root=data["processed_root"],
        channel_indices=data.get("channel_indices"),
    )

    iou_thresholds = args.iou_threshold or [float(config["evaluation"]["iou_threshold"])]
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / f"{args.split}_eval"
    )
    report = {"initialization": transfer_report, "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split}
    for iou in iou_thresholds:
        metrics = summarize_detection(
            cases,
            predictions,
            iou_threshold=float(iou),
            fp_targets=config["evaluation"]["fp_per_scan"],
        )
        iou_dir = output_root if len(iou_thresholds) == 1 else output_root / f"iou_{iou:.2f}"
        save_detection_artifacts(iou_dir, predictions, metrics)
        report[f"iou_{iou:.2f}"] = metrics_table(metrics)
        print(json.dumps({"iou": iou, "metrics": metrics_table(metrics)}, indent=2))

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
