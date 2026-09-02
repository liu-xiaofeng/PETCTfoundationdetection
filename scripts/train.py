#!/usr/bin/env python
"""Train the PlainConvUNet 3D RetinaNet detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mae_petct_detection.config import load_config, resolve_path
from mae_petct_detection.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D PET/CT lesion detection.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "petct_scratch.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    data = config["data"]
    if args.data_root:
        data["raw_root"] = str(resolve_path(args.data_root, config))
    data["processed_root"] = str(resolve_path(args.processed_root or data["processed_root"], config))
    data["annotations"] = str(resolve_path(args.annotations or data["annotations"], config))
    data["splits_dir"] = str(resolve_path(data["splits_dir"], config))
    config["training"]["output_dir"] = str(resolve_path(args.output_dir or config["training"]["output_dir"], config))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_steps_per_epoch is not None:
        config["training"]["max_steps_per_epoch"] = args.max_steps_per_epoch
    if args.resume:
        config["training"]["resume"] = str(Path(args.resume).resolve())
    elif config["training"].get("resume"):
        config["training"]["resume"] = str(resolve_path(config["training"]["resume"], config))

    foundation = config.setdefault("foundation", {})
    checkpoint = args.pretrained_checkpoint or foundation.get("checkpoint")
    if checkpoint:
        foundation["checkpoint"] = str(resolve_path(checkpoint, config))
    train(config)


if __name__ == "__main__":
    main()
