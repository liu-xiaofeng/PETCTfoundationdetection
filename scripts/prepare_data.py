#!/usr/bin/env python
"""Preprocess AutoPET FDG PET/CT and build box annotations for the paper splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mae_petct_detection.annotations import build_annotations
from mae_petct_detection.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess AutoPET volumes and write detection annotations.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-root", default=None, help="AutoPET FDG root containing PET/CT/seg NIfTI files.")
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-reuse", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    dataset_root = resolve_path(args.data_root or data["raw_root"], config)
    processed_root = resolve_path(args.processed_root or data["processed_root"], config)
    splits_dir = resolve_path(args.splits_dir or data["splits_dir"], config)
    output_path = resolve_path(args.output or data["annotations"], config)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            f"AutoPET root not found: {dataset_root}. Pass --data-root pointing to folders with PET.nii.gz, "
            "CT_resample.nii.gz, and tumorSeg.nii.gz."
        )

    payload = build_annotations(
        dataset_root=dataset_root,
        processed_root=processed_root,
        splits_dir=splits_dir,
        output_path=output_path,
        workers=int(args.workers if args.workers is not None else data.get("annotation_workers", 1)),
        reuse_existing=not args.no_reuse,
    )
    summary = {
        name: {
            "scans": len(cases),
            "positive": sum(bool(case.get("boxes")) for case in cases),
            "lesions": sum(len(case.get("boxes", [])) for case in cases),
        }
        for name, cases in payload["splits"].items()
    }
    print(json.dumps({"annotations": str(output_path), "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
