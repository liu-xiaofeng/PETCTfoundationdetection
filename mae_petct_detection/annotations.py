from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .preprocess import (
    MIN_LESION_VOXELS,
    SPACING_XYZ,
    collect_cases,
    preprocess_case,
    read_split_ids,
    save_processed_case,
)


def load_split_lists(splits_dir: str | Path) -> dict[str, list[str]]:
    splits_dir = Path(splits_dir)
    mapping = {"training": "train.txt", "validation": "val.txt", "test": "test.txt"}
    splits = {}
    for name, filename in mapping.items():
        path = splits_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = read_split_ids(path)
    return splits


def _prepare_one(job: dict[str, Any]) -> dict[str, Any]:
    processed_dir = Path(job["processed_dir"])
    image_path = processed_dir / "image.npy"
    if job["reuse_existing"] and image_path.exists() and (processed_dir / "meta.json").exists():
        with (processed_dir / "meta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return meta

    payload = preprocess_case(
        job["pet_path"],
        job["ct_path"],
        job["seg_path"],
        spacing_xyz=job["spacing_xyz"],
        min_lesion_voxels=job["min_lesion_voxels"],
    )
    save_processed_case(processed_dir, payload)
    meta = {
        "case_id": job["case_id"],
        "subject_id": job["subject_id"],
        "split": job["split"],
        "image_path": str(Path(job["case_id"]) / "image.npy"),
        "image_shape": payload["image_shape"],
        "boxes": payload["boxes"],
        "lesion_voxels": payload["lesion_voxels"],
        "has_lesion": bool(payload["boxes"]),
        "target_spacing_xyz": payload["target_spacing_xyz"],
        "crop_zyx": payload["crop_zyx"],
    }
    with (processed_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    return meta


def build_annotations(
    dataset_root: str | Path,
    processed_root: str | Path,
    splits_dir: str | Path,
    output_path: str | Path,
    spacing_xyz: tuple[float, float, float] = SPACING_XYZ,
    min_lesion_voxels: int = MIN_LESION_VOXELS,
    workers: int = 1,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    processed_root = Path(processed_root)
    catalog = collect_cases(dataset_root)
    splits = load_split_lists(splits_dir)

    jobs = []
    missing = []
    for split_name, case_ids in splits.items():
        for case_id in case_ids:
            source = catalog.get(case_id)
            if source is None:
                missing.append(case_id)
                continue
            jobs.append(
                {
                    "case_id": case_id,
                    "subject_id": source["subject_id"],
                    "split": split_name,
                    "pet_path": source["pet_path"],
                    "ct_path": source["ct_path"],
                    "seg_path": source["seg_path"],
                    "processed_dir": str(processed_root / case_id),
                    "spacing_xyz": list(spacing_xyz),
                    "min_lesion_voxels": int(min_lesion_voxels),
                    "reuse_existing": bool(reuse_existing),
                }
            )
    if missing:
        raise RuntimeError(f"{len(missing)} split cases were not found under {dataset_root}; first={missing[0]}")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_prepare_one, jobs, chunksize=1))
    else:
        results = [_prepare_one(job) for job in jobs]

    annotated: dict[str, list[dict[str, Any]]] = {name: [] for name in splits}
    for meta in results:
        annotated[meta["split"]].append(meta)

    payload = {
        "coordinate_order": "zyxzyx",
        "processed_root": str(processed_root),
        "preprocessing": {
            "target_spacing_xyz": list(spacing_xyz),
            "body_threshold_hu": -500,
            "body_padding_voxels": 3,
            "normalization": "per-volume z-score per modality",
            "min_lesion_voxels": min_lesion_voxels,
            "channel_order": ["PET", "CT"],
        },
        "splits": annotated,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def load_annotations(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_case_image_path(case: dict[str, Any], processed_root: str | Path | None = None) -> Path:
    image_path = Path(case["image_path"])
    if image_path.is_absolute():
        return image_path
    root = Path(processed_root or case.get("processed_root") or ".")
    return root / image_path
