"""AutoPET I/O: body crop, resample to 2x2x3 mm, z-score, and lesion boxes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

SPACING_XYZ = (2.0, 2.0, 3.0)
BODY_HU_THRESHOLD = -500.0
BODY_PADDING_VOXELS = 3
MIN_LESION_VOXELS = 3
PET_NAME = "PET.nii.gz"
CT_NAME = "CT_resample.nii.gz"
SEG_NAME = "tumorSeg.nii.gz"


def collect_cases(dataset_root: str | Path) -> dict[str, dict[str, str]]:
    """Index AutoPET folders that contain PET, CT, and tumorSeg NIfTI files."""

    dataset_root = Path(dataset_root)
    cases: dict[str, dict[str, str]] = {}
    for pet_path in sorted(dataset_root.rglob(PET_NAME)):
        scan_dir = pet_path.parent
        ct_path = scan_dir / CT_NAME
        seg_path = scan_dir / SEG_NAME
        if not (ct_path.exists() and seg_path.exists()):
            continue
        case_id = scan_dir.relative_to(dataset_root).as_posix()
        cases[case_id] = {
            "case_id": case_id,
            "subject_id": case_id.split("/", 1)[0],
            "pet_path": str(pet_path),
            "ct_path": str(ct_path),
            "seg_path": str(seg_path),
        }
    return cases


def read_split_ids(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _resample_to_spacing(
    image: sitk.Image,
    spacing_xyz: Sequence[float],
    interpolator: int,
) -> sitk.Image:
    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    original_size = np.asarray(image.GetSize(), dtype=np.float64)
    target_spacing = np.asarray(tuple(spacing_xyz), dtype=np.float64)
    new_size = np.maximum(1, np.round(original_size * original_spacing / target_spacing)).astype(int).tolist()
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        tuple(float(value) for value in target_spacing),
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def _body_slices(ct: np.ndarray, padding: int = BODY_PADDING_VOXELS) -> tuple[slice, slice, slice]:
    labels, count = ndimage.label(ct > BODY_HU_THRESHOLD)
    if count == 0:
        return tuple(slice(0, size) for size in ct.shape)  # type: ignore[return-value]
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    coords = np.where(labels == int(sizes.argmax()))
    slices = []
    for axis, indices in enumerate(coords):
        start = max(0, int(indices.min()) - padding)
        stop = min(ct.shape[axis], int(indices.max()) + padding + 1)
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def _zscore(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    mean = float(volume.mean())
    std = float(volume.std())
    return (volume - mean) / (std + 1e-8)


def _component_boxes(mask: np.ndarray, min_voxels: int) -> tuple[list[list[float]], list[int]]:
    labels, count = ndimage.label(mask > 0)
    if count == 0:
        return [], []
    objects = ndimage.find_objects(labels)
    sizes = ndimage.sum(mask > 0, labels, index=np.arange(1, count + 1))
    boxes: list[list[float]] = []
    voxel_counts: list[int] = []
    for index, slices in enumerate(objects):
        n_voxels = int(sizes[index])
        if slices is None or n_voxels < min_voxels:
            continue
        starts = [float(item.start) for item in slices]
        stops = [float(item.stop) for item in slices]
        boxes.append(starts + stops)
        voxel_counts.append(n_voxels)
    return boxes, voxel_counts


def preprocess_case(
    pet_path: str | Path,
    ct_path: str | Path,
    seg_path: str | Path | None = None,
    spacing_xyz: Sequence[float] = SPACING_XYZ,
    min_lesion_voxels: int = MIN_LESION_VOXELS,
) -> dict[str, Any]:
    """Resample PET/CT(/seg), remove blank CT borders, and z-score each modality."""

    pet = sitk.ReadImage(str(pet_path))
    ct = sitk.ReadImage(str(ct_path))
    pet_r = _resample_to_spacing(pet, spacing_xyz, sitk.sitkLinear)
    ct_r = sitk.Resample(ct, pet_r, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    pet_arr = sitk.GetArrayFromImage(pet_r).astype(np.float32)
    ct_arr = sitk.GetArrayFromImage(ct_r).astype(np.float32)
    crop = _body_slices(ct_arr)
    pet_arr = pet_arr[crop]
    ct_arr = ct_arr[crop]
    image = np.stack([_zscore(pet_arr), _zscore(ct_arr)], axis=0).astype(np.float32)

    boxes: list[list[float]] = []
    voxel_counts: list[int] = []
    if seg_path is not None and Path(seg_path).exists():
        seg = sitk.ReadImage(str(seg_path))
        seg_r = sitk.Resample(seg, pet_r, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0, seg.GetPixelID())
        seg_arr = sitk.GetArrayFromImage(seg_r)[crop] > 0
        boxes, voxel_counts = _component_boxes(seg_arr, min_lesion_voxels)

    return {
        "image": image,
        "boxes": boxes,
        "lesion_voxels": voxel_counts,
        "image_shape": list(image.shape[1:]),
        "target_spacing_xyz": [float(value) for value in spacing_xyz],
        "crop_zyx": [[int(item.start), int(item.stop)] for item in crop],
    }


def save_processed_case(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "image.npy"
    np.save(image_path, payload["image"])
    return image_path
