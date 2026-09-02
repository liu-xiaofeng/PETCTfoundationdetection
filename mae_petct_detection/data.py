from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .annotations import resolve_case_image_path


def _pad_image(image: np.ndarray, spatial_size: Sequence[int]) -> np.ndarray:
    output_shape = (image.shape[0],) + tuple(
        max(int(current), int(target)) for current, target in zip(image.shape[1:], spatial_size)
    )
    padded = np.zeros(output_shape, dtype=image.dtype)
    if image.shape[0] > 1:
        # Match the PET/CT MAE pretraining FOV padding: PET=0, z-scored CT=-1.
        padded[1] = -1
    source = (slice(None),) + tuple(slice(0, size) for size in image.shape[1:])
    padded[source] = image
    return padded


def _positive_crop_start(
    box: np.ndarray,
    image_shape: np.ndarray,
    crop_shape: np.ndarray,
    rng: random.Random,
) -> np.ndarray:
    start = np.zeros(3, dtype=np.int64)
    for axis in range(3):
        max_start = max(0, int(image_shape[axis] - crop_shape[axis]))
        lower = max(0, int(math.ceil(float(box[axis + 3]))) - int(crop_shape[axis]))
        upper = min(int(math.floor(float(box[axis]))), max_start)
        if lower <= upper:
            start[axis] = rng.randint(lower, upper)
        else:
            center = 0.5 * (float(box[axis]) + float(box[axis + 3]))
            start[axis] = int(np.clip(round(center - crop_shape[axis] / 2), 0, max_start))
    return start


class DetectionPatchDataset(Dataset):
    """Random 96 x 128 x 128 patches. Truncated lesions are not used as targets."""

    def __init__(
        self,
        cases: list[dict[str, Any]],
        patch_size: Sequence[int] = (96, 128, 128),
        positive_crop_probability: float = 0.5,
        samples_per_epoch: int = 400,
        random_flips: bool = True,
        channel_indices: Sequence[int] | None = None,
        processed_root: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        if not cases:
            raise ValueError("Training dataset is empty")
        self.cases = cases
        self.patch_size = np.asarray(patch_size, dtype=np.int64)
        self.positive_crop_probability = positive_crop_probability
        self.samples_per_epoch = samples_per_epoch
        self.random_flips = random_flips
        self.channel_indices = tuple(int(index) for index in channel_indices) if channel_indices is not None else None
        self.processed_root = processed_root
        self.seed = seed
        self.epoch = 0
        self.positive_indices = [index for index, case in enumerate(cases) if case.get("boxes")]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + self.epoch * self.samples_per_epoch + index + worker_id * 9973)

        use_positive = bool(self.positive_indices) and rng.random() < self.positive_crop_probability
        case = self.cases[rng.choice(self.positive_indices) if use_positive else rng.randrange(len(self.cases))]
        image = np.load(resolve_case_image_path(case, self.processed_root), mmap_mode="r")
        image = _pad_image(np.asarray(image), self.patch_size)
        if self.channel_indices is not None:
            image = image[list(self.channel_indices)]
        image_shape = np.asarray(image.shape[1:], dtype=np.int64)
        boxes = np.asarray(case.get("boxes", []), dtype=np.float32).reshape(-1, 6)

        if use_positive and len(boxes):
            start = _positive_crop_start(boxes[rng.randrange(len(boxes))], image_shape, self.patch_size, rng)
        else:
            max_start = np.maximum(0, image_shape - self.patch_size)
            start = np.asarray([rng.randint(0, int(value)) for value in max_start], dtype=np.int64)
        stop = start + self.patch_size
        image_patch = np.asarray(
            image[:, start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]],
            dtype=np.float32,
        ).copy()

        if len(boxes):
            fully_inside = np.all(boxes[:, :3] >= start, axis=1) & np.all(boxes[:, 3:] <= stop, axis=1)
            boxes = boxes[fully_inside]
            boxes[:, :3] -= start
            boxes[:, 3:] -= start

        if self.random_flips:
            for axis in range(3):
                if rng.random() < 0.5:
                    image_patch = np.flip(image_patch, axis=axis + 1).copy()
                    if len(boxes):
                        low = boxes[:, axis].copy()
                        high = boxes[:, axis + 3].copy()
                        boxes[:, axis] = self.patch_size[axis] - high
                        boxes[:, axis + 3] = self.patch_size[axis] - low

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return torch.from_numpy(image_patch), target


def detection_collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
