from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .annotations import load_annotations, resolve_case_image_path
from .data import DetectionPatchDataset, detection_collate
from .evaluation import save_detection_artifacts, summarize_detection
from .model import build_detector


def _distributed_context() -> tuple[bool, int, int, torch.device]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        return True, dist.get_rank(), dist.get_world_size(), device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def _reduce_mean(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.item())


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    path: Path,
    detector: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "network": detector.network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


@torch.no_grad()
def predict_cases(
    detector: Any,
    cases: list[dict[str, Any]],
    device: torch.device,
    processed_root: str | Path | None = None,
    channel_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    detector.eval()
    predictions = []
    for case in cases:
        image_array = np.asarray(np.load(resolve_case_image_path(case, processed_root)), dtype=np.float32)
        if channel_indices is not None:
            image_array = image_array[channel_indices]
        image = torch.from_numpy(np.ascontiguousarray(image_array))
        output = detector([image], use_inferer=True)[0]
        predictions.append(
            {
                "case_id": case["case_id"],
                "boxes": output[detector.target_box_key].detach().cpu().tolist(),
                "scores": output[detector.pred_score_key].detach().cpu().tolist(),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions


def train(config: dict[str, Any]) -> None:
    distributed, rank, world_size, device = _distributed_context()
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = True

    data_config = config["data"]
    training_config = config["training"]
    payload = load_annotations(data_config["annotations"])
    processed_root = data_config.get("processed_root") or payload.get("processed_root")
    splits = payload["splits"]
    training_cases = list(splits["training"])
    selection_cases = list(splits["validation"])
    if not training_cases or not selection_cases:
        raise RuntimeError("Training and validation splits must both be non-empty")

    output_dir = Path(training_config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
    if distributed:
        dist.barrier()

    dataset = DetectionPatchDataset(
        training_cases,
        patch_size=data_config["patch_size"],
        positive_crop_probability=float(data_config.get("positive_crop_probability", 0.5)),
        samples_per_epoch=int(training_config["samples_per_epoch"]),
        random_flips=bool(data_config.get("random_flips", True)),
        channel_indices=data_config.get("channel_indices"),
        processed_root=processed_root,
        seed=seed,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=int(data_config.get("workers", 4)),
        pin_memory=True,
        persistent_workers=int(data_config.get("workers", 4)) > 0,
        collate_fn=detection_collate,
    )

    detector, transfer_report = build_detector(config, device)
    if rank == 0:
        with (output_dir / "initialization.json").open("w", encoding="utf-8") as handle:
            json.dump(transfer_report, handle, indent=2)

    optimizer = torch.optim.Adam(
        [parameter for parameter in detector.parameters() if parameter.requires_grad],
        lr=float(training_config["learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_metric = -1.0
    resume_path = training_config.get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        detector.network.load_state_dict(checkpoint["network"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        best_metric = float(checkpoint.get("best_metric", -1.0))

    train_model: Any = detector
    if distributed:
        train_model = DistributedDataParallel(detector, device_ids=[device.index])

    history: list[dict[str, Any]] = []
    max_steps = training_config.get("max_steps_per_epoch")
    for epoch in range(start_epoch, int(training_config["epochs"])):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_model.train()
        running_loss = 0.0
        running_classification = 0.0
        running_regression = 0.0
        valid_steps = 0
        started = time.time()

        for step, (images, targets) in enumerate(loader):
            if max_steps is not None and step >= int(max_steps):
                break
            image_list = [image.to(device, non_blocking=True) for image in images]
            target_list = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    losses = train_model(image_list, target_list)
                    classification_loss = losses[detector.cls_key]
                    regression_loss = losses[detector.box_reg_key]
                    loss = classification_loss + regression_loss
            except ValueError as exc:
                if "NaN" not in str(exc) and "Inf" not in str(exc):
                    raise
                continue
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            running_classification += float(classification_loss.detach())
            running_regression += float(regression_loss.detach())
            valid_steps += 1

        local_denominator = max(valid_steps, 1)
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": _reduce_mean(running_loss / local_denominator, device, world_size),
            "classification_loss": _reduce_mean(running_classification / local_denominator, device, world_size),
            "regression_loss": _reduce_mean(running_regression / local_denominator, device, world_size),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }

        selection_interval = int(training_config.get("validation_interval", 20))
        should_select = selection_interval > 0 and (epoch + 1) % selection_interval == 0
        if distributed:
            dist.barrier()
        if rank == 0 and should_select:
            limit = training_config.get("validation_max_cases")
            selected_cases = selection_cases[: int(limit)] if limit else selection_cases
            predictions = predict_cases(
                detector,
                selected_cases,
                device,
                processed_root=processed_root,
                channel_indices=data_config.get("channel_indices"),
            )
            metrics = summarize_detection(
                selected_cases,
                predictions,
                iou_threshold=float(config["evaluation"]["iou_threshold"]),
                fp_targets=config["evaluation"]["fp_per_scan"],
            )
            epoch_dir = output_dir / "validation" / f"epoch_{epoch + 1:03d}"
            save_detection_artifacts(epoch_dir, predictions, metrics)
            metric = float(metrics["selection_metric"])
            row["val_froc_mean"] = metric
            row["val_afroc_auc"] = float(metrics["curves"]["all"]["afroc_auc"])
            if metric > best_metric:
                best_metric = metric
                _save_checkpoint(output_dir / "best.pt", detector, optimizer, epoch + 1, best_metric, config)
        if distributed:
            dist.barrier()

        if rank == 0:
            row["best_val_froc_mean"] = best_metric
            history.append(row)
            _write_history(output_dir / "history.csv", history)
            _save_checkpoint(output_dir / "last.pt", detector, optimizer, epoch + 1, best_metric, config)
            print(json.dumps(row), flush=True)

    if distributed:
        dist.destroy_process_group()
