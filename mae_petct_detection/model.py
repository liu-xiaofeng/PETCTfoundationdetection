"""PlainConvUNet encoder-decoder backbone with RetinaNet heads on P2-P5."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import RetinaNet
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.losses.focal_loss import sigmoid_focal_loss
from torch import nn

# Decoder stage index -> FPN level. P1 (stride 1) is omitted as in RetinaNet.
# Stage 0: stride 16 / 320 ch, stage 1: stride 8 / 256, stage 2: stride 4 / 128, stage 3: stride 2 / 64.
LEVEL_SPECS: dict[str, tuple[int, int]] = {
    "p5": (0, 320),
    "p4": (1, 256),
    "p3": (2, 128),
    "p2": (3, 64),
}
LEVEL_ORDER = ("p2", "p3", "p4", "p5")


def build_plain_conv_unet(input_channels: int = 2) -> PlainConvUNet:
    return PlainConvUNet(
        input_channels=input_channels,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1]] + [[2, 2, 2]] * 5,
        n_conv_per_stage=[2] * 6,
        num_classes=2,
        n_conv_per_stage_decoder=[2] * 5,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=False,
    )


class PlainConvUNetPyramid(nn.Module):
    """U-Net decoder features at P2-P5, projected to 128 channels for shared heads."""

    out_channels = 128

    def __init__(self, input_channels: int = 2, pyramid_levels: Sequence[str] = LEVEL_ORDER) -> None:
        super().__init__()
        levels = tuple(str(level).lower() for level in pyramid_levels)
        unknown = [level for level in levels if level not in LEVEL_SPECS]
        if unknown:
            raise ValueError(f"Unsupported pyramid levels: {unknown}")
        self.pyramid_levels = tuple(level for level in LEVEL_ORDER if level in levels)
        if not self.pyramid_levels:
            raise ValueError("At least one pyramid level is required")
        self.nnunet = build_plain_conv_unet(input_channels=input_channels)
        self.laterals = nn.ModuleDict()
        for level in self.pyramid_levels:
            _, in_channels = LEVEL_SPECS[level]
            self.laterals[level] = nn.Sequential(
                nn.Conv3d(in_channels, self.out_channels, kernel_size=1, bias=True),
                nn.Conv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=True),
            )
        for module in self.laterals.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        skips = self.nnunet.encoder(images)
        decoded = skips[-1]
        max_stage = max(LEVEL_SPECS[level][0] for level in self.pyramid_levels)
        collected: dict[str, torch.Tensor] = {}
        for stage_index in range(max_stage + 1):
            decoded = self.nnunet.decoder.transpconvs[stage_index](decoded)
            decoded = torch.cat((decoded, skips[-(stage_index + 2)]), dim=1)
            decoded = self.nnunet.decoder.stages[stage_index](decoded)
            for level in self.pyramid_levels:
                if LEVEL_SPECS[level][0] == stage_index:
                    collected[level] = self.laterals[level](decoded)
        return OrderedDict((level, collected[level]) for level in self.pyramid_levels)


class PositiveNormalizedFocalLoss(nn.Module):
    """RetinaNet focal loss averaged over the number of positive anchors."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        loss = sigmoid_focal_loss(logits, targets, gamma=self.gamma, alpha=self.alpha)
        n_pos = targets.sum().clamp(min=1.0)
        return loss.sum() / n_pos


class PositiveNormalizedSmoothL1(nn.Module):
    """Smooth-L1 with beta=1, summed over the 6 box deltas and divided by N+."""

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predicted.numel() == 0 or predicted.shape[0] == 0:
            return predicted.sum() * 0.0
        loss = F.smooth_l1_loss(predicted, target, beta=1.0, reduction="none")
        return loss.sum() / predicted.shape[0]


def load_backbone_weights(backbone: PlainConvUNetPyramid, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if not isinstance(source, dict):
        raise TypeError("Pretrained checkpoint does not contain a state dictionary")

    target = backbone.nnunet.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for source_key, value in source.items():
        key = source_key
        for prefix in ("module.", "model.", "nnunet."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("seg_outputs") or ".seg_layers." in key:
            continue
        if key in target and target[key].shape == value.shape:
            transferred[key] = value
        elif key in target:
            skipped_shape.append(source_key)

    missing, unexpected = backbone.nnunet.load_state_dict(transferred, strict=False)
    encoder_loaded = sum(name.startswith("encoder.") for name in transferred)
    if encoder_loaded == 0:
        raise RuntimeError(
            "No PlainConvUNet encoder tensors were transferred. "
            "Provide the MAE-pretrained nnU-Net checkpoint from the companion PET/CT foundation-model repository."
        )
    return {
        "loaded_tensors": len(transferred),
        "loaded_encoder_tensors": encoder_loaded,
        "missing_tensors": len([name for name in missing if "seg_layers" not in name]),
        "unexpected_tensors": len(unexpected),
        "shape_mismatches": skipped_shape,
    }


def build_detector(config: dict[str, Any], device: torch.device) -> tuple[RetinaNetDetector, dict[str, Any]]:
    model_config = config["model"]
    channel_indices = config.get("data", {}).get("channel_indices")
    input_channels = int(
        model_config.get(
            "input_channels",
            len(channel_indices) if channel_indices is not None else 2,
        )
    )
    if channel_indices is not None and input_channels != len(channel_indices):
        raise ValueError("model.input_channels must match data.channel_indices")

    pyramid_levels = model_config.get("pyramid_levels", list(LEVEL_ORDER))
    feature_extractor = PlainConvUNetPyramid(input_channels=input_channels, pyramid_levels=pyramid_levels)
    foundation = config.get("foundation") or {}
    if foundation.get("enabled", False):
        if input_channels != 2:
            raise ValueError("MAE pretrained initialization is defined for two-channel PET+CT input")
        checkpoint = foundation.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "foundation.enabled is true but foundation.checkpoint is empty. "
                "Download the PlainConvUNet MAE weights from the companion foundation-model repository."
            )
        transfer_report = load_backbone_weights(feature_extractor, checkpoint)
        transfer_report["initialization"] = "mae_pretrained"
    else:
        feature_extractor.nnunet.apply(InitWeights_He(1e-2))
        transfer_report = {"initialization": "random", "loaded_tensors": 0}

    transfer_report["pyramid_levels"] = list(feature_extractor.pyramid_levels)
    n_levels = len(feature_extractor.pyramid_levels)
    feature_map_scales = model_config.get("feature_map_scales") or [1] * n_levels
    if len(feature_map_scales) != n_levels:
        raise ValueError("model.feature_map_scales must match the number of pyramid levels")

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=feature_map_scales,
        base_anchor_shapes=model_config["base_anchor_shapes"],
        indexing="ij",
    )
    network = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
        feature_extractor=feature_extractor,
        size_divisible=[32, 32, 32],
    )
    detector = RetinaNetDetector(network=network, anchor_generator=anchor_generator).to(device)
    detector.set_cls_loss(
        PositiveNormalizedFocalLoss(
            alpha=float(model_config.get("focal_alpha", 0.25)),
            gamma=float(model_config.get("focal_gamma", 2.0)),
        )
    )
    detector.set_box_regression_loss(PositiveNormalizedSmoothL1(), encode_gt=True, decode_pred=False)

    matcher = str(model_config.get("matcher", "atss")).lower()
    if matcher == "atss":
        detector.set_atss_matcher(
            num_candidates=int(model_config.get("atss_num_candidates", 4)),
            center_in_gt=False,
        )
    elif matcher == "iou":
        detector.set_regular_matcher(
            fg_iou_thresh=float(model_config.get("fg_iou_thresh", 0.5)),
            bg_iou_thresh=float(model_config.get("bg_iou_thresh", 0.4)),
            allow_low_quality_matches=True,
        )
    else:
        raise ValueError(f"Unsupported model.matcher: {matcher}")
    transfer_report["matcher"] = matcher

    detector.set_target_keys(box_key="boxes", label_key="labels")
    apply_inference_settings(detector, config)
    return detector, transfer_report


def apply_inference_settings(detector: RetinaNetDetector, config: dict[str, Any]) -> None:
    inference = config["inference"]
    detector.set_box_selector_parameters(
        score_thresh=float(inference["score_threshold"]),
        topk_candidates_per_level=int(inference["topk_candidates_per_level"]),
        nms_thresh=float(inference["nms_threshold"]),
        detections_per_img=int(inference["detections_per_image"]),
    )
    detector.set_sliding_window_inferer(
        roi_size=config["data"]["patch_size"],
        overlap=float(inference["overlap"]),
        sw_batch_size=int(inference["sw_batch_size"]),
        mode=str(inference.get("blend_mode", "constant")),
        sw_device=next(detector.network.parameters()).device,
        device="cpu",
    )
