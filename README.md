# Whole-Body FDG PET/CT 3D Multi-Lesion Detection

Training and evaluation code for:

> Liu et al., *Quantifying Multi-Lesion Detection and Localization in Whole-Body FDG PET/CT with a Multimodal Foundation Model*, Physics in Medicine & Biology (manuscript).

The detector is a one-stage 3D RetinaNet with a PlainConvUNet (nnU-Net v2 3D full-resolution) encoder–decoder backbone. This repository contains **detection training, inference, and FROC/AFROC evaluation only**. It does **not** include MAE pretraining code, foundation-model weights, run outputs, or intermediate files.

Four matched configurations from the paper are supported:

| Config | Input | Initialization | Epochs | Learning rate |
|---|---|---|---:|---:|
| `configs/petct_scratch.yaml` | PET+CT | random | 300 | `1e-4` |
| `configs/pet_only.yaml` | PET | random | 300 | `1e-4` |
| `configs/ct_only.yaml` | CT | random | 300 | `1e-4` |
| `configs/petct_pretrained.yaml` | PET+CT | MAE PlainConvUNet weights (optional) | 100 | `1e-5` |

Pretrained initialization loads an off-the-shelf PlainConvUNet checkpoint from the companion PET/CT foundation-model repository. Those weights are **not** shipped here.

## Requirements

- Python 3.10+
- CUDA GPU (paper: NVIDIA A100)
- AutoPET FDG PET/CT with tumor masks ([Gatidis et al., *Scientific Data* 2022](https://doi.org/10.1038/s41597-022-01718-3))

```bash
pip install -r requirements.txt
```

Expected per-scan files under `--data-root`:

```
<data-root>/<case_id>/PET.nii.gz
<data-root>/<case_id>/CT_resample.nii.gz
<data-root>/<case_id>/tumorSeg.nii.gz
```

`case_id` may be a single folder or `subject/scan` for multi-scan subjects. Subject IDs for the paper split are listed in `splits/`.

## Quick start

### 1. Preprocess and build boxes

Blank CT borders are removed, PET/CT/masks are resampled to `2.0 x 2.0 x 3.0 mm`, and each modality is z-scored per volume. Connected components with fewer than 3 voxels are dropped; remaining components become axis-aligned boxes.

```bash
python scripts/prepare_data.py --config configs/default.yaml \
  --data-root /path/to/AutoPET_FDG \
  --processed-root data/processed
```

This writes `data/annotations.json` and cached `data/processed/<case_id>/image.npy` (PET, CT). Cached arrays are gitignored.

### 2. Train

Each epoch has 200 iterations of batch size 2 (`96 x 128 x 128` patches, Z/Y/X). Optimizer is Adam.

```bash
# PET+CT from scratch
python scripts/train.py --config configs/petct_scratch.yaml --seed 42

# PET-only / CT-only
python scripts/train.py --config configs/pet_only.yaml --seed 42
python scripts/train.py --config configs/ct_only.yaml --seed 42

# PET+CT fine-tuned from MAE PlainConvUNet weights
python scripts/train.py --config configs/petct_pretrained.yaml \
  --pretrained-checkpoint /path/to/plainconvunet_mae.pth --seed 42
```

The paper reports mean ± SD over three independent seeds. Repeat with `--seed 42, 43, 44` (or any three seeds).

Checkpoints: `outputs/<run>/last.pt` and `best.pt` (selected on validation FROC at 1/2/4 FP/scan). The 200-case test split is never used for model selection.

### 3. Evaluate

Inference uses overlapping `96 x 128 x 128` windows (50% overlap), keeps candidates with score > 0.5 at each pyramid level, and applies 3D NMS (IoU 0.1). FROC/AFROC then sweep the post-NMS scores. A predicted box is a hit if IoU ≥ 0.25 with an unmatched ground-truth box (one-to-one, score-ordered).

```bash
python scripts/evaluate.py --config configs/petct_scratch.yaml \
  --checkpoint outputs/petct_scratch/best.pt --split test
```

IoU robustness (0.10 / 0.25 / 0.50) without retraining:

```bash
python scripts/evaluate.py --config configs/petct_scratch.yaml \
  --checkpoint outputs/petct_scratch/best.pt --split test \
  --iou-threshold 0.10 0.25 0.50
```

Outputs (gitignored): `predictions.json`, `metrics.json`, FROC/AFROC plots, and a size-stratified table.

## Paper ↔ code

| Paper | This repository |
|---|---|
| 1,014 studies; 700 / 114 / 200 subject-independent split | `splits/train.txt`, `val.txt`, `test.txt` |
| Spacing `2.0 x 2.0 x 3.0 mm`; volume-wise z-score; blank CT borders removed | `mae_petct_detection/preprocess.py` |
| Patch `96 x 128 x 128` (Z, Y, X); inference overlap 50% | `configs/default.yaml` |
| Components < 3 voxels excluded | `min_lesion_voxels: 3` |
| PlainConvUNet; heads on P2–P5; P1 omitted | `mae_petct_detection/model.py` |
| Heads: four shared `3 x 3 x 3` convs, 128 channels | MONAI `RetinaNet` heads |
| K-means anchors `(3.1, 5.9, 5.1)`, `(12.0, 20.3, 18.7)`, `(41.0, 52.2, 55.3)` (ZYX voxels) | `model.base_anchor_shapes` |
| ATSS, 4 candidates, center need not lie in the GT box | `matcher: atss`, `atss_num_candidates: 4` |
| Focal loss α=0.25, γ=2; Smooth-L1 (β=1); `L = L_cls + L_box` | `PositiveNormalizedFocalLoss`, `PositiveNormalizedSmoothL1` |
| Score > 0.5 per level; NMS IoU 0.1 | `inference.score_threshold`, `nms_threshold` |
| Match IoU ≥ 0.25 | `evaluation.iou_threshold` |
| Scratch 300 epochs, `1e-4`; pretrained 100 epochs, `1e-5`; 200 iter/epoch; batch 2; Adam | training configs |
| Size bins `<100`, `100–299`, `300–499`, `500–999`, `1000–2499`, `≥2500` voxels | `mae_petct_detection/evaluation.py` |
| Size-stratified FROC uses global matching; FP = unmatched to any GT | `summarize_detection()` |
| Ablations: generic anchors / no ATSS / no P2 | `configs/ablation_*.yaml` |

## Ablations

```bash
python scripts/train.py --config configs/ablation_scratch_generic_anchors.yaml
python scripts/train.py --config configs/ablation_scratch_no_atss.yaml
python scripts/train.py --config configs/ablation_scratch_no_p2.yaml
```

The same three ablations exist for MAE initialization (`configs/ablation_pretrained_*.yaml`).

## Tests

```bash
python -m pytest
```

## Layout

```
configs/                 # Paper hyperparameters
splits/                  # Train / val / test case IDs
mae_petct_detection/     # Data, model, training, FROC/AFROC
scripts/prepare_data.py
scripts/train.py
scripts/evaluate.py
tests/
```

## Citation

Please cite the detection paper and the AutoPET dataset. MAE pretrained PlainConvUNet initialization is described in the companion PET/CT foundation-model paper and is obtained from that project’s repository, not from this one.

## License

MIT — see [LICENSE](LICENSE).
