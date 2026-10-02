# Degradation-Aware Deepfake Detection

Code and released models for **degradation-aware training of a deepfake detector** (EfficientNet-B0),
evaluated for robustness to real-world image degradations (compression, down-scaling, blur, noise),
cross-dataset generalisation and prediction stability.

The study is a controlled comparison: **the architecture, data split, optimiser, schedule and decision-threshold
rule are identical across models**. The only thing that changes is how the model is trained.

| Variant | Script | What it is |
|---|---|---|
| **Model A** (baseline) | [`model_a_baseline_efficientnet_b0.py`](model_a_baseline_efficientnet_b0.py) | ImageNet-pretrained torchvision EfficientNet-B0, supervised REAL vs FAKE on clean frames. No degradation augmentation, no consistency loss. |
| **Model B** (proposed) | [`model_b_degradation_aware_efficientnet_b0.py`](model_b_degradation_aware_efficientnet_b0.py) | Same network, trained with random degradation augmentation **plus** a prediction-consistency loss between the clean and the degraded view of each image. |
| **Model B1** (ablation) | same script, `RUN_VARIANT = "B1"` | Degradation augmentation only (consistency weight λ = 0). Isolates the contribution of the consistency term. |

## Pretrained models

All trained checkpoints are public on Kaggle:

**https://www.kaggle.com/models/syedazmulhasansabbir/degradation-aware-deepfake-detection**

Checkpoints follow the naming produced by the scripts, e.g. `ModelA_Baseline_EfficientNetB0_Best.pth`,
`ModelB_DegradationAware_EfficientNetB0_Best.pth` and `ModelB1_DegAugOnly_EfficientNetB0_Best.pth`.
Each `*_Best.pth` is the epoch with the best **clean validation ROC-AUC**.

## Method

**Network.** `torchvision.models.efficientnet_b0` (ImageNet weights), final layer replaced by a single logit
(`BCEWithLogitsLoss`). Label convention: `REAL = 0`, `FAKE = 1`, `P(FAKE) = sigmoid(logit)`.

**Degradation-aware training (Model B).** For every training image, one geometric/colour augmentation
(resize 256 → random crop 224, horizontal flip, light colour jitter) is applied, and then a *degraded view of the
same crop* is generated:

| Per-sample policy | Probability |
|---|---|
| No degradation | 0.20 |
| One degradation | 0.50 |
| Combination (2 ops 80 % / 3 ops 20 %) | 0.30 |

Operations and training ranges: JPEG quality 30–90, down-up scaling 0.40–0.90, Gaussian blur σ 0.3–2.0,
additive Gaussian noise σ 2–12 (0–255 scale), brightness/contrast/saturation 0.75–1.25. JPEG is always applied last,
as in real pipelines.

**Loss.** Clean and degraded views go through the network in one forward pass (2× batch):

```
L = 0.5 · [ BCE(z_clean, y) + BCE(z_deg, y) ]  +  λ · mean | σ(z_clean) − σ(z_deg) |        (λ = 0.2)
```

The clean prediction is used as a stop-gradient anchor (`CONSISTENCY_DETACH_CLEAN = True`) so the degraded view is
pulled toward the clean one rather than both collapsing toward 0.5. Class imbalance is handled with a BCE
`pos_weight` (only if the majority/minority ratio exceeds 1.5); no resampling.

**Shared training setup (A and B).** AdamW (lr 3e-4, weight decay 1e-2), 1 warm-up epoch then cosine decay,
gradient clipping 1.0, mixed precision, batch size 64 source images, up to 20 epochs with early stopping
(patience 5) on validation AUC, seed 42.

## Evaluation protocol

* **Data / split.** FaceForensics++ extracted frames. If the dataset has no train/val/test folders, a
  **source-video-level** 80/10/10 split is created (fake pairs link their target and source IDs, so no source video
  crosses splits; the script asserts zero group overlap). Model A writes `*_Split_Assignment.csv`, and Model B
  **reuses it automatically** so both models see identical splits.
* **Threshold.** Chosen once on **clean validation** (Youden's J) and reused for every test condition, including
  degraded and cross-dataset sets. No threshold is tuned on test data.
* **Degradation conditions (17 test conditions).** Clean; JPEG (quality 70/40/20); resolution (×0.75/0.5/0.25);
  Gaussian blur (σ 1/2/3); noise (σ 5/10/20); combined (blur+JPEG, down+JPEG, down+noise+JPEG) at
  mild/medium/severe; and an **unseen codec** (WebP, quality 30) that is never used in training. The *severe* levels
  of JPEG, resolution, blur and noise lie **outside** the training ranges by design. Degradations are deterministic
  per image (seeded from path + condition), so Model A and Model B see exactly the same degraded pixels.
* **Cross-dataset (trained on FF++ only).** Celeb-DF, DFDC, a diffusion-generated set and WildDeepfake, with up to
  5,000 images per class, using the FF++ validation threshold.
* **Statistics.** 95 % confidence intervals for AUC via video-cluster bootstrap (200 resamples), plus prediction
  stability (mean |Δp| and decision-flip rate vs. the clean prediction) and calibration (ECE, Brier, reliability
  diagrams).
* **Metrics.** ROC-AUC, average precision, accuracy, precision, recall, F1, specificity, balanced accuracy, ECE, Brier,
  and AUC/F1 drop and retention relative to clean.

## Results

> Replace this table with the numbers from your `*_Paper_Summary_Table.csv` files, or delete this section and
> point to the paper.

| Model | Clean AUC | Mean degraded AUC | Mean AUC drop | Unseen-codec (WebP) AUC |
|---|---|---|---|---|
| Model A (baseline) | TBD | TBD | TBD | TBD |
| Model B1 (deg-aug only) | TBD | TBD | TBD | TBD |
| Model B (deg-aug + consistency) | TBD | TBD | TBD | TBD |

## Repository structure

```
.
├── model_a_baseline_efficientnet_b0.py             # Model A: training + full evaluation (one Kaggle cell)
├── model_b_degradation_aware_efficientnet_b0.py    # Model B / B1: training + full evaluation + A-vs-B comparison
├── predict.py                                      # inference with a released checkpoint
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Quick start: inference with a released checkpoint

1. Download a `*_Best.pth` file from the [Kaggle model page](https://www.kaggle.com/models/syedazmulhasansabbir/degradation-aware-deepfake-detection).
2. Install dependencies: `pip install -r requirements.txt`
3. Run:

```bash
python predict.py \
    --ckpt ModelB_DegradationAware_EfficientNetB0_Best.pth \
    --input path/to/image_or_folder \
    --out predictions.csv
```

The script prints `P(FAKE)` and a REAL/FAKE decision per image. The decision threshold used in the paper (from clean
FF++ validation) is stored in the `threshold` column of `<Model>_Final_Clean_Test_Metrics.csv`; pass it with
`--threshold`. The default is 0.5.

Or in Python:

```python
import torch, torch.nn as nn, torchvision
from torchvision import transforms as T
from PIL import Image

model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
ck = torch.load("ModelB_DegradationAware_EfficientNetB0_Best.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"]); model.eval()

tf = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
with torch.inference_mode():
    p_fake = torch.sigmoid(model(tf(Image.open("face.jpg").convert("RGB")).unsqueeze(0))).item()
```

> Checkpoints are full training checkpoints (model, optimiser, config, history), which is why `weights_only=False`
> is needed. Only load checkpoints from sources you trust.

Inputs are expected to be **face-region frames** like the FF++ frames used for training. Detect and crop faces first
if you start from raw video.

## Reproducing the experiments

Both scripts are single self-contained files written for a **Kaggle GPU notebook** (paste into one cell, or save as
a script). They are restart-safe: per-epoch checkpoints are written atomically, and re-running the cell resumes from
the latest valid checkpoint. They stop gracefully before Kaggle's 12-hour limit (`MAX_RUNTIME_HOURS = 11`).

1. **Model A** (baseline): attach the FF++ frames dataset, set `DATA_ROOT`, and run
   `model_a_baseline_efficientnet_b0.py`. Internet must be on to download ImageNet weights (or set
   `PRETRAINED_WEIGHTS_FILE`).
2. **Model B**: run `model_b_degradation_aware_efficientnet_b0.py` with `RUN_VARIANT = "B"`. If Model A's outputs are
   attached as a Kaggle input or present in `/kaggle/working`, Model B reuses its split and automatically produces
   the A-vs-B comparison tables and figures. Set `MODEL_PATH_OVERRIDES` if the auto-search for checkpoints fails.
3. **Ablation (B1)**: run Model B again with `RUN_VARIANT = "B1"` (λ = 0).

Datasets are configured at the top of each script (`DATA_ROOT`, `CROSS_FOLDERS`, `CROSS_PKL`). Edit these paths if you
run outside Kaggle. Set `RUN_CROSS_DATASET = False` to skip cross-dataset evaluation.

### Outputs

Each run writes to `/kaggle/working/<RunName>/`:

| Folder | Contents |
|---|---|
| `checkpoints/` | per-epoch checkpoints and `*_Best.pth` |
| `metrics/` | training history, robustness metrics for all conditions, performance-drop tables, confusion matrices, cross-dataset metrics, bootstrap CIs, prediction stability, paper-ready summary tables (`.csv` and `.tex`), split assignment, run configuration |
| `predictions/` | per-image predictions for every test condition and cross-dataset set |
| `figures/` | loss/AUC curves, ROC, confusion matrix, AUC vs. severity, per-type curves, heatmap, bootstrap CIs, reliability diagrams, score distributions, cross-dataset AUC (PNG and PDF, 300 dpi) |
| `logs/` | log of unreadable or failed images |

## Datasets

Training uses FaceForensics++ frames. Cross-dataset tests use Celeb-DF, DFDC, a diffusion-generated image set and
WildDeepfake. These are third-party datasets under their own licences. This repository does not redistribute any
images, so obtain them from their original sources and follow their terms of use.

## Citation

If you use this code or the released models, please cite:

```bibtex
@article{TODO_key,
  title   = {TODO: paper title},
  author  = {TODO: authors},
  journal = {TODO},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

Open an issue on this repository for questions or problems.
