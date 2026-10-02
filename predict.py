"""Run a released checkpoint (Model A / B / B1) on one image or a folder of images.

Usage:
    python predict.py --ckpt ModelB_DegradationAware_EfficientNetB0_Best.pth --input path/to/image_or_folder
    python predict.py --ckpt <ckpt> --input <dir> --threshold 0.47 --out predictions.csv

Output: P(FAKE) per image (label convention: REAL = 0, FAKE = 1, prob = sigmoid(logit)).
The decision threshold used in the paper was selected on clean FF++ validation (Youden's J) and is
stored in the `threshold` column of `<Model>_Final_Clean_Test_Metrics.csv`; pass it via --threshold.
Defaults to 0.5 if omitted.
"""
import argparse, os, sys
import torch, torch.nn as nn, torchvision
from torchvision import transforms as T
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Identical to the evaluation transform used in training/evaluation scripts: resize 256 -> center-crop 224.
TF = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model(ckpt_path: str, device: torch.device) -> nn.Module:
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)  # single logit
    # Training checkpoints are dicts (model/optimizer/config/...) that include non-tensor objects,
    # hence weights_only=False. Only load checkpoints from sources you trust.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    return model.to(device).eval()


def list_images(path: str):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(IMG_EXTS)]
    return sorted(out)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="path to a *_Best.pth checkpoint")
    ap.add_argument("--input", required=True, help="image file or folder (searched recursively)")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold on P(FAKE) (default 0.5)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=None, help="optional CSV path for the predictions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, device)
    paths = list_images(args.input)
    if not paths:
        sys.exit(f"No images found in {args.input}")

    rows = []
    for i in range(0, len(paths), args.batch_size):
        chunk, tensors, ok = paths[i:i + args.batch_size], [], []
        for p in chunk:
            try:
                tensors.append(TF(Image.open(p).convert("RGB"))); ok.append(p)
            except Exception as e:
                print(f"skipping unreadable image {p}: {e}", file=sys.stderr)
        if not tensors:
            continue
        x = torch.stack(tensors).to(device)
        probs = torch.sigmoid(model(x).squeeze(1)).float().cpu().tolist()
        rows += list(zip(ok, probs))

    for p, pr in rows:
        print(f"{pr:.4f}\t{'FAKE' if pr >= args.threshold else 'REAL'}\t{p}")
    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image_path", "prob_fake", "predicted_label"])
            for p, pr in rows:
                w.writerow([p, f"{pr:.6f}", int(pr >= args.threshold)])
        print(f"saved {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
