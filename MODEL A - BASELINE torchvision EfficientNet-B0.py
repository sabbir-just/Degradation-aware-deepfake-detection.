# =====================================================================
# MODEL A - BASELINE: torchvision EfficientNet-B0, supervised REAL vs FAKE
# FF++ extracted frames.  NO KD / degradation / consistency / frequency /
# temporal / attention / fusion.  ONE Kaggle cell, restart-safe.
# =====================================================================
import os, sys, re, io, json, time, math, random, pickle, shutil, hashlib, glob, zlib, platform, warnings, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
T_START = time.time()

# ============================ 1. CONFIG ==============================
DATA_ROOT  = "/kaggle/input/datasets/syedazmulhasansabbir/face-forensics-frames/Face Forensics Frames"
RUN_NAME   = "ModelA_Baseline_EfficientNetB0"
OUT_DIR    = f"/kaggle/working/{RUN_NAME}"
SEED       = 42
IMG_SIZE, RESIZE_SIZE = 224, 256          # train: resize->RandomCrop(IMG); eval: resize->CenterCrop(IMG)
BATCH_SIZE, EVAL_BATCH_SIZE = 64, 128
EPOCHS, PATIENCE, MIN_DELTA = 20, 5, 1e-4  # early stopping on VAL AUC
LR, WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP = 3e-4, 1e-2, 1, 1.0
SPLIT_RATIOS = (0.80, 0.10, 0.10)          # used only if the dataset has no train/val/test folders
NUM_WORKERS  = min(4, os.cpu_count() or 2)
MAX_RUNTIME_HOURS = 11.0                   # graceful stop before Kaggle's 12h hard limit
USE_JPEG_DRAFT    = True                   # PIL DCT-domain downscale while decoding (big speedup for 1280x720 JPEGs)
IMBALANCE_TRIGGER = 1.5                    # use BCE pos_weight if majority/minority ratio exceeds this
PRETRAINED_WEIGHTS_FILE = None             # optional local efficientnet_b0 .pth if Kaggle internet is OFF
ALLOW_RANDOM_INIT = False                  # keep False: baseline must be ImageNet-pretrained
# --- robustness / cross-dataset / paper-extras settings (evaluation only; identical to Model B) ---
RUN_CROSS_DATASET = True
CROSS_N_PER_CLASS = 5000
CROSS_PKL = "/kaggle/input/datasets/asmabagum/dataset-scan/dataset_scans_v40.pkl"
CROSS_KEYS = {"celeb": "CelebDF_PKL", "dfdc": "DFDC_PKL", "diffusion": "Diffusion_PKL"}   # PKL-based (set to {} to disable)
CROSS_FOLDERS = {                                # folder-based: class sub-folders labelled by name (REAL_HINTS -> 0, else 1)
    "CelebDF": "/kaggle/input/datasets/syedazmulhasansabbir/celeb-df-frames/content/drive/MyDrive/dataset/celeb_df_frames",
    "DFDC": "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/validation",
    "Diffusion": "/kaggle/input/datasets/syedazmulhasansabbir/diffusion/diffusion",
    "WildDeepfake": "/kaggle/input/datasets/maysuni/wild-deepfake/test",
}
N_BOOT = 200                               # video-cluster bootstrap resamples for AUC CIs
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REAL_HINTS = ("real", "orig", "orginal", "pristine", "youtube", "authentic", "genuine", "natural")
SPLIT_DIR_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                   "test": "test", "testing": "test"}

for sub in ["checkpoints", "metrics", "predictions", "figures", "logs"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
CK_DIR, MET_DIR, PRED_DIR, FIG_DIR, LOG_DIR = [os.path.join(OUT_DIR, s) for s in
                                                ["checkpoints", "metrics", "predictions", "figures", "logs"]]
FAIL_TSV = os.path.join(LOG_DIR, "failed_images_raw.tsv")
if os.path.exists(FAIL_TSV) and not any(f.endswith(".pth") for f in os.listdir(CK_DIR)): os.remove(FAIL_TSV)
def log_fail(path, stage, err):
    try:
        with open(FAIL_TSV, "a") as f:
            f.write(f"{path}\t{stage}\t{str(err).replace(chr(9), ' ').replace(chr(10), ' ')[:200]}\n")
    except Exception: pass

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
torch.backends.cudnn.benchmark = True
print("=" * 90 + "\nENVIRONMENT\n" + "=" * 90)
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | device {dev} "
      f"({torch.cuda.get_device_name(0) if use_amp else 'cpu'}) | AMP {use_amp} | workers {NUM_WORKERS}")

# ====================== 2. DATASET DISCOVERY =========================
print("\n" + "=" * 90 + "\nDATASET DISCOVERY\n" + "=" * 90)

def subdirs(p):
    return sorted(e.name for e in os.scandir(p) if e.is_dir())

def has_images_directly(p):
    return any(e.is_file() and e.name.lower().endswith(IMG_EXTS) for e in os.scandir(p))

root = DATA_ROOT
assert os.path.isdir(root), f"DATA_ROOT not found: {root}"
while len(subdirs(root)) == 1 and not has_images_directly(root):      # e.g. 'X/X/...' double nesting
    root = os.path.join(root, subdirs(root)[0]); print("descending into single sub-folder:", root)
print("dataset root:", root)
top = subdirs(root)
print("top-level folders:", top)

def scan_tree(folder):
    out, stack = [], [folder]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False): stack.append(e.path)
                    elif e.name.lower().endswith(IMG_EXTS): out.append(e.path)
        except Exception as ex:
            print("  scan error:", d, ex)
    return out

presplit = any(n.lower() in SPLIT_DIR_NAMES for n in top)
tasks = []   # (split or None, class_folder, folder_path)
if presplit:
    for n in top:
        if n.lower() in SPLIT_DIR_NAMES:
            for c in subdirs(os.path.join(root, n)):
                tasks.append((SPLIT_DIR_NAMES[n.lower()], c, os.path.join(root, n, c)))
    print("EXISTING train/val/test folders detected -> using them as-is.")
else:
    tasks = [(None, c, os.path.join(root, c)) for c in top]
    print("No train/val/test folders -> will create a source-video-level split.")

IDX_CACHE = os.path.join(OUT_DIR, "file_index_cache.pkl")
cache_key = (root, tuple((s, c) for s, c, _ in tasks))
rows = None
if os.path.exists(IDX_CACHE):
    try:
        with open(IDX_CACHE, "rb") as f: cc = pickle.load(f)
        if cc["key"] == cache_key: rows = cc["rows"]; print("loaded cached file index:", len(rows), "files")
    except Exception: rows = None
if rows is None:
    print("scanning files (parallel per class folder) ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex:
        res = list(ex.map(lambda t: scan_tree(t[2]), tasks))
    rows = [(p, c, s) for (s, c, _), ps in zip(tasks, res) for p in ps]
    with open(IDX_CACHE, "wb") as f: pickle.dump({"key": cache_key, "rows": rows}, f)
    print(f"scan done in {time.time()-t0:.0f}s: {len(rows)} files")

df = pd.DataFrame(rows, columns=["path", "cls", "split"]).sort_values("path").reset_index(drop=True)
assert len(df) > 0, "No images found."

# ---- class-folder -> label mapping (REAL=0, FAKE=1; FAKE is the positive class)
cls_names = sorted(df.cls.unique())
cls2label = {c: (0 if any(h in c.lower() for h in REAL_HINTS) else 1) for c in cls_names}
print("\nclass-folder -> label (0=REAL, 1=FAKE):")
for c in cls_names: print(f"   {c:<20} -> {cls2label[c]} ({'REAL' if cls2label[c]==0 else 'FAKE'})   n={int((df.cls==c).sum()):,}")
df["label"] = df.cls.map(cls2label).astype(int)
assert df.label.nunique() == 2, "Could not find both REAL and FAKE class folders - check REAL_HINTS."
print("supported extensions:", IMG_EXTS, "| found:", dict(collections.Counter(os.path.splitext(p)[1].lower() for p in df.path)))
print(f"total images: {len(df):,} | REAL: {(df.label==0).sum():,} | FAKE: {(df.label==1).sum():,}")

# ---- image-size probe (is this face crops or full frames?)
samp = df.sample(min(150, len(df)), random_state=SEED).path.tolist()
sizes = []
for p in samp:
    try: sizes.append(Image.open(p).size)
    except Exception: pass
if sizes:
    ws, hs = np.array([s[0] for s in sizes]), np.array([s[1] for s in sizes])
    print(f"image size probe (n={len(sizes)}): width median {np.median(ws):.0f} [{ws.min()}-{ws.max()}], "
          f"height median {np.median(hs):.0f} [{hs.min()}-{hs.max()}]")
    if np.median(ws) >= 640:
        print("  WARNING: images look like FULL FRAMES, not tight face crops. They are squashed to "
              f"{RESIZE_SIZE}x{RESIZE_SIZE}, so faces will be small. Baseline still valid, but be aware.")

# ---- source-video grouping (FF++ naming) + split
RE_PAIR   = re.compile(r"^(\d+)_(\d+)_(\d+)$")            # fake: TARGET_SOURCE_FRAME
RE_REAL_M = re.compile(r"^real_(\d+)\.mp4_(\d+)$")        # real_044.mp4_22
RE_SINGLE = re.compile(r"^(\d+)_(\d+)$")                  # real: VIDEO_FRAME
RE_GENERIC = re.compile(r"^(.*?)[_\-\.]*(?:frame)?[_\-\.]*\d+$", re.I)

def parse_ids(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = RE_PAIR.match(stem)
    if m: return "pair", int(m[1]), int(m[2])
    m = RE_REAL_M.match(stem) or RE_SINGLE.match(stem)
    if m: return "single", int(m[1]), None
    g = RE_GENERIC.match(stem)
    return "other", (g[1] if g else stem), None

parsed = [parse_ids(p) for p in df.path]
df["ptype"] = [p[0] for p in parsed]
print("\nfilename-pattern counts:", df.ptype.value_counts().to_dict())

parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[max(ra, rb)] = min(ra, rb)
for t, a, b in parsed:
    if t == "pair": union(a, b)
    elif t == "single": find(a)
groups = []
for (t, a, b), cname in zip(parsed, df.cls):
    if t in ("pair", "single"): groups.append(f"vid{find(a):04d}")
    else: groups.append(f"other::{cname}::{a}")
df["group"] = groups
comp_sizes = collections.Counter(collections.Counter(find(x) for x in list(parent)).values())
print("video-id clusters (a fake pair links its target+source ids): size->count", dict(comp_sizes))
n_other = int((df.ptype == "other").sum())
if n_other / len(df) > 0.01:
    print(f"  WARNING: {n_other:,} files ({n_other/len(df):.1%}) did not match FF++ naming; they are grouped by "
          "filename prefix only. Leakage prevention is NOT guaranteed for those.")
if max(comp_sizes) > 20:
    print("  WARNING: a large connected video cluster exists -> split sizes may be uneven.")

if presplit:
    df["split"] = df.split
else:
    g_sizes = df.groupby("group").size()
    glist = sorted(g_sizes.index); random.Random(SEED).shuffle(glist)
    tot = len(df); targets = {s: r * tot for s, r in zip(("train", "val", "test"), SPLIT_RATIOS)}
    cur = {s: 0 for s in targets}; assign = {}
    for g in glist:
        s = max(targets, key=lambda k: targets[k] - cur[k]); assign[g] = s; cur[s] += int(g_sizes[g])
    df["split"] = df.group.map(assign)

gs = {s: set(df[df.split == s].group) for s in ("train", "val", "test")}
overlap = {k: len(gs[a] & gs[b]) for k, (a, b) in {"train&val": ("train", "val"), "train&test": ("train", "test"),
                                                   "val&test": ("val", "test")}.items()}
print("\nsource-group overlap between splits:", overlap)
if not presplit: assert all(v == 0 for v in overlap.values()), "LEAKAGE: groups cross splits!"
elif any(overlap.values()): print("  NOTE: existing splits share source groups (by filename inference) -> possible leakage in provided splits.")
overall_fake = df.label.mean()
for s in ("train", "val", "test"):
    d = df[df.split == s]
    print(f"  {s:<5}: {len(d):>8,} imgs | REAL {int((d.label==0).sum()):>7,} | FAKE {int((d.label==1).sum()):>7,} "
          f"| fake ratio {d.label.mean():.3f} | groups {d.group.nunique():,}")
    if abs(d.label.mean() - overall_fake) > 0.03: print(f"    WARNING: {s} class ratio deviates >3pts from overall ({overall_fake:.3f})")
split_hash = hashlib.md5("\n".join((df.path + "|" + df.split).tolist()).encode()).hexdigest()
df.to_csv(os.path.join(MET_DIR, f"{RUN_NAME}_Split_Assignment.csv"), index=False)
print("split hash:", split_hash)
tr_df, va_df, te_df = [df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")]

# class imbalance handling (train split only)
n_real, n_fake = int((tr_df.label == 0).sum()), int((tr_df.label == 1).sum())
ratio = max(n_real, n_fake) / max(1, min(n_real, n_fake))
pos_weight = (n_real / n_fake) if ratio > IMBALANCE_TRIGGER else 1.0
print(f"\nclass imbalance (train): REAL {n_real:,} vs FAKE {n_fake:,} (ratio {ratio:.2f}) -> "
      f"BCE pos_weight = {pos_weight:.4f} " + ("(imbalance handling ON; no resampling used)" if pos_weight != 1.0 else "(balanced enough; OFF)"))

# ============================ 3. DATA ================================
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tf_train = T.Compose([T.Resize((RESIZE_SIZE, RESIZE_SIZE)), T.RandomCrop(IMG_SIZE), T.RandomHorizontalFlip(),
                      T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15), T.ToTensor(), T.Normalize(MEAN, STD)])
tf_eval = T.Compose([T.Resize((RESIZE_SIZE, RESIZE_SIZE)), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])

class FrameDS(Dataset):
    def __init__(self, d, tf): self.paths, self.labels, self.tf = d.path.tolist(), d.label.values.astype(np.float32), tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for k in range(5):                       # skip corrupted files, never crash the run
            j = (i + k) % len(self.paths)
            try:
                img = Image.open(self.paths[j])
                if USE_JPEG_DRAFT and img.format == "JPEG": img.draft("RGB", (RESIZE_SIZE, RESIZE_SIZE))
                return self.tf(img.convert("RGB")), torch.tensor(self.labels[j]), j
            except Exception as e: log_fail(self.paths[j], "train" if self.tf is tf_train else "val", e)
        raise RuntimeError("5 consecutive unreadable images")

def winit(w): np.random.seed(SEED + w); random.seed(SEED + w)
def make_loader(d, tf, bs, shuffle, persistent=False):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=use_amp, worker_init_fn=winit)
    if NUM_WORKERS > 0: kw.update(persistent_workers=persistent, prefetch_factor=4)
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(FrameDS(d, tf), batch_size=bs, shuffle=shuffle, drop_last=shuffle, generator=g, **kw)
train_loader = make_loader(tr_df, tf_train, BATCH_SIZE, True, persistent=True)
val_loader   = make_loader(va_df, tf_eval, EVAL_BATCH_SIZE, False)
test_loader  = make_loader(te_df, tf_eval, EVAL_BATCH_SIZE, False)
steps_per_epoch = len(train_loader)

# ============================ 4. MODEL ===============================
def build_model():
    try:
        m = torchvision.models.efficientnet_b0(weights="DEFAULT")
        print("loaded ImageNet-pretrained EfficientNet-B0 (weights='DEFAULT')")
    except Exception as e:
        print("pretrained download failed:", repr(e)[:200])
        if PRETRAINED_WEIGHTS_FILE and os.path.exists(PRETRAINED_WEIGHTS_FILE):
            m = torchvision.models.efficientnet_b0(weights=None)
            m.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE, map_location="cpu")); print("loaded local weights file")
        elif ALLOW_RANDOM_INIT:
            m = torchvision.models.efficientnet_b0(weights=None); print("!! RANDOM INIT (not a valid pretrained baseline)")
        else:
            raise RuntimeError("Cannot get ImageNet weights. Turn Kaggle Internet ON, or set PRETRAINED_WEIGHTS_FILE.")
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)     # single logit + BCEWithLogitsLoss
    return m

model = build_model().to(dev).to(memory_format=torch.channels_last)
n_tot = sum(p.numel() for p in model.parameters()); n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nparams total {n_tot:,} | trainable {n_tr:,} | features blocks: {len(model.features)} | head: {model.classifier}")

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps, warm = EPOCHS * steps_per_epoch, max(1, WARMUP_EPOCHS * steps_per_epoch)
def lr_lambda(s):
    if s < warm: return (s + 1) / warm
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
CONFIG = dict(DATA_ROOT=root, IMG_SIZE=IMG_SIZE, RESIZE_SIZE=RESIZE_SIZE, BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS,
              PATIENCE=PATIENCE, LR=LR, WEIGHT_DECAY=WEIGHT_DECAY, WARMUP_EPOCHS=WARMUP_EPOCHS, SEED=SEED,
              pos_weight=pos_weight, split_hash=split_hash, mean=MEAN, std=STD, label_convention="REAL=0, FAKE=1, prob=sigmoid(logit)=P(FAKE)")
with open(os.path.join(MET_DIR, f"{RUN_NAME}_Run_Config.json"), "w") as f: json.dump(CONFIG, f, indent=2)

# ============================ 5. EVAL HELPERS ========================
def ece_score(y, p, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def compute_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    two = len(np.unique(y)) == 2
    return dict(n=int(len(y)), threshold=float(thr), auc=float(roc_auc_score(y, p)) if two else float("nan"),
                ap=float(average_precision_score(y, p)) if two else float("nan"), accuracy=float((pred == y).mean()),
                precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                specificity=float(spec), balanced_acc=float((rec + spec) / 2), ece=ece_score(y, p),
                brier=float(np.mean((p - y) ** 2)), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

@torch.inference_mode()
def predict(loader):
    model.eval(); L, Y, I = [], [], []
    for x, y, idx in loader:
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            lg = model(x).squeeze(1)
        L.append(lg.float().cpu()); Y.append(y); I.append(idx)
    L, Y, I = torch.cat(L).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()
    _, u = np.unique(I, return_index=True)               # drop duplicates caused by skipped corrupt files
    return L[u], Y[u].astype(int), I[u]

def val_quick(loader, d):
    lg, y, _ = predict(loader)
    p = 1 / (1 + np.exp(-lg))
    loss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(lg), torch.tensor(y, dtype=torch.float32),
                                                                pos_weight=torch.tensor([pos_weight])))
    m = compute_metrics(y, p, 0.5)
    return dict(val_loss=loss, val_acc=m["accuracy"], val_precision=m["precision"], val_recall=m["recall"], val_f1=m["f1"],
                val_auc=m["auc"], val_ap=m["ap"], val_specificity=m["specificity"], val_bal_acc=m["balanced_acc"])

# ============================ 6. RESUME ==============================
CK_RE = re.compile(rf"^{RUN_NAME}_Epoch(\d{{3}})\.pth$")
def ck_path(ep): return os.path.join(CK_DIR, f"{RUN_NAME}_Epoch{ep:03d}.pth")
BEST_PATH = os.path.join(CK_DIR, f"{RUN_NAME}_Best.pth")
def save_atomic(obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

class SplitMismatch(Exception): pass
start_epoch, best_auc, best_epoch, no_improve, history = 0, -1.0, 0, 0, []
cands = sorted([(int(m[1]), f) for f in os.listdir(CK_DIR) if (m := CK_RE.match(f))], reverse=True)
for ep_n, fn in cands:
    try:
        ck = torch.load(os.path.join(CK_DIR, fn), map_location="cpu", weights_only=False)
        if ck["config"]["split_hash"] != split_hash:
            raise SplitMismatch("checkpoint was trained with a DIFFERENT split (dataset/seed changed). "
                               f"Delete {OUT_DIR} to start fresh.")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, best_epoch = ck["epoch"], ck["best_val_auc"], ck["best_epoch"]
        no_improve, history = ck["epochs_no_improve"], ck["history"]
        print(f"\nRESUMED from {fn} (epoch {start_epoch}, best val AUC {best_auc:.5f} @ epoch {best_epoch})"); break
    except SplitMismatch: raise
    except Exception as e:
        print(f"  could not load {fn} ({repr(e)[:100]}); trying an older one")
else:
    print("\nno valid checkpoint found -> starting from pretrained EfficientNet-B0")

# ============================ 7. TRAINING ============================
print("\n" + "=" * 90 + "\nTRAINING\n" + "=" * 90)
epoch_times, stopped_for_time = [], False
finished = start_epoch >= EPOCHS or no_improve >= PATIENCE
if finished: print("training already complete in the loaded checkpoint -> skipping to final evaluation")
for ep in range(start_epoch + 1, EPOCHS + 1):
    if finished: break
    elapsed_h = (time.time() - T_START) / 3600
    if epoch_times and elapsed_h + 1.15 * np.mean(epoch_times) / 3600 > MAX_RUNTIME_HOURS - 0.6:   # reserve ~0.6h for robustness eval
        stopped_for_time = True; print(f"time budget reached ({elapsed_h:.2f}h) -> stopping. Re-run the cell to resume."); break
    model.train(); t0 = time.time(); seen = 0; loss_sum = 0.0; correct = 0
    for it, (x, y, _) in enumerate(train_loader, 1):
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.to(dev, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logit = model(x).squeeze(1)
        loss = criterion(logit.float(), y)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        bs = y.size(0); seen += bs; loss_sum += loss.item() * bs; correct += ((logit.detach() > 0).float() == y).sum().item()
        if it % 200 == 0 or it == steps_per_epoch:
            sp = seen / (time.time() - t0)
            print(f"  ep {ep:02d} [{it}/{steps_per_epoch}] loss {loss_sum/seen:.4f} acc {correct/seen:.4f} "
                  f"| {sp:.0f} img/s | epoch ETA {(len(train_loader.dataset)-seen)/sp/60:.1f} min", flush=True)
    tr_loss, tr_acc = loss_sum / seen, correct / seen
    vm = val_quick(val_loader, va_df)
    improved = vm["val_auc"] > best_auc + MIN_DELTA
    if improved: best_auc, best_epoch, no_improve = vm["val_auc"], ep, 0
    else: no_improve += 1
    history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, lr=optimizer.param_groups[0]["lr"], **vm,
                        epoch_minutes=(time.time() - t0) / 60))
    ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
              scaler=scaler.state_dict(), epoch=ep, best_val_auc=best_auc, best_epoch=best_epoch,
              epochs_no_improve=no_improve, history=history, config=CONFIG)
    save_atomic(ck, ck_path(ep))                                   # unique per-epoch checkpoint
    if improved: save_atomic(ck, BEST_PATH)                        # best by VALIDATION AUC only
    pd.DataFrame(history).to_csv(os.path.join(MET_DIR, f"{RUN_NAME}_Training_History.csv"), index=False)
    epoch_times.append(time.time() - t0)
    print(f"EPOCH {ep:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {vm['val_loss']:.4f} "
          f"AUC {vm['val_auc']:.5f} acc {vm['val_acc']:.4f} | best AUC {best_auc:.5f} @ {best_epoch} "
          f"{'*' if improved else f'(no improve {no_improve}/{PATIENCE})'} | {epoch_times[-1]/60:.1f} min | "
          f"elapsed {(time.time()-T_START)/3600:.2f}h", flush=True)
    if no_improve >= PATIENCE:
        print(f"early stopping: no val-AUC improvement for {PATIENCE} epochs"); break

# ============================ 8. FINAL EVALUATION ====================
# ============================ 8. FINAL EVALUATION ====================
print("\n" + "=" * 90 + "\nFINAL EVALUATION (best checkpoint by VAL AUC; test touched only here)\n" + "=" * 90)
assert os.path.exists(BEST_PATH), "No best checkpoint was saved."
bck = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(bck["model"]); model.eval()
print(f"loaded best checkpoint: epoch {bck['epoch']} (val AUC {bck['best_val_auc']:.5f})")
if stopped_for_time: print("NOTE: training stopped early by the time budget. Results are for the best-so-far checkpoint.")
PRIMARY = PFX = "ModelA"; FULL_NAME = "ModelA_Baseline"
LABEL, COLOR = {PRIMARY: "Model A (baseline)"}, {PRIMARY: "#4C72B0"}
LBL = lambda k: LABEL[k].split(" (")[0]
models = collections.OrderedDict([(PRIMARY, model)])
GPU_NAME = torch.cuda.get_device_name(0) if use_amp else "cpu"
CONFIG_FULL = dict(CONFIG, run=RUN_NAME, variant="A", python=platform.python_version(), torch=torch.__version__,
                   torchvision=torchvision.__version__, cuda=torch.version.cuda, gpu=GPU_NAME, params_total=n_tot,
                   params_trainable=n_tr, lambda_consistency=0.0, cls_loss="BCE(clean), pos_weight",
                   best_epoch=best_epoch, best_val_auc=best_auc, cross_folders=CROSS_FOLDERS, n_boot=N_BOOT)

# ---- degradation machinery: IDENTICAL to Model B (same ops, same per-image seeds -> same degraded pixels) ----
geo_eval = T.Compose([T.Resize((RESIZE_SIZE, RESIZE_SIZE)), T.CenterCrop(IMG_SIZE)])
to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

def apply_op(img, op, val, rng=None):
    if op in ("jpeg", "webp"):
        buf = io.BytesIO(); img.save(buf, "JPEG" if op == "jpeg" else "WEBP", quality=int(val)); buf.seek(0)
        out = Image.open(buf); out.load(); return out.convert("RGB")
    if op == "down":
        w, h = img.size
        return img.resize((max(8, round(w * val)), max(8, round(h * val))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if op == "blur": return img.filter(ImageFilter.GaussianBlur(radius=float(val)))
    if op == "noise":
        a = np.asarray(img, dtype=np.float32)
        n = (rng.normal(0.0, float(val), a.shape) if rng is not None else np.random.normal(0.0, float(val), a.shape))
        return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))
    if op == "color":
        b, c, s = val
        img = ImageEnhance.Brightness(img).enhance(b); img = ImageEnhance.Contrast(img).enhance(c)
        return ImageEnhance.Color(img).enhance(s)
    raise ValueError(op)

def degrade_ops(img, ops, rng=None):
    for op, v in ops: img = apply_op(img, op, v, rng)
    return img

COND = collections.OrderedDict()
COND["clean"] = dict(type="clean", sev="none", family="clean", ops=[])
for q, s in [(70, "mild"), (40, "medium"), (20, "severe")]: COND[f"jpeg_{s}"] = dict(type="JPEG", sev=s, family="in", ops=[("jpeg", q)])
for sc, s in [(0.75, "mild"), (0.5, "medium"), (0.25, "severe")]: COND[f"resolution_{s}"] = dict(type="Resolution", sev=s, family="in", ops=[("down", sc)])
for sg, s in [(1.0, "mild"), (2.0, "medium"), (3.0, "severe")]: COND[f"blur_{s}"] = dict(type="Blur", sev=s, family="in", ops=[("blur", sg)])
for sg, s in [(5, "mild"), (10, "medium"), (20, "severe")]: COND[f"noise_{s}"] = dict(type="Noise", sev=s, family="in", ops=[("noise", sg)])
COND["combined_mild"]   = dict(type="Combined", sev="mild", family="in", ops=[("blur", 0.8), ("jpeg", 70)])
COND["combined_medium"] = dict(type="Combined", sev="medium", family="in", ops=[("down", 0.5), ("jpeg", 40)])
COND["combined_severe"] = dict(type="Combined", sev="severe", family="in", ops=[("down", 0.4), ("noise", 10), ("jpeg", 20)])
try:
    apply_op(Image.new("RGB", (32, 32)), "webp", 30); COND["webp_medium"] = dict(type="WebP(unseen codec)", sev="medium", family="out", ops=[("webp", 30)])
except Exception as e: print("WebP encoder unavailable, skipping unseen-codec condition:", repr(e)[:80])
SEVS = ["mild", "medium", "severe"]
print(f"test-time conditions ({len(COND)}): {list(COND)}")
CONFIG_FULL["test_conditions"] = {k: dict(type=v["type"], sev=v["sev"], ops=v["ops"]) for k, v in COND.items()}
json.dump(CONFIG_FULL, open(os.path.join(MET_DIR, f"{PFX}_Configuration.json"), "w"), indent=2, default=str)

def cond_title(n):
    c = COND[n]; return ("Clean_Test" if n == "clean" else f"{c['type'].split('(')[0]}_{c['sev'].capitalize()}")
def pred_file(mkey, n): return os.path.join(PRED_DIR, f"{FULL_NAME}_{cond_title(n)}_Predictions.csv")

def load_rgb(p):
    img = Image.open(p)
    if USE_JPEG_DRAFT and img.format == "JPEG": img.draft("RGB", (RESIZE_SIZE, RESIZE_SIZE))
    return img.convert("RGB")

class EvalDS(Dataset):
    def __init__(self, d, cond="clean", stage="eval"):
        self.paths, self.labels, self.cond, self.ops, self.stage = d.path.tolist(), d.label.values.astype(np.float32), cond, COND[cond]["ops"], stage
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for k in range(5):
            j = (i + k) % len(self.paths)
            try:
                base = geo_eval(load_rgb(self.paths[j]))
                if self.ops:
                    rng = np.random.default_rng((zlib.crc32(f"{self.paths[j]}|{self.cond}".encode()) ^ SEED) & 0xFFFFFFFF)
                    base = degrade_ops(base, self.ops, rng)
                return to_tensor(base), torch.tensor(self.labels[j]), j
            except Exception as e: log_fail(self.paths[j], self.stage if self.cond == "clean" else f"{self.stage}:{self.cond}", e)
        raise RuntimeError("5 consecutive unreadable images")

def make_eval_loader(ds, bs):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=use_amp, worker_init_fn=winit)
    if NUM_WORKERS > 0: kw.update(persistent_workers=False, prefetch_factor=4)
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(ds, batch_size=bs, shuffle=False, generator=g, **kw)

def sigmoid(z): return 1 / (1 + np.exp(-z))

@torch.inference_mode()
def run_models(models, loader):
    for m in models.values(): m.eval()
    outs = {k: [] for k in models}; Y, I = [], []
    for x, y, idx in loader:
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
        for k, m in models.items():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outs[k].append(m(x).squeeze(1).float().cpu())
        Y.append(y); I.append(idx)
    I = torch.cat(I).numpy(); Y = torch.cat(Y).numpy().astype(int); _, u = np.unique(I, return_index=True)
    return {k: torch.cat(v).numpy()[u] for k, v in outs.items()}, Y[u], I[u]

@torch.inference_mode()
def time_inference(m, loader, nb=40):
    m.eval(); tt, n = 0.0, 0
    for b, (x, y, idx) in enumerate(loader):
        if b >= nb: break
        x = x.to(dev).contiguous(memory_format=torch.channels_last)
        if use_amp: torch.cuda.synchronize()
        t = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp): m(x)
        if use_amp: torch.cuda.synchronize()
        if b >= 3: tt += time.time() - t; n += x.size(0)
    return tt / max(1, n)

# ---- threshold from CLEAN validation only (Youden J), reused for every test condition ----
o, vy, vi = run_models(models, val_loader)
fv, tv, thv = roc_curve(vy, sigmoid(o[PRIMARY])); THR = {PRIMARY: float(thv[np.argmax(tv - fv)])}
print("validation-selected threshold (clean val, reused for ALL test conditions):", {k: round(v, 4) for k, v in THR.items()})
pd.DataFrame(dict(image_path=va_df.path.values[vi], ground_truth=vy, predicted_probability=sigmoid(o[PRIMARY]))).to_csv(
    os.path.join(PRED_DIR, f"{PFX}_Val_Predictions.csv"), index=False)

RES, PROBS = {}, {}
def meta_table(d, idx): return d.iloc[idx][["path", "group"]].reset_index(drop=True)
def write_pred(mkey, cname, y, p, idx):
    t = meta_table(te_df, idx); t.insert(0, "image_path", t.pop("path")); t.insert(1, "sample_id", t.image_path.map(os.path.basename))
    t = t.rename(columns={"group": "source_video_id"}); t["ground_truth"] = y; t["predicted_probability"] = p
    t["predicted_label"] = (p >= THR[mkey]).astype(int); t["degradation_type"] = COND[cname]["type"]; t["degradation_severity"] = COND[cname]["sev"]
    t.to_csv(pred_file(mkey, cname), index=False); return t

def try_cached(cname):
    out = {}
    for k in models:
        f = pred_file(k, cname)
        if not os.path.exists(f): return None
        t = pd.read_csv(f)
        if len(t) == 0 or "predicted_probability" not in t: return None
        out[k] = t
    return out

t0 = time.time(); clean_loader = make_eval_loader(EvalDS(te_df, "clean", "test"), EVAL_BATCH_SIZE)
o1, ty, ti = run_models(models, clean_loader); total_clean_time = time.time() - t0
ms_per_img = 1000 * time_inference(model, clean_loader)
print(f"clean test inference: {total_clean_time:.1f}s total for {len(ty):,} imgs (incl. data loading) | model-only {ms_per_img:.3f} ms/img (batch {EVAL_BATCH_SIZE})")
p = sigmoid(o1[PRIMARY]); PROBS[(PRIMARY, "clean")] = (ty, p, ti); write_pred(PRIMARY, "clean", ty, p, ti)
RES[(PRIMARY, "clean")] = dict(model=PRIMARY, condition="clean", type="clean", severity="none", family="clean", **compute_metrics(ty, p, THR[PRIMARY]))
shutil.copyfile(pred_file(PRIMARY, "clean"), os.path.join(PRED_DIR, f"{PFX}_Test_Predictions_Clean.csv"))

print("\ndegraded test conditions ...")
for n_c, cname in enumerate([c for c in COND if c != "clean"], 1):
    if (time.time() - T_START) / 3600 > MAX_RUNTIME_HOURS - 0.15:
        print(f"time budget nearly exhausted -> skipping remaining conditions from {cname}"); break
    cached = try_cached(cname); t0 = time.time()
    if cached is not None:
        for k, t in cached.items(): PROBS[(k, cname)] = (t.ground_truth.values.astype(int), t.predicted_probability.values, None)
        src = "cached"
    else:
        oo, yy, ii = run_models(models, make_eval_loader(EvalDS(te_df, cname, "test"), EVAL_BATCH_SIZE))
        for k in models:
            p = sigmoid(oo[k]); PROBS[(k, cname)] = (yy, p, ii); write_pred(k, cname, yy, p, ii)
        src = "computed"
    for k in models:
        yy, p, _ = PROBS[(k, cname)]
        RES[(k, cname)] = dict(model=k, condition=cname, type=COND[cname]["type"], severity=COND[cname]["sev"],
                               family=COND[cname]["family"], **compute_metrics(yy, p, THR[k]))
    print(f"  [{n_c:02d}/{len(COND)-1}] {cname:<18} {src:<8} {time.time()-t0:5.0f}s | " +
          " | ".join(f"{k} AUC {RES[(k, cname)]['auc']:.4f} F1 {RES[(k, cname)]['f1']:.4f}" for k in models), flush=True)

# ---- tables: robustness, drops (same schema as Model B) ----
R = pd.DataFrame(list(RES.values()))
DROP_METRICS = ["auc", "f1", "accuracy", "precision", "recall", "balanced_acc", "ap"]
clean_ref = {k: RES[(k, "clean")] for k in models}
for mk in DROP_METRICS: R[f"{mk}_drop"] = [clean_ref[m][mk] - v for m, v in zip(R.model, R[mk])]
R["auc_retention"] = [v / clean_ref[m]["auc"] if clean_ref[m]["auc"] else np.nan for m, v in zip(R.model, R.auc)]
R["f1_retention"] = [v / clean_ref[m]["f1"] if clean_ref[m]["f1"] else np.nan for m, v in zip(R.model, R.f1)]
Rd = R[R.condition != "clean"]
def drop_summary(Rm):
    rows = []
    for k, g in Rm[Rm.family == "in"].groupby("model"):
        for s in SEVS:
            gg = g[g.severity == s]
            if len(gg): rows.append(dict(row_type="severity_mean", model=k, condition=f"MEAN_{s}", type="all in-family types", severity=s,
                                         **{f"{m}_drop": gg[f"{m}_drop"].mean() for m in DROP_METRICS}, auc=gg.auc.mean(), f1=gg.f1.mean()))
        rows.append(dict(row_type="overall_mean", model=k, condition="MEAN_ALL_IN_FAMILY", type="all in-family types", severity="all",
                         **{f"{m}_drop": g[f"{m}_drop"].mean() for m in DROP_METRICS}, auc=g.auc.mean(), f1=g.f1.mean()))
    return pd.DataFrame(rows)
summ = drop_summary(Rd)
rob_cols = ["model", "condition", "type", "severity", "family", "n", "threshold", "auc", "ap", "accuracy", "precision", "recall", "f1",
            "specificity", "balanced_acc", "ece", "brier", "tp", "tn", "fp", "fn"]
drop_cols = [c for c in R.columns if c.endswith("_drop") or c.endswith("_retention")]
P_ = R[R.model == PRIMARY]
pd.DataFrame(history).to_csv(os.path.join(MET_DIR, f"{PFX}_Training_History.csv"), index=False)
P_[rob_cols].to_csv(os.path.join(MET_DIR, f"{PFX}_Robustness_Metrics.csv"), index=False)
P_[["model", "condition", "type", "severity", "family", "auc", "f1", "accuracy", "balanced_acc", "ap"] + drop_cols].to_csv(
    os.path.join(MET_DIR, f"{PFX}_Degradation_Comparison.csv"), index=False)
pd.concat([P_[P_.condition != "clean"].assign(row_type="condition")[["row_type", "model", "condition", "type", "severity"] + drop_cols],
           summ[summ.model == PRIMARY]], ignore_index=True).to_csv(os.path.join(MET_DIR, f"{PFX}_Performance_Drop.csv"), index=False)
pc = PROBS[(PRIMARY, "clean")]
mc05 = compute_metrics(pc[0], pc[1], 0.5); mcv = compute_metrics(pc[0], pc[1], THR[PRIMARY])
pd.DataFrame([dict(rule="fixed 0.5", **mc05), dict(rule="val-selected (Youden, clean val)", **mcv)]).assign(
    params_total=n_tot, params_trainable=n_tr, ms_per_image_model_only=ms_per_img, clean_test_total_seconds=total_clean_time,
    best_epoch=best_epoch, best_val_auc=best_auc).to_csv(os.path.join(MET_DIR, f"{PFX}_Final_Clean_Test_Metrics.csv"), index=False)
cmv = np.array([[mcv["tn"], mcv["fp"]], [mcv["fn"], mcv["tp"]]])
pd.DataFrame(cmv, index=["true_REAL", "true_FAKE"], columns=["pred_REAL", "pred_FAKE"]).to_csv(os.path.join(MET_DIR, f"{PFX}_Confusion_Matrix_Clean.csv"))
P_[P_.condition != "clean"][["model", "condition", "type", "severity", "tp", "tn", "fp", "fn"]].to_csv(
    os.path.join(MET_DIR, f"{PFX}_Confusion_Matrix_Degraded.csv"), index=False)

# per-class-folder breakdown on clean test (kept from the original Model A script)
tt = te_df.iloc[pc[2]][["path", "cls"]].reset_index(drop=True); tt["label"] = pc[0]; tt["prob_fake"] = pc[1]
real_t, rows_b = tt[tt.label == 0], []
for c in sorted(tt.cls.unique()):
    s = tt[tt.cls == c]; r = dict(class_folder=c, label=cls2label[c], n=len(s), mean_prob_fake=float(s.prob_fake.mean()),
                                  acc_at_val_thr=float(((s.prob_fake >= THR[PRIMARY]).astype(int) == s.label).mean()))
    if cls2label[c] == 1 and len(real_t):
        z = pd.concat([real_t, s]); r["auc_vs_real"] = float(roc_auc_score(z.label, z.prob_fake))
    rows_b.append(r)
bd = pd.DataFrame(rows_b); bd.to_csv(os.path.join(MET_DIR, f"{PFX}_Test_PerClassFolder.csv"), index=False)
print("\nper-class-folder clean-test breakdown:\n" + bd.round(4).to_string(index=False))

print(f"\nCLEAN TEST (val-selected thr): AUC {RES[(PRIMARY,'clean')]['auc']:.4f} F1 {RES[(PRIMARY,'clean')]['f1']:.4f} Acc {RES[(PRIMARY,'clean')]['accuracy']:.4f}")
print(f"{PRIMARY} clean @0.5: AUC {mc05['auc']:.4f} Acc {mc05['accuracy']:.4f} F1 {mc05['f1']:.4f} P {mc05['precision']:.4f} R {mc05['recall']:.4f} "
      f"Spec {mc05['specificity']:.4f} BalAcc {mc05['balanced_acc']:.4f} AP {mc05['ap']:.4f} | TP {mc05['tp']} TN {mc05['tn']} FP {mc05['fp']} FN {mc05['fn']}")
print("(ECE/Brier are affected by the class-imbalance pos_weight; AUC/AP are not.)")
s = summ[(summ.model == PRIMARY) & (summ.row_type == "severity_mean")]
print(f"  {PRIMARY:<8} clean AUC {RES[(PRIMARY,'clean')]['auc']:.4f} | " + " | ".join(f"{r.severity}: AUC {r.auc:.4f} (drop {r.auc_drop:.4f}) F1 {r.f1:.4f}" for r in s.itertuples()))

# ============================ 9. FIGURES =============================
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
def savefig(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
h = pd.DataFrame(history)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(h.epoch, h.train_loss, marker="o", ms=3, label="train (classification)"); ax.plot(h.epoch, h.val_loss, marker="o", ms=3, label="val")
ax.axvline(best_epoch, ls="--", c="gray"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (BCE)"); ax.legend(); savefig(fig, "FigA01_ModelA_Loss_Curves")
fig, ax = plt.subplots(figsize=(5.5, 4)); ax.plot(h.epoch, h.val_auc, marker="o", ms=3, color=COLOR[PRIMARY]); ax.axvline(best_epoch, ls="--", c="gray")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation ROC-AUC (clean)"); savefig(fig, "FigA02_ModelA_Validation_AUC")
fig, ax = plt.subplots(figsize=(5, 4.5)); y_, p_, _ = PROBS[(PRIMARY, "clean")]; f_, t_, _ = roc_curve(y_, p_)
ax.plot(f_, t_, color=COLOR[PRIMARY], label=f"{LABEL[PRIMARY]} AUC={RES[(PRIMARY,'clean')]['auc']:.4f}"); ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right", fontsize=8); ax.set_title("Clean FF++ test", fontsize=9)
savefig(fig, "FigA03_ModelA_Clean_Test_ROC")
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cmv, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, f"{cmv[i,j]:,}", ha="center", va="center", color="white" if cmv[i, j] > cmv.max() / 2 else "black")
ax.set_xticks([0, 1], ["REAL", "FAKE"]); ax.set_yticks([0, 1], ["REAL", "FAKE"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"{LABEL[PRIMARY]}\nclean test, thr={THR[PRIMARY]:.3f}", fontsize=9); savefig(fig, "FigA04_ModelA_Clean_Confusion_Matrix")

def sev_curve(metric, fname, ylabel):
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); ys, lo, hi = [RES[(PRIMARY, "clean")][metric]], [0], [0]
    for s in SEVS:
        v = Rd[(Rd.model == PRIMARY) & (Rd.family == "in") & (Rd.severity == s)][metric].values
        if len(v): ys.append(v.mean()); lo.append(v.mean() - v.min()); hi.append(v.max() - v.mean())
        else: ys.append(np.nan); lo.append(0); hi.append(0)
    ax.errorbar(range(4), ys, yerr=[lo, hi], marker="o", capsize=3, color=COLOR[PRIMARY], label=LABEL[PRIMARY])
    ax.set_xticks(range(4), ["clean"] + SEVS); ax.set_xlabel("Degradation severity (mean over in-family types; bars = min-max)"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); savefig(fig, fname)
sev_curve("auc", "FigA05_AUC_vs_Degradation_Severity", "ROC-AUC"); sev_curve("f1", "FigA06_F1_vs_Degradation_Severity", "F1")

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
sv = summ[(summ.model == PRIMARY) & (summ.row_type == "severity_mean")].set_index("severity")
for a, (m_, nm) in zip(axs, [("auc_drop", "AUC drop from clean"), ("f1_drop", "F1 drop from clean")]):
    a.bar(np.arange(3), [sv[m_].get(s, np.nan) for s in SEVS], 0.6, color=COLOR[PRIMARY], label=LABEL[PRIMARY])
    a.set_xticks(np.arange(3), SEVS); a.set_ylabel(nm + " (lower = more robust)"); a.set_xlabel("Severity (mean over in-family types)")
axs[0].legend(fontsize=8); savefig(fig, "FigA07_ModelA_Performance_Drop")

TYPES = ["JPEG", "Resolution", "Blur", "Noise", "Combined"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for a, (m_, nm) in zip(axs, [("auc", "ROC-AUC"), ("f1", "F1")]):
    vals = [Rd[(Rd.model == PRIMARY) & (Rd.type == t_) & (Rd.family == "in")][m_].mean() for t_ in TYPES]
    a.bar(np.arange(len(TYPES)), vals, 0.6, color=COLOR[PRIMARY], label=LABEL[PRIMARY])
    a.set_xticks(np.arange(len(TYPES)), TYPES, rotation=20); a.set_ylabel(f"mean {nm} over severities")
    lo_ = np.nanmin([Rd[(Rd.family == "in")][m_].min(), 0.5]); a.set_ylim(max(0, lo_ - 0.05), 1.0)
axs[0].legend(fontsize=8); savefig(fig, "FigA08_Degradation_Type_Comparison")

reps = [c for c in ["jpeg_medium", "blur_medium", "noise_medium", "combined_severe"] if c in COND]
fig, axs = plt.subplots(2, 2, figsize=(8, 7)); axs = axs.ravel()
for a, c in zip(axs, reps):
    if (PRIMARY, c) in PROBS:
        y_, p_, _ = PROBS[(PRIMARY, c)]; f_, t_, _ = roc_curve(y_, p_); a.plot(f_, t_, color=COLOR[PRIMARY], label=f"{LBL(PRIMARY)} AUC={RES[(PRIMARY,c)]['auc']:.3f}")
    a.plot([0, 1], [0, 1], "k:", lw=1); a.set_title(c, fontsize=10); a.legend(fontsize=7, loc="lower right"); a.set_xlabel("FPR"); a.set_ylabel("TPR")
for a in axs[len(reps):]: a.axis("off")
savefig(fig, "FigA09_Robustness_ROC")

# ================= 10. CROSS-DATASET EVAL (same 4 sets, same sampling as Model B) =================
cross_rows = []
if RUN_CROSS_DATASET and (time.time() - T_START) / 3600 < MAX_RUNTIME_HOURS - 0.75:
    print("\n" + "=" * 90 + "\nCROSS-DATASET (unseen-domain) EVALUATION - thresholds from FF++ clean val\n" + "=" * 90)
    try:
        allp = []
        try:
            assert CROSS_KEYS and os.path.exists(CROSS_PKL), f"scan PKL not found: {CROSS_PKL}"
            scan = pickle.load(open(CROSS_PKL, "rb"))
            for key, nm in CROSS_KEYS.items():
                d_ = pd.DataFrame(scan[key], columns=["path", "label", "group"])
                if key == "dfdc" and d_.path.str.contains("/validation/").any(): d_ = d_[d_.path.str.contains("/validation/")]
                low = d_.path.str.lower()
                bad = int(((low.str.contains("/real/") & (d_.label == 1)) | (low.str.contains("/fake/") & (d_.label == 0))).sum())
                if bad: print(f"  WARNING {nm}: {bad} entries where folder name contradicts PKL label")
                parts = []
                for lab in (0, 1):
                    s_ = d_[d_.label == lab]; parts.append(s_.sample(min(CROSS_N_PER_CLASS, len(s_)), random_state=SEED))
                d_ = pd.concat(parts).reset_index(drop=True); d_["dataset"] = nm; allp.append(d_)
        except Exception as e: print("PKL-based cross datasets skipped:", repr(e)[:150])
        for nm, root_ in CROSS_FOLDERS.items():
            if not os.path.isdir(root_): print(f"  {nm}: folder not found -> skipped ({root_})"); continue
            rows_ = []
            for c in subdirs(root_):
                lab = 0 if any(hh in c.lower() for hh in REAL_HINTS) else 1
                cr_ = os.path.join(root_, c)
                rows_ += [(p, lab, f"{c}::" + (os.path.basename(os.path.dirname(p)) if os.path.dirname(p) != cr_
                                               else os.path.basename(p).rsplit("_", 2)[0])) for p in scan_tree(cr_)]
            d_ = pd.DataFrame(rows_, columns=["path", "label", "group"]).sort_values("path").reset_index(drop=True)
            if d_.label.nunique() < 2: print(f"  {nm}: need both real and fake folders, found {subdirs(root_)} -> skipped"); continue
            print(f"  {nm}: folders {subdirs(root_)} | REAL {(d_.label==0).sum():,} FAKE {(d_.label==1).sum():,}")
            parts = [d_[d_.label == lab].sample(min(CROSS_N_PER_CLASS, int((d_.label == lab).sum())), random_state=SEED) for lab in (0, 1)]
            d_ = pd.concat(parts).reset_index(drop=True); d_["dataset"] = nm; allp.append(d_)
        for d_ in allp:
            nm = d_.dataset.iloc[0]
            ok = d_.path.map(os.path.exists); d_ = d_[ok].reset_index(drop=True)
            if len(d_) == 0 or d_.label.nunique() < 2: print(f"  {nm}: skipped (missing files / one class)"); continue
            t0 = time.time(); oo, yy, ii = run_models(models, make_eval_loader(EvalDS(d_, "clean", f"cross:{nm}"), EVAL_BATCH_SIZE))
            tab = d_.iloc[ii][["path", "label", "group", "dataset"]].reset_index(drop=True)
            for k in models:
                p = sigmoid(oo[k]); m_ = compute_metrics(yy, p, THR[k]); cross_rows.append(dict(dataset=nm, model=k, **m_)); tab[f"prob_{k}"] = p
            tab.to_csv(os.path.join(PRED_DIR, f"{PFX}_CrossDataset_{nm}_Predictions.csv"), index=False)
            print(f"  {nm:<12} n={len(yy):,} ({time.time()-t0:.0f}s) | " + " | ".join(f"{k} AUC {[r for r in cross_rows if r['dataset']==nm and r['model']==k][0]['auc']:.4f}" for k in models))
        if cross_rows:
            pd.DataFrame(cross_rows).to_csv(os.path.join(MET_DIR, f"{PFX}_CrossDataset_Metrics.csv"), index=False)
            print("(trained on FF++ only; unseen-domain results with FF++-validation thresholds)")
    except Exception as e: print("cross-dataset evaluation skipped:", repr(e)[:200])
else: print("\ncross-dataset evaluation skipped (disabled or not enough time budget).")

# ============ 10b. PAPER EXTRAS (CIs, stability, tables, extra figures) ============
print("\n" + "=" * 90 + "\nPAPER EXTRAS: bootstrap CIs, prediction stability, tables, extra figures\n" + "=" * 90)
def paper_item(fn):
    try: fn(); print("  done:", fn.__name__)
    except Exception as e: plt.close("all"); print(f"  SKIPPED {fn.__name__}: {repr(e)[:150]}")
    return fn
PR, stab, boot_rows = {}, [], []

@paper_item
def load_pred_tables():
    for c in COND:
        f = pred_file(PRIMARY, c)
        if os.path.exists(f):
            PR[(PRIMARY, c)] = pd.read_csv(f, usecols=["image_path", "source_video_id", "ground_truth", "predicted_probability"]
                                           ).drop_duplicates("image_path").set_index("image_path")
conds_done = [c for c in COND if (PRIMARY, c) in PR]
print("conditions available for paper extras:", len(conds_done))

@paper_item
def prediction_stability():
    base = PR[(PRIMARY, "clean")]
    for c in conds_done:
        if c == "clean": continue
        t = PR[(PRIMARY, c)]; j = base.index.intersection(t.index)
        pb, pq = base.loc[j, "predicted_probability"].values, t.loc[j, "predicted_probability"].values
        stab.append(dict(model=PRIMARY, condition=c, type=COND[c]["type"], severity=COND[c]["sev"], family=COND[c]["family"], n=len(j),
                         mean_abs_prob_shift=float(np.abs(pb - pq).mean()), mean_signed_shift=float((pq - pb).mean()),
                         decision_flip_rate=float(((pb >= THR[PRIMARY]) != (pq >= THR[PRIMARY])).mean())))
    pd.DataFrame(stab).to_csv(os.path.join(MET_DIR, f"{PFX}_Prediction_Stability.csv"), index=False)
S = pd.DataFrame(stab)

@paper_item
def bootstrap_auc_ci():                            # resample SOURCE-VIDEO clusters
    if (time.time() - T_START) / 3600 > MAX_RUNTIME_HOURS - 0.3: print("  time budget low -> bootstrap skipped"); return
    for c in conds_done:
        t = PR[(PRIMARY, c)]; y = t.ground_truth.values.astype(int); g = t.source_video_id.astype(str).values; pp = t.predicted_probability.values
        ug, inv = np.unique(g, return_inverse=True); members = [np.where(inv == i)[0] for i in range(len(ug))]
        rng = np.random.default_rng(SEED); bs = []
        for _ in range(N_BOOT):
            ii = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
            if len(np.unique(y[ii])) < 2: continue
            bs.append(roc_auc_score(y[ii], pp[ii]))
        if not bs: continue
        a = np.array(bs)
        boot_rows.append(dict(condition=c, model=PRIMARY, auc=roc_auc_score(y, pp), ci95_lo=np.percentile(a, 2.5),
                              ci95_hi=np.percentile(a, 97.5), n_boot=len(a), n_groups=len(ug)))
    pd.DataFrame(boot_rows).to_csv(os.path.join(MET_DIR, f"{PFX}_AUC_Bootstrap_CI.csv"), index=False)

@paper_item
def paper_tables():
    W = R.pivot_table(index=["condition", "type", "severity", "family"], columns="model", values=["auc", "f1", "accuracy", "ap", "balanced_acc"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]; W = W.reset_index()
    W["_o"] = W.condition.map({c: i for i, c in enumerate(COND)}); W = W.sort_values("_o").drop(columns="_o")
    W.to_csv(os.path.join(MET_DIR, f"{PFX}_Table_Metrics_By_Condition.csv"), index=False)
    r_ = dict(model=LABEL[PRIMARY], clean_auc=RES[(PRIMARY, "clean")]["auc"], clean_f1=RES[(PRIMARY, "clean")]["f1"], clean_acc=RES[(PRIMARY, "clean")]["accuracy"])
    for s in SEVS:
        x = summ[(summ.model == PRIMARY) & (summ.row_type == "severity_mean") & (summ.severity == s)]
        if len(x): r_.update({f"{s}_auc": x.auc.iloc[0], f"{s}_auc_drop": x.auc_drop.iloc[0], f"{s}_f1": x.f1.iloc[0], f"{s}_f1_drop": x.f1_drop.iloc[0]})
    o_ = summ[(summ.model == PRIMARY) & (summ.row_type == "overall_mean")]
    if len(o_): r_.update(mean_degraded_auc=o_.auc.iloc[0], mean_auc_drop=o_.auc_drop.iloc[0], mean_f1_drop=o_.f1_drop.iloc[0])
    wo = R[(R.model == PRIMARY) & (R.family == "out")]
    if len(wo): r_["unseen_codec_webp_auc"] = wo.auc.iloc[0]
    T_ = pd.DataFrame([r_]); T_.to_csv(os.path.join(MET_DIR, f"{PFX}_Paper_Summary_Table.csv"), index=False)
    try: T_.round(4).to_latex(os.path.join(MET_DIR, f"{PFX}_Paper_Summary_Table.tex"), index=False, float_format="%.4f")
    except Exception as e: print("  LaTeX export skipped:", repr(e)[:80])

@paper_item
def figA10_per_type():
    TY = [("JPEG", "jpeg"), ("Resolution", "resolution"), ("Blur", "blur"), ("Noise", "noise"), ("Combined", "combined")]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for a, (tn, tk) in zip(axs, TY):
        ys = [RES[(PRIMARY, "clean")]["auc"]] + [RES[(PRIMARY, f"{tk}_{s}")]["auc"] if (PRIMARY, f"{tk}_{s}") in RES else np.nan for s in SEVS]
        a.plot(range(4), ys, marker="o", color=COLOR[PRIMARY], label=LABEL[PRIMARY]); a.set_xticks(range(4), ["clean"] + SEVS, rotation=20); a.set_title(tn)
    axs[0].set_ylabel("ROC-AUC"); axs[0].legend(fontsize=7); savefig(fig, "FigA10_AUC_vs_Severity_Per_Type")

@paper_item
def figA11_heatmap():
    cl = [c for c in COND if (PRIMARY, c) in RES]
    M = np.array([[RES[(PRIMARY, c)]["auc"] for c in cl]]); vmin = max(0.5, np.nanmin(M))
    fig, a = plt.subplots(figsize=(max(8, 0.75 * len(cl)), 2.6)); im = a.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=1.0)
    a.set_xticks(range(len(cl)), cl, rotation=60, ha="right", fontsize=8); a.set_yticks([0], [LBL(PRIMARY)]); a.set_title("ROC-AUC", fontsize=9)
    for j in range(M.shape[1]): a.text(j, 0, f"{M[0, j]:.3f}", ha="center", va="center", fontsize=6, color="white" if M[0, j] < (vmin + 1) / 2 else "black")
    fig.colorbar(im, ax=a, fraction=0.02); savefig(fig, "FigA11_AUC_Heatmap_All_Conditions")

@paper_item
def figA12_ci_dots():
    BR = pd.DataFrame(boot_rows); cl2 = [c for c in COND if c in set(BR.condition)]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(cl2)), 4)); b = BR.set_index("condition").loc[cl2]
    ax.errorbar(np.arange(len(cl2)), b.auc, yerr=[b.auc - b.ci95_lo, b.ci95_hi - b.auc], fmt="o", ms=3, capsize=2, color=COLOR[PRIMARY], label=LABEL[PRIMARY])
    ax.set_xticks(range(len(cl2)), cl2, rotation=60, ha="right", fontsize=8); ax.set_ylabel("ROC-AUC (95% video-cluster bootstrap CI)"); ax.legend(fontsize=8)
    savefig(fig, "FigA12_AUC_With_Bootstrap_CI")

@paper_item
def figA14_stability():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for a, (m_, nm) in zip(axs, [("mean_abs_prob_shift", "Mean |p_deg - p_clean|"), ("decision_flip_rate", "Decision flip rate")]):
        s_ = S[(S.model == PRIMARY) & (S.family == "in")].groupby("severity")[m_].mean().reindex(SEVS)
        a.bar(np.arange(3), s_.values, 0.6, color=COLOR[PRIMARY], label=LABEL[PRIMARY]); a.set_xticks(np.arange(3), SEVS)
        a.set_ylabel(nm + " (lower = more stable)"); a.set_xlabel("Severity (mean over in-family types)")
    axs[0].legend(fontsize=8); savefig(fig, "FigA14_Prediction_Stability")

@paper_item
def figA15_reliability():
    rc = [c for c in ["clean", "combined_severe"] if c in conds_done]
    fig, axs = plt.subplots(1, len(rc), figsize=(4.6 * len(rc), 4), squeeze=False)
    for a, c in zip(axs[0], rc):
        t = PR[(PRIMARY, c)]; p_, y_ = t.predicted_probability.values, t.ground_truth.values; bn = np.minimum((p_ * 10).astype(int), 9); xs, ys = [], []
        for b in range(10):
            m = bn == b
            if m.sum() >= 20: xs.append(p_[m].mean()); ys.append(y_[m].mean())
        a.plot(xs, ys, marker="o", ms=3, color=COLOR[PRIMARY], label=f"{LBL(PRIMARY)} ECE={RES[(PRIMARY, c)]['ece']:.3f}")
        a.plot([0, 1], [0, 1], "k:"); a.set_title(c, fontsize=10); a.set_xlabel("Mean predicted P(FAKE)"); a.set_ylabel("Observed FAKE fraction"); a.legend(fontsize=7)
    savefig(fig, "FigA15_Reliability_Diagram")

@paper_item
def figA16_training_curves():
    hh = pd.DataFrame(history); fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(hh.epoch, hh.train_acc, label="train"); axs[0].plot(hh.epoch, hh.val_acc, label="val (clean)"); axs[0].set_ylabel("Accuracy")
    axs[1].plot(hh.epoch, hh.train_loss, label="train loss"); axs[1].plot(hh.epoch, hh.val_loss, label="val loss"); axs[1].set_ylabel("Loss")
    for c_, l_ in [("val_f1", "val F1"), ("val_ap", "val PR-AUC"), ("val_auc", "val ROC-AUC")]:
        if c_ in hh and hh[c_].notna().any(): axs[2].plot(hh.epoch, hh[c_], label=l_)
    axs[2].set_ylabel("Score")
    for a in axs: a.set_xlabel("Epoch"); a.legend(fontsize=8); a.axvline(best_epoch, ls="--", c="gray", lw=1)
    savefig(fig, "FigA16_Training_Validation_Curves")

@paper_item
def figA17_cross_dataset():
    cd = pd.DataFrame(cross_rows); dsn = list(cd.dataset.unique()); xl = ["FF++ (in-domain)"] + dsn
    vals = [RES[(PRIMARY, "clean")]["auc"]] + [cd[(cd.dataset == d) & (cd.model == PRIMARY)].auc.iloc[0] for d in dsn]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(np.arange(len(xl)), vals, 0.6, color=COLOR[PRIMARY], label=LABEL[PRIMARY])
    ax.set_xticks(np.arange(len(xl)), xl, rotation=15); ax.set_ylabel("ROC-AUC (FF++ val thresholds)"); ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, ls=":", c="k", lw=1); ax.legend(fontsize=8); savefig(fig, "FigA17_Cross_Dataset_AUC")

@paper_item
def figA18_score_distributions():
    sc = [c for c in ["clean", "combined_severe"] if c in conds_done]
    fig, axs = plt.subplots(1, len(sc), figsize=(4.2 * len(sc), 2.8), squeeze=False, sharex=True)
    for q, c in enumerate(sc):
        t = PR[(PRIMARY, c)]; a = axs[0, q]
        for lab, nm_, col in [(0, "REAL", "#4C72B0"), (1, "FAKE", "#C44E52")]:
            a.hist(t.predicted_probability[t.ground_truth == lab], bins=40, range=(0, 1), alpha=0.6, color=col, label=nm_, density=True)
        a.axvline(THR[PRIMARY], ls="--", c="k", lw=1); a.set_title(f"{LBL(PRIMARY)} | {c}", fontsize=9); a.set_xlabel("Predicted P(FAKE)")
    axs[0, 0].set_ylabel("Density"); axs[0, 0].legend(fontsize=7); savefig(fig, "FigA18_Score_Distributions")

# ============================ 11. SUMMARY ============================
if os.path.exists(FAIL_TSV):
    fl = pd.read_csv(FAIL_TSV, sep="\t", header=None, names=["image_path", "stage", "error_message"], on_bad_lines="skip")
else: fl = pd.DataFrame(columns=["image_path", "stage", "error_message"])
fl.drop_duplicates().to_csv(os.path.join(LOG_DIR, f"{PFX}_Failed_Images.csv"), index=False)
print("\n" + "=" * 90 + "\nSUMMARY\n" + "=" * 90)
print(f"dataset: {len(df):,} imgs | train {len(tr_df):,} / val {len(va_df):,} / test {len(te_df):,} | train REAL {n_real:,} FAKE {n_fake:,} | group overlap {overlap}")
print(f"device: {GPU_NAME} | params {n_tot:,} (trainable {n_tr:,}) | epochs run {len(history)} | best epoch {best_epoch} | best val AUC {best_auc:.5f}")
print(f"{PRIMARY} clean test: AUC {RES[(PRIMARY,'clean')]['auc']:.4f} | F1 {RES[(PRIMARY,'clean')]['f1']:.4f} | Acc {RES[(PRIMARY,'clean')]['accuracy']:.4f} (val-selected thr {THR[PRIMARY]:.3f})")
s = summ[(summ.model == PRIMARY) & (summ.row_type == "overall_mean")]
if len(s): print(f"{PRIMARY} mean in-family degraded AUC {s.auc.iloc[0]:.4f} | mean AUC drop {s.auc_drop.iloc[0]:.4f} | mean F1 drop {s.f1_drop.iloc[0]:.4f}")
for r in cross_rows: print(f"  cross-dataset {r['dataset']:<12} AUC {r['auc']:.4f} F1 {r['f1']:.4f} Acc {r['accuracy']:.4f}")
print(f"failed/unreadable image log rows: {len(fl)} | total runtime {(time.time()-T_START)/3600:.2f} h")
print("best checkpoint:", BEST_PATH, "\noutput dir:", OUT_DIR)
for sd in ("checkpoints", "metrics", "predictions", "figures", "logs"):
    fs = sorted(os.listdir(os.path.join(OUT_DIR, sd)))
    n_ep = len([f for f in fs if "_Epoch" in f and f.endswith(".pth")])
    shown = [f for f in fs if not ("_Epoch" in f and f.endswith(".pth"))]
    if sd == "predictions": shown = f"{len(shown)} prediction CSVs"
    print(f"  {sd}/: {shown}" + (f" + {n_ep} epoch checkpoints" if n_ep else ""))