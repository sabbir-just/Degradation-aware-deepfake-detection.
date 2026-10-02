# =====================================================================
# MODEL B - DEGRADATION-AWARE EfficientNet-B0  (+ prediction consistency)
# Same architecture / split / optimizer / schedule / threshold rule as
# Model A.  Only difference: degradation augmentation + consistency loss.
# ONE Kaggle cell, restart-safe.  RUN_VARIANT="B1" = ablation (no consistency).
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
DATA_ROOT = "/kaggle/input/datasets/syedazmulhasansabbir/face-forensics-frames/Face Forensics Frames"
RUN_VARIANT = "B"                 # "B" = degradation aug + consistency (main) | "B1" = degradation aug only (ablation, lambda=0)
LAMBDA_CONSISTENCY = 0.2          # weight of consistency loss (forced to 0 for B1)
CONSISTENCY_DETACH_CLEAN = True   # clean prediction is the (stop-grad) anchor -> degraded view is pulled toward it; avoids drifting to 0.5
SEED = 42
# --- identical to Model A (fair comparison) ---
IMG_SIZE, RESIZE_SIZE = 224, 256
BATCH_SIZE, EVAL_BATCH_SIZE = 64, 128          # BATCH_SIZE = #source images; each also gets a degraded view (2x forward)
EPOCHS, PATIENCE, MIN_DELTA = 20, 5, 1e-4
LR, WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP = 3e-4, 1e-2, 1, 1.0
SPLIT_RATIOS = (0.80, 0.10, 0.10)
IMBALANCE_TRIGGER = 1.5
NUM_WORKERS = min(4, os.cpu_count() or 2)
MAX_RUNTIME_HOURS = 11.0
USE_JPEG_DRAFT = True
PRETRAINED_WEIGHTS_FILE, ALLOW_RANDOM_INIT = None, False
# --- training-time degradation policy (per sample; the degraded view of the SAME cropped/flipped image) ---
P_NONE, P_SINGLE = 0.20, 0.50                   # remaining 0.30 = combination (2 ops 80% / 3 ops 20%)
TRAIN_RANGES = dict(jpeg_q=(30, 90), down_scale=(0.40, 0.90), blur_sigma=(0.3, 2.0),
                    noise_sigma=(2.0, 12.0), color=(0.75, 1.25))     # pixel units 0-255 for noise
# --- reuse Model A artifacts if found (checkpoint / split). Set explicit paths here if auto-search fails ---
MODEL_PATH_OVERRIDES = {"ModelA": None, "ModelB": None, "ModelB1": None}
MODEL_A_SPLIT_CSV = None
USE_MODEL_A_SPLIT_IF_FOUND = True               # guarantees IDENTICAL train/val/test with Model A
ALLOW_SPLIT_MISMATCH = False                    # exclude other models trained on a different split (leakage risk)
# --- optional cross-dataset (runs only if time remains) ---
RUN_CROSS_DATASET = True
CROSS_PKL = "/kaggle/input/datasets/asmabagum/dataset-scan/dataset_scans_v40.pkl"   # unused while CROSS_KEYS is empty
CROSS_N_PER_CLASS = 5000
CROSS_KEYS = {"celeb": "CelebDF_PKL", "dfdc": "DFDC_PKL", "diffusion": "Diffusion_PKL"}   # PKL-based (set to {} to disable)
CROSS_FOLDERS = {                                # folder-based: class sub-folders labelled by name (REAL_HINTS -> 0, else 1)
    "CelebDF": "/kaggle/input/datasets/syedazmulhasansabbir/celeb-df-frames/content/drive/MyDrive/dataset/celeb_df_frames",
    "DFDC": "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/validation",
    "Diffusion": "/kaggle/input/datasets/syedazmulhasansabbir/diffusion/diffusion",
    "WildDeepfake": "/kaggle/input/datasets/maysuni/wild-deepfake/test",
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REAL_HINTS = ("real", "orig", "orginal", "pristine", "youtube", "authentic", "genuine", "natural")
SPLIT_DIR_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                   "test": "test", "testing": "test"}

assert RUN_VARIANT in ("B", "B1")
PRIMARY = "ModelB" if RUN_VARIANT == "B" else "ModelB1"
FULL = {"ModelA": "ModelA_Baseline", "ModelB": "ModelB_DegradationAware", "ModelB1": "ModelB1_DegAugOnly"}
LABEL = {"ModelA": "Model A (baseline)", "ModelB": "Model B (deg-aware + consist.)", "ModelB1": "Model B1 (deg-aug only)"}
COLOR = {"ModelA": "#4C72B0", "ModelB": "#DD8452", "ModelB1": "#55A868"}
RUN_NAME = {k: f"{v}_EfficientNetB0" for k, v in FULL.items()}
PFX = PRIMARY                                    # prefix for required CSV names (ModelB_*)
LAMBDA = 0.0 if RUN_VARIANT == "B1" else float(LAMBDA_CONSISTENCY)
OUT_DIR = f"/kaggle/working/{RUN_NAME[PRIMARY]}"
for sub in ["checkpoints", "metrics", "predictions", "figures", "logs"]:
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
CK_DIR, MET_DIR, PRED_DIR, FIG_DIR, LOG_DIR = [os.path.join(OUT_DIR, s) for s in
                                                ["checkpoints", "metrics", "predictions", "figures", "logs"]]
FAIL_TSV = os.path.join(LOG_DIR, "failed_images_raw.tsv")
if os.path.exists(FAIL_TSV) and not any(f.endswith(".pth") for f in os.listdir(CK_DIR)): os.remove(FAIL_TSV)

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
torch.backends.cudnn.benchmark = True
GPU_NAME = torch.cuda.get_device_name(0) if use_amp else "cpu"
print("=" * 90 + f"\nMODEL {RUN_VARIANT}  |  {RUN_NAME[PRIMARY]}\n" + "=" * 90)
print(f"python {platform.python_version()} | torch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"CUDA {torch.version.cuda} | GPU {GPU_NAME} | AMP {use_amp} | workers {NUM_WORKERS} | seed {SEED}")
print(f"lambda_consistency = {LAMBDA} | detach clean anchor = {CONSISTENCY_DETACH_CLEAN}")

def log_fail(path, stage, err):
    try:
        with open(FAIL_TSV, "a") as f:
            f.write(f"{path}\t{stage}\t{str(err).replace(chr(9), ' ').replace(chr(10), ' ')[:200]}\n")
    except Exception: pass

def find_file(fname, run_dir):
    for sd in ("checkpoints", "metrics", "predictions", ""):
        c = os.path.join("/kaggle/working", run_dir, sd, fname)
        if os.path.exists(c): return c
    pats = [f"/kaggle/input/*/{fname}", f"/kaggle/input/*/*/{fname}", f"/kaggle/input/*/*/*/{fname}",
            f"/kaggle/input/*/{run_dir}/{fname}", f"/kaggle/input/*/{run_dir}/*/{fname}",
            f"/kaggle/input/*/*/{run_dir}/*/{fname}", f"/kaggle/input/datasets/*/*/{run_dir}/*/{fname}",
            f"/kaggle/input/datasets/*/*/*/{run_dir}/*/{fname}", f"/kaggle/input/models/*/*/*/*/*/{fname}",
            f"/kaggle/input/models/*/*/*/*/*/{run_dir}/*/{fname}"]
    for p in pats:
        h = sorted(glob.glob(p))
        if h: return h[0]
    return None

# ================== 2. DATASET DISCOVERY + SPLIT =====================
print("\n" + "=" * 90 + "\nDATASET DISCOVERY\n" + "=" * 90)
RE_PAIR   = re.compile(r"^(\d+)_(\d+)_(\d+)$")
RE_REAL_M = re.compile(r"^real_(\d+)\.mp4_(\d+)$")
RE_SINGLE = re.compile(r"^(\d+)_(\d+)$")
RE_GENERIC = re.compile(r"^(.*?)[_\-\.]*(?:frame)?[_\-\.]*\d+$", re.I)

def subdirs(p): return sorted(e.name for e in os.scandir(p) if e.is_dir())
def has_images_directly(p): return any(e.is_file() and e.name.lower().endswith(IMG_EXTS) for e in os.scandir(p))

def scan_tree(folder):
    out, stack = [], [folder]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False): stack.append(e.path)
                    elif e.name.lower().endswith(IMG_EXTS): out.append(e.path)
        except Exception as ex: print("  scan error:", d, ex)
    return out

def parse_ids(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = RE_PAIR.match(stem)
    if m: return "pair", int(m[1]), int(m[2])
    m = RE_REAL_M.match(stem) or RE_SINGLE.match(stem)
    if m: return "single", int(m[1]), None
    g = RE_GENERIC.match(stem)
    return "other", (g[1] if g else stem), None

def discover_and_split():
    root = DATA_ROOT
    assert os.path.isdir(root), f"DATA_ROOT not found: {root}"
    while len(subdirs(root)) == 1 and not has_images_directly(root):
        root = os.path.join(root, subdirs(root)[0]); print("descending into single sub-folder:", root)
    print("dataset root:", root); top = subdirs(root); print("top-level folders:", top)
    presplit = any(n.lower() in SPLIT_DIR_NAMES for n in top); tasks = []
    if presplit:
        for n in top:
            if n.lower() in SPLIT_DIR_NAMES:
                for c in subdirs(os.path.join(root, n)): tasks.append((SPLIT_DIR_NAMES[n.lower()], c, os.path.join(root, n, c)))
        print("EXISTING train/val/test folders detected -> using them as-is.")
    else:
        tasks = [(None, c, os.path.join(root, c)) for c in top]
        print("No train/val/test folders -> creating a source-video-level split.")
    cache = os.path.join(OUT_DIR, "file_index_cache.pkl"); key = (root, tuple((s, c) for s, c, _ in tasks)); rows = None
    if os.path.exists(cache):
        try:
            cc = pickle.load(open(cache, "rb"))
            if cc["key"] == key and len(cc["rows"]) > 0: rows = cc["rows"]; print("loaded cached file index:", len(rows))
        except Exception: rows = None
    if rows is None:
        t0 = time.time(); print("scanning files (parallel per class folder) ...")
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex: res = list(ex.map(lambda t: scan_tree(t[2]), tasks))
        rows = [(p, c, s) for (s, c, _), ps in zip(tasks, res) for p in ps]
        pickle.dump({"key": key, "rows": rows}, open(cache, "wb")); print(f"scan done in {time.time()-t0:.0f}s: {len(rows)} files")
    d = pd.DataFrame(rows, columns=["path", "cls", "split"]).sort_values("path").reset_index(drop=True)
    assert len(d) > 0, "No images found."
    cls_names = sorted(d.cls.unique())
    cls2label = {c: (0 if any(h in c.lower() for h in REAL_HINTS) else 1) for c in cls_names}
    print("\nclass-folder -> label (0=REAL, 1=FAKE):")
    for c in cls_names: print(f"   {c:<20} -> {cls2label[c]} ({'REAL' if cls2label[c]==0 else 'FAKE'})   n={int((d.cls==c).sum()):,}")
    d["label"] = d.cls.map(cls2label).astype(int)
    assert d.label.nunique() == 2, "Could not find both REAL and FAKE class folders."
    parsed = [parse_ids(p) for p in d.path]; d["ptype"] = [p[0] for p in parsed]
    print("filename-pattern counts:", d.ptype.value_counts().to_dict())
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for t, a, b in parsed:
        if t == "pair":
            ra, rb = find(a), find(b)
            if ra != rb: parent[max(ra, rb)] = min(ra, rb)
        elif t == "single": find(a)
    d["group"] = [f"vid{find(a):04d}" if t in ("pair", "single") else f"other::{cn}::{a}"
                  for (t, a, b), cn in zip(parsed, d.cls)]
    comp = collections.Counter(collections.Counter(find(x) for x in list(parent)).values())
    print("video-id clusters size->count:", dict(comp))
    n_other = int((d.ptype == "other").sum())
    if n_other / len(d) > 0.01:
        print(f"  WARNING: {n_other:,} files ({n_other/len(d):.1%}) did not match FF++ naming; grouped by prefix only -> leakage prevention NOT guaranteed for them.")
    if presplit: d["split"] = d.split
    else:
        gsz = d.groupby("group").size(); gl = sorted(gsz.index); random.Random(SEED).shuffle(gl)
        tot = len(d); tg = {s: r * tot for s, r in zip(("train", "val", "test"), SPLIT_RATIOS)}
        cur = {s: 0 for s in tg}; asg = {}
        for g in gl:
            s = max(tg, key=lambda k: tg[k] - cur[k]); asg[g] = s; cur[s] += int(gsz[g])
        d["split"] = d.group.map(asg)
    return d

a_split = MODEL_A_SPLIT_CSV or (find_file(f"{RUN_NAME['ModelA']}_Split_Assignment.csv", RUN_NAME["ModelA"]) if USE_MODEL_A_SPLIT_IF_FOUND else None)
df = None
if a_split and os.path.exists(a_split):
    try:
        df = pd.read_csv(a_split).sort_values("path").reset_index(drop=True)
        assert {"path", "split", "label", "group"}.issubset(df.columns)
        miss = [p for p in df.sample(min(50, len(df)), random_state=SEED).path if not os.path.exists(p)]
        assert not miss, f"{len(miss)}/50 sampled paths from Model A split missing on disk"
        if "cls" not in df.columns: df["cls"] = "NA"
        print(f"REUSING Model A split assignment (identical train/val/test): {a_split}\n  files: {len(df):,}")
    except Exception as e:
        print("could not reuse Model A split:", repr(e)[:200], "-> re-deriving the split"); df = None
if df is None: df = discover_and_split()

print(f"\ntotal images: {len(df):,} | REAL: {(df.label==0).sum():,} | FAKE: {(df.label==1).sum():,} | extensions: "
      f"{dict(collections.Counter(os.path.splitext(p)[1].lower() for p in df.path))}")
gs = {s: set(df[df.split == s].group) for s in ("train", "val", "test")}
overlap = {"train&val": len(gs["train"] & gs["val"]), "train&test": len(gs["train"] & gs["test"]), "val&test": len(gs["val"] & gs["test"])}
print("source-group overlap between splits:", overlap)
if any(overlap.values()): print("  WARNING: splits share source groups -> possible leakage.")
for s in ("train", "val", "test"):
    d_ = df[df.split == s]
    print(f"  {s:<5}: {len(d_):>8,} imgs | REAL {int((d_.label==0).sum()):>7,} | FAKE {int((d_.label==1).sum()):>7,} | fake ratio {d_.label.mean():.3f} | groups {d_.group.nunique():,}")
split_hash = hashlib.md5("\n".join((df.path + "|" + df.split).tolist()).encode()).hexdigest()
print("split hash:", split_hash)
df.to_csv(os.path.join(MET_DIR, f"{RUN_NAME[PRIMARY]}_Split_Assignment.csv"), index=False)
tr_df, va_df, te_df = [df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
n_real, n_fake = int((tr_df.label == 0).sum()), int((tr_df.label == 1).sum())
ratio = max(n_real, n_fake) / max(1, min(n_real, n_fake))
pos_weight = (n_real / n_fake) if ratio > IMBALANCE_TRIGGER else 1.0
print(f"class imbalance (train): REAL {n_real:,} vs FAKE {n_fake:,} (ratio {ratio:.2f}) -> BCE pos_weight {pos_weight:.4f} (no resampling)")

# =================== 3. TRANSFORMS + DEGRADATIONS ====================
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
geo_train = T.Compose([T.Resize((RESIZE_SIZE, RESIZE_SIZE)), T.RandomCrop(IMG_SIZE), T.RandomHorizontalFlip(),
                       T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)])      # identical to Model A
geo_eval = T.Compose([T.Resize((RESIZE_SIZE, RESIZE_SIZE)), T.CenterCrop(IMG_SIZE)])          # identical to Model A
to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

def apply_op(img, op, val, rng=None):
    """All degradations act on the 224x224 model-input image (PIL RGB)."""
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

OP_ORDER = ["down", "blur", "noise", "color", "jpeg"]            # JPEG last, as in real pipelines
def sample_train_ops():
    r = random.random()
    if r < P_NONE: return []
    k = 1 if r < P_NONE + P_SINGLE else (2 if random.random() < 0.8 else 3)
    chosen = set(random.sample(OP_ORDER, k)); R = TRAIN_RANGES; ops = []
    for o in OP_ORDER:
        if o not in chosen: continue
        if o == "jpeg": v = random.randint(*R["jpeg_q"])
        elif o == "down": v = random.uniform(*R["down_scale"])
        elif o == "blur": v = random.uniform(*R["blur_sigma"])
        elif o == "noise": v = random.uniform(*R["noise_sigma"])
        else: v = tuple(random.uniform(*R["color"]) for _ in range(3))
        ops.append((o, v))
    return ops

# ---- fixed TEST-time degradation conditions (evaluation only; deterministic per image) ----
COND = collections.OrderedDict()
COND["clean"] = dict(type="clean", sev="none", family="clean", ops=[])
for q, s in [(70, "mild"), (40, "medium"), (20, "severe")]: COND[f"jpeg_{s}"] = dict(type="JPEG", sev=s, family="in", ops=[("jpeg", q)])
for sc, s in [(0.75, "mild"), (0.5, "medium"), (0.25, "severe")]: COND[f"resolution_{s}"] = dict(type="Resolution", sev=s, family="in", ops=[("down", sc)])
for sg, s in [(1.0, "mild"), (2.0, "medium"), (3.0, "severe")]: COND[f"blur_{s}"] = dict(type="Blur", sev=s, family="in", ops=[("blur", sg)])
for sg, s in [(5, "mild"), (10, "medium"), (20, "severe")]: COND[f"noise_{s}"] = dict(type="Noise", sev=s, family="in", ops=[("noise", sg)])
COND["combined_mild"]   = dict(type="Combined", sev="mild", family="in", ops=[("blur", 0.8), ("jpeg", 70)])
COND["combined_medium"] = dict(type="Combined", sev="medium", family="in", ops=[("down", 0.5), ("jpeg", 40)])
COND["combined_severe"] = dict(type="Combined", sev="severe", family="in", ops=[("down", 0.4), ("noise", 10), ("jpeg", 20)])
try:                                                            # out-of-family codec check (never used in training)
    apply_op(Image.new("RGB", (32, 32)), "webp", 30); COND["webp_medium"] = dict(type="WebP(unseen codec)", sev="medium", family="out", ops=[("webp", 30)])
except Exception as e: print("WebP encoder unavailable, skipping unseen-codec condition:", repr(e)[:80])
SEVS = ["mild", "medium", "severe"]
print(f"\ntest-time conditions ({len(COND)}): {list(COND)}")
print(f"training degradation policy: p_none={P_NONE} p_single={P_SINGLE} p_combo={round(1-P_NONE-P_SINGLE,2)} ranges={TRAIN_RANGES}")
print("(severe levels for JPEG/resolution/blur/noise lie OUTSIDE the training ranges by design)")

def cond_title(n):
    c = COND[n]; return ("Clean_Test" if n == "clean" else f"{c['type'].split('(')[0]}_{c['sev'].capitalize()}")
def pred_file(mkey, n): return os.path.join(PRED_DIR, f"{FULL[mkey]}_{cond_title(n)}_Predictions.csv")

# ============================ 4. DATA ================================
def load_rgb(p):
    img = Image.open(p)
    if USE_JPEG_DRAFT and img.format == "JPEG": img.draft("RGB", (RESIZE_SIZE, RESIZE_SIZE))
    return img.convert("RGB")

class TrainDS(Dataset):
    def __init__(self, d): self.paths, self.labels = d.path.tolist(), d.label.values.astype(np.float32)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for k in range(5):
            j = (i + k) % len(self.paths)
            try:
                base = geo_train(load_rgb(self.paths[j]))               # SAME crop/flip/jitter for both views
                xc = to_tensor(base); xd = to_tensor(degrade_ops(base, sample_train_ops()))
                return xc, xd, torch.tensor(self.labels[j]), j
            except Exception as e: log_fail(self.paths[j], "train", e)
        raise RuntimeError("5 consecutive unreadable images")

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
                    base = degrade_ops(base, self.ops, rng)                # identical degraded pixels for every model
                return to_tensor(base), torch.tensor(self.labels[j]), j
            except Exception as e: log_fail(self.paths[j], self.stage if self.cond == "clean" else f"{self.stage}:{self.cond}", e)
        raise RuntimeError("5 consecutive unreadable images")

def winit(w): np.random.seed(SEED + w); random.seed(SEED + w)
def make_loader(ds, bs, shuffle=False, persistent=False):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=use_amp, worker_init_fn=winit)
    if NUM_WORKERS > 0: kw.update(persistent_workers=persistent, prefetch_factor=4)
    g = torch.Generator(); g.manual_seed(SEED)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle, generator=g, **kw)
train_loader = make_loader(TrainDS(tr_df), BATCH_SIZE, True, True)
val_loader   = make_loader(EvalDS(va_df, "clean", "val"), EVAL_BATCH_SIZE)
steps_per_epoch = len(train_loader)

# ============================ 5. MODEL ===============================
def build_arch(pretrained):
    if pretrained:
        try:
            m = torchvision.models.efficientnet_b0(weights="DEFAULT"); print("loaded ImageNet-pretrained EfficientNet-B0")
        except Exception as e:
            print("pretrained download failed:", repr(e)[:200])
            if PRETRAINED_WEIGHTS_FILE and os.path.exists(PRETRAINED_WEIGHTS_FILE):
                m = torchvision.models.efficientnet_b0(weights=None); m.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE, map_location="cpu"))
            elif ALLOW_RANDOM_INIT: m = torchvision.models.efficientnet_b0(weights=None); print("!! RANDOM INIT")
            else: raise RuntimeError("Cannot get ImageNet weights. Turn Kaggle Internet ON or set PRETRAINED_WEIGHTS_FILE.")
    else: m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)          # identical head to Model A
    return m

model = build_arch(True).to(dev).to(memory_format=torch.channels_last)
n_tot = sum(p.numel() for p in model.parameters()); n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nparams total {n_tot:,} | trainable {n_tr:,} | head: {model.classifier}")
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps, warm = EPOCHS * steps_per_epoch, max(1, WARMUP_EPOCHS * steps_per_epoch)
def lr_lambda(s):
    if s < warm: return (s + 1) / warm
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
CONFIG = dict(run=RUN_NAME[PRIMARY], variant=RUN_VARIANT, lambda_consistency=LAMBDA, detach_clean=CONSISTENCY_DETACH_CLEAN,
              consistency="mean |sigmoid(z_clean) - sigmoid(z_deg)|", cls_loss="0.5*(BCE(clean)+BCE(deg)), pos_weight",
              data_root=DATA_ROOT, img_size=IMG_SIZE, resize_size=RESIZE_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS,
              patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS, grad_clip=GRAD_CLIP, seed=SEED,
              pos_weight=pos_weight, split_hash=split_hash, mean=MEAN, std=STD, label_convention="REAL=0, FAKE=1, prob=P(FAKE)",
              p_none=P_NONE, p_single=P_SINGLE, train_ranges=TRAIN_RANGES, test_conditions={k: dict(type=v["type"], sev=v["sev"], ops=v["ops"]) for k, v in COND.items()},
              python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda, gpu=GPU_NAME,
              params_total=n_tot, params_trainable=n_tr)
json.dump(CONFIG, open(os.path.join(MET_DIR, f"{PFX}_Configuration.json"), "w"), indent=2, default=str)

# ======================= 6. METRICS / INFERENCE ======================
def ece_score(y, p, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = ((p > edges[i]) & (p <= edges[i + 1])) if i else ((p >= edges[i]) & (p <= edges[i + 1]))
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def compute_metrics(y, p, thr):
    pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0; spec = tn / (tn + fp) if tn + fp else 0.0
    two = len(np.unique(y)) == 2
    return dict(n=int(len(y)), threshold=float(thr), auc=float(roc_auc_score(y, p)) if two else float("nan"),
                ap=float(average_precision_score(y, p)) if two else float("nan"), accuracy=float((pred == y).mean()),
                precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                specificity=float(spec), balanced_acc=float((rec + spec) / 2), ece=ece_score(y, p),
                brier=float(np.mean((p - y) ** 2)), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

@torch.inference_mode()
def run_models(models, loader):
    """One pass over `loader`, all models see the SAME batch. Returns ({name: logits}, labels, idx) de-duplicated by idx."""
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

def sigmoid(z): return 1 / (1 + np.exp(-z))

def val_epoch():
    o, y, _ = run_models({"m": model}, val_loader); lg = o["m"]; p = sigmoid(lg)
    loss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(lg), torch.tensor(y, dtype=torch.float32),
                                                                pos_weight=torch.tensor([pos_weight])))
    m = compute_metrics(y, p, 0.5)
    return dict(val_loss=loss, val_acc=m["accuracy"], val_precision=m["precision"], val_recall=m["recall"], val_f1=m["f1"],
                val_auc=m["auc"], val_ap=m["ap"], val_specificity=m["specificity"], val_bal_acc=m["balanced_acc"])

# ============================ 7. RESUME ==============================
class ConfigMismatch(Exception): pass
CK_RE = re.compile(rf"^{RUN_NAME[PRIMARY]}_Epoch(\d{{3}})\.pth$")
def ck_path(ep): return os.path.join(CK_DIR, f"{RUN_NAME[PRIMARY]}_Epoch{ep:03d}.pth")
BEST_PATH = os.path.join(CK_DIR, f"{RUN_NAME[PRIMARY]}_Best.pth")
def save_atomic(obj, path): tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

start_epoch, best_auc, best_epoch, no_improve, history = 0, -1.0, 0, 0, []
for ep_n, fn in sorted([(int(m[1]), f) for f in os.listdir(CK_DIR) if (m := CK_RE.match(f))], reverse=True):
    try:
        ck = torch.load(os.path.join(CK_DIR, fn), map_location="cpu", weights_only=False)
        c = ck["config"]
        if c["split_hash"] != split_hash or abs(c["lambda_consistency"] - LAMBDA) > 1e-12:
            raise ConfigMismatch(f"checkpoint config differs (split or lambda). Delete {OUT_DIR} to start fresh.")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_auc, best_epoch = ck["epoch"], ck["best_val_auc"], ck["best_epoch"]
        no_improve, history = ck["epochs_no_improve"], ck["history"]
        print(f"\nRESUMED from {fn} (epoch {start_epoch}, best val AUC {best_auc:.5f} @ epoch {best_epoch})"); break
    except ConfigMismatch: raise
    except Exception as e: print(f"  could not load {fn} ({repr(e)[:100]}); trying an older one")
else: print("\nno valid checkpoint -> starting from pretrained EfficientNet-B0")

# ============================ 8. TRAINING ============================
print("\n" + "=" * 90 + "\nTRAINING (clean + degraded view per sample)\n" + "=" * 90)
epoch_times, stopped_for_time, total_nonfinite = [], False, 0
finished = start_epoch >= EPOCHS or no_improve >= PATIENCE
if finished: print("training already complete in loaded checkpoint -> skipping to final evaluation")
for ep in range(start_epoch + 1, EPOCHS + 1):
    if finished: break
    el = (time.time() - T_START) / 3600
    if epoch_times and el + 1.15 * np.mean(epoch_times) / 3600 > MAX_RUNTIME_HOURS - 0.6:   # reserve ~0.6h for robustness eval
        stopped_for_time = True; print(f"time budget reached ({el:.2f}h) -> stopping. Re-run the cell to resume."); break
    model.train(); t0 = time.time(); seen = 0; s_tot = s_cls = s_con = 0.0; c_clean = c_deg = 0; nonfin = 0; consec = 0
    for it, (xc, xd, y, idx) in enumerate(train_loader, 1):
        bs = y.size(0)
        try:
            xc = xc.to(dev, non_blocking=True); xd = xd.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            x = torch.cat([xc, xd], 0).contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x).squeeze(1)
            lg = logits.float(); lc, ld = lg[:bs], lg[bs:]
            l_cls = 0.5 * (criterion(lc, y) + criterion(ld, y))
            pc = torch.sigmoid(lc); pc = pc.detach() if CONSISTENCY_DETACH_CLEAN else pc
            l_con = (pc - torch.sigmoid(ld)).abs().mean()
            loss = l_cls + LAMBDA * l_con
            if not torch.isfinite(loss):
                nonfin += 1; consec += 1; total_nonfinite += 1
                for j in idx.tolist(): log_fail(tr_df.path[j], "train_nonfinite_loss", f"epoch {ep} iter {it}")
                optimizer.zero_grad(set_to_none=True)
                if consec > 50: raise RuntimeError("50 consecutive non-finite losses - aborting (not hiding numerical problems)")
                continue
            consec = 0
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); scheduler.step()
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache(); nonfin += 1
            log_fail("batch", "train_cuda_oom", f"epoch {ep} iter {it}"); print("  CUDA OOM -> batch skipped"); continue
        seen += bs; s_tot += loss.item() * bs; s_cls += l_cls.item() * bs; s_con += l_con.item() * bs
        c_clean += (lc.detach() > 0).eq(y > 0.5).sum().item(); c_deg += (ld.detach() > 0).eq(y > 0.5).sum().item()
        if it % 200 == 0 or it == steps_per_epoch:
            sp = seen / (time.time() - t0)
            print(f"  ep {ep:02d} [{it}/{steps_per_epoch}] total {s_tot/seen:.4f} cls {s_cls/seen:.4f} cons {s_con/seen:.4f} "
                  f"| acc clean {c_clean/seen:.4f} deg {c_deg/seen:.4f} | {sp:.0f} img/s | ETA {(len(train_loader.dataset)-seen)/sp/60:.1f} min", flush=True)
    assert seen > 0, "no batch trained this epoch"
    vm = val_epoch(); improved = vm["val_auc"] > best_auc + MIN_DELTA
    if improved: best_auc, best_epoch, no_improve = vm["val_auc"], ep, 0
    else: no_improve += 1
    history.append(dict(epoch=ep, train_total_loss=s_tot / seen, train_cls_loss=s_cls / seen, train_cons_loss=s_con / seen,
                        train_acc_clean=c_clean / seen, train_acc_degraded=c_deg / seen, lr=optimizer.param_groups[0]["lr"],
                        **vm, nonfinite_or_oom_batches=nonfin, epoch_minutes=(time.time() - t0) / 60))
    ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
              epoch=ep, best_val_auc=best_auc, best_epoch=best_epoch, epochs_no_improve=no_improve, history=history,
              config=CONFIG, seed=SEED)
    save_atomic(ck, ck_path(ep))
    if improved: save_atomic(ck, BEST_PATH)                                    # selection by VALIDATION AUC only
    pd.DataFrame(history).to_csv(os.path.join(MET_DIR, f"{PFX}_Training_History.csv"), index=False)
    epoch_times.append(time.time() - t0)
    print(f"EPOCH {ep:02d}/{EPOCHS} | train total {s_tot/seen:.4f} (cls {s_cls/seen:.4f}, cons {s_con/seen:.4f}) acc {c_clean/seen:.4f} | "
          f"val loss {vm['val_loss']:.4f} AUC {vm['val_auc']:.5f} acc {vm['val_acc']:.4f} F1 {vm['val_f1']:.4f} | best {best_auc:.5f}@{best_epoch} "
          f"{'*' if improved else f'(no improve {no_improve}/{PATIENCE})'} | {epoch_times[-1]/60:.1f} min | elapsed {(time.time()-T_START)/3600:.2f}h", flush=True)
    if no_improve >= PATIENCE: print(f"early stopping: no val-AUC improvement for {PATIENCE} epochs"); break
if total_nonfinite: print(f"WARNING: {total_nonfinite} non-finite/OOM batches skipped this session (see failed-images log)")

# ============ 9. LOAD BEST + OTHER MODELS (A, B/B1) FOR COMPARISON ===
print("\n" + "=" * 90 + "\nFINAL EVALUATION (best checkpoint by VAL AUC; test used only here)\n" + "=" * 90)
assert os.path.exists(BEST_PATH), "No best checkpoint was saved."
bck = torch.load(BEST_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(bck["model"]); model.eval()
print(f"{PRIMARY}: best checkpoint epoch {bck['epoch']} (val AUC {bck['best_val_auc']:.5f})")
if stopped_for_time: print("NOTE: training stopped by the time budget; results are for the best-so-far checkpoint.")
models = collections.OrderedDict([(PRIMARY, model)])
others = ["ModelA"] + (["ModelB1"] if PRIMARY == "ModelB" else ["ModelB"])
skipped_models = {}
for k in others:
    pth = MODEL_PATH_OVERRIDES.get(k) or find_file(f"{RUN_NAME[k]}_Best.pth", RUN_NAME[k])
    if not pth or not os.path.exists(pth):
        skipped_models[k] = "checkpoint not found"; print(f"{k}: checkpoint not found -> not included in comparison"); continue
    try:
        ckx = torch.load(pth, map_location="cpu", weights_only=False); mx = build_arch(False)
        mx.load_state_dict(ckx["model"]); mx = mx.to(dev).to(memory_format=torch.channels_last).eval()
        same = ckx.get("config", {}).get("split_hash") == split_hash
        print(f"{k}: loaded {pth} | epoch {ckx.get('epoch')} | split matches this run: {same}")
        if not same and not ALLOW_SPLIT_MISMATCH:
            skipped_models[k] = "different split (leakage risk)"; print(f"  -> EXCLUDED: trained on a different split; test sets would not be comparable/clean"); del mx; continue
        models[k] = mx
    except Exception as e: skipped_models[k] = repr(e)[:100]; print(f"{k}: failed to load ({repr(e)[:120]})")
have_cmp = len(models) > 1
print("models evaluated:", list(models), "| skipped:", skipped_models)

# ===== 10. CLEAN VAL (thresholds) + CLEAN TEST + ROBUSTNESS EVAL =====
o, vy, vi = run_models(models, val_loader)
THR = {}
for k, lg in o.items():
    f_, t_, th_ = roc_curve(vy, sigmoid(lg)); THR[k] = float(th_[np.argmax(t_ - f_)])        # Youden J on CLEAN validation only
print("validation-selected thresholds (clean val, reused for ALL test conditions):", {k: round(v, 4) for k, v in THR.items()})

RES, PROBS = {}, {}       # RES[(model, cond)] = metrics ; PROBS[(model, cond)] = (y, p, idx)
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

# clean test: primary alone first (timed) so total inference time is meaningful
t0 = time.time(); clean_loader = make_loader(EvalDS(te_df, "clean", "test"), EVAL_BATCH_SIZE)
o1, ty, ti = run_models({PRIMARY: model}, clean_loader); total_clean_time = time.time() - t0
ms_per_img = 1000 * time_inference(model, clean_loader)
print(f"clean test inference: {total_clean_time:.1f}s total for {len(ty):,} imgs (incl. data loading) | model-only {ms_per_img:.3f} ms/img (batch {EVAL_BATCH_SIZE})")
clean_out = {PRIMARY: o1[PRIMARY]}
if len(models) > 1:
    o2, ty2, ti2 = run_models({k: m for k, m in models.items() if k != PRIMARY}, clean_loader)
    assert (ty2 == ty).all(); clean_out.update(o2)
for k in models:
    p = sigmoid(clean_out[k]); PROBS[(k, "clean")] = (ty, p, ti); write_pred(k, "clean", ty, p, ti)
    RES[(k, "clean")] = dict(model=k, condition="clean", type="clean", severity="none", family="clean", **compute_metrics(ty, p, THR[k]))
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
        ld_ = make_loader(EvalDS(te_df, cname, "test"), EVAL_BATCH_SIZE)
        oo, yy, ii = run_models(models, ld_)
        for k in models:
            p = sigmoid(oo[k]); PROBS[(k, cname)] = (yy, p, ii); write_pred(k, cname, yy, p, ii)
        src = "computed"
    for k in models:
        yy, p, _ = PROBS[(k, cname)]
        RES[(k, cname)] = dict(model=k, condition=cname, type=COND[cname]["type"], severity=COND[cname]["sev"],
                               family=COND[cname]["family"], **compute_metrics(yy, p, THR[k]))
    print(f"  [{n_c:02d}/{len(COND)-1}] {cname:<18} {src:<8} {time.time()-t0:5.0f}s | " +
          " | ".join(f"{k} AUC {RES[(k, cname)]['auc']:.4f} F1 {RES[(k, cname)]['f1']:.4f}" for k in models), flush=True)

# ============ 11. TABLES: robustness, drops, comparisons =============
R = pd.DataFrame(list(RES.values()))
for c in ["auc", "ap", "accuracy", "precision", "recall", "f1", "balanced_acc"]: pass
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
P_[rob_cols].to_csv(os.path.join(MET_DIR, f"{PFX}_Robustness_Metrics.csv"), index=False)
P_[["model", "condition", "type", "severity", "family", "auc", "f1", "accuracy", "balanced_acc", "ap"] + drop_cols].to_csv(
    os.path.join(MET_DIR, f"{PFX}_Degradation_Comparison.csv"), index=False)
pd.concat([P_[P_.condition != "clean"].assign(row_type="condition")[["row_type", "model", "condition", "type", "severity"] + drop_cols],
           summ[summ.model == PRIMARY]], ignore_index=True).to_csv(os.path.join(MET_DIR, f"{PFX}_Performance_Drop.csv"), index=False)
# clean test metrics (fixed 0.5 and validation-selected threshold) + efficiency
pc = PROBS[(PRIMARY, "clean")]
mc05 = compute_metrics(pc[0], pc[1], 0.5); mcv = compute_metrics(pc[0], pc[1], THR[PRIMARY])
pd.DataFrame([dict(rule="fixed 0.5", **mc05), dict(rule="val-selected (Youden, clean val)", **mcv)]).assign(
    params_total=n_tot, params_trainable=n_tr, ms_per_image_model_only=ms_per_img, clean_test_total_seconds=total_clean_time,
    best_epoch=best_epoch, best_val_auc=best_auc).to_csv(os.path.join(MET_DIR, f"{PFX}_Final_Clean_Test_Metrics.csv"), index=False)
cmv = np.array([[mcv["tn"], mcv["fp"]], [mcv["fn"], mcv["tp"]]])
pd.DataFrame(cmv, index=["true_REAL", "true_FAKE"], columns=["pred_REAL", "pred_FAKE"]).to_csv(os.path.join(MET_DIR, f"{PFX}_Confusion_Matrix_Clean.csv"))
P_[P_.condition != "clean"][["model", "condition", "type", "severity", "tp", "tn", "fp", "fn"]].to_csv(
    os.path.join(MET_DIR, f"{PFX}_Confusion_Matrix_Degraded.csv"), index=False)
if have_cmp:
    cmp_name = f"ModelA_vs_{PFX}"
    R[R.condition == "clean"][rob_cols].to_csv(os.path.join(MET_DIR, f"{cmp_name}_Clean_Comparison.csv"), index=False)
    R[rob_cols + drop_cols].to_csv(os.path.join(MET_DIR, f"{cmp_name}_Robustness_Comparison.csv"), index=False)
    wide = Rd.pivot_table(index=["condition", "type", "severity", "family"], columns="model", values=["auc_drop", "f1_drop", "auc", "f1"]).reset_index()
    wide.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in wide.columns]
    if "ModelA" in models and PRIMARY in models:
        wide["auc_drop_reduction_vs_A"] = wide["auc_drop_ModelA"] - wide[f"auc_drop_{PRIMARY}"]     # >0 => primary loses LESS AUC than A
        wide["f1_drop_reduction_vs_A"] = wide["f1_drop_ModelA"] - wide[f"f1_drop_{PRIMARY}"]
    pd.concat([wide.assign(row_type="condition"), summ], ignore_index=True).to_csv(os.path.join(MET_DIR, f"{cmp_name}_Performance_Drop.csv"), index=False)

print("\nCLEAN TEST (val-selected thr): " + " | ".join(f"{k}: AUC {RES[(k,'clean')]['auc']:.4f} F1 {RES[(k,'clean')]['f1']:.4f} Acc {RES[(k,'clean')]['accuracy']:.4f}" for k in models))
print(f"{PRIMARY} clean @0.5: AUC {mc05['auc']:.4f} Acc {mc05['accuracy']:.4f} F1 {mc05['f1']:.4f} P {mc05['precision']:.4f} R {mc05['recall']:.4f} "
      f"Spec {mc05['specificity']:.4f} BalAcc {mc05['balanced_acc']:.4f} AP {mc05['ap']:.4f} | TP {mc05['tp']} TN {mc05['tn']} FP {mc05['fp']} FN {mc05['fn']}")
print("\nmean in-family AUC / F1 per severity (higher = more robust):")
for k in models:
    s = summ[(summ.model == k) & (summ.row_type == "severity_mean")]
    print(f"  {k:<8} clean AUC {RES[(k,'clean')]['auc']:.4f} | " + " | ".join(f"{r.severity}: AUC {r.auc:.4f} (drop {r.auc_drop:.4f}) F1 {r.f1:.4f}" for r in s.itertuples()))

# ============================ 12. FIGURES ============================
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
sfx = "" if have_cmp else " (Model-only; no other model found)"
def savefig(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
h = pd.DataFrame(history)
fig, ax = plt.subplots(figsize=(6, 4)); 
for c, l in [("train_total_loss", "total"), ("train_cls_loss", "classification"), ("train_cons_loss", "consistency (unweighted)")]: ax.plot(h.epoch, h[c], marker="o", ms=3, label=l)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.legend(); savefig(fig, "FigB01_ModelB_Loss_Components")
fig, ax = plt.subplots(figsize=(5.5, 4)); ax.plot(h.epoch, h.val_auc, marker="o", ms=3, color=COLOR[PRIMARY]); ax.axvline(best_epoch, ls="--", c="gray")
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation ROC-AUC (clean)"); savefig(fig, "FigB02_ModelB_Validation_AUC")
fig, ax = plt.subplots(figsize=(5, 4.5))
for k in models:
    y_, p_, _ = PROBS[(k, "clean")]; f_, t_, _ = roc_curve(y_, p_); ax.plot(f_, t_, color=COLOR[k], label=f"{LABEL[k]} AUC={RES[(k,'clean')]['auc']:.4f}")
ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right", fontsize=8); ax.set_title("Clean FF++ test" + sfx, fontsize=9)
savefig(fig, "FigB03_ModelB_Clean_Test_ROC")
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cmv, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, f"{cmv[i,j]:,}", ha="center", va="center", color="white" if cmv[i, j] > cmv.max() / 2 else "black")
ax.set_xticks([0, 1], ["REAL", "FAKE"]); ax.set_yticks([0, 1], ["REAL", "FAKE"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"{LABEL[PRIMARY]}\nclean test, thr={THR[PRIMARY]:.3f}", fontsize=9); savefig(fig, "FigB04_ModelB_Clean_Confusion_Matrix")

def sev_curve(metric, fname, ylabel):
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); xs = ["clean"] + SEVS
    for k in models:
        ys, lo, hi = [RES[(k, "clean")][metric]], [0], [0]
        for s in SEVS:
            v = Rd[(Rd.model == k) & (Rd.family == "in") & (Rd.severity == s)][metric].values
            if len(v): ys.append(v.mean()); lo.append(v.mean() - v.min()); hi.append(v.max() - v.mean())
            else: ys.append(np.nan); lo.append(0); hi.append(0)
        ax.errorbar(range(4), ys, yerr=[lo, hi], marker="o", capsize=3, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(range(4), xs); ax.set_xlabel("Degradation severity (mean over in-family types; bars = min-max)"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8); ax.set_title(sfx.strip() or "", fontsize=8); savefig(fig, fname)
sev_curve("auc", "FigB05_AUC_vs_Degradation_Severity", "ROC-AUC"); sev_curve("f1", "FigB06_F1_vs_Degradation_Severity", "F1")

fig, axs = plt.subplots(1, 2, figsize=(9.5, 4)); w = 0.8 / len(models)
for a, (m_, nm) in zip(axs, [("auc_drop", "AUC drop from clean"), ("f1_drop", "F1 drop from clean")]):
    for i, k in enumerate(models):
        s = summ[(summ.model == k) & (summ.row_type == "severity_mean")].set_index("severity")
        a.bar(np.arange(3) + i * w, [s[m_].get(sv, np.nan) for sv in SEVS], w, color=COLOR[k], label=LABEL[k])
    a.set_xticks(np.arange(3) + w * (len(models) - 1) / 2, SEVS); a.set_ylabel(nm + " (lower = more robust)"); a.set_xlabel("Severity (mean over in-family types)")
axs[0].legend(fontsize=8); savefig(fig, "FigB07_ModelA_vs_ModelB_Performance_Drop")

TYPES = ["JPEG", "Resolution", "Blur", "Noise", "Combined"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4)); w = 0.8 / len(models)
for a, (m_, nm) in zip(axs, [("auc", "ROC-AUC"), ("f1", "F1")]):
    for i, k in enumerate(models):
        vals = [Rd[(Rd.model == k) & (Rd.type == t_) & (Rd.family == "in")][m_].mean() for t_ in TYPES]
        a.bar(np.arange(len(TYPES)) + i * w, vals, w, color=COLOR[k], label=LABEL[k])
    a.set_xticks(np.arange(len(TYPES)) + w * (len(models) - 1) / 2, TYPES, rotation=20); a.set_ylabel(f"mean {nm} over severities")
    lo_ = np.nanmin([Rd[(Rd.family == "in")][m_].min(), 0.5]); a.set_ylim(max(0, lo_ - 0.05), 1.0)
axs[0].legend(fontsize=8); savefig(fig, "FigB08_Degradation_Type_Comparison")

reps = [c for c in ["jpeg_medium", "blur_medium", "noise_medium", "combined_severe"] if c in COND]
fig, axs = plt.subplots(2, 2, figsize=(8, 7)); axs = axs.ravel()
for a, c in zip(axs, reps):
    for k in models:
        if (k, c) not in PROBS: continue
        y_, p_, _ = PROBS[(k, c)]; f_, t_, _ = roc_curve(y_, p_); a.plot(f_, t_, color=COLOR[k], label=f"{LABEL[k].split(' (')[0]} AUC={RES[(k,c)]['auc']:.3f}")
    a.plot([0, 1], [0, 1], "k:", lw=1); a.set_title(c, fontsize=10); a.legend(fontsize=7, loc="lower right"); a.set_xlabel("FPR"); a.set_ylabel("TPR")
for a in axs[len(reps):]: a.axis("off")
savefig(fig, "FigB09_Robustness_ROC_Comparison")

# ================= 13. OPTIONAL CROSS-DATASET EVAL ===================
cross_rows = []
if RUN_CROSS_DATASET and (time.time() - T_START) / 3600 < MAX_RUNTIME_HOURS - 0.75:
    print("\n" + "=" * 90 + "\nOPTIONAL: CROSS-DATASET (unseen-domain) EVALUATION - thresholds from FF++ clean val\n" + "=" * 90)
    try:
        allp = []
        if CROSS_KEYS:
            assert os.path.exists(CROSS_PKL), f"scan PKL not found: {CROSS_PKL}"
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
        for nm, root_ in CROSS_FOLDERS.items():
            if not os.path.isdir(root_): print(f"  {nm}: folder not found -> skipped ({root_})"); continue
            rows_ = []
            for c in subdirs(root_):
                lab = 0 if any(h in c.lower() for h in REAL_HINTS) else 1
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
            t0 = time.time(); oo, yy, ii = run_models(models, make_loader(EvalDS(d_, "clean", f"cross:{nm}"), EVAL_BATCH_SIZE))
            tab = d_.iloc[ii][["path", "label", "group", "dataset"]].reset_index(drop=True)
            for k in models:
                p = sigmoid(oo[k]); m_ = compute_metrics(yy, p, THR[k]); cross_rows.append(dict(dataset=nm, model=k, **m_)); tab[f"prob_{k}"] = p
            tab.to_csv(os.path.join(PRED_DIR, f"{PFX}_CrossDataset_{nm}_Predictions.csv"), index=False)
            print(f"  {nm:<10} n={len(yy):,} ({time.time()-t0:.0f}s) | " + " | ".join(f"{k} AUC {[r for r in cross_rows if r['dataset']==nm and r['model']==k][0]['auc']:.4f}" for k in models))
        if cross_rows:
            cdf = pd.DataFrame(cross_rows); cdf.to_csv(os.path.join(MET_DIR, f"{PFX}_CrossDataset_Metrics.csv"), index=False)
            if have_cmp: cdf.to_csv(os.path.join(MET_DIR, f"ModelA_vs_{PFX}_CrossDataset.csv"), index=False)
            print("(models trained on FF++ only; these are unseen-domain results with FF++-validation thresholds)")
    except Exception as e: print("cross-dataset evaluation skipped:", repr(e)[:200])
else: print("\ncross-dataset evaluation skipped (disabled or not enough time budget).")

# ============ 13b. PAPER EXTRAS (CIs, stability, tables, extra figures) ============
print("\n" + "=" * 90 + "\nPAPER EXTRAS: bootstrap CIs, prediction stability, tables, extra figures\n" + "=" * 90)
N_BOOT = 200                                       # video-cluster bootstrap resamples (paired across models)
def paper_item(fn):
    try: fn(); print("  done:", fn.__name__)
    except Exception as e: plt.close("all"); print(f"  SKIPPED {fn.__name__}: {repr(e)[:150]}")
    return fn
PR, stab, boot_rows, diff_rows = {}, [], [], []
refs = [k for k in models if k != PRIMARY]; ref = refs[0] if refs else None
LBL = lambda k: LABEL[k].split(" (")[0]

@paper_item
def load_pred_tables():
    for k in models:
        for c in COND:
            f = pred_file(k, c)
            if os.path.exists(f):
                PR[(k, c)] = pd.read_csv(f, usecols=["image_path", "source_video_id", "ground_truth", "predicted_probability"]
                                         ).drop_duplicates("image_path").set_index("image_path")
conds_done = [c for c in COND if all((k, c) in PR for k in models)]
print("conditions available for paper extras:", len(conds_done))

@paper_item
def prediction_stability():                        # how much do probabilities / decisions move under degradation?
    for k in models:
        base = PR[(k, "clean")]
        for c in conds_done:
            if c == "clean": continue
            t = PR[(k, c)]; j = base.index.intersection(t.index)
            pb, pq = base.loc[j, "predicted_probability"].values, t.loc[j, "predicted_probability"].values
            stab.append(dict(model=k, condition=c, type=COND[c]["type"], severity=COND[c]["sev"], family=COND[c]["family"], n=len(j),
                             mean_abs_prob_shift=float(np.abs(pb - pq).mean()), mean_signed_shift=float((pq - pb).mean()),
                             decision_flip_rate=float(((pb >= THR[k]) != (pq >= THR[k])).mean())))
    pd.DataFrame(stab).to_csv(os.path.join(MET_DIR, f"{PFX}_Prediction_Stability.csv"), index=False)
S = pd.DataFrame(stab)

@paper_item
def bootstrap_auc_ci():                            # resample SOURCE-VIDEO clusters; same resamples for every model (paired)
    if (time.time() - T_START) / 3600 > MAX_RUNTIME_HOURS - 0.3: print("  time budget low -> bootstrap skipped"); return
    for c in conds_done:
        j = PR[(PRIMARY, c)].index
        for k in models: j = j.intersection(PR[(k, c)].index)
        y = PR[(PRIMARY, c)].loc[j, "ground_truth"].values.astype(int)
        g = PR[(PRIMARY, c)].loc[j, "source_video_id"].astype(str).values
        P = {k: PR[(k, c)].loc[j, "predicted_probability"].values for k in models}
        ug, inv = np.unique(g, return_inverse=True); members = [np.where(inv == i)[0] for i in range(len(ug))]
        rng = np.random.default_rng(SEED); bs = {k: [] for k in models}
        for _ in range(N_BOOT):
            ii = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
            if len(np.unique(y[ii])) < 2: continue
            for k in models: bs[k].append(roc_auc_score(y[ii], P[k][ii]))
        if not bs[PRIMARY]: continue
        for k in models:
            a = np.array(bs[k])
            boot_rows.append(dict(condition=c, model=k, auc=roc_auc_score(y, P[k]), ci95_lo=np.percentile(a, 2.5),
                                  ci95_hi=np.percentile(a, 97.5), n_boot=len(a), n_groups=len(ug)))
        for o in refs:
            d = np.array(bs[PRIMARY]) - np.array(bs[o]); lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
            diff_rows.append(dict(vs=o, condition=c, type=COND[c]["type"], severity=COND[c]["sev"], family=COND[c]["family"],
                                  auc_diff=roc_auc_score(y, P[PRIMARY]) - roc_auc_score(y, P[o]), ci95_lo=lo, ci95_hi=hi,
                                  frac_boot_diff_le0=float((d <= 0).mean()), significant_95=bool(lo > 0 or hi < 0)))
    pd.DataFrame(boot_rows).to_csv(os.path.join(MET_DIR, f"{PFX}_AUC_Bootstrap_CI.csv"), index=False)
    if diff_rows: pd.DataFrame(diff_rows).to_csv(os.path.join(MET_DIR, f"{PFX}_AUC_Difference_CI.csv"), index=False)

@paper_item
def paper_tables():
    W = R.pivot_table(index=["condition", "type", "severity", "family"], columns="model", values=["auc", "f1", "accuracy", "ap", "balanced_acc"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]; W = W.reset_index()
    W["_o"] = W.condition.map({c: i for i, c in enumerate(COND)}); W = W.sort_values("_o").drop(columns="_o")
    if ref:
        W[f"auc_gain_{PRIMARY}_vs_{ref}"] = W[f"auc_{PRIMARY}"] - W[f"auc_{ref}"]; W[f"f1_gain_{PRIMARY}_vs_{ref}"] = W[f"f1_{PRIMARY}"] - W[f"f1_{ref}"]
    W.to_csv(os.path.join(MET_DIR, f"{PFX}_Table_Metrics_By_Condition.csv"), index=False)
    rows = []
    for k in models:
        r_ = dict(model=LABEL[k], clean_auc=RES[(k, "clean")]["auc"], clean_f1=RES[(k, "clean")]["f1"], clean_acc=RES[(k, "clean")]["accuracy"])
        for s in SEVS:
            x = summ[(summ.model == k) & (summ.row_type == "severity_mean") & (summ.severity == s)]
            if len(x): r_.update({f"{s}_auc": x.auc.iloc[0], f"{s}_auc_drop": x.auc_drop.iloc[0], f"{s}_f1": x.f1.iloc[0], f"{s}_f1_drop": x.f1_drop.iloc[0]})
        o_ = summ[(summ.model == k) & (summ.row_type == "overall_mean")]
        if len(o_): r_.update(mean_degraded_auc=o_.auc.iloc[0], mean_auc_drop=o_.auc_drop.iloc[0], mean_f1_drop=o_.f1_drop.iloc[0])
        wo = R[(R.model == k) & (R.family == "out")]
        if len(wo): r_["unseen_codec_webp_auc"] = wo.auc.iloc[0]
        rows.append(r_)
    T_ = pd.DataFrame(rows); T_.to_csv(os.path.join(MET_DIR, f"{PFX}_Paper_Summary_Table.csv"), index=False)
    try: T_.round(4).to_latex(os.path.join(MET_DIR, f"{PFX}_Paper_Summary_Table.tex"), index=False, float_format="%.4f")
    except Exception as e: print("  LaTeX export skipped:", repr(e)[:80])

@paper_item
def figB10_per_type():
    TY = [("JPEG", "jpeg"), ("Resolution", "resolution"), ("Blur", "blur"), ("Noise", "noise"), ("Combined", "combined")]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for a, (tn, tk) in zip(axs, TY):
        for k in models:
            ys = [RES[(k, "clean")]["auc"]] + [RES[(k, f"{tk}_{s}")]["auc"] if (k, f"{tk}_{s}") in RES else np.nan for s in SEVS]
            a.plot(range(4), ys, marker="o", color=COLOR[k], label=LABEL[k])
        a.set_xticks(range(4), ["clean"] + SEVS, rotation=20); a.set_title(tn)
    axs[0].set_ylabel("ROC-AUC"); axs[0].legend(fontsize=7); savefig(fig, "FigB10_AUC_vs_Severity_Per_Type")

@paper_item
def figB11_heatmap():
    cl = [c for c in COND if all((k, c) in RES for k in models)]
    M = np.array([[RES[(k, c)]["auc"] for c in cl] for k in models]); vmin = max(0.5, np.nanmin(M))
    fig, axs = plt.subplots(2 if ref else 1, 1, figsize=(max(8, 0.75 * len(cl)), 1.1 * len(models) + (1.4 if ref else 0) + 2.2), squeeze=False)
    a = axs[0, 0]; im = a.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=1.0)
    a.set_xticks(range(len(cl)), cl, rotation=60, ha="right", fontsize=8); a.set_yticks(range(len(models)), [LBL(k) for k in models]); a.set_title("ROC-AUC", fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < (vmin + 1) / 2 else "black")
    fig.colorbar(im, ax=a, fraction=0.02)
    if ref:
        D = np.array([[RES[(PRIMARY, c)]["auc"] - RES[(ref, c)]["auc"] for c in cl]]); v = max(1e-3, np.abs(D).max()); b = axs[1, 0]
        im2 = b.imshow(D, cmap="RdBu", aspect="auto", vmin=-v, vmax=v); b.set_yticks([0], [f"{LBL(PRIMARY)} - {LBL(ref)}"])
        b.set_xticks(range(len(cl)), cl, rotation=60, ha="right", fontsize=8); b.set_title("AUC difference (blue = primary better)", fontsize=9)
        for j in range(D.shape[1]): b.text(j, 0, f"{D[0, j]:+.3f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im2, ax=b, fraction=0.02)
    savefig(fig, "FigB11_AUC_Heatmap_All_Conditions")

@paper_item
def figB12_ci_dots():
    BR = pd.DataFrame(boot_rows); cl2 = [c for c in COND if c in set(BR.condition)]; w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(cl2)), 4))
    for i, k in enumerate(models):
        b = BR[BR.model == k].set_index("condition").loc[cl2]; x = np.arange(len(cl2)) + (i - (len(models) - 1) / 2) * w
        ax.errorbar(x, b.auc, yerr=[b.auc - b.ci95_lo, b.ci95_hi - b.auc], fmt="o", ms=3, capsize=2, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(range(len(cl2)), cl2, rotation=60, ha="right", fontsize=8); ax.set_ylabel("ROC-AUC (95% video-cluster bootstrap CI)"); ax.legend(fontsize=8)
    savefig(fig, "FigB12_AUC_With_Bootstrap_CI")

@paper_item
def figB13_forest():
    DF = pd.DataFrame(diff_rows); DF = DF[DF.vs == ref].reset_index(drop=True); y_ = np.arange(len(DF))
    col = ["#2a9d8f" if s and d > 0 else ("#c0392b" if s else "gray") for s, d in zip(DF.significant_95, DF.auc_diff)]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(DF) + 1.5))
    ax.errorbar(DF.auc_diff, y_, xerr=[DF.auc_diff - DF.ci95_lo, DF.ci95_hi - DF.auc_diff], fmt="none", ecolor="gray", capsize=2)
    ax.scatter(DF.auc_diff, y_, c=col, s=18, zorder=3); ax.axvline(0, c="k", lw=1)
    ax.set_yticks(y_, DF.condition, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel(f"AUC({LBL(PRIMARY)}) - AUC({LBL(ref)}), 95% CI  (>0 favours {LBL(PRIMARY)}; colour = CI excludes 0)", fontsize=8)
    savefig(fig, "FigB13_AUC_Difference_Forest")

@paper_item
def figB14_stability():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4)); w = 0.8 / len(models)
    for a, (m_, nm) in zip(axs, [("mean_abs_prob_shift", "Mean |p_deg - p_clean|"), ("decision_flip_rate", "Decision flip rate")]):
        for i, k in enumerate(models):
            s_ = S[(S.model == k) & (S.family == "in")].groupby("severity")[m_].mean().reindex(SEVS)
            a.bar(np.arange(3) + i * w, s_.values, w, color=COLOR[k], label=LABEL[k])
        a.set_xticks(np.arange(3) + w * (len(models) - 1) / 2, SEVS); a.set_ylabel(nm + " (lower = more stable)"); a.set_xlabel("Severity (mean over in-family types)")
    axs[0].legend(fontsize=8); savefig(fig, "FigB14_Prediction_Stability")

@paper_item
def figB15_reliability():
    rc = [c for c in ["clean", "combined_severe"] if c in conds_done]
    fig, axs = plt.subplots(1, len(rc), figsize=(4.6 * len(rc), 4), squeeze=False)
    for a, c in zip(axs[0], rc):
        for k in models:
            t = PR[(k, c)]; p_, y_ = t.predicted_probability.values, t.ground_truth.values; bn = np.minimum((p_ * 10).astype(int), 9); xs, ys = [], []
            for b in range(10):
                m = bn == b
                if m.sum() >= 20: xs.append(p_[m].mean()); ys.append(y_[m].mean())
            a.plot(xs, ys, marker="o", ms=3, color=COLOR[k], label=f"{LBL(k)} ECE={RES[(k, c)]['ece']:.3f}")
        a.plot([0, 1], [0, 1], "k:"); a.set_title(c, fontsize=10); a.set_xlabel("Mean predicted P(FAKE)"); a.set_ylabel("Observed FAKE fraction"); a.legend(fontsize=7)
    savefig(fig, "FigB15_Reliability_Diagram")

@paper_item
def figB16_training_curves():
    hh = pd.DataFrame(history); fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(hh.epoch, hh.train_acc_clean, label="train (clean view)"); axs[0].plot(hh.epoch, hh.train_acc_degraded, label="train (degraded view)")
    axs[0].plot(hh.epoch, hh.val_acc, label="val (clean)"); axs[0].set_ylabel("Accuracy")
    axs[1].plot(hh.epoch, hh.train_cls_loss, label="train cls loss"); axs[1].plot(hh.epoch, hh.val_loss, label="val loss"); axs[1].set_ylabel("Loss")
    axs[2].plot(hh.epoch, hh.val_f1, label="val F1"); axs[2].plot(hh.epoch, hh.val_ap, label="val PR-AUC"); axs[2].plot(hh.epoch, hh.val_auc, label="val ROC-AUC"); axs[2].set_ylabel("Score")
    for a in axs: a.set_xlabel("Epoch"); a.legend(fontsize=8); a.axvline(best_epoch, ls="--", c="gray", lw=1)
    savefig(fig, "FigB16_Training_Validation_Curves")

@paper_item
def figB17_cross_dataset():
    cd = pd.DataFrame(cross_rows); dsn = list(cd.dataset.unique()); xl = ["FF++ (in-domain)"] + dsn; w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, k in enumerate(models):
        vals = [RES[(k, "clean")]["auc"]] + [cd[(cd.dataset == d) & (cd.model == k)].auc.iloc[0] for d in dsn]
        ax.bar(np.arange(len(xl)) + i * w, vals, w, color=COLOR[k], label=LABEL[k])
    ax.set_xticks(np.arange(len(xl)) + w * (len(models) - 1) / 2, xl); ax.set_ylabel("ROC-AUC (FF++ val thresholds)"); ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, ls=":", c="k", lw=1); ax.legend(fontsize=8); savefig(fig, "FigB17_Cross_Dataset_AUC")

@paper_item
def figB18_score_distributions():
    sc = [c for c in ["clean", "combined_severe"] if c in conds_done]
    fig, axs = plt.subplots(len(models), len(sc), figsize=(4.2 * len(sc), 2.8 * len(models)), squeeze=False, sharex=True)
    for r, k in enumerate(models):
        for q, c in enumerate(sc):
            t = PR[(k, c)]; a = axs[r, q]
            for lab, nm_, col in [(0, "REAL", "#4C72B0"), (1, "FAKE", "#C44E52")]:
                a.hist(t.predicted_probability[t.ground_truth == lab], bins=40, range=(0, 1), alpha=0.6, color=col, label=nm_, density=True)
            a.axvline(THR[k], ls="--", c="k", lw=1); a.set_title(f"{LBL(k)} | {c}", fontsize=9)
            if r == len(models) - 1: a.set_xlabel("Predicted P(FAKE)")
            if q == 0: a.set_ylabel("Density")
    axs[0, 0].legend(fontsize=7); savefig(fig, "FigB18_Score_Distributions")

print("paper-extra CSVs:", sorted(f for f in os.listdir(MET_DIR) if any(s in f for s in ("Bootstrap", "Difference_CI", "Stability", "Paper_Summary", "By_Condition"))))

# ============================ 14. SUMMARY ============================
if os.path.exists(FAIL_TSV):
    fl = pd.read_csv(FAIL_TSV, sep="\t", header=None, names=["image_path", "stage", "error_message"], on_bad_lines="skip")
else: fl = pd.DataFrame(columns=["image_path", "stage", "error_message"])
fl.drop_duplicates().to_csv(os.path.join(LOG_DIR, f"{PFX}_Failed_Images.csv"), index=False)
print("\n" + "=" * 90 + "\nSUMMARY\n" + "=" * 90)
print(f"dataset: {len(df):,} imgs | train {len(tr_df):,} / val {len(va_df):,} / test {len(te_df):,} | train REAL {n_real:,} FAKE {n_fake:,} | group overlap {overlap}")
print(f"device: {GPU_NAME} | params {n_tot:,} (trainable {n_tr:,}) | lambda {LAMBDA} | best epoch {best_epoch} | best val AUC {best_auc:.5f}")
print(f"{PRIMARY} clean test: AUC {RES[(PRIMARY,'clean')]['auc']:.4f} | F1 {RES[(PRIMARY,'clean')]['f1']:.4f} | Acc {RES[(PRIMARY,'clean')]['accuracy']:.4f} (val-selected thr {THR[PRIMARY]:.3f})")
for k in models:
    s = summ[(summ.model == k) & (summ.row_type == "overall_mean")]
    if len(s): print(f"{k:<8} mean in-family degraded AUC {s.auc.iloc[0]:.4f} | mean AUC drop {s.auc_drop.iloc[0]:.4f} | mean F1 drop {s.f1_drop.iloc[0]:.4f}")
if have_cmp and "ModelA" in models:
    print("\nMODEL A vs " + PRIMARY + " (AUC / F1; drop = clean - degraded):")
    for c in COND:
        a_, b_ = RES[("ModelA", c)], RES[(PRIMARY, c)]
        print(f"  {c:<18} A {a_['auc']:.4f}/{a_['f1']:.4f} | {PRIMARY} {b_['auc']:.4f}/{b_['f1']:.4f}" +
              ("" if c == "clean" else f" | AUC drop A {a_['auc_drop'] if 'auc_drop' in a_ else clean_ref['ModelA']['auc']-a_['auc']:.4f} vs {PRIMARY} {clean_ref[PRIMARY]['auc']-b_['auc']:.4f}"))
else: print("\nModel A comparison NOT produced:", skipped_models.get("ModelA", "n/a"), "- set MODEL_PATH_OVERRIDES['ModelA'] or attach Model A outputs as a Kaggle input.")
print(f"failed/unreadable image log rows: {len(fl)} | total runtime {(time.time()-T_START)/3600:.2f} h")
print("best checkpoint:", BEST_PATH, "\noutput dir:", OUT_DIR)
for sd in ("checkpoints", "metrics", "predictions", "figures", "logs"):
    fs = sorted(os.listdir(os.path.join(OUT_DIR, sd)))
    n_ep = len([f for f in fs if "_Epoch" in f and f.endswith(".pth")])
    shown = [f for f in fs if not ("_Epoch" in f and f.endswith(".pth"))]
    if sd == "predictions": shown = f"{len(shown)} prediction CSVs"
    print(f"  {sd}/: {shown}" + (f" + {n_ep} epoch checkpoints" if n_ep else ""))