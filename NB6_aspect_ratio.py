# ===================================================================================================
# NB6  |  DegradeGuard revision experiments  |  ONE Kaggle cell (GPU T4, Internet ON, 4 CPUs, "Save & Run All")
# CONFIG block (only this block differs between notebooks) -> shared COMMON STATS -> shared ENGINE -> main()
# ===================================================================================================
NBTAG = "NB6"
SMOKE_TEST = False                       # True: 300 train imgs, 1 epoch, 300 test imgs, 3 conditions, <5 min validation run
RUN_LIST = [
    {'name': 'A_arpres_s42', 'mode': 'A', 'seed': 42, 'preproc': 'arpres', 'prio': 'P1'},
    {'name': 'B_arpres_s42', 'mode': 'B', 'seed': 42, 'preproc': 'arpres', 'prio': 'P1'},
]
REF_PAIRS = [('B_arpres_s42', 'A_arpres_s42')]              # (model, reference) pairs for paired bootstrap CIs
LAMBDA_PLOT = False
ORIG_CKPT = False                         # evaluate the original full-protocol checkpoints (A_orig / B_orig)
ANALYSIS_MODELS = []
POST_STAGES = []
ANALYSIS_EST_H = 0.0

import os, json
# ---- paths (Kaggle) ----
DATA_ROOT = os.environ.get("DG_DATA_ROOT", "/kaggle/input/datasets/syedazmulhasansabbir/face-forensics-frames/Face Forensics Frames")
CROSS_FOLDERS = json.loads(os.environ["DG_CROSS_FOLDERS"]) if "DG_CROSS_FOLDERS" in os.environ else {
    "CelebDF": "/kaggle/input/datasets/syedazmulhasansabbir/celeb-df-frames/content/drive/MyDrive/dataset/celeb_df_frames",
    "DFDC": "/kaggle/input/datasets/itamargr/dfdc-faces-of-the-train-sample/validation",
    "Diffusion": "/kaggle/input/datasets/syedazmulhasansabbir/diffusion/diffusion",
    "WildDeepfake": "/kaggle/input/datasets/maysuni/wild-deepfake/test"}
CROSS_USE_PKL = False                                   # optional PKL index (default off)
CROSS_PKL = "/kaggle/input/datasets/asmabagum/dataset-scan/dataset_scans_v40.pkl"
CROSS_KEYS = {"celeb": "CelebDF_PKL", "dfdc": "DFDC_PKL", "diffusion": "Diffusion_PKL"}
ORIG_MODEL_DIR = os.environ.get("DG_ORIG_DIR", "/kaggle/input/models/syedazmulhasansabbir/degradation-aware-deepfake-detection")
MODEL_A_CKPT = None; MODEL_B_CKPT = None                # explicit overrides for the original checkpoints (NB5)
PRETRAINED_WEIGHTS_FILE = None                          # local efficientnet_b0 .pth if Internet is OFF
# ---- seeds ----
SPLIT_SEED = 42; EVAL_SEED = 42; SPLIT_RATIOS = (0.80, 0.10, 0.10)
# ---- light protocol (identical optimiser/schedule to the original scripts) ----
STRIDE, EPOCHS, PATIENCE, MIN_DELTA = 2, 10, 3, 1e-4
IMG_SIZE, RESIZE_SIZE = 224, 256
BATCH_SIZE, EVAL_BATCH_SIZE = int(os.environ.get("DG_BATCH", 64)), 128
LR, WEIGHT_DECAY, WARMUP_EPOCHS, GRAD_CLIP = 3e-4, 1e-2, 1, 1.0
IMBALANCE_TRIGGER = 1.5; USE_JPEG_DRAFT = True; NUM_WORKERS = min(4, os.cpu_count() or 2)
P_NONE, P_SINGLE = 0.20, 0.50
TRAIN_RANGES = dict(jpeg_q=(30, 90), down_scale=(0.40, 0.90), blur_sigma=(0.3, 2.0), noise_sigma=(2.0, 12.0), color=(0.75, 1.25))
# ---- time safety ----
HARD_STOP_H, PLAN_BUDGET_H, EVAL_RESERVE_H = 10.8, 8.5, 1.2
EVAL_EST_H, COLD_START_H = 0.9, 0.3
PRIOR_MIN_PER_EPOCH = dict(single=10.0, two=12.5)       # priors; replaced by measured speed after each run
# ---- evaluation ----
VALSUB_N, MIXVAL_N, N_BOOT, CROSS_N_PER_CLASS = 5000, 3000, 1000, 5000
SMOKE_N = int(os.environ.get("DG_SMOKE_N", 300)); SMOKE_CONDS = ["clean", "jpeg_medium", "combined_severe"]
# ---- analysis suite (NB5) ----
LAT_ITERS, LAT_WARM, LAT_REPS, E2E_N, DEGOP_REPS = 200, 20, 5, 200, 100
META_TRAIN_N, META_TEST_N, CHAR_PIX_N, DEG_STAT_N = 3500, 1500, 2000, 150


# ---------- shared statistics helpers (identical in engine and NB7) ----------
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve

def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def ece_score(y, p, bins=15):                       # identical to the scripts
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = ((p > edges[i]) & (p <= edges[i + 1])) if i else ((p >= edges[i]) & (p <= edges[i + 1]))
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)

def compute_metrics(y, p, thr):                     # identical to the scripts
    pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0; spec = tn / (tn + fp) if tn + fp else 0.0
    two = len(np.unique(y)) == 2
    return dict(n=int(len(y)), threshold=float(thr), auc=float(roc_auc_score(y, p)) if two else float("nan"),
                ap=float(average_precision_score(y, p)) if two else float("nan"), accuracy=float((pred == y).mean()),
                precision=float(prec), recall=float(rec), f1=float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                specificity=float(spec), balanced_acc=float((rec + spec) / 2), ece=ece_score(y, p),
                brier=float(np.mean((p - y) ** 2)), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))

def roc_extras(y, p):                               # threshold-free extras
    if len(np.unique(y)) < 2: return dict(tpr_at_fpr1=np.nan, tpr_at_fpr5=np.nan, eer=np.nan, best_f1=np.nan)
    f, t, _ = roc_curve(y, p); fnr = 1 - t; i = int(np.argmin(np.abs(fnr - f)))
    pr, rc, _ = precision_recall_curve(y, p); f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return dict(tpr_at_fpr1=float(np.interp(0.01, f, t)), tpr_at_fpr5=float(np.interp(0.05, f, t)),
                eer=float((f[i] + fnr[i]) / 2), best_f1=float(f1.max()))

def full_metrics(y, p, thr):
    d = compute_metrics(y, p, thr); d.update(roc_extras(y, p)); return d

def youden(y, p):
    f, t, th = roc_curve(y, p); return float(th[np.argmax(t - f)])

# ---- paired video-cluster bootstrap (same cluster draws for every model) ----
def boot_setup(groups, n_boot, seed):
    ug, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    draws = np.random.default_rng(seed).integers(0, len(ug), (n_boot, len(ug)))
    return members, draws

def boot_auc(y, p, members, draws):
    out = np.full(len(draws), np.nan)
    for b, dr in enumerate(draws):
        ii = np.concatenate([members[i] for i in dr])
        if y[ii].min() == y[ii].max(): continue
        out[b] = roc_auc_score(y[ii], p[ii])
    return out

def ci(a):
    a = a[~np.isnan(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)

def ci_and_diff_tables(B, PT, pairs, conds, infam):
    """B[(model,cond)] = bootstrap AUC array; PT[(model,cond)] = point AUC.
    Returns (ci_rows, diff_rows). MEAN_IN_FAMILY = per-resample mean AUC over the in-family conditions present."""
    models = sorted({m for m, _ in B}); ci_rows, diff_rows = [], []
    def get(m, c):
        if c == "MEAN_IN_FAMILY":
            xs = [x for x in infam if (m, x) in B]
            if not xs: return None, np.nan
            return np.nanmean(np.stack([B[(m, x)] for x in xs]), 0), float(np.mean([PT[(m, x)] for x in xs]))
        return (B[(m, c)], PT[(m, c)]) if (m, c) in B else (None, np.nan)
    for m in models:
        for c in list(conds) + ["MEAN_IN_FAMILY"]:
            b, pt = get(m, c)
            if b is None: continue
            lo, hi = ci(b); ci_rows.append(dict(model=m, condition=c, auc=pt, ci95_lo=lo, ci95_hi=hi))
    for m, ref in pairs:
        for c in list(conds) + ["MEAN_IN_FAMILY"]:
            b, pt = get(m, c); br, pr_ = get(ref, c)
            if b is None or br is None: continue
            d = b - br; lo, hi = ci(d)
            diff_rows.append(dict(model=m, ref=ref, condition=c, auc_model=pt, auc_ref=pr_, auc_diff=pt - pr_, ci95_lo=lo, ci95_hi=hi,
                                  ci_excludes_0=bool(lo > 0 or hi < 0), frac_boot_diff_le0=float(np.nanmean(d <= 0))))
    return ci_rows, diff_rows

# =====================================================================
# ENGINE (identical in every notebook; only the CONFIG block above differs)
# Scientific definitions are copied from MODEL_A / MODEL_B scripts: labels REAL=0/FAKE=1,
# metrics, ECE(15 bins), 16 test conditions, CRC32-seeded noise, JPEG-last ordering,
# Youden threshold on CLEAN validation.  cudnn.benchmark => runs are NOT bitwise deterministic.
# =====================================================================
import os, sys, re, io, json, time, math, random, pickle, shutil, hashlib, glob, zlib, platform, warnings, collections, gc, traceback, zipfile, copy
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
T_START = time.time()
def elapsed_h(): return (time.time() - T_START) / 3600

EXPECT_HASH = "7037c5769e42792d40a4682dd6ba41c0"
EXPECT_SIZES, EXPECT_GROUPS, EXPECT_POSW = (208718, 26170, 25905), (399, 52, 49), 0.5151
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REAL_HINTS = ("real", "orig", "orginal", "pristine", "youtube", "authentic", "genuine", "natural")
SPLIT_DIR_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# ---------------- smoke-test overrides ----------------
if SMOKE_TEST:
    EPOCHS, PATIENCE, N_BOOT, VALSUB_N, MIXVAL_N, CROSS_N_PER_CLASS = 1, 1, 50, SMOKE_N, SMOKE_N, 50
    LAT_ITERS, LAT_WARM, LAT_REPS, E2E_N = 3, 1, 2, 10
    META_TRAIN_N, META_TEST_N, CHAR_PIX_N, DEG_STAT_N, DEGOP_REPS = 15, 8, 30, 10, 3

# ---------------- folders / logging ----------------
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("WORK_DIR", "./work")
OUT = f"{WORK}/RESULTS_{NBTAG}"; CK_DIR = f"{WORK}/checkpoints"
MET_DIR, PRED_DIR, FIG_DIR, LOG_DIR = [f"{OUT}/{s}" for s in ("metrics", "predictions", "figures", "logs")]
for d_ in (MET_DIR, PRED_DIR, FIG_DIR, LOG_DIR, CK_DIR): os.makedirs(d_, exist_ok=True)
_LOGF = open(f"{LOG_DIR}/run.log", "a")
def log(*a):
    s = f"[{elapsed_h():5.2f}h] " + " ".join(str(x) for x in a); print(s, flush=True)
    try: _LOGF.write(s + "\n"); _LOGF.flush()
    except Exception: pass
def log_err(stage):
    tb = traceback.format_exc(); log(f"!! ERROR in {stage}: {tb.strip().splitlines()[-1]}")
    with open(f"{LOG_DIR}/errors.log", "a") as f: f.write(f"==== {stage} @ {elapsed_h():.2f}h ====\n{tb}\n")
def safe(stage, fn, *a, **k):
    try: return fn(*a, **k)
    except Exception: log_err(stage); return None
MAN = dict(nbtag=NBTAG, smoke=SMOKE_TEST, runs={}, stages={}, dropped_by_planner=[], warnings=[])
def save_manifest():
    MAN["elapsed_h"] = round(elapsed_h(), 3)
    with open(f"{OUT}/MANIFEST.json", "w") as f: json.dump(MAN, f, indent=2, default=str)
def make_zip():
    zp = f"{WORK}/{NBTAG}_results.zip"
    try:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for r_, _, fs in os.walk(OUT):
                for fn in fs:
                    if not fn.endswith(".pth"): z.write(os.path.join(r_, fn), os.path.relpath(os.path.join(r_, fn), WORK))
        return zp
    except Exception: log_err("make_zip")

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); use_amp = dev.type == "cuda"
torch.backends.cudnn.benchmark = True
GPU_NAME = torch.cuda.get_device_name(0) if use_amp else "cpu"
def _winit(base, w): np.random.seed((base + w) % (2**32)); random.seed(base + w)
log(f"{NBTAG} | torch {torch.__version__} tv {torchvision.__version__} | {GPU_NAME} | AMP {use_amp} | workers {NUM_WORKERS} | smoke {SMOKE_TEST}")

# ---------------- training modes ----------------
MODES = {
 "A":        dict(view="single", aug=None,     sup_deg=False, lam=0.0),
 "A_jpeg":   dict(view="single", aug="jpeg",   sup_deg=False, lam=0.0),
 "A_randaug":dict(view="single", aug="randaug",sup_deg=False, lam=0.0),
 "A_deg1":   dict(view="single", aug="deg",    sup_deg=False, lam=0.0),
 "B":        dict(view="two",    aug="deg",    sup_deg=True,  lam=0.2),
 "B1":       dict(view="two",    aug="deg",    sup_deg=True,  lam=0.0),
 "C":        dict(view="two",    aug="deg",    sup_deg=False, lam=0.2),
}

# ====================== DATA DISCOVERY + SPLIT (from MODEL_A script) ======================
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
RE_PAIR = re.compile(r"^(\d+)_(\d+)_(\d+)$"); RE_REAL_M = re.compile(r"^real_(\d+)\.mp4_(\d+)$")
RE_SINGLE = re.compile(r"^(\d+)_(\d+)$"); RE_GENERIC = re.compile(r"^(.*?)[_\-\.]*(?:frame)?[_\-\.]*\d+$", re.I)
def parse_ids(path):
    stem = os.path.splitext(os.path.basename(path))[0]; m = RE_PAIR.match(stem)
    if m: return "pair", int(m[1]), int(m[2])
    m = RE_REAL_M.match(stem) or RE_SINGLE.match(stem)
    if m: return "single", int(m[1]), None
    g = RE_GENERIC.match(stem); return "other", (g[1] if g else stem), None

def discover_and_split():
    root = DATA_ROOT; assert os.path.isdir(root), f"DATA_ROOT not found: {root}"
    while len(subdirs(root)) == 1 and not has_images_directly(root): root = os.path.join(root, subdirs(root)[0]); log("descending into", root)
    top = subdirs(root); log("dataset root:", root, "| top-level folders:", top)
    presplit = any(n.lower() in SPLIT_DIR_NAMES for n in top); tasks = []
    if presplit:
        for n in top:
            if n.lower() in SPLIT_DIR_NAMES:
                for c in subdirs(os.path.join(root, n)): tasks.append((SPLIT_DIR_NAMES[n.lower()], c, os.path.join(root, n, c)))
    else: tasks = [(None, c, os.path.join(root, c)) for c in top]
    cache = f"{WORK}/file_index_cache.pkl"; key = (root, tuple((s, c) for s, c, _ in tasks)); rows = None
    if os.path.exists(cache):
        try:
            cc = pickle.load(open(cache, "rb"))
            if cc["key"] == key and len(cc["rows"]) > 0: rows = cc["rows"]; log("loaded cached file index:", len(rows))
        except Exception: rows = None
    if rows is None:
        t0 = time.time(); log("scanning files ...")
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(tasks)))) as ex: res = list(ex.map(lambda t: scan_tree(t[2]), tasks))
        rows = [(p, c, s) for (s, c, _), ps in zip(tasks, res) for p in ps]; pickle.dump({"key": key, "rows": rows}, open(cache, "wb"))
        log(f"scan done {time.time()-t0:.0f}s: {len(rows)} files")
    d = pd.DataFrame(rows, columns=["path", "cls", "split"]).sort_values("path").reset_index(drop=True); assert len(d) > 0
    cls2label = {c: (0 if any(h in c.lower() for h in REAL_HINTS) else 1) for c in sorted(d.cls.unique())}
    for c, l in cls2label.items(): log(f"   class folder {c:<18} -> {l} ({'REAL' if l == 0 else 'FAKE'}) n={int((d.cls == c).sum()):,}")
    d["label"] = d.cls.map(cls2label).astype(int); assert d.label.nunique() == 2
    parsed = [parse_ids(p) for p in d.path]; d["ptype"] = [p[0] for p in parsed]; parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for t, a, b in parsed:
        if t == "pair":
            ra, rb = find(a), find(b)
            if ra != rb: parent[max(ra, rb)] = min(ra, rb)
        elif t == "single": find(a)
    d["group"] = [f"vid{find(a):04d}" if t in ("pair", "single") else f"other::{cn}::{a}" for (t, a, b), cn in zip(parsed, d.cls)]
    d["video_key"] = [(f"{a}_{b}" if t == "pair" else (str(a) if t == "single" else f"o::{a}")) for t, a, b in parsed]
    if presplit: d["split"] = d.split
    else:
        gsz = d.groupby("group").size(); gl = sorted(gsz.index); random.Random(SPLIT_SEED).shuffle(gl)
        tot = len(d); tg = {s: r * tot for s, r in zip(("train", "val", "test"), SPLIT_RATIOS)}; cur = {s: 0 for s in tg}; asg = {}
        for g in gl: s = max(tg, key=lambda k: tg[k] - cur[k]); asg[g] = s; cur[s] += int(gsz[g])
        d["split"] = d.group.map(asg)
    return d, root, cls2label

def setup_data():
    global df, tr_df, va_df, te_df, pos_weight, split_hash, cls2label, DATA_ROOT_USED, SPLIT_INFO
    df, DATA_ROOT_USED, cls2label = discover_and_split()
    split_hash = hashlib.md5("\n".join((df.path + "|" + df.split).tolist()).encode()).hexdigest()
    sizes = tuple(int((df.split == s).sum()) for s in ("train", "val", "test")); groups = tuple(df[df.split == s].group.nunique() for s in ("train", "val", "test"))
    log("split sizes", sizes, "groups", groups, "hash", split_hash)
    if not SMOKE_TEST:
        if split_hash == EXPECT_HASH: log("split hash matches the paper:", EXPECT_HASH)
        else:
            msg = f"!!!!!! WARNING: split hash {split_hash} != paper hash {EXPECT_HASH} (probably a different Kaggle mount path)"; log(msg); MAN["warnings"].append(msg)
            assert sizes == EXPECT_SIZES and groups == EXPECT_GROUPS, f"split sizes/groups {sizes}/{groups} differ from paper {EXPECT_SIZES}/{EXPECT_GROUPS}"
    ov = {k: len(set(df[df.split == a].group) & set(df[df.split == b].group)) for k, (a, b) in {"tr&va": ("train", "val"), "tr&te": ("train", "test"), "va&te": ("val", "test")}.items()}
    assert not any(ov.values()), f"group leakage {ov}"
    df[["path", "cls", "label", "group", "split"]].to_csv(f"{MET_DIR}/Split_Assignment.csv", index=False)
    open(f"{MET_DIR}/split_hash.txt", "w").write(f"{split_hash}\npaper_hash={EXPECT_HASH}\nsizes(train,val,test)={sizes}\ngroups={groups}\nsplit_seed={SPLIT_SEED}\n")
    SPLIT_INFO = dict(hash=split_hash, sizes=sizes, groups=groups, root=DATA_ROOT_USED); MAN["split"] = SPLIT_INFO
    df["folder"] = df.cls
    tr_df, va_df, te_df = [df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")]
    n_real, n_fake = int((tr_df.label == 0).sum()), int((tr_df.label == 1).sum()); ratio = max(n_real, n_fake) / max(1, min(n_real, n_fake))
    pos_weight = (n_real / n_fake) if ratio > IMBALANCE_TRIGGER else 1.0
    log(f"pos_weight (FULL train split) = {pos_weight:.4f} (paper {EXPECT_POSW})")
    if SMOKE_TEST:       # tiny subsets, hash already computed on the full split
        va_df = va_df.sample(min(SMOKE_N, len(va_df)), random_state=EVAL_SEED).sort_values("path").reset_index(drop=True)
        te_df = te_df.sample(min(SMOKE_N, len(te_df)), random_state=EVAL_SEED).sort_values("path").reset_index(drop=True)

def tr_run_df():
    if SMOKE_TEST: return tr_df.sample(min(SMOKE_N, len(tr_df)), random_state=EVAL_SEED).sort_values("path").reset_index(drop=True)
    return tr_df.iloc[::STRIDE].reset_index(drop=True)            # every STRIDE-th frame of the sorted file list (deterministic)

def group_stratified_sample(d, n, seed):
    if len(d) <= n: return d.reset_index(drop=True)
    frac = n / len(d); parts = []
    for g, x in d.groupby("group"): parts.append(x.sample(max(1, int(round(len(x) * frac))), random_state=seed))
    return pd.concat(parts).sort_values("path").reset_index(drop=True)

# ====================== TRANSFORMS + DEGRADATIONS (from MODEL_B script) ======================
def make_geo(preproc):
    rs = T.Resize(RESIZE_SIZE) if preproc == "arpres" else T.Resize((RESIZE_SIZE, RESIZE_SIZE))   # arpres: shorter side -> 256 (lateral content cropped)
    return dict(train=T.Compose([rs, T.RandomCrop(IMG_SIZE), T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)]),
                eval=T.Compose([rs, T.CenterCrop(IMG_SIZE)]))
GEO = {"squash": make_geo("squash"), "arpres": make_geo("arpres")}
to_tensor = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
RANDAUG = T.RandAugment(num_ops=2, magnitude=9)

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
        b, c, s = val; img = ImageEnhance.Brightness(img).enhance(b); img = ImageEnhance.Contrast(img).enhance(c)
        return ImageEnhance.Color(img).enhance(s)
    raise ValueError(op)
def degrade_ops(img, ops, rng=None):
    for op, v in ops: img = apply_op(img, op, v, rng)
    return img
OP_ORDER = ["down", "blur", "noise", "color", "jpeg"]
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

COND = collections.OrderedDict(); COND["clean"] = dict(type="clean", sev="none", family="clean", ops=[])
for q, s in [(70, "mild"), (40, "medium"), (20, "severe")]: COND[f"jpeg_{s}"] = dict(type="JPEG", sev=s, family="in", ops=[("jpeg", q)])
for sc, s in [(0.75, "mild"), (0.5, "medium"), (0.25, "severe")]: COND[f"resolution_{s}"] = dict(type="Resolution", sev=s, family="in", ops=[("down", sc)])
for sg, s in [(1.0, "mild"), (2.0, "medium"), (3.0, "severe")]: COND[f"blur_{s}"] = dict(type="Blur", sev=s, family="in", ops=[("blur", sg)])
for sg, s in [(5, "mild"), (10, "medium"), (20, "severe")]: COND[f"noise_{s}"] = dict(type="Noise", sev=s, family="in", ops=[("noise", sg)])
COND["combined_mild"] = dict(type="Combined", sev="mild", family="in", ops=[("blur", 0.8), ("jpeg", 70)])
COND["combined_medium"] = dict(type="Combined", sev="medium", family="in", ops=[("down", 0.5), ("jpeg", 40)])
COND["combined_severe"] = dict(type="Combined", sev="severe", family="in", ops=[("down", 0.4), ("noise", 10), ("jpeg", 20)])
try: apply_op(Image.new("RGB", (32, 32)), "webp", 30); COND["webp_medium"] = dict(type="WebP(unseen codec)", sev="medium", family="out", ops=[("webp", 30)])
except Exception as e: log("WebP encoder unavailable:", repr(e)[:80])
SEVS = ["mild", "medium", "severe"]; INFAM = [c for c in COND if COND[c]["family"] == "in"]
TEST_CONDS = [c for c in COND if (not SMOKE_TEST or c in SMOKE_CONDS)]
INFAM_T = [c for c in INFAM if c in TEST_CONDS]

def load_rgb(p):
    img = Image.open(p)
    if USE_JPEG_DRAFT and img.format == "JPEG": img.draft("RGB", (RESIZE_SIZE, RESIZE_SIZE))
    return img.convert("RGB")

class TrainDS(Dataset):
    def __init__(self, d, mode, geo): self.paths, self.labels, self.spec, self.geo = d.path.tolist(), d.label.values.astype(np.float32), MODES[mode], geo
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for k in range(5):
            j = (i + k) % len(self.paths)
            try:
                base = self.geo(load_rgb(self.paths[j])); sp = self.spec
                if sp["view"] == "two": return to_tensor(base), to_tensor(degrade_ops(base, sample_train_ops())), torch.tensor(self.labels[j]), j
                if sp["aug"] == "jpeg" and random.random() < 0.5: base = apply_op(base, "jpeg", random.randint(30, 90))
                elif sp["aug"] == "randaug": base = RANDAUG(base)
                elif sp["aug"] == "deg": base = degrade_ops(base, sample_train_ops())
                return to_tensor(base), torch.zeros(1), torch.tensor(self.labels[j]), j
            except Exception as e: log_fail(self.paths[j], "train", e)
        raise RuntimeError("5 consecutive unreadable images")
def log_fail(path, stage, err):
    try:
        with open(f"{LOG_DIR}/failed_images.tsv", "a") as f: f.write(f"{path}\t{stage}\t{str(err).replace(chr(9),' ').replace(chr(10),' ')[:200]}\n")
    except Exception: pass

class EvalDS(Dataset):
    """cond = key of COND, or 'mix' (each image gets a fixed in-family condition chosen by CRC32 of its path)."""
    def __init__(self, d, cond, geo, stage="eval"): self.paths, self.labels, self.cond, self.geo, self.stage = d.path.tolist(), d.label.values.astype(np.float32), cond, geo, stage
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for k in range(5):
            j = (i + k) % len(self.paths)
            try:
                p = self.paths[j]; base = self.geo(load_rgb(p)); cn = self.cond
                if cn == "mix": cn = INFAM[zlib.crc32(p.encode()) % len(INFAM)]
                ops = COND[cn]["ops"]
                if ops:
                    rng = np.random.default_rng((zlib.crc32(f"{p}|{cn}".encode()) ^ EVAL_SEED) & 0xFFFFFFFF); base = degrade_ops(base, ops, rng)
                return to_tensor(base), torch.tensor(self.labels[j]), j
            except Exception as e: log_fail(self.paths[j], f"{self.stage}:{self.cond}", e)
        raise RuntimeError("5 consecutive unreadable images")
def eval_loader(d, cond, preproc, stage="eval", bs=None):
    kw = dict(num_workers=NUM_WORKERS, pin_memory=use_amp, worker_init_fn=partial(_winit, EVAL_SEED))
    if NUM_WORKERS > 0: kw.update(prefetch_factor=4)
    return DataLoader(EvalDS(d, cond, GEO[preproc]["eval"], stage), batch_size=bs or EVAL_BATCH_SIZE, shuffle=False, **kw)

# ====================== MODEL ======================
def build_arch(pretrained):
    if pretrained:
        try: m = torchvision.models.efficientnet_b0(weights="DEFAULT"); log("loaded ImageNet-pretrained EfficientNet-B0")
        except Exception as e:
            log("pretrained download failed:", repr(e)[:150])
            if PRETRAINED_WEIGHTS_FILE and os.path.exists(PRETRAINED_WEIGHTS_FILE):
                m = torchvision.models.efficientnet_b0(weights=None); m.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE, map_location="cpu"))
            elif SMOKE_TEST: m = torchvision.models.efficientnet_b0(weights=None); log("!! SMOKE: random init")
            else: raise RuntimeError("Cannot get ImageNet weights: turn Internet ON or set PRETRAINED_WEIGHTS_FILE")
    else: m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1); return m

def to_dev(m): return m.to(dev).to(memory_format=torch.channels_last)

@torch.inference_mode()
def run_models(models, loader):
    """One pass over `loader`; all models see the SAME batch. -> ({name: logits}, labels, idx) de-duplicated by idx."""
    for m in models.values(): m.eval()
    outs = {k: [] for k in models}; Y, I = [], []
    for x, y, idx in loader:
        x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)
        for k, m in models.items():
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp): outs[k].append(m(x).squeeze(1).float().cpu())
        Y.append(y); I.append(idx)
    I = torch.cat(I).numpy(); Y = torch.cat(Y).numpy().astype(int); _, u = np.unique(I, return_index=True)
    return {k: torch.cat(v).numpy()[u] for k, v in outs.items()}, Y[u], I[u]

def val_epoch(model, loader):
    o, y, _ = run_models({"m": model}, loader); lg = o["m"]; p = sigmoid(lg)
    loss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(lg), torch.tensor(y, dtype=torch.float32), pos_weight=torch.tensor([pos_weight])))
    m = compute_metrics(y, p, 0.5)
    return dict(val_loss=loss, val_acc=m["accuracy"], val_f1=m["f1"], val_auc=m["auc"], val_ap=m["ap"])


# ====================== PLANNER ======================
REG = collections.OrderedDict()          # name -> dict(path, preproc, mode, seed, lam, status, orig)
SPEED = {}                                # measured min/epoch per run type (updated after each run)
def run_type(r): return MODES[r["mode"]]["view"]
def est_train_h(r):
    m = SPEED.get(run_type(r), PRIOR_MIN_PER_EPOCH[run_type(r)]); return COLD_START_H + EPOCHS * m / 60
def planner():
    runs = list(RUN_LIST); groups = {r.get("preproc", "squash") for r in runs} | ({"squash"} if ORIG_CKPT else set())
    ev = EVAL_EST_H * max(1, len(groups)); ana = ANALYSIS_EST_H
    def total(rs): return sum(est_train_h(r) for r in rs) + ev + ana
    print("\n" + "=" * 90 + "\nPLANNER (priors; updated with measured speed after each run)\n" + "=" * 90)
    kept = list(runs)
    while kept and total(kept) > PLAN_BUDGET_H:
        d = kept.pop(); MAN["dropped_by_planner"].append(d["name"]); log(f"PLANNER: dropping {d['name']} (est total {total(kept + [d]):.2f}h > budget {PLAN_BUDGET_H}h)")
    cum = 0.0
    for r in runs:
        e = est_train_h(r); st = "KEEP" if r in kept else "DROP"; cum += e if r in kept else 0
        print(f"  {r['name']:<18} mode={r['mode']:<9} seed={r['seed']} prio={r.get('prio','')} est {e:4.2f} h  cum {cum:4.2f} h  [{st}]")
    print(f"  shared evaluation est {ev:.2f} h | analysis est {ana:.2f} h | TOTAL est {total(kept):.2f} h (budget {PLAN_BUDGET_H} h, hard stop {HARD_STOP_H} h)\n", flush=True)
    return kept

def save_atomic(obj, path): tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)

# ====================== TRAINING ======================
def train_run(r):
    name, mode, seed = r["name"], r["mode"], r["seed"]; spec = MODES[mode]; lam = float(r.get("lam", spec["lam"])); preproc = r.get("preproc", "squash")
    best_path, info_path, res_path = f"{CK_DIR}/{name}_best.pth", f"{MET_DIR}/RunInfo_{name}.json", f"{CK_DIR}/{name}_resume.pth"
    if os.path.exists(best_path) and os.path.exists(info_path):
        info = json.load(open(info_path)); log(f"[{name}] already trained in this session -> reusing"); return info
    log(f"[{name}] START mode={mode} seed={seed} lambda={lam} preproc={preproc} two_view={spec['view']=='two'} sup_deg={spec['sup_deg']}")
    seed_all(seed); trd = tr_run_df(); n_two = spec["view"] == "two"
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=use_amp, worker_init_fn=partial(_winit, seed))
    if NUM_WORKERS > 0: kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(TrainDS(trd, mode, GEO[preproc]["train"]), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=g, **kw)
    val_loader = eval_loader(va_df, "clean", preproc, "val"); mix_loader = eval_loader(MIXVAL_DF, "mix", preproc, "mixval")
    spe = len(train_loader); assert spe > 0, "no training batches (BATCH_SIZE > train subset?)"
    model = to_dev(build_arch(True)); n_params = sum(p.numel() for p in model.parameters())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps, warm = EPOCHS * spe, max(1, WARMUP_EPOCHS * spe)
    def lr_lambda(s):
        if s < warm: return (s + 1) / warm
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda); scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cfg = dict(seed=seed, mode=mode, lam=lam, epochs=EPOCHS, stride=STRIDE, split_hash=split_hash, preproc=preproc, smoke=SMOKE_TEST)
    start_ep, best_auc, best_ep, no_imp, hist = 0, -1.0, 0, 0, []
    if os.path.exists(res_path):
        try:
            ck = torch.load(res_path, map_location="cpu", weights_only=False); assert ck["cfg"] == cfg
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
            start_ep, best_auc, best_ep, no_imp, hist = ck["epoch"], ck["best_auc"], ck["best_ep"], ck["no_imp"], ck["hist"]; log(f"[{name}] RESUMED at epoch {start_ep}")
        except Exception as e: log(f"[{name}] resume failed ({repr(e)[:80]}) -> fresh start")
    if use_amp: torch.cuda.reset_peak_memory_stats()
    status, stop_reason, t_run0, ep_times, nonfin_tot = "COMPLETE", "max_epochs", time.time(), [], 0
    for ep in range(start_ep + 1, EPOCHS + 1):
        if ep_times and elapsed_h() + 1.15 * np.mean(ep_times) / 3600 + EVAL_RESERVE_H > HARD_STOP_H:
            status, stop_reason = "TRUNCATED", "time_budget"; log(f"[{name}] TRUNCATED: next epoch would breach the time limit"); break
        model.train(); t0 = time.time(); seen = 0; s_tot = s_cls = s_con = 0.0; c_c = c_d = 0; nonfin = consec = 0
        for it, (xc, xd, y, idx) in enumerate(train_loader, 1):
            bs = y.size(0)
            try:
                xc = xc.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
                x = torch.cat([xc, xd.to(dev, non_blocking=True)], 0) if n_two else xc
                x = x.contiguous(memory_format=torch.channels_last); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp): logits = model(x).squeeze(1)
                lg = logits.float()
                if n_two:
                    lc, ld = lg[:bs], lg[bs:]; pc = torch.sigmoid(lc).detach(); l_con = (pc - torch.sigmoid(ld)).abs().mean()
                    l_cls = 0.5 * (crit(lc, y) + crit(ld, y)) if spec["sup_deg"] else crit(lc, y)
                    loss = l_cls + lam * l_con
                else: lc, ld = lg, None; l_con = torch.zeros((), device=dev); l_cls = crit(lc, y); loss = l_cls
                if not torch.isfinite(loss):
                    nonfin += 1; consec += 1; nonfin_tot += 1; opt.zero_grad(set_to_none=True)
                    if consec > 50: raise RuntimeError("50 consecutive non-finite losses")
                    continue
                consec = 0; scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt); scaler.update(); sched.step()
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); nonfin += 1; log(f"[{name}] CUDA OOM -> batch skipped"); continue
            seen += bs; s_tot += loss.item() * bs; s_cls += l_cls.item() * bs; s_con += l_con.item() * bs
            c_c += (lc.detach() > 0).eq(y > 0.5).sum().item(); c_d += ((ld.detach() > 0).eq(y > 0.5).sum().item() if ld is not None else 0)
            if it % 200 == 0 or it == spe:
                sp = seen / (time.time() - t0); print(f"  [{name}] ep {ep:02d} [{it}/{spe}] loss {s_tot/seen:.4f} cls {s_cls/seen:.4f} con {s_con/seen:.4f} | acc clean {c_c/seen:.4f} | {sp:.0f} src img/s | ETA {(spe*BATCH_SIZE-seen)/sp/60:.1f} min", flush=True)
        assert seen > 0, "no batch trained this epoch"
        vm = val_epoch(model, val_loader)
        try:      # logged only (never used for selection): AUC on the fixed degraded-mix validation subset
            o_, y_, _ = run_models({"m": model}, mix_loader); mix_auc = float(roc_auc_score(y_, o_["m"])) if len(np.unique(y_)) == 2 else float("nan")
        except Exception: mix_auc = float("nan")
        improved = vm["val_auc"] > best_auc + MIN_DELTA
        if improved: best_auc, best_ep, no_imp = vm["val_auc"], ep, 0; save_atomic(model.state_dict(), best_path)
        else: no_imp += 1
        ep_min = (time.time() - t0) / 60; ep_times.append(ep_min * 60)
        hist.append(dict(epoch=ep, train_total_loss=s_tot / max(seen, 1), train_cls_loss=s_cls / max(seen, 1), train_cons_loss=s_con / max(seen, 1), train_acc_clean=c_c / max(seen, 1),
                         train_acc_degraded=(c_d / max(seen, 1)) if n_two else float("nan"), lr=opt.param_groups[0]["lr"], **vm, mixval_auc=mix_auc, nonfinite_batches=nonfin, epoch_minutes=ep_min))
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), epoch=ep, best_auc=best_auc, best_ep=best_ep, no_imp=no_imp, hist=hist, cfg=cfg), res_path)
        pd.DataFrame(hist).to_csv(f"{MET_DIR}/Training_History_{name}.csv", index=False)
        log(f"[{name}] EPOCH {ep}/{EPOCHS} train {s_tot/max(seen,1):.4f} | val AUC {vm['val_auc']:.5f} (best {best_auc:.5f}@{best_ep}) {'*' if improved else f'(no improve {no_imp}/{PATIENCE})'} | mixval AUC {mix_auc:.4f} | {ep_min:.1f} min")
        if no_imp >= PATIENCE: stop_reason = "early_stop"; log(f"[{name}] early stopping"); break
    if not os.path.exists(best_path): save_atomic(model.state_dict(), best_path); best_ep = best_ep or len(hist)
    peak = float(torch.cuda.max_memory_allocated() / 2**20) if use_amp else float("nan")
    train_min = float(sum(h["epoch_minutes"] for h in hist)); wall_min = (time.time() - t_run0) / 60
    if hist: SPEED[run_type(r)] = float(np.mean([h["epoch_minutes"] for h in hist[-max(1, len(hist) // 2):]]))
    info = dict(name=name, mode=mode, seed=seed, lam=lam, preproc=preproc, status=status, stop_reason=stop_reason, best_epoch=best_ep, best_val_auc=best_auc,
                epochs_run=len(hist), train_minutes=train_min, wall_minutes_this_session=wall_min, params=n_params, peak_gpu_mem_mb=peak, n_train_images=len(trd),
                steps_per_epoch=spe, nonfinite_batches=nonfin_tot, history=hist, cfg=cfg)
    json.dump(info, open(info_path, "w"), indent=2, default=str)
    try: os.remove(res_path)
    except Exception: pass
    del train_loader, val_loader, mix_loader, model, opt, sched, scaler; gc.collect()
    if use_amp: torch.cuda.empty_cache()
    log(f"[{name}] DONE status={status} best epoch {best_ep} val AUC {best_auc:.5f} | {train_min:.0f} train-min"); return info

def run_all_training(plan):
    for r in plan:
        name = r["name"]
        try:
            e = est_train_h(r)
            if not SMOKE_TEST and elapsed_h() + e + EVAL_RESERVE_H > HARD_STOP_H:
                MAN["runs"][name] = dict(status="SKIPPED_TIME", est_h=e); log(f"[{name}] SKIPPED_TIME (elapsed {elapsed_h():.2f}h + est {e:.2f}h + reserve {EVAL_RESERVE_H}h > {HARD_STOP_H}h)"); save_manifest(); continue
            info = train_run(r); MAN["runs"][name] = {k: v for k, v in info.items() if k not in ("history",)}
            REG[name] = dict(path=f"{CK_DIR}/{name}_best.pth", preproc=info["preproc"], mode=info["mode"], seed=info["seed"], lam=info["lam"], status=info["status"], orig=False)
        except Exception:
            log_err(f"train:{name}"); MAN["runs"][name] = dict(status="FAILED")
            gc.collect(); (torch.cuda.empty_cache() if use_amp else None)
        save_manifest(); make_zip()

def register_orig():
    """Original full-protocol checkpoints (NB5): locate by recursive glob, identify A/B by FILE name."""
    root = ORIG_MODEL_DIR; cands = sorted(set(glob.glob(f"{root}/**/*.pth", recursive=True)) | set(glob.glob(f"{root}/**/*.pt", recursive=True)))
    print("\nORIGINAL CHECKPOINT DIRECTORY TREE:", root)
    for r_, _, fs in os.walk(root):
        for fn in sorted(fs): print("   ", os.path.relpath(os.path.join(r_, fn), root), f"({os.path.getsize(os.path.join(r_, fn))/2**20:.1f} MB)")
    def pick(kind):
        ov = MODEL_A_CKPT if kind == "A" else MODEL_B_CKPT
        if ov: return ov
        sc = []
        for p in cands:
            n = os.path.basename(p).lower()
            isA = ("modela" in n or "baseline" in n); isB = (("modelb" in n or "degradation" in n) and "modelb1" not in n and "degaug" not in n)
            if (kind == "A" and isA) or (kind == "B" and isB): sc.append((("best" in n) * 2 + ("epoch" not in n), p))
        return max(sc)[1] if sc else None
    for kind, nm in (("A", "A_orig"), ("B", "B_orig")):
        p = pick(kind)
        if not p: log(f"!! {nm}: checkpoint not found under {root} (set MODEL_{kind}_CKPT)"); MAN["warnings"].append(f"{nm} not found"); continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            sd = ck["model"] if isinstance(ck, dict) and "model" in ck else (ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}; m = build_arch(False); m.load_state_dict(sd)
            local = f"{CK_DIR}/{nm}_best.pth"; save_atomic(m.state_dict(), local)
            hist = ck.get("history") if isinstance(ck, dict) else None
            REG[nm] = dict(path=local, preproc="squash", mode=kind, seed=42, lam=(0.2 if kind == "B" else 0.0), status="ORIGINAL", orig=True, src=p,
                           history=hist, epoch=(ck.get("epoch") if isinstance(ck, dict) else None), best_epoch=(ck.get("best_epoch") if isinstance(ck, dict) else None))
            log(f"{nm}: loaded {p} (keys: {'full checkpoint' if isinstance(ck, dict) and 'model' in ck else 'bare state_dict'})"); del ck, m
        except Exception: log_err(f"load_orig:{nm}")


# ====================== PREDICTION STORE + EVALUATION ======================
def pred_path(m, split, cond): return f"{PRED_DIR}/{m}/{split}__{cond}.csv.gz"
def save_pred(m, split, cond, d, idx, y, lg):
    os.makedirs(f"{PRED_DIR}/{m}", exist_ok=True)
    pd.DataFrame(dict(image_path=d.path.values[idx], source_video_id=d.group.values[idx], label=y.astype(int), logit=lg.astype(np.float32),
                      prob=sigmoid(lg), folder=d.folder.values[idx])).to_csv(pred_path(m, split, cond), index=False, compression="gzip")
def load_pred(m, split, cond):
    p = pred_path(m, split, cond)
    try: return pd.read_csv(p) if os.path.exists(p) else None
    except Exception: return None
def eval_set(nets, preproc, d, split, cond, stage):
    need = collections.OrderedDict((k, v) for k, v in nets.items() if not os.path.exists(pred_path(k, split, cond)))
    if need:
        t0 = time.time(); out, y, idx = run_models(need, eval_loader(d, cond, preproc, stage))
        for k in need: save_pred(k, split, cond, d, idx, y, out[k])
        return f"{split}:{cond} n={len(y):,} {time.time()-t0:.0f}s"
    return f"{split}:{cond} cached"
def time_ok(extra=0.35): return elapsed_h() < HARD_STOP_H - extra

EXT = collections.OrderedDict()
def build_ext():
    n_per = CROSS_N_PER_CLASS
    for nm, root_ in CROSS_FOLDERS.items():
        if SMOKE_TEST and len(EXT) >= 2: break
        idxf = f"{MET_DIR}/External_Index_{nm}.csv"
        try:
            if os.path.exists(idxf): d_ = pd.read_csv(idxf)
            else:
                if not os.path.isdir(root_): log(f"  ext {nm}: folder not found -> skipped ({root_})"); continue
                rows_ = []
                for c in subdirs(root_):
                    lab = 0 if any(h in c.lower() for h in REAL_HINTS) else 1; cr_ = os.path.join(root_, c)
                    for p in scan_tree(cr_):
                        dn = os.path.dirname(p)
                        rows_.append((p, lab, f"{c}::" + (os.path.basename(dn) if dn != cr_ else os.path.basename(p).rsplit("_", 2)[0]), os.path.relpath(dn, root_)))
                d_ = pd.DataFrame(rows_, columns=["path", "label", "group", "folder"]).sort_values("path").reset_index(drop=True)
                if d_.label.nunique() < 2: log(f"  ext {nm}: need real+fake folders, found {subdirs(root_)} -> skipped"); continue
                log(f"  ext {nm}: REAL {(d_.label==0).sum():,} FAKE {(d_.label==1).sum():,} folders {sorted(d_.folder.unique())[:8]}")
                d_ = pd.concat([d_[d_.label == lab].sample(min(n_per, int((d_.label == lab).sum())), random_state=EVAL_SEED) for lab in (0, 1)]).reset_index(drop=True)
                d_.to_csv(idxf, index=False)
            d_ = d_[d_.path.map(os.path.exists)].reset_index(drop=True)
            if len(d_) == 0 or d_.label.nunique() < 2: log(f"  ext {nm}: missing files / one class -> skipped"); continue
            EXT[nm] = d_
        except Exception: log_err(f"build_ext:{nm}")
    if CROSS_USE_PKL and os.path.exists(CROSS_PKL):
        try:
            scan = pickle.load(open(CROSS_PKL, "rb"))
            for key, nm in CROSS_KEYS.items():
                d_ = pd.DataFrame(scan[key], columns=["path", "label", "group"])
                d_ = pd.concat([d_[d_.label == lab].sample(min(n_per, int((d_.label == lab).sum())), random_state=EVAL_SEED) for lab in (0, 1)]).reset_index(drop=True)
                d_["folder"] = d_.path.map(lambda p: os.path.basename(os.path.dirname(p))); EXT[nm] = d_
        except Exception: log_err("build_ext:pkl")
    log("external datasets:", {k: len(v) for k, v in EXT.items()})

def shared_eval():
    names = list(REG)
    if not names: log("no models to evaluate"); return
    groups = collections.OrderedDict()
    for n in names: groups.setdefault(REG[n]["preproc"], []).append(n)
    build_ext(); print("\n" + "=" * 90 + f"\nSHARED EVALUATION: {names} | preproc groups {dict(groups)}\n" + "=" * 90, flush=True)
    for preproc, ns in groups.items():
        nets = collections.OrderedDict()
        for n in ns:
            m = build_arch(False); m.load_state_dict(torch.load(REG[n]["path"], map_location="cpu")); nets[n] = to_dev(m).eval()
        steps = [("val", va_df, "clean", "val")]
        steps += [("test", te_df, "clean", "test")]
        steps += [(f"ext_{nm}", d_, "clean", f"cross:{nm}") for nm, d_ in EXT.items()]
        steps += [("test", te_df, c, "test") for c in TEST_CONDS if c != "clean"]
        steps += [("valsub", VALSUB_DF, c, "valsub") for c in TEST_CONDS]
        for i, (split, d, cond, stage) in enumerate(steps, 1):
            if not time_ok(): log(f"time budget nearly exhausted -> skipping remaining evaluation steps from {split}:{cond}"); MAN["warnings"].append(f"eval truncated at {split}:{cond}"); break
            try: log(f"  eval [{preproc}] {i}/{len(steps)} " + eval_set(nets, preproc, d, split, cond, stage), "|", ", ".join(ns))
            except Exception: log_err(f"eval:{split}:{cond}")
        del nets; gc.collect(); (torch.cuda.empty_cache() if use_amp else None)
    make_zip()

# ====================== TABLES ======================
DROP_METRICS = ["auc", "f1", "accuracy", "precision", "recall", "balanced_acc", "ap"]
def A_(t, PID): return t.image_path.map(PID).values, t.label.values.astype(int), t.prob.values, t.logit.values
def get_thr(n):
    t = load_pred(n, "val", "clean")
    return youden(t.label.values.astype(int), t.prob.values) if t is not None and t.label.nunique() == 2 else float("nan")

def build_tables():
    global THR, R, SUMM, PA
    names = [n for n in REG if os.path.exists(pred_path(n, "test", "clean")) and os.path.exists(pred_path(n, "val", "clean"))]
    THR = {n: get_thr(n) for n in names}; pd.DataFrame([dict(model=n, thr_youden_clean_val=THR[n]) for n in names]).to_csv(f"{MET_DIR}/Thresholds_CleanVal_Youden.csv", index=False)
    log("clean-val Youden thresholds:", {k: round(v, 4) for k, v in THR.items()})
    PID = {p: i for i, p in enumerate(te_df.path)}; PA = {}; rows = []
    for n in names:
        for c in TEST_CONDS:
            t = load_pred(n, "test", c)
            if t is None: continue
            PA[(n, c)] = A_(t, PID); _, y, p, _ = PA[(n, c)]
            rows.append(dict(model=n, condition=c, type=COND[c]["type"], severity=COND[c]["sev"], family=COND[c]["family"], **full_metrics(y, p, THR[n])))
    R = pd.DataFrame(rows)
    for mk in DROP_METRICS: R[f"{mk}_drop"] = R.model.map(R[R.condition == "clean"].set_index("model")[mk]) - R[mk]
    R["auc_retention"] = R.auc / R.model.map(R[R.condition == "clean"].set_index("model")["auc"]); R["f1_retention"] = R.f1 / R.model.map(R[R.condition == "clean"].set_index("model")["f1"])
    R.to_csv(f"{MET_DIR}/Robustness_Metrics.csv", index=False)
    # clean-test metrics: fixed 0.5 and val-selected
    cl = []
    for n in names:
        _, y, p, _ = PA[(n, "clean")]
        for rule, th in (("fixed_0.5", 0.5), ("clean_val_youden", THR[n])): cl.append(dict(model=n, rule=rule, **full_metrics(y, p, th)))
    pd.DataFrame(cl).to_csv(f"{MET_DIR}/Clean_Test_Metrics.csv", index=False)
    # validation-subset metrics (all conditions, clean-val threshold) -> used by the recalibration analysis
    vs = []
    for n in names:
        for c in TEST_CONDS:
            t = load_pred(n, "valsub", c)
            if t is not None and t.label.nunique() == 2: vs.append(dict(model=n, condition=c, **full_metrics(t.label.values.astype(int), t.prob.values, THR[n])))
    pd.DataFrame(vs).to_csv(f"{MET_DIR}/ValSub_Metrics.csv", index=False)
    # prediction stability
    st = []
    for n in names:
        if (n, "clean") not in PA: continue
        i0, _, p0, _ = PA[(n, "clean")]
        for c in TEST_CONDS:
            if c == "clean" or (n, c) not in PA: continue
            i1, _, p1, _ = PA[(n, c)]; com = np.intersect1d(i0, i1); a = p0[np.searchsorted(i0, com)]; b = p1[np.searchsorted(i1, com)]
            st.append(dict(model=n, condition=c, type=COND[c]["type"], severity=COND[c]["sev"], family=COND[c]["family"], n=len(com), mean_abs_prob_shift=float(np.abs(a - b).mean()),
                           mean_signed_shift=float((b - a).mean()), decision_flip_rate=float(((a >= THR[n]) != (b >= THR[n])).mean())))
    pd.DataFrame(st).to_csv(f"{MET_DIR}/Prediction_Stability.csv", index=False)
    # per-folder AUC vs REAL (FF++ test folders, all conditions; external sub-folders, clean)
    pf = []
    def folder_aucs(t, n, split, cond):
        real = t[t.label == 0]
        if len(real) == 0: return
        for f, g in t[t.label == 1].groupby("folder"):
            z = pd.concat([real, g]); pf.append(dict(model=n, set=split, condition=cond, folder=f, n_fake=len(g), n_real=len(real), auc=float(roc_auc_score(z.label, z.prob))))
    for n in names:
        for c in TEST_CONDS:
            t = load_pred(n, "test", c)
            if t is not None: folder_aucs(t, n, "FF++ test", c)
    # cross-dataset
    cr = []
    for n in names:
        for nm in EXT:
            t = load_pred(n, f"ext_{nm}", "clean")
            if t is None or t.label.nunique() < 2: continue
            cr.append(dict(dataset=nm, model=n, **full_metrics(t.label.values.astype(int), t.prob.values, THR[n]))); folder_aucs(t, n, nm, "clean")
    pd.DataFrame(pf).to_csv(f"{MET_DIR}/PerFolder_AUC.csv", index=False)
    CR = pd.DataFrame(cr); CR.to_csv(f"{MET_DIR}/CrossDataset_Metrics.csv", index=False)
    # summary
    rows = []
    for n in names:
        g = R[R.model == n]; inf = g[g.family == "in"]; c0 = g[g.condition == "clean"].iloc[0]; info = MAN["runs"].get(n, {})
        if not info and os.path.exists(f"{MET_DIR}/RunInfo_{n}.json"): info = json.load(open(f"{MET_DIR}/RunInfo_{n}.json"))
        hist = REG[n].get("history") if REG[n].get("orig") else None
        tm = info.get("train_minutes", (sum(h.get("epoch_minutes", 0) for h in hist) if hist else float("nan")))
        r_ = dict(model=n, mode=REG[n]["mode"], seed=REG[n]["seed"], lam=REG[n]["lam"], preproc=REG[n]["preproc"], status=REG[n]["status"], clean_auc=c0.auc, clean_f1=c0.f1, clean_acc=c0.accuracy,
                  mean_degraded_auc=inf.auc.mean(), mean_auc_drop=inf.auc_drop.mean(), mean_f1_drop=inf.f1_drop.mean(), mean_tpr_at_fpr1_degraded=inf.tpr_at_fpr1.mean(), mean_tpr_at_fpr1_clean=c0.tpr_at_fpr1,
                  webp_auc=(g[g.family == "out"].auc.iloc[0] if (g.family == "out").any() else np.nan), threshold=THR[n],
                  best_epoch=info.get("best_epoch", REG[n].get("best_epoch")), train_minutes=tm, params=sum(p.numel() for p in build_arch(False).parameters()) if n == names[0] else None)
        for nm in EXT:
            x = CR[(CR.model == n) & (CR.dataset == nm)]; r_[f"cross_auc_{nm}"] = x.auc.iloc[0] if len(x) else np.nan
        rows.append(r_)
    SUMM = pd.DataFrame(rows); SUMM["params"] = SUMM.params.ffill().bfill()
    SUMM.to_csv(f"{MET_DIR}/Summary_AllModels.csv", index=False); print(SUMM.round(4).to_string(index=False), flush=True)
    return names

def bootstrap_stage(names):
    keys = [(n, c) for n in names for c in TEST_CONDS if (n, c) in PA]
    if not keys: return
    common = None
    for k in keys: common = PA[k][0] if common is None else np.intersect1d(common, PA[k][0])
    log(f"bootstrap: {len(common):,} aligned test images, {N_BOOT} resamples (cluster-level, seed {EVAL_SEED})")
    grp = te_df.group.values[common]; members, draws = boot_setup(grp, N_BOOT, EVAL_SEED); B, PT = {}, {}
    y = PA[keys[0]][1][np.searchsorted(PA[keys[0]][0], common)]
    for c in TEST_CONDS:
        for n in names:
            if (n, c) not in PA: continue
            if not time_ok(0.3): log("time low -> bootstrap truncated"); break
            i_, _, p_, _ = PA[(n, c)]; p = p_[np.searchsorted(i_, common)]; PT[(n, c)] = float(roc_auc_score(y, p)); B[(n, c)] = boot_auc(y, p, members, draws)
        log(f"  bootstrap done: {c}")
    pairs = [(m, r) for m, r in REF_PAIRS if m in names and r in names]
    ci_rows, diff_rows = ci_and_diff_tables(B, PT, pairs, TEST_CONDS, INFAM_T)
    pd.DataFrame(ci_rows).to_csv(f"{MET_DIR}/AUC_Bootstrap_CI.csv", index=False); pd.DataFrame(diff_rows).to_csv(f"{MET_DIR}/AUC_Paired_Diff.csv", index=False)
    global DIFF_ROWS; DIFF_ROWS = pd.DataFrame(diff_rows)
    if len(DIFF_ROWS): print(DIFF_ROWS[DIFF_ROWS.condition.isin(["clean", "MEAN_IN_FAMILY"])].round(4).to_string(index=False), flush=True)

# ====================== FIGURES ======================
def savefig(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{FIG_DIR}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
PALETTE = plt.cm.tab10.colors
def col(n, names): return PALETTE[list(names).index(n) % 10]
def figures(names):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def heat():
        P = R.pivot(index="model", columns="condition", values="auc").reindex(index=names, columns=[c for c in TEST_CONDS]); M = P.values; vmin = max(0.4, np.nanmin(M))
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * M.shape[1]), 0.6 * len(names) + 2.5)); im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=vmin, vmax=1.0)
        ax.set_xticks(range(M.shape[1]), P.columns, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(len(names)), names, fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] < (vmin + 1) / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.02); ax.set_title("Test ROC-AUC (clean-val thresholds irrelevant for AUC)"); savefig(fig, "Fig_AUC_Heatmap_Models_x_Conditions")
    safe("fig:heat", heat)
    def sev():
        TY = [("JPEG", "jpeg"), ("Resolution", "resolution"), ("Blur", "blur"), ("Noise", "noise"), ("Combined", "combined")]
        fig, axs = plt.subplots(1, 5, figsize=(17, 3.5), sharey=True)
        for a, (tn, tk) in zip(axs, TY):
            for n in names:
                ys = [R[(R.model == n) & (R.condition == c)].auc.iloc[0] if ((R.model == n) & (R.condition == c)).any() else np.nan for c in ["clean"] + [f"{tk}_{s}" for s in SEVS]]
                a.plot(range(4), ys, marker="o", ms=3, color=col(n, names), label=n)
            a.set_xticks(range(4), ["clean"] + SEVS, rotation=20); a.set_title(tn)
        axs[0].set_ylabel("ROC-AUC"); axs[0].legend(fontsize=6); savefig(fig, "Fig_AUC_vs_Severity_PerFamily")
    safe("fig:sev", sev)
    def forest():
        if not len(DIFF_ROWS): return
        prs = DIFF_ROWS[["model", "ref"]].drop_duplicates().values.tolist()[:6]
        fig, axs = plt.subplots(1, len(prs), figsize=(4.8 * len(prs), 0.3 * len(TEST_CONDS) + 2.5), squeeze=False)
        for a, (m, rf) in zip(axs[0], prs):
            D = DIFF_ROWS[(DIFF_ROWS.model == m) & (DIFF_ROWS.ref == rf)].reset_index(drop=True); y_ = np.arange(len(D))
            cc = ["#2a9d8f" if (s and d > 0) else ("#c0392b" if s else "gray") for s, d in zip(D.ci_excludes_0, D.auc_diff)]
            a.errorbar(D.auc_diff, y_, xerr=[D.auc_diff - D.ci95_lo, D.ci95_hi - D.auc_diff], fmt="none", ecolor="gray", capsize=2); a.scatter(D.auc_diff, y_, c=cc, s=16, zorder=3)
            a.axvline(0, c="k", lw=1); a.set_yticks(y_, D.condition, fontsize=7); a.invert_yaxis(); a.set_title(f"{m} - {rf}", fontsize=9); a.set_xlabel("paired AUC difference (95% cluster-bootstrap CI)", fontsize=8)
        savefig(fig, "Fig_Paired_AUC_Difference_Forest")
    safe("fig:forest", forest)
    def cross():
        cc_ = [c for c in SUMM.columns if c.startswith("cross_auc_")]
        if not cc_: return
        labs = ["FF++ clean"] + [c.replace("cross_auc_", "") for c in cc_]; w = 0.8 / len(names); fig, ax = plt.subplots(figsize=(7.5, 4))
        for i, n in enumerate(names):
            r_ = SUMM[SUMM.model == n].iloc[0]; ax.bar(np.arange(len(labs)) + i * w, [r_.clean_auc] + [r_[c] for c in cc_], w, color=col(n, names), label=n)
        ax.set_xticks(np.arange(len(labs)) + w * (len(names) - 1) / 2, labs, rotation=15); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, ls=":", c="k", lw=1); ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=6)
        savefig(fig, "Fig_Cross_Dataset_AUC")
    safe("fig:cross", cross)
    if LAMBDA_PLOT: safe("fig:lambda", lambda_sweep)

def lambda_sweep():
    rows = [SUMM.assign(src=NBTAG)]
    for p in glob.glob(f"{WORK}/RESULTS_NB1/metrics/Summary_AllModels.csv") + glob.glob("/kaggle/input/**/RESULTS_NB1/metrics/Summary_AllModels.csv", recursive=True):
        try: rows.append(pd.read_csv(p).assign(src="NB1")); break
        except Exception: pass
    S = pd.concat(rows, ignore_index=True); S = S[S["mode"].isin(["B", "B1"])].drop_duplicates("model").sort_values("lam")
    S.to_csv(f"{MET_DIR}/Lambda_Table_partial.csv", index=False)
    cc_ = [c for c in S.columns if c.startswith("cross_auc_")]; S["mean_cross_auc"] = S[cc_].mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4)); xs = np.arange(len(S))
    for k, lab in (("clean_auc", "clean AUC"), ("mean_degraded_auc", "mean degraded AUC"), ("webp_auc", "WebP AUC"), ("mean_cross_auc", "mean cross-dataset AUC")): ax.plot(xs, S[k], marker="o", label=lab)
    ax.set_xticks(xs, [f"{l:g}" for l in S.lam]); ax.set_xlabel("consistency weight lambda"); ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=8); savefig(fig, "Fig_Lambda_Sweep")

# ====================== README / MAIN ======================
def write_readme():
    txt = f"""RESULTS_{NBTAG}  (DegradeGuard revision experiments)
Generated by the {NBTAG} notebook; elapsed {elapsed_h():.2f} h; smoke={SMOKE_TEST}.
PROTOCOL: train split subsampled every {STRIDE}-th frame of the sorted file list, {EPOCHS} epochs (cosine planned for {EPOCHS}), patience {PATIENCE}, batch {BATCH_SIZE} source images,
AdamW lr {LR} wd {WEIGHT_DECAY}, AMP, grad-clip {GRAD_CLIP}, checkpoint = best CLEAN-val AUC. pos_weight from the FULL train split.
SEEDS: SPLIT_SEED={SPLIT_SEED} (paper split, hash asserted/logged in metrics/split_hash.txt); TRAIN_SEED per run in MANIFEST.json; EVAL_SEED={EVAL_SEED} (CRC32 noise, external sampling, bootstrap).
NOTE: cudnn.benchmark=True and AMP make training non-bitwise-deterministic; identical seeds reproduce results only approximately.
THRESHOLDS: Youden's J on CLEAN validation, reused for every condition/dataset. No test-set tuning.
FILES: metrics/Robustness_Metrics.csv (per model x condition), Summary_AllModels.csv, CrossDataset_Metrics.csv, PerFolder_AUC.csv, Prediction_Stability.csv,
AUC_Bootstrap_CI.csv + AUC_Paired_Diff.csv (1000 paired cluster-bootstrap resamples; MEAN_IN_FAMILY = mean AUC over the 15 in-family conditions), ValSub_Metrics.csv,
Thresholds_CleanVal_Youden.csv, Split_Assignment.csv, RunInfo_*.json, Training_History_*.csv; predictions/<model>/<split>__<condition>.csv.gz (image_path, source_video_id, label, logit, prob, folder);
figures/*.png|pdf; logs/run.log, logs/errors.log; MANIFEST.json (statuses: COMPLETE / TRUNCATED / SKIPPED_TIME / FAILED / DROPPED_BY_PLANNER).
"""
    open(f"{OUT}/README.txt", "w").write(txt)

def print_tree():
    zp = f"{WORK}/{NBTAG}_results.zip"; print("\nZIP:", zp, f"{os.path.getsize(zp)/2**20:.1f} MB" if os.path.exists(zp) else "(missing)")
    for r_, _, fs in os.walk(OUT):
        lvl = os.path.relpath(r_, OUT).count(os.sep) if r_ != OUT else 0
        if lvl <= 1 or "predictions" not in r_: print("  " * lvl + os.path.basename(r_) + "/", f"({len(fs)} files)")

def main():
    global MIXVAL_DF, VALSUB_DF, DIFF_ROWS
    DIFF_ROWS = pd.DataFrame(); MAN["config"] = dict(RUN_LIST=RUN_LIST, EPOCHS=EPOCHS, STRIDE=STRIDE, PATIENCE=PATIENCE, N_BOOT=N_BOOT, HARD_STOP_H=HARD_STOP_H, REF_PAIRS=REF_PAIRS)
    setup_data(); MIXVAL_DF = va_df.sample(min(MIXVAL_N, len(va_df)), random_state=EVAL_SEED).sort_values("path").reset_index(drop=True); VALSUB_DF = group_stratified_sample(va_df, VALSUB_N, EVAL_SEED)
    log(f"val subsets: degraded-mix {len(MIXVAL_DF)} | group-stratified valsub {len(VALSUB_DF)} ({VALSUB_DF.group.nunique()} groups)"); save_manifest()
    plan = planner()
    for r in RUN_LIST:
        if r not in plan: MAN["runs"][r["name"]] = dict(status="DROPPED_BY_PLANNER")
    save_manifest(); run_all_training(plan)
    if ORIG_CKPT: safe("register_orig", register_orig)
    safe("shared_eval", shared_eval); save_manifest()
    names = safe("tables", build_tables) or []
    if names:
        safe("bootstrap", bootstrap_stage, names); safe("figures", figures, names)
    write_readme(); save_manifest(); make_zip()
    for st in POST_STAGES:
        log(f"POST STAGE {st} ..."); t0 = time.time(); safe(f"post:{st}", globals()[st]); MAN["stages"][st] = f"{(time.time()-t0)/60:.1f} min"; save_manifest(); make_zip()
    write_readme(); save_manifest(); make_zip(); print_tree(); log("ALL DONE")


# ============================ RUN ============================
main()
