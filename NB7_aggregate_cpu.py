# ===================================================================================================
# NB7 | CPU-only aggregation | attach the RESULTS_NB* outputs of NB1..NB6 as Kaggle inputs (or run after copying them to /kaggle/working)
# ===================================================================================================
NBTAG = "NB7"; NB7_BOOT = 1000
import os
RESULTS_ROOT = os.environ.get("DG_RESULTS_ROOT", "/kaggle/input/datasets/syedazmulhasansabbir/results-from-nb-degradeguard")   # contains NB1_results/RESULTS_NB1 ... NB6_results/RESULTS_NB6
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
# NB7 - CPU aggregation of the RESULTS_* folders produced by NB1..NB6 (no GPU, no training)
# =====================================================================
import os, sys, re, json, glob, time, zipfile, shutil, traceback, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
T_START = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.environ.get("WORK_DIR", "./work")
OUT = f"{WORK}/RESULTS_NB7"; TAB = f"{OUT}/tables"; FIGD = f"{OUT}/figures"; TEX = f"{OUT}/latex"; LOGD = f"{OUT}/logs"
for d_ in (TAB, FIGD, TEX, LOGD): os.makedirs(d_, exist_ok=True)
def log(*a):
    s = f"[{(time.time()-T_START)/60:6.1f}min] " + " ".join(str(x) for x in a); print(s, flush=True); open(f"{LOGD}/run.log", "a").write(s + "\n")
def safe(stage, fn, *a, **k):
    try: return fn(*a, **k)
    except Exception:
        tb = traceback.format_exc(); log("!! ERROR", stage, tb.strip().splitlines()[-1]); open(f"{LOGD}/errors.log", "a").write(f"==== {stage}\n{tb}\n")

# ---------- locate inputs ----------
RES = {}
for tag in ("NB1", "NB2", "NB3", "NB4", "NB5", "NB6"):
    for p in (f"{RESULTS_ROOT}/{tag}_results/RESULTS_{tag}", f"{RESULTS_ROOT}/RESULTS_{tag}", f"{WORK}/RESULTS_{tag}"):
        if os.path.isdir(f"{p}/metrics"): RES[tag] = p; break
for pat in (f"{WORK}/RESULTS_NB*", "/kaggle/input/**/RESULTS_NB*"):     # fallback: any other attached location
    for p in sorted(glob.glob(pat, recursive=True)):
        tag = os.path.basename(p).replace("RESULTS_", "")
        if os.path.isdir(p) and re.match(r"^NB[1-6]$", tag) and tag not in RES and os.path.isdir(f"{p}/metrics"): RES[tag] = p
for tag in ("NB1", "NB2", "NB3", "NB4", "NB5", "NB6"):
    if tag not in RES: log(f"WARNING: RESULTS_{tag} not found under {RESULTS_ROOT} (tables that need it will be skipped)")
log("found result folders:", RES)
def rd(tag, rel):
    p = f"{RES[tag]}/{rel}" if tag in RES else None
    return pd.read_csv(p) if p and os.path.exists(p) else None
def cat(rel):
    parts = [rd(t, rel).assign(nb=t) for t in RES if rd(t, rel) is not None]
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=[c for c in ("model", "condition", "dataset", "folder", "set") if c in parts[0].columns] if parts else None) if parts else pd.DataFrame()
ROB = cat("metrics/Robustness_Metrics.csv"); SUM = cat("metrics/Summary_AllModels.csv"); CRS = cat("metrics/CrossDataset_Metrics.csv")
log("models available:", sorted(SUM.model.unique()) if len(SUM) else "NONE")
INFAM = [c for c in ROB.condition.unique() if (ROB[ROB.condition == c].family.iloc[0] == "in")] if len(ROB) else []
CONDS = list(ROB.condition.unique()) if len(ROB) else []

def booktabs(df, caption, label, floatfmt="{:.4f}"):
    cols = list(df.columns); esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    def f(v):
        if isinstance(v, (float, np.floating)): return "--" if np.isnan(v) else floatfmt.format(v)
        return esc(v)
    L = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
         " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"] + [" & ".join(f(v) for v in r) + r" \\" for r in df.itertuples(index=False)] + [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L)
TABLES = {}
def emit(name, df, caption, fmt="{:.4f}"):
    if df is None or len(df) == 0: log(f"  table {name}: no data -> skipped"); return
    df.to_csv(f"{TAB}/{name}.csv", index=False); open(f"{TEX}/{name}.tex", "w").write(booktabs(df, caption, f"tab:{name}", fmt)); TABLES[name] = df; log(f"  table {name}: {df.shape}")

# ---------- 1. multi-seed ----------
def seeds_table():
    A = [f"A_s{s}" for s in (42, 43, 44)]; B = [f"B_s{s}" for s in (42, 43, 44)]
    def val(m, c, k="auc"):
        x = ROB[(ROB.model == m) & (ROB.condition == c)]; return float(x[k].iloc[0]) if len(x) else np.nan
    def sv(m, k):
        x = SUM[SUM.model == m]; return float(x[k].iloc[0]) if len(x) and k in x else np.nan
    cross = [c for c in SUM.columns if c.startswith("cross_auc_")]; rows = []
    metrics = [("clean_auc", "Clean test AUC"), ("mean_degraded_auc", "Mean in-family degraded AUC"), ("mean_f1_drop", "Mean F1 drop"), ("webp_auc", "WebP (unseen codec) AUC")] + [(c, c.replace("cross_auc_", "Cross AUC ")) for c in cross]
    for k, nm in metrics:
        r = dict(metric=nm)
        for tag, ms in (("A", A), ("B", B)):
            v = [sv(m, k) for m in ms]
            for s, x in zip((42, 43, 44), v): r[f"{tag}_s{s}"] = x
            vv = np.array(v, float); r[f"{tag}_mean"] = np.nanmean(vv) if np.isfinite(vv).any() else np.nan; r[f"{tag}_std"] = np.nanstd(vv, ddof=1) if np.isfinite(vv).sum() > 1 else np.nan
        va, vb = np.array([sv(m, k) for m in A], float), np.array([sv(m, k) for m in B], float); ok = np.isfinite(va) & np.isfinite(vb)
        if k == "mean_f1_drop": r["B_better_in_k_seeds"] = f"{int((vb[ok] < va[ok]).sum())} of {int(ok.sum())}"
        else: r["B_better_in_k_seeds"] = f"{int((vb[ok] > va[ok]).sum())} of {int(ok.sum())}"
        rows.append(r)
    emit("T1_multiseed_summary", pd.DataFrame(rows), "Light protocol, three training seeds (A: baseline, B: DegradeGuard). Mean and sample std over seeds.")
    rows = []
    for c in CONDS:
        va = np.array([val(m, c) for m in A]); vb = np.array([val(m, c) for m in B]); ok = np.isfinite(va) & np.isfinite(vb)
        rows.append(dict(condition=c, **{f"A_s{s}": x for s, x in zip((42, 43, 44), va)}, **{f"B_s{s}": x for s, x in zip((42, 43, 44), vb)}, A_mean=np.nanmean(va) if ok.any() else np.nan, B_mean=np.nanmean(vb) if ok.any() else np.nan,
                         B_gt_A_seeds=f"{int((vb[ok] > va[ok]).sum())} of {int(ok.sum())}"))
    emit("T1b_multiseed_per_condition_auc", pd.DataFrame(rows), "Per-condition test AUC at seeds 42/43/44 and the number of seeds in which B beats A.")

# ---------- 2. ablation with paired CIs (recomputed from raw predictions; all models share the same test images) ----------
def pred_arr(model, cond):
    for tag, root in RES.items():
        p = f"{root}/predictions/{model}/test__{cond}.csv.gz"
        if os.path.exists(p):
            t = pd.read_csv(p, usecols=["image_path", "source_video_id", "label", "prob"]).sort_values("image_path"); return t
    return None
ABL = ["A_s42", "A_jpeg_s42", "A_randaug_s42", "A_deg1_s42", "C_s42", "B1_s42", "B_s42"]
def bootstrap_models(models, refs_pairs, tag, conds=None, n_boot=1000):
    conds = conds or CONDS; data = {}; common = None
    for m in models:
        for c in conds:
            t = pred_arr(m, c)
            if t is not None: data[(m, c)] = t; common = set(t.image_path) if common is None else common & set(t.image_path)
    if not data: return None, None
    common = sorted(common); log(f"[{tag}] bootstrap on {len(common):,} aligned images, {len(data)} (model,cond) arrays, {n_boot} resamples")
    first = next(iter(data.values())).set_index("image_path").loc[common]; y = first.label.values.astype(int); members, draws = boot_setup(first.source_video_id.values, n_boot, 42); B, PT = {}, {}
    for (m, c), t in data.items():
        p = t.set_index("image_path").loc[common].prob.values; PT[(m, c)] = float(roc_auc_score(y, p)); B[(m, c)] = boot_auc(y, p, members, draws)
    return ci_and_diff_tables(B, PT, [(a, b) for a, b in refs_pairs if any(k[0] == a for k in PT) and any(k[0] == b for k in PT)], conds, INFAM)
def ablation_table():
    ms = [m for m in ABL if len(SUM) and m in set(SUM.model)]
    if len(ms) < 2: log("ablation: fewer than 2 models available"); return
    pairs = [(m, "A_s42") for m in ms if m != "A_s42"] + [(m, "B_s42") for m in ms if m not in ("B_s42",)]
    cir, dfr = bootstrap_models(ms, pairs, "ablation", n_boot=NB7_BOOT)
    if cir is None: return
    pd.DataFrame(dfr).to_csv(f"{TAB}/ablation_all_paired_differences.csv", index=False); D = pd.DataFrame(dfr)
    rows = []
    for m in ms:
        s = SUM[SUM.model == m].iloc[0]; r = dict(model=m, clean_auc=s.clean_auc, mean_degraded_auc=s.mean_degraded_auc, mean_f1_drop=s.mean_f1_drop, webp_auc=s.webp_auc)
        for ref, tg in (("A_s42", "vsA"), ("B_s42", "vsB")):
            x = D[(D.model == m) & (D.ref == ref) & (D.condition == "MEAN_IN_FAMILY")]; xc = D[(D.model == m) & (D.ref == ref) & (D.condition == "clean")]
            r[f"d_meanAUC_{tg}"] = x.auc_diff.iloc[0] if len(x) else np.nan; r[f"ci_{tg}"] = f"[{x.ci95_lo.iloc[0]:+.3f}, {x.ci95_hi.iloc[0]:+.3f}]" if len(x) else ""; r[f"sig_{tg}"] = bool(x.ci_excludes_0.iloc[0]) if len(x) else ""
            r[f"d_cleanAUC_{tg}"] = xc.auc_diff.iloc[0] if len(xc) else np.nan
        for c in SUM.columns:
            if c.startswith("cross_auc_"): r[c] = s[c]
        rows.append(r)
    emit("T2_ablation_paired", pd.DataFrame(rows), "Ablation with paired video-cluster bootstrap (1000 resamples) of the mean in-family AUC difference versus A and versus B (seed 42).")
    pd.DataFrame(cir).to_csv(f"{TAB}/ablation_auc_ci.csv", index=False)
    def fig():
        S = SUM[SUM.model.isin(ms)].set_index("model").reindex(ms); fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(ms)); w = 0.27
        ax.bar(x - w, S.clean_auc, w, label="clean"); ax.bar(x, S.mean_degraded_auc, w, label="mean degraded"); ax.bar(x + w, S.webp_auc, w, label="WebP (unseen)")
        ax.set_xticks(x, ms, rotation=25); ax.set_ylim(0.4, 1.0); ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIGD}/Fig_Ablation_Bars.png", dpi=300); fig.savefig(f"{FIGD}/Fig_Ablation_Bars.pdf"); plt.close(fig)
    safe("fig ablation", fig)

# ---------- 3. lambda ----------
def lambda_table():
    mp = [("B1_s42", 0.0), ("B_lam0.05_s42", 0.05), ("B_s42", 0.2), ("B_lam0.5_s42", 0.5), ("B_lam1.0_s42", 1.0)]; rows = []
    for m, l in mp:
        x = SUM[SUM.model == m]
        if len(x):
            r = x.iloc[0]; cc = [c for c in SUM.columns if c.startswith("cross_auc_")]; rows.append(dict(lam=l, model=m, clean_auc=r.clean_auc, mean_degraded_auc=r.mean_degraded_auc, mean_f1_drop=r.mean_f1_drop, webp_auc=r.webp_auc, mean_cross_auc=float(np.nanmean([r[c] for c in cc])) if cc else np.nan, **{c: r[c] for c in cc}))
    L = pd.DataFrame(rows); emit("T3_lambda_sweep", L, "Consistency weight sweep (seed 42, light protocol); lambda=0 is the degradation-only ablation B1.")
    if len(L) > 1:
        def fig():
            fig, ax = plt.subplots(figsize=(6, 4)); xs = np.arange(len(L))
            for k, lab in (("clean_auc", "clean"), ("mean_degraded_auc", "mean degraded"), ("webp_auc", "WebP"), ("mean_cross_auc", "mean cross-dataset")): ax.plot(xs, L[k], marker="o", label=lab)
            ax.set_xticks(xs, [f"{l:g}" for l in L.lam]); ax.set_xlabel("lambda"); ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIGD}/Fig_Lambda_Sweep.png", dpi=300); fig.savefig(f"{FIGD}/Fig_Lambda_Sweep.pdf"); plt.close(fig)
        safe("fig lambda", fig)

# ---------- 4. aspect ratio ----------
def aspect_table():
    rows = []
    for m, lab in (("A_s42", "A squash"), ("A_arpres_s42", "A arpres"), ("B_s42", "B squash"), ("B_arpres_s42", "B arpres")):
        x = SUM[SUM.model == m]
        if len(x): r = x.iloc[0]; rows.append(dict(setting=lab, clean_auc=r.clean_auc, mean_degraded_auc=r.mean_degraded_auc, mean_f1_drop=r.mean_f1_drop, webp_auc=r.webp_auc, **{c: r[c] for c in SUM.columns if c.startswith("cross_auc_")}))
    emit("T4_aspect_ratio", pd.DataFrame(rows), "Effect of squashing whole frames to 256x256 (squash) versus aspect-preserving shorter-side resize + centre crop (arpres). Note: arpres discards lateral frame content.")
    pairs = [("B_s42", "A_s42"), ("B_arpres_s42", "A_arpres_s42")]; cir, dfr = bootstrap_models(["A_s42", "B_s42", "A_arpres_s42", "B_arpres_s42"], pairs, "aspect", n_boot=NB7_BOOT)
    if dfr: emit("T4b_aspect_paired_B_minus_A", pd.DataFrame(dfr)[lambda d: d.condition.isin(["clean", "MEAN_IN_FAMILY", "webp_medium"])], "Paired B minus A AUC differences within each preprocessing variant.")

# ---------- 5. NB5 analysis tables ----------
def nb5_tables():
    th = rd("NB5", "metrics/Threshold_Analysis.csv")
    if th is not None:
        keep = th[th.condition.isin(["clean", "jpeg_medium", "blur_severe", "noise_medium", "combined_severe", "webp_medium"])]
        emit("T5_recalibration_threshold", keep.pivot_table(index=["model", "condition"], columns="rule", values="f1").reset_index(), "F1 under four threshold rules (fixed 0.5; clean-val Youden; re-calibrated on same-condition degraded validation subset; ORACLE test-set threshold = upper bound only).")
        emit("T5b_threshold_all", th, "All threshold-rule metrics for every condition.")
    ca = rd("NB5", "metrics/Calibration_TemperatureScaling.csv")
    if ca is not None: emit("T5c_calibration", ca[ca.condition.isin(["clean", "combined_severe", "webp_medium"])], "ECE/Brier before and after temperature scaling (fit on clean val or on degraded val subset).")
    for nm, rel, cap in (("T6_efficiency_inference", "metrics/Efficiency_Inference.csv", "Parameters, FLOPs, model-only and end-to-end latency."), ("T6b_degradation_op_cpu_ms", "metrics/Efficiency_Degradation_Op_CPU_ms.csv", "CPU cost of each degradation op on a 224x224 image."),
                         ("T6c_training_peak_memory", "metrics/Efficiency_Training_Peak_Memory.csv", "Peak GPU memory of one training step."), ("T6d_training_light_protocol", "metrics/Efficiency_Training_Light_Protocol.csv", "Training resources of the light-protocol runs."),
                         ("T7_external_characterisation", "metrics/External_Characterisation.csv", "Resolution, resize factor, file size, JPEG quality, sharpness and noise of external datasets versus FF++."),
                         ("T7b_external_gap_vs_auc", "metrics/External_Gap_vs_AUC.csv", "Gap to FF++ statistics next to AUC of original A/B and the B-minus-A gain."), ("T7c_external_per_folder_auc", "metrics/External_PerManipulationFolder_AUC.csv", "Per-manipulation sub-folder AUC on external datasets."),
                         ("T8_ffpp_dataset_facts", "metrics/FFpp_Dataset_Facts_PerFolder.csv", "FF++ copy: resolution, file size, estimated JPEG quality, frames per video."), ("T8b_shortcut_classifiers", "metrics/Shortcut_Classifier_AUC.csv", "Metadata/pixel-statistics shortcut classifiers (AUC well above 0.5 = format shortcut).")):
        x = rd("NB5", rel)
        if x is not None: emit(nm, x.T.reset_index().rename(columns={"index": "field"}) if nm == "T6_efficiency_inference" else x, cap)

# ---------- 6. per-condition main comparison (B vs A, 3 seeds mean) ----------
def final_cond_table():
    if not len(ROB): return
    P = ROB[ROB.model.isin([f"A_s{s}" for s in (42, 43, 44)] + [f"B_s{s}" for s in (42, 43, 44)])].copy()
    if not len(P): return
    P["arm"] = P.model.str[0]; G = P.groupby(["condition", "arm"]).auc.agg(["mean", "std"]).unstack("arm"); G.columns = [f"{a}_{b}" for a, b in G.columns]; G = G.reset_index()
    emit("T9_per_condition_mean_over_seeds", G, "Test AUC per condition, mean and std over available seeds.")

def reviewer_map():
    have = lambda *n: all(x in TABLES for x in n)
    L = ["# REVIEWER_MAP (auto-generated by NB7)", "", f"Result folders found: {', '.join(sorted(RES)) or 'NONE'}", "", "| Comment | Evidence | Where | Status |", "|---|---|---|---|"]
    def row(c, ev, wh, ok): L.append(f"| {c} | {ev} | {wh} | {'AVAILABLE' if ok else 'MISSING (run the notebook / see README_RUN_ORDER.md)'} |")
    row("R1-1 weak baseline", "A_jpeg / A_randaug / A_deg1 vs B (paired CIs)", "tables/T2_ablation_paired.*, NB2+NB4 metrics", have("T2_ablation_paired"))
    row("R1-2 cross-dataset AUC low", "External characterisation, gap vs AUC, per-manipulation AUC", "T7*, NB5 metrics/External_*", have("T7_external_characterisation"))
    row("R1-3 threshold re-calibration", "4 threshold rules incl. WebP; temperature scaling", "T5*, NB5 Threshold_Analysis.csv", have("T5_recalibration_threshold"))
    row("R1-4 ablation + lambda", "A/B1/B/C and lambda 0/0.05/0.2/0.5/1.0", "T2, T3, Fig_Lambda_Sweep", have("T2_ablation_paired") and have("T3_lambda_sweep"))
    row("R2-1 novelty / comparisons", "compute-matched (A_deg1) and standard-augmentation baselines only; literature comparison = writing", "T2", have("T2_ablation_paired"))
    row("R2-2 degradation-only vs consistency-only vs full", "B1 vs C vs B", "T2", have("T2_ablation_paired"))
    row("R2-3 seeds / paired stats", "3 seeds, paired cluster-bootstrap CIs", "T1, T1b, T2", have("T1_multiseed_summary"))
    row("R2-4 dataset details / aspect ratio", "FF++ facts, shortcut classifiers, squash vs arpres", "T8*, T4*, NB5 dataset_manifest.json", have("T8_ffpp_dataset_facts") or have("T4_aspect_ratio"))
    row("R2-5 collapse investigation + stronger baselines", "radial spectra, stats under degradation, Spearman, A_jpeg/A_randaug", "NB5 Radial_Power_Spectrum.csv, Stats_Under_Degradation.csv, Spearman_*.csv, T2", have("T2_ablation_paired"))
    row("R2-6 cross-dataset analysis", "dataset-specific statistics and per-folder AUC", "T7*", have("T7_external_characterisation"))
    row("R2-7 threshold + calibration (incl. A)", "F1 vs threshold curves, ECE/Brier, reliability diagrams for A and B", "T5*, NB5 figures", have("T5c_calibration"))
    row("R2-8 efficiency accounting", "params, FLOPs, latency, e2e, peak memory, training time", "T6*", have("T6_efficiency_inference"))
    row("R3-1 data availability", "Split_Assignment.csv + split_hash.txt + dataset_manifest.json", "any RESULTS_*/metrics", any(os.path.exists(f"{p}/metrics/Split_Assignment.csv") for p in RES.values()))
    L += ["", "NOT covered by experiments (writing only): R2-1 literature comparison text, R3-2 author contributions, R3-3 outdated references, R3-4 preprints.",
          "", "Caveats to state in the paper: light-protocol runs use every 2nd training frame and 10 epochs (not comparable in absolute terms to the original full-protocol numbers); cudnn.benchmark => non-bitwise-deterministic;",
          "the ORACLE threshold is an upper bound, never a headline; n=4 external datasets makes gap/gain correlations descriptive only."]
    open(f"{OUT}/REVIEWER_MAP.md", "w").write("\n".join(L)); log("REVIEWER_MAP.md written")

for st in (seeds_table, ablation_table, lambda_table, aspect_table, nb5_tables, final_cond_table, reviewer_map): safe(st.__name__, st)
# final zip: tables, latex, figures, reviewer map + copies of every notebook's metrics/figures (not predictions)
zp = f"{WORK}/FINAL_PAPER_RESULTS.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for r_, _, fs in os.walk(OUT):
        for fn in fs: z.write(os.path.join(r_, fn), os.path.join("RESULTS_NB7", os.path.relpath(os.path.join(r_, fn), OUT)))
    for tag, root in RES.items():
        for sub in ("metrics", "figures"):
            for r_, _, fs in os.walk(f"{root}/{sub}"):
                for fn in fs: z.write(os.path.join(r_, fn), os.path.join(f"RESULTS_{tag}", os.path.relpath(os.path.join(r_, fn), root)))
        if os.path.exists(f"{root}/MANIFEST.json"): z.write(f"{root}/MANIFEST.json", f"RESULTS_{tag}/MANIFEST.json")
print("FINAL ZIP PATH:", zp, "| download it from the notebook Output tab;",  f"{os.path.getsize(zp)/2**20:.1f} MB")
for r_, _, fs in os.walk(OUT): print("  " * os.path.relpath(r_, OUT).count(os.sep) + os.path.basename(r_) + "/", sorted(fs)[:12])
