"""
masa_runner_grouped.py — MASA grouped 80/20 5-Fold using popsign splits
========================================================================
Loads fold_splits.json from the popsign grouped-kfold pipeline and maps
each .h5 path to its corresponding .npy path via canonical keys.

Uses the SAME FIXED dataset / model / trainer as the LOSO run.

Handles edge cases:
  • Multiple MASA NPYs per canonical key → stateful allocation
  • Popsign has more H5s than MASA NPYs for a key → drop from TEST only
    (keep in train to not lose training signal; eval from best_model.pt)

Run:
    python masa_runner_grouped.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import sys
import json
import random
import csv
import time
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from masa_dataset import label_from_filename, user_from_path, find_npy_files
from masa_train   import train_one_fold, Tee, save_confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

POPSIGN_SPLITS_JSON = (
    "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/"
    "both_hands/kfold/code_30fps/80-20/results_grouped/fold_splits.json"
)

POPSIGN_H5_ROOT = (
    "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/ISL_Goa_Data_h5_30fps"
)

MASA_NPY_ROOT = (
    "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/MASA/Data/ISL_GOA/Pose"
)

RESULTS_DIR = (
    "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/new_masa/results_grouped_masa"
)

config = {
    "max_frames"      : 150,
    "num_workers"     : 6,

    "batch_size"      : 64,
    "epochs"          : 60,
    "warmup_epochs"   : 3,
    "min_lr_ratio"    : 0.05,
    "lr"              : 2e-4,
    "weight_decay"    : 5e-4,
    "label_smoothing" : 0.1,
    "use_mixup"       : True,
    "mixup_alpha"     : 0.2,
    "use_amp"         : True,
    "seed"            : 42,
    "num_folds"       : 5,

    "model_dim"       : 256,
    "nhead"           : 8,
    "num_layers"      : 4,
    "dim_feedforward" : 512,
    "dropout"         : 0.2,
    "drop_path_rate"  : 0.1,

    "mask_ratio"      : 0.4,
    "recon_weight"    : 0.1,
    "decoder_layers"  : 2,
}

START_FROM_FOLD = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

set_seed(config["seed"])
torch.backends.cudnn.benchmark = True


# ============================================================
# SETUP / LOGGING
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
log_f = open(os.path.join(RESULTS_DIR, f"masa_grouped_kfold_{ts}.log"), "w")
sys.stdout = Tee(sys.stdout, log_f)
sys.stderr = Tee(sys.stderr, log_f)

print("=" * 70)
print(f"  MASA GROUPED 5-FOLD (reusing popsign split)  |  {ts}")
print(f"  POPSIGN SPLITS JSON : {POPSIGN_SPLITS_JSON}")
print(f"  MASA NPY ROOT       : {MASA_NPY_ROOT}")
print(f"  RESULTS_DIR         : {RESULTS_DIR}")
print(f"  START_FROM_FOLD     : {START_FROM_FOLD}")
print(f"  Seed                : {config['seed']}")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")
if device.type == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")


# ============================================================
# CANONICAL KEY MATCHER (unchanged)
# ============================================================

USER_RE_FOLDER    = re.compile(r'user0*(\d+)', re.IGNORECASE)
USER_RE_FILENAME  = re.compile(r'user0*(\d+)', re.IGNORECASE)
SESSION_RE_A      = re.compile(r'session0*(\d+)', re.IGNORECASE)
CLIP_RE_A         = re.compile(r'clip0*(\d+)',    re.IGNORECASE)
PATTERN_B_SC_RE   = re.compile(r's(\d+)-(\d+)', re.IGNORECASE)
PATTERN_C_RE      = re.compile(r'^([A-Za-z]+)__(\d+)$')


def _extract_user_num(path: str):
    path_norm = path.replace("\\", "/")
    parts     = path_norm.split("/")
    for part in parts[:-1]:
        m = USER_RE_FOLDER.search(part)
        if m:
            return int(m.group(1))
    base = os.path.basename(path_norm)
    m    = USER_RE_FILENAME.search(base)
    if m:
        return int(m.group(1))
    return None


def _extract_word_session_clip(path: str):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    m_sess_a = SESSION_RE_A.search(stem)
    m_clip_a = CLIP_RE_A.search(stem)

    if m_sess_a and m_clip_a:
        session  = int(m_sess_a.group(1))
        clip     = int(m_clip_a.group(1))
        before   = stem[: m_sess_a.start()].rstrip("_")
        word_raw = before.split("_")[-1]
        word     = word_raw.lower()
        if word:
            return word, session, clip

    if m_clip_a and not m_sess_a:
        clip     = int(m_clip_a.group(1))
        before   = stem[: m_clip_a.start()].rstrip("_")
        word_raw = before.split("_")[-1]
        word     = word_raw.lower()
        if word:
            return word, 0, clip

    m_b = PATTERN_B_SC_RE.search(stem)
    if m_b:
        session = int(m_b.group(1))
        clip    = int(m_b.group(2))
        first_token = stem.split("__", 1)[0].lstrip("_")
        word = None
        if first_token and first_token.isalpha() \
                and not first_token.lower().startswith("user") \
                and "mnt" not in first_token.lower():
            word = first_token.lower()
        else:
            after    = stem[m_b.end():]
            m_suffix = re.match(r'_([A-Za-z]+)', after)
            if m_suffix:
                word = m_suffix.group(1).lower()
        if word:
            return word, session, clip

    m_c = PATTERN_C_RE.match(stem)
    if m_c:
        word = m_c.group(1).lower()
        num  = int(m_c.group(2))
        return word, num, 0

    return None, None, None


def canonical_key(path: str):
    user = _extract_user_num(path)
    word, session, clip = _extract_word_session_clip(path)
    if None in (user, word, session, clip):
        return None
    return (user, word, session, clip)


# ============================================================
# BUILD H5 → NPY MAPPING
# ============================================================

print("\n  Indexing MASA .npy files by canonical key …")
all_npy_paths = find_npy_files(MASA_NPY_ROOT)
print(f"  Total .npy files found: {len(all_npy_paths)}")

npy_by_key = defaultdict(list)
n_unparseable = 0
for p in all_npy_paths:
    k = canonical_key(p)
    if k is None:
        n_unparseable += 1
    else:
        npy_by_key[k].append(p)

n_dup_keys = sum(1 for ps in npy_by_key.values() if len(ps) > 1)
print(f"  Unique canonical keys : {len(npy_by_key)}")
print(f"  Unparseable .npy files: {n_unparseable}")
print(f"  Duplicate keys (resolved): {n_dup_keys}")

# Pre-sort candidates per key once for deterministic allocation order
_sorted_cands_by_key = {
    k: sorted(cs, key=lambda p: ("/Round1/" in p, len(p), p))
    for k, cs in npy_by_key.items()
}
_used_per_key = defaultdict(int)   # key → next candidate index to hand out


def h5_to_npy(h5_path: str):
    """
    Stateful H5 → NPY mapper. For canonical keys with multiple NPY
    candidates, hands out each candidate to consecutive H5 requests.
    When requests exceed candidates, cycles (and the caller should flag).
    """
    k = canonical_key(h5_path)
    if k is None:
        return None, None
    cands = _sorted_cands_by_key.get(k)
    if not cands:
        return None, None
    idx = _used_per_key[k]
    if idx >= len(cands):
        # More H5s than NPYs for this key — cycle but mark as "reused"
        chosen = cands[idx % len(cands)]
        reused = True
    else:
        chosen = cands[idx]
        reused = False
    _used_per_key[k] += 1
    return chosen, reused


# ============================================================
# LOAD POPSIGN SPLITS & REMAP
# ============================================================

if not os.path.isfile(POPSIGN_SPLITS_JSON):
    raise FileNotFoundError(f"Cannot find popsign splits: {POPSIGN_SPLITS_JSON}")

with open(POPSIGN_SPLITS_JSON, "r") as f:
    popsign_splits = json.load(f)

fold_keys = sorted([k for k in popsign_splits if k.startswith("fold_")],
                   key=lambda s: int(s.split("_")[1]))
num_folds = len(fold_keys)

if num_folds != config["num_folds"]:
    print(f"\n  ⚠ Config says num_folds={config['num_folds']} but JSON has {num_folds}. Using {num_folds}.")
    config["num_folds"] = num_folds

fold_train_paths = [[] for _ in range(num_folds)]
fold_test_paths  = [[] for _ in range(num_folds)]

total_train_h5 = 0
total_test_h5  = 0
missing_train  = 0
missing_test   = 0
missing_examples = []

# Track reused NPYs per fold for reporting
reused_train_per_fold = [0] * num_folds
reused_test_per_fold  = [0] * num_folds

for f, fk in enumerate(fold_keys):
    split    = popsign_splits[fk]
    h5_train = split["train"]
    h5_test  = split["test"]
    total_train_h5 += len(h5_train)
    total_test_h5  += len(h5_test)

    # Reset allocation counter for each fold so the same NPYs can be
    # re-handed out across folds (but WITHIN a fold, train and test
    # get distinct NPYs when multiple candidates exist).
    _used_per_key.clear()

    # Map training H5s first — they get first pick.
    for h5p in h5_train:
        npy, reused = h5_to_npy(h5p)
        if npy is None:
            missing_train += 1
            if len(missing_examples) < 10:
                missing_examples.append(h5p)
            continue
        if reused:
            reused_train_per_fold[f] += 1
        fold_train_paths[f].append(npy)

    # Then map test H5s — they get whatever remains.
    for h5p in h5_test:
        npy, reused = h5_to_npy(h5p)
        if npy is None:
            missing_test += 1
            if len(missing_examples) < 10:
                missing_examples.append(h5p)
            continue
        if reused:
            reused_test_per_fold[f] += 1
        fold_test_paths[f].append(npy)

total_h5      = total_train_h5 + total_test_h5
total_missing = missing_train + missing_test
coverage      = 100.0 * (total_h5 - total_missing) / max(total_h5, 1)

print(f"\n  Popsign split loaded.")
print(f"    Train entries across folds : {total_train_h5}")
print(f"    Test  entries across folds : {total_test_h5}")
print(f"    Unmapped train entries     : {missing_train}")
print(f"    Unmapped test  entries     : {missing_test}")
print(f"    Coverage                   : {coverage:.4f}%")

if any(reused_train_per_fold) or any(reused_test_per_fold):
    print(f"\n  ⚠ Some NPYs were REUSED (popsign had more H5s than MASA NPYs for a key):")
    for f in range(num_folds):
        r_tr = reused_train_per_fold[f]
        r_te = reused_test_per_fold[f]
        if r_tr or r_te:
            print(f"    Fold {f}: {r_tr} reused in train, {r_te} reused in test")

if missing_examples:
    print(f"\n  First missing examples (up to 10):")
    for p in missing_examples: print(f"    {p}")

if coverage < 99.0:
    raise RuntimeError(f"Coverage too low ({coverage:.2f}%). Fix parser first.")


# ============================================================
# OVERLAP RESOLUTION (auto-fix conflicting NPYs)
# ============================================================

print("\n  Resolving train/test overlaps (caused by H5-NPY cardinality mismatch)…")

overlap_resolution_csv = os.path.join(RESULTS_DIR, "overlap_resolution.csv")
with open(overlap_resolution_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "conflicting_npy", "action", "note"])

    total_removed_from_test = 0
    for f_idx in range(num_folds):
        train_set = set(fold_train_paths[f_idx])
        test_set  = set(fold_test_paths[f_idx])
        overlap   = train_set & test_set

        if overlap:
            print(f"    Fold {f_idx}: {len(overlap)} overlapping NPY(s) — "
                  f"dropping from test (keeping in train)")
            # Drop overlaps from TEST (preserves training signal,
            # only lose a few test samples per fold)
            fold_test_paths[f_idx] = [
                p for p in fold_test_paths[f_idx] if p not in overlap
            ]
            total_removed_from_test += len(overlap)
            for npy in sorted(overlap):
                writer.writerow([
                    f_idx, npy, "removed_from_test",
                    "conflict: same NPY appeared in both train and test",
                ])
        else:
            print(f"    Fold {f_idx}: no overlaps ✓")

print(f"\n  Total test entries removed due to conflicts: {total_removed_from_test}")
print(f"  Overlap resolution log saved: {overlap_resolution_csv}")


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

print("\n  Final sanity checks…")

# Overlap must now be zero
for f_idx in range(num_folds):
    overlap = set(fold_train_paths[f_idx]) & set(fold_test_paths[f_idx])
    assert len(overlap) == 0, \
        f"Fold {f_idx}: overlap STILL present ({len(overlap)} items) after resolution!"
print("  Overlap check passed ✓")

print("\n  Grouped-kfold protocol verification "
      "(session-level grouping, users intentionally overlap):")

for f_idx in range(num_folds):
    train_users = sorted(set(_extract_user_num(p) for p in fold_train_paths[f_idx]))
    test_users  = sorted(set(_extract_user_num(p) for p in fold_test_paths[f_idx]))
    shared      = sorted(set(train_users) & set(test_users))
    print(f"    Fold {f_idx}: "
          f"train={len(train_users)} users, "
          f"test={len(test_users)} users, "
          f"shared={len(shared)} users "
          f"({'✓ session-level grouping' if shared else '⚠ no user overlap'})")

print("  Grouped-kfold protocol verified ✓")

print("\n  Fold statistics (after remap + conflict resolution):")
for f_idx in range(num_folds):
    train_users = sorted(set(_extract_user_num(p) for p in fold_train_paths[f_idx]))
    test_users  = sorted(set(_extract_user_num(p) for p in fold_test_paths[f_idx]))
    print(f"    Fold {f_idx}: "
          f"Train={len(fold_train_paths[f_idx])} files ({len(train_users)} users), "
          f"Test={len(fold_test_paths[f_idx])} files ({len(test_users)} users)")

# ============================================================
# LABEL ENCODER
# ============================================================

all_mapped = set()
for f_idx in range(num_folds):
    all_mapped.update(fold_train_paths[f_idx])
    all_mapped.update(fold_test_paths[f_idx])
all_mapped = sorted(all_mapped)

all_labels  = [label_from_filename(p) for p in all_mapped]
all_classes = sorted(set(all_labels))
label_encoder = LabelEncoder().fit(all_classes)
np.save(os.path.join(RESULTS_DIR, "class_names.npy"),
        np.array(label_encoder.classes_))

print(f"\n  Total classes : {len(all_classes)}")
print(f"  First 10      : {all_classes[:10]}{'...' if len(all_classes) > 10 else ''}")


# ============================================================
# SAVE MASA SPLITS
# ============================================================

masa_splits = {
    f"fold_{f_idx}": {"train": fold_train_paths[f_idx], "test": fold_test_paths[f_idx]}
    for f_idx in range(num_folds)
}
splits_path = os.path.join(RESULTS_DIR, "fold_splits_masa.json")
with open(splits_path, "w") as f:
    json.dump(masa_splits, f, indent=2)
print(f"  MASA fold splits saved : {splits_path}")


# ============================================================
# TRAINING LOOP
# ============================================================

fold_results  = []
overall_start = time.time()

for fold_idx in range(num_folds):
    if fold_idx < START_FROM_FOLD:
        print(f"  Skipping fold {fold_idx} (< START_FROM_FOLD).")
        continue

    print("\n" + "=" * 60)
    print(f"  RUNNING MASA GROUPED FOLD {fold_idx}")
    print("=" * 60)

    train_paths = fold_train_paths[fold_idx]
    test_paths  = fold_test_paths[fold_idx]
    fold_dir    = os.path.join(RESULTS_DIR, f"fold_{fold_idx}")

    fold_start = time.time()
    result = train_one_fold(
        fold_idx      = fold_idx,
        train_paths   = train_paths,
        test_paths    = test_paths,
        label_encoder = label_encoder,
        class_names   = list(label_encoder.classes_),
        fold_log_dir  = fold_dir,
        device        = device,
        config        = config,
        tb_root       = os.path.join(RESULTS_DIR, "runs"),
        dry_run       = False,
    )
    fold_time = time.time() - fold_start
    result["fold_time_s"] = fold_time
    fold_results.append(result)

    # ── Per-user accuracy ─────────────────────────────────
    user_y_true = defaultdict(list)
    user_y_pred = defaultdict(list)
    for path, yt, yp in zip(test_paths, result["y_true"], result["y_pred"]):
        user = user_from_path(path)
        user_y_true[user].append(yt)
        user_y_pred[user].append(yp)

    user_acc_path = os.path.join(fold_dir, "per_user_accuracy.csv")
    with open(user_acc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "n_samples", "top1_accuracy(%)"])
        for user in sorted(user_y_true.keys()):
            yt  = np.array(user_y_true[user])
            yp  = np.array(user_y_pred[user])
            acc = (yt == yp).mean() * 100.0
            writer.writerow([user, len(yt), f"{acc:.2f}"])
    print(f"  Per-user accuracy saved: {user_acc_path}")

    avg = (time.time() - overall_start) / len(fold_results)
    remain = (num_folds - fold_idx - 1) * avg
    print(f"\n  ⏱ Fold {fold_idx} time: {fold_time/60:.1f} min | "
          f"Avg: {avg/60:.1f} min | ETA remaining: {remain/60:.1f} min "
          f"({remain/3600:.1f} h)")


# ============================================================
# SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("  FINAL MASA GROUPED K-FOLD SUMMARY")
print("=" * 70)

print(
    f"\n  {'Fold':<6} {'TrTop1':>7} {'TrTop5':>7} "
    f"{'TeTop1':>7} {'TeTop5':>7} {'MacroF1':>8} {'WtdF1':>7}"
)
print(f"  {'-'*62}")
for r in fold_results:
    print(
        f"  {r['fold']:<6} "
        f"{r['train_top1_acc']:6.2f}%  "
        f"{r['train_top5_acc']:6.2f}%  "
        f"{r['top1_acc']:6.2f}%  "
        f"{r['top5_acc']:6.2f}%  "
        f"{r['macro_f1']:7.2f}%  "
        f"{r['weighted_f1']:6.2f}%"
    )
print(f"  {'-'*62}")

metric_keys = [
    ("train_top1_acc", "Train Top-1 Acc"),
    ("train_top5_acc", "Train Top-5 Acc"),
    ("top1_acc",       "Test  Top-1 Acc"),
    ("top5_acc",       "Test  Top-5 Acc"),
    ("macro_f1",       "Macro F1"),
    ("weighted_f1",    "Weighted F1"),
]
print()
for key, label in metric_keys:
    vals = [r[key] for r in fold_results]
    print(f"  {label:<22}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")


# ============================================================
# CSV
# ============================================================

csv_path = os.path.join(RESULTS_DIR, "kfold_summary.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "fold",
        "train_top1(%)","train_top5(%)",
        "test_top1(%)","test_top5(%)",
        "macro_f1(%)","weighted_f1(%)","test_loss","fold_time_min",
    ])
    for r in fold_results:
        w.writerow([
            r["fold"],
            f"{r['train_top1_acc']:.2f}", f"{r['train_top5_acc']:.2f}",
            f"{r['top1_acc']:.2f}",       f"{r['top5_acc']:.2f}",
            f"{r['macro_f1']:.2f}",       f"{r['weighted_f1']:.2f}",
            f"{r['test_loss']:.4f}",
            f"{r.get('fold_time_s',0)/60:.2f}",
        ])
    for stat, fn in [("MEAN", np.mean), ("STD", np.std)]:
        w.writerow(
            [stat] +
            [f"{fn([r[k] for r in fold_results]):.2f}" for k,_ in metric_keys] +
            ["—","—"]
        )
print(f"\n  Summary saved: {csv_path}")


# ============================================================
# AGGREGATED CONFUSION MATRIX
# ============================================================

if fold_results:
    all_y_true = np.concatenate([r["y_true"] for r in fold_results])
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    cm = confusion_matrix(all_y_true, all_y_pred,
                          labels=list(range(len(label_encoder.classes_))))
    save_confusion_matrix(
        cm, list(label_encoder.classes_),
        os.path.join(RESULTS_DIR, "confusion_matrix_aggregated.png"),
        title="MASA Grouped K-Fold Confusion Matrix",
    )
    print("  Aggregated confusion matrix saved.")


# ============================================================
# MEAN CURVES
# ============================================================

if fold_results:
    max_ep = max(len(r["train_losses"]) for r in fold_results)
    def pad_to(lst, length):
        return lst + [lst[-1]] * (length - len(lst))

    mean_train_loss = np.mean([pad_to(r["train_losses"], max_ep) for r in fold_results], axis=0)
    mean_test_loss  = np.mean([pad_to(r["test_losses"],  max_ep) for r in fold_results], axis=0)
    mean_train_top1 = np.mean([pad_to(r["train_top1s"],  max_ep) for r in fold_results], axis=0)
    mean_test_top1  = np.mean([pad_to(r["test_top1s"],   max_ep) for r in fold_results], axis=0)
    mean_train_top5 = np.mean([pad_to(r["train_top5s"],  max_ep) for r in fold_results], axis=0)
    mean_test_top5  = np.mean([pad_to(r["test_top5s"],   max_ep) for r in fold_results], axis=0)
    mean_recon      = np.mean([pad_to(r["recon_losses"], max_ep) for r in fold_results], axis=0)

    def save_curve(series_list, title, ylabel, path):
        fig, ax = plt.subplots(figsize=(10, 5))
        for color, lbl, data in series_list:
            ax.plot(data, color=color, linewidth=2, label=lbl)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()
        print(f"  Saved: {os.path.basename(path)}")

    save_curve(
        [("steelblue","Mean Train CE",mean_train_loss),
         ("coral","Mean Test CE",mean_test_loss)],
        "Mean Cross-Entropy Loss","Loss",
        os.path.join(RESULTS_DIR, "mean_loss_curve.png"))
    save_curve(
        [("steelblue","Train Top-1",mean_train_top1),
         ("coral","Test Top-1",mean_test_top1)],
        "Mean Top-1 Accuracy","Accuracy (%)",
        os.path.join(RESULTS_DIR, "mean_top1_curve.png"))
    save_curve(
        [("steelblue","Train Top-5",mean_train_top5),
         ("coral","Test Top-5",mean_test_top5)],
        "Mean Top-5 Accuracy","Accuracy (%)",
        os.path.join(RESULTS_DIR, "mean_top5_curve.png"))
    save_curve(
        [("mediumseagreen","Mean Recon Loss (×λ)",mean_recon)],
        "Mean Reconstruction Loss","Loss",
        os.path.join(RESULTS_DIR, "mean_recon_curve.png"))


# ============================================================
# BAR CHARTS
# ============================================================

if fold_results:
    x     = np.arange(len(fold_results))
    xlbls = [f"Fold {r['fold']}" for r in fold_results]

    def save_bar(key, ylabel, title, filename, color="steelblue"):
        vals = [r[key] for r in fold_results]; mv = np.mean(vals)
        fig, ax = plt.subplots(figsize=(max(8, len(fold_results)*1.2), 6))
        bars = ax.bar(x, vals, 0.5, color=color, edgecolor="black")
        ax.axhline(mv, color="crimson", ls="--", lw=1.5, label=f"Mean = {mv:.2f}%")
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
        ax.set_ylim(0, 115); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=100); plt.close()
        print(f"  Bar chart: {filename}")

    save_bar("train_top1_acc","Accuracy (%)","Train Top-1 per Fold","kfold_bar_train_top1.png")
    save_bar("train_top5_acc","Accuracy (%)","Train Top-5 per Fold","kfold_bar_train_top5.png")
    save_bar("top1_acc",      "Accuracy (%)","Test Top-1 per Fold", "kfold_bar_test_top1.png", color="coral")
    save_bar("top5_acc",      "Accuracy (%)","Test Top-5 per Fold", "kfold_bar_test_top5.png", color="coral")
    save_bar("macro_f1",      "Macro F1 (%)","Test Macro F1 per Fold","kfold_bar_macro_f1.png",color="mediumseagreen")


total = time.time() - overall_start
print(f"\n  ALL DONE. Total wall time: {total/3600:.2f} h")
print(f"  Results in: {RESULTS_DIR}")
print(f"  TensorBoard: tensorboard --logdir={RESULTS_DIR}/runs")