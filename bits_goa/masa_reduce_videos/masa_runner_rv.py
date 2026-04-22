"""
rq2_runner_masa.py — MASA RQ2: Progressive Training Data Reduction (LOSO)
===========================================================================
Analyzes relationship between training data size and signer-independent
recognition under LOSO.

Cap-based subsampling (step 2, matching popsign RQ2 logic):
    Round 1 : cap=10  → nominal 14×10 = 140 videos/word
    Round 2 : cap=8   → nominal 14×8  = 112
    Round 3 : cap=6   → nominal 14×6  = 84
    Round 4 : cap=4   → nominal 14×4  = 56
    Round 5 : cap=2   → nominal 14×2  = 28
    Round 6 : cap=1   → nominal 14×1  = 14

    Words with < cap recordings per user contribute all available.
    Test set is fixed per fold (held-out user's full recordings) across all caps.

Reproducibility: seed = base_seed + round_idx*1000 + fold_idx

Uses the production MASA stack:
    masa_dataset.py  (unchanged)
    masa_model.py    (unchanged)
    masa_train.py    (unchanged) — train_one_fold saves best_model.pt
After training each fold, best_model.pt is loaded for detailed evaluation.

Outputs (under RESULTS_DIR):
    cap_10/  cap_8/  cap_6/  cap_4/  cap_2/  cap_1/
        fold_0/ ... fold_14/
            best_model.pt
            eval/
                confusion_matrix.png
                top_confused_pairs.png/.csv
                per_class_accuracy.csv/_bar.png
                classification_report.txt
                metrics.txt
            per_user_accuracy.csv
        round_summary.csv
    rq2_summary.csv
    rq2_accuracy_curve.png
    test_coverage.csv
    class_names.npy
    config.json
    rq2_log_TIMESTAMP.txt
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # pin to GPU 0

import sys
import csv
import json
import random
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, f1_score,
    precision_score, recall_score, classification_report,
)
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from masa_dataset import (
    find_npy_files, label_from_filename, user_from_path,
    MASADataset, collate_fn_masa, FEAT_DIM,
)
from masa_model import MASAClassifier
from masa_train import train_one_fold, Tee, save_confusion_matrix


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/MASA/Data/ISL_GOA/Pose"
RESULTS_DIR = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/new_masa/results_rq2_masa"

CAP_VALUES = [10, 8, 6, 4, 2, 1]   # step 2, descending

config = {
    # Data
    "max_frames"      : 150,
    "num_workers"     : 6,

    # Training — proven healthy from LOSO run
    "batch_size"      : 64,
    "epochs"          : 60,     
    "warmup_epochs"   : 2,
    "min_lr_ratio"    : 0.05,
    "lr"              : 2e-4,
    "weight_decay"    : 5e-4,
    "label_smoothing" : 0.1,
    "use_mixup"       : True,
    "mixup_alpha"     : 0.2,
    "use_amp"         : True,
    "seed"            : 42,

    # Model
    "model_dim"       : 256,
    "nhead"           : 8,
    "num_layers"      : 4,
    "dim_feedforward" : 512,
    "dropout"         : 0.2,
    "drop_path_rate"  : 0.1,

    # MASA auxiliary
    "mask_ratio"      : 0.4,
    "recon_weight"    : 0.1,
    "decoder_layers"  : 2,
}

# Resume support — set > 0 to skip already-completed rounds
START_FROM_ROUND = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ============================================================
# FILE GROUPING
# ============================================================

def group_by_user_and_word(paths):
    """dict[user][word] → list of .npy paths."""
    uwp = defaultdict(lambda: defaultdict(list))
    for p in paths:
        uwp[user_from_path(p)][label_from_filename(p)].append(p)
    return uwp


# ============================================================
# CAP-BASED SUBSAMPLING  (mirrors popsign RQ2 logic exactly)
# ============================================================

def sample_train_paths(train_users, user_word_paths, cap,
                       round_idx, fold_idx, base_seed):
    """
    For each training user and each word, retain at most `cap` recordings.
    Words with < cap contribute all available.

    Deterministic seed = base_seed + round_idx*1000 + fold_idx.
    """
    seed = base_seed + round_idx * 1000 + fold_idx
    rng  = np.random.default_rng(seed=seed)

    selected        = []
    per_word_counts = defaultdict(int)

    for user in sorted(train_users):
        for word in sorted(user_word_paths[user].keys()):
            paths = sorted(user_word_paths[user][word])  # deterministic order
            n     = len(paths)

            if n <= cap:
                chosen = paths
            else:
                idx    = rng.choice(n, size=cap, replace=False)
                chosen = [paths[i] for i in idx]

            selected.extend(chosen)
            per_word_counts[word] += len(chosen)

    counts = list(per_word_counts.values()) if per_word_counts else [0]
    stats  = {
        "total_files"     : len(selected),
        "mean_per_word"   : float(np.mean(counts)),
        "min_per_word"    : int(np.min(counts)),
        "max_per_word"    : int(np.max(counts)),
        "words_at_nominal": int(sum(
            1 for c in counts if c == cap * len(train_users)
        )),
        "words_below"     : int(sum(
            1 for c in counts if c < cap * len(train_users)
        )),
    }
    return selected, stats


# ============================================================
# EVALUATION HELPERS  (parity with popsign RQ2 eval)
# ============================================================

@torch.no_grad()
def _collect_masa_preds(model, loader, device, top_k=5, use_amp=True):
    """Run MASA model over a loader, return y_true, y_pred, y_topk, avg_loss."""
    import torch.nn.functional as F
    model.eval()
    y_true, y_pred, y_topk = [], [], []
    total_loss, total_n    = 0.0, 0

    for s1, s2, s3, labels, lengths, padding_mask in loader:
        s1    = s1.to(device)
        s2    = s2.to(device)
        s3    = s3.to(device)
        labels = labels.to(device)
        padding_mask = padding_mask.to(device)

        with autocast(enabled=use_amp and torch.cuda.is_available()):
            logits, _ = model(s1, s2, s3, padding_mask)
            loss      = F.cross_entropy(logits, labels, reduction="sum")

        preds = logits.argmax(dim=1)
        k     = min(top_k, logits.size(1))
        topk  = logits.topk(k, dim=1).indices

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_topk.append(topk.cpu().numpy())
        total_loss += loss.item()
        total_n    += labels.size(0)

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_topk, axis=0),
        total_loss / max(total_n, 1),
    )


def _save_top_confused_pairs(cm, class_names, save_path_png, n=20):
    """Save top-n off-diagonal confusion pairs as CSV + horizontal barplot."""
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    pairs = pairs[:n]

    csv_path = save_path_png.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("true_class,predicted_as,count\n")
        for t, p, c in pairs:
            f.write(f"{t},{p},{c}\n")

    if pairs:
        labels = [f"{t}→{p}" for t, p, _ in pairs]
        counts = [c for _, _, c in pairs]
        plt.figure(figsize=(14, 6))
        plt.barh(labels[::-1], counts[::-1],
                 color="steelblue", edgecolor="black")
        plt.xlabel("Count")
        plt.title(f"Top {n} Most Confused Class Pairs")
        plt.tight_layout()
        plt.savefig(save_path_png, dpi=100)
        plt.close()


def _save_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    rows = []
    for i, name in enumerate(class_names):
        mask      = (y_true == i)
        n_samples = int(mask.sum())
        acc       = (y_pred[mask] == i).mean() * 100.0 if n_samples > 0 else 0.0
        rows.append((name, n_samples, acc))
    rows.sort(key=lambda x: x[2])

    csv_path = os.path.join(save_dir, "per_class_accuracy.csv")
    with open(csv_path, "w") as f:
        f.write("class,n_samples,accuracy(%)\n")
        for name, n, acc in rows:
            f.write(f"{name},{n},{acc:.2f}\n")

    names = [r[0] for r in rows]
    accs  = [r[2] for r in rows]
    fig_h = max(8, len(names) * 0.18)

    plt.figure(figsize=(12, fig_h))
    colors = ["#d73027" if a < 50 else "#4575b4" for a in accs]
    plt.barh(names, accs, color=colors, edgecolor="none", height=0.7)
    plt.axvline(x=np.mean(accs), color="black", linestyle="--",
                linewidth=1, label=f"Mean={np.mean(accs):.1f}%")
    plt.xlabel("Accuracy (%)")
    plt.title("Per-Class Accuracy (sorted)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy_bar.png"), dpi=80)
    plt.close()
    return rows


def _print_worst_best_classes(rows, n=5):
    print(f"\n    --- Bottom {n} classes ---")
    for name, n_s, acc in rows[:n]:
        print(f"      {name:<25s} n={n_s:4d}  acc={acc:.1f}%")
    print(f"    --- Top {n} classes ---")
    for name, n_s, acc in rows[-n:][::-1]:
        print(f"      {name:<25s} n={n_s:4d}  acc={acc:.1f}%")


def _save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        digits=4, zero_division=0,
    )
    with open(save_path, "w") as f:
        f.write(report)
    return report


def _top_k_accuracy(y_true, y_topk):
    correct = (y_topk == y_true[:, None]).any(axis=1).sum()
    return 100.0 * correct / max(len(y_true), 1)


# ============================================================
# DETAILED FOLD EVALUATION  (loads best_model.pt)
# ============================================================

def evaluate_fold_masa(fold_idx, fold_log_dir, test_paths,
                       label_encoder, class_names, device, config):
    """
    Load best_model.pt from fold_log_dir and run full evaluation.
    Saves: confusion matrix, top confused pairs, per-class accuracy,
           classification report, metrics.txt
    Returns metrics dict including y_true / y_pred.
    """
    eval_dir = os.path.join(fold_log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Dataset + loader
    test_set = MASADataset(
        test_paths, label_encoder,
        n=config["max_frames"], augment=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size  = config["batch_size"],
        shuffle     = False,
        num_workers = config["num_workers"],
        pin_memory  = True,
        collate_fn  = collate_fn_masa,
    )

    num_classes = len(class_names)

    model = MASAClassifier(
        feat_dim        = FEAT_DIM,
        num_classes     = num_classes,
        model_dim       = config["model_dim"],
        nhead           = config["nhead"],
        num_layers      = config["num_layers"],
        dim_feedforward = config["dim_feedforward"],
        dropout         = config["dropout"],
        drop_path_rate  = config["drop_path_rate"],
        mask_ratio      = config["mask_ratio"],
        recon_weight    = config["recon_weight"],
        decoder_layers  = config["decoder_layers"],
    ).to(device)

    ckpt_path = os.path.join(fold_log_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No best_model.pt at {ckpt_path}. "
            "train_one_fold must have completed this fold first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    # masa_train.py saves {"model_state": ..., "config": ...}
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred, y_topk, test_loss = _collect_masa_preds(
        model, test_loader, device, top_k=5,
        use_amp=config.get("use_amp", True),
    )

    top1  = (y_true == y_pred).mean() * 100.0
    top5  = _top_k_accuracy(y_true, y_topk)
    mac_p = precision_score(y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_r = recall_score(   y_true, y_pred, average="macro",    zero_division=0) * 100
    mac_f = f1_score(       y_true, y_pred, average="macro",    zero_division=0) * 100
    wtd_f = f1_score(       y_true, y_pred, average="weighted", zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    save_confusion_matrix(
        cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx}",
    )
    _save_top_confused_pairs(
        cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"),
        n=20,
    )
    rows = _save_per_class_accuracy(y_true, y_pred, class_names, eval_dir)
    _print_worst_best_classes(rows, n=5)
    _save_classification_report(
        y_true, y_pred, class_names,
        os.path.join(eval_dir, "classification_report.txt"),
    )

    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(f"Fold        : {fold_idx}\n")
        f.write(f"Top-1       : {top1:.2f}%\n")
        f.write(f"Top-5       : {top5:.2f}%\n")
        f.write(f"Macro P     : {mac_p:.2f}%\n")
        f.write(f"Macro R     : {mac_r:.2f}%\n")
        f.write(f"Macro F1    : {mac_f:.2f}%\n")
        f.write(f"Weighted F1 : {wtd_f:.2f}%\n")
        f.write(f"Test Loss   : {test_loss:.4f}\n")

    # Free the eval model
    del model, test_loader, test_set
    torch.cuda.empty_cache()

    return {
        "top1_acc"   : top1,
        "top5_acc"   : top5,
        "macro_p"    : mac_p,
        "macro_r"    : mac_r,
        "macro_f1"   : mac_f,
        "weighted_f1": wtd_f,
        "test_loss"  : test_loss,
        "y_true"     : y_true,
        "y_pred"     : y_pred,
    }


# ============================================================
# RQ2 CURVE PLOT
# ============================================================

def plot_rq2_curve(round_summaries, save_path):
    vids  = [s["actual_mean_per_word"] for s in round_summaries]
    top1s = [s["mean_top1"]            for s in round_summaries]
    top5s = [s["mean_top5"]            for s in round_summaries]
    f1s   = [s["mean_macro_f1"]        for s in round_summaries]
    stds  = [s["std_top1"]             for s in round_summaries]

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(vids, top1s, marker="o", color="steelblue",
            linewidth=2.5, markersize=9, label="Top-1 Accuracy")
    ax.fill_between(
        vids,
        [t - s for t, s in zip(top1s, stds)],
        [t + s for t, s in zip(top1s, stds)],
        color="steelblue", alpha=0.15, label="Top-1 ± 1 std (across folds)",
    )
    ax.plot(vids, top5s, marker="s", color="coral",
            linewidth=2.5, markersize=9, label="Top-5 Accuracy")
    ax.plot(vids, f1s,   marker="^", color="mediumseagreen",
            linewidth=2.5, markersize=9, label="Macro F1")

    for x, y in zip(vids, top1s):
        ax.annotate(
            f"{y:.1f}%", (x, y),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9,
            color="steelblue", fontweight="bold",
        )

    ax.set_xlabel(
        "Actual Mean Training Videos per Word "
        "(averaged across folds × words)",
        fontsize=12,
    )
    ax.set_ylabel("Accuracy / F1  (%)", fontsize=12)
    ax.set_title(
        "MASA RQ2 — Effect of Training Data Size on LOSO Recognition\n"
        "(Test set fixed per fold: held-out signer's full recordings)",
        fontsize=13,
    )
    ax.set_xticks(vids)
    ax.set_xticklabels([f"{v:.0f}" for v in vids], fontsize=10)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  RQ2 curve saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(config["seed"])
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Save config ───────────────────────────────────────────
    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump({
            **config,
            "CAP_VALUES" : args.cap_values,
            "DATA_ROOT"  : args.data_root,
            "RESULTS_DIR": args.results_dir,
        }, f, indent=4)

    # ── Logging ───────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(
        os.path.join(args.results_dir, f"rq2_log_{ts}.txt"), "w"
    )
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"  MASA RQ2 — Data Efficiency LOSO  |  {ts}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Cap values  : {args.cap_values}")
    print(f"  Epochs/fold : {config['epochs']}")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── Discover + group ──────────────────────────────────────
    all_paths = find_npy_files(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .npy files found under: {args.data_root}")
    print(f"\n  Total .npy files : {len(all_paths)}")

    user_word_paths = group_by_user_and_word(all_paths)
    users           = sorted(user_word_paths.keys())
    num_folds       = len(users)
    print(f"  Users ({num_folds}) : {users}")

    print(f"\n  Per-user recording stats:")
    print(f"  {'User':<12} {'Words':>6} {'MinReps':>8} {'MaxReps':>8} {'MeanReps':>9}")
    print(f"  {'-'*50}")
    for u in users:
        counts = [len(v) for v in user_word_paths[u].values()]
        print(f"  {u:<12} {len(counts):>6} {min(counts):>8} "
              f"{max(counts):>8} {np.mean(counts):>9.1f}")

    # ── Label encoder (global, across all users) ─────────────
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))
    num_classes = len(all_classes)
    label_encoder = LabelEncoder().fit(all_classes)
    np.save(os.path.join(args.results_dir, "class_names.npy"),
            np.array(label_encoder.classes_))
    print(f"\n  Word classes : {num_classes}")

    # ── LOSO fold assignments ─────────────────────────────────
    fold_assignments = []
    for i, test_user in enumerate(users):
        fold_assignments.append({
            "fold_idx"   : i,
            "test_user"  : test_user,
            "train_users": [u for u in users if u != test_user],
        })

    # Test paths per fold — fixed, never subsampled
    test_paths_per_fold    = {}
    missing_words_per_fold = {}

    print(f"\n  {'Fold':<6} {'Test User':<12} {'Test Files':>10}  {'Missing Words':>14}")
    print(f"  {'-'*50}")
    for fa in fold_assignments:
        fi = fa["fold_idx"]
        test_paths_per_fold[fi] = [
            p for word_paths in user_word_paths[fa["test_user"]].values()
            for p in word_paths
        ]
        test_labels = set(
            label_from_filename(p) for p in test_paths_per_fold[fi]
        )
        missing_words_per_fold[fi] = sorted(set(all_classes) - test_labels)
        print(f"  {fi:<6} {fa['test_user']:<12} "
              f"{len(test_paths_per_fold[fi]):>10}  "
              f"{len(missing_words_per_fold[fi]):>14}")

    # ── Test coverage CSV (once) ─────────────────────────────
    cov_csv = os.path.join(args.results_dir, "test_coverage.csv")
    with open(cov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "test_user", "test_files",
                    "n_missing_words", "missing_words"])
        for fa in fold_assignments:
            fi = fa["fold_idx"]
            w.writerow([
                fi, fa["test_user"],
                len(test_paths_per_fold[fi]),
                len(missing_words_per_fold[fi]),
                "|".join(missing_words_per_fold[fi]),
            ])
    print(f"\n  Test coverage saved: {cov_csv}")

    # ── Global CSV header ────────────────────────────────────
    global_csv = os.path.join(args.results_dir, "rq2_summary.csv")
    if not os.path.isfile(global_csv):
        with open(global_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "cap",
                "nominal_train_per_word",
                "actual_mean_per_word",
                "actual_min_per_word",
                "actual_max_per_word",
                "mean_top1(%)", "std_top1(%)",
                "mean_top5(%)",
                "mean_macro_f1(%)", "std_macro_f1(%)",
                "mean_weighted_f1(%)",
            ])

    # ============================================================
    # CAP SWEEP
    # ============================================================
    round_summaries = []
    overall_start   = time.time()

    for round_idx, cap in enumerate(args.cap_values):
        if round_idx < args.start_from_round:
            print(f"\n  ⏭  Skipping round {round_idx + 1} (cap={cap}) — before start_from_round.")
            continue

        nominal_per_word = (num_folds - 1) * cap
        round_dir        = os.path.join(args.results_dir, f"cap_{cap}")
        os.makedirs(round_dir, exist_ok=True)

        print(f"\n\n{'#'*70}")
        print(f"  ROUND {round_idx + 1}/{len(args.cap_values)}  "
              f"|  cap={cap}  |  nominal train/word={nominal_per_word}")
        print(f"  Output folder : {round_dir}")
        print(f"{'#'*70}")

        fold_results = []
        round_start  = time.time()

        for fa in fold_assignments:
            fold_idx    = fa["fold_idx"]
            test_user   = fa["test_user"]
            train_users = fa["train_users"]
            fold_log_dir = os.path.join(round_dir, f"fold_{fold_idx}")

            # ── Sample with cap ──────────────────────────────
            train_paths, stats = sample_train_paths(
                train_users     = train_users,
                user_word_paths = user_word_paths,
                cap             = cap,
                round_idx       = round_idx,
                fold_idx        = fold_idx,
                base_seed       = config["seed"],
            )
            test_paths = test_paths_per_fold[fold_idx]

            print(f"\n  Fold {fold_idx} | test={test_user} | cap={cap}"
                  f" | train_files={stats['total_files']} | test_files={len(test_paths)}")
            print(f"    Train vids/word — nominal={nominal_per_word} "
                  f"| actual mean={stats['mean_per_word']:.1f} "
                  f"min={stats['min_per_word']} max={stats['max_per_word']} "
                  f"| words_at_nominal={stats['words_at_nominal']} "
                  f"below={stats['words_below']}")

            # ── Train ────────────────────────────────────────
            fold_start = time.time()
            _ = train_one_fold(
                fold_idx      = fold_idx,
                train_paths   = train_paths,
                test_paths    = test_paths,
                label_encoder = label_encoder,
                class_names   = list(label_encoder.classes_),
                fold_log_dir  = fold_log_dir,
                device        = device,
                config        = config,
                tb_root       = os.path.join(args.results_dir, "runs",
                                             f"cap_{cap}"),
                dry_run       = False,
            )
            fold_train_time = time.time() - fold_start

            # ── Detailed eval from best checkpoint ───────────
            print(f"\n  Evaluating fold {fold_idx} from best_model.pt …")
            em = evaluate_fold_masa(
                fold_idx      = fold_idx,
                fold_log_dir  = fold_log_dir,
                test_paths    = test_paths,
                label_encoder = label_encoder,
                class_names   = list(label_encoder.classes_),
                device        = device,
                config        = config,
            )

            # ── Per-user accuracy CSV ────────────────────────
            user_y_true = defaultdict(list)
            user_y_pred = defaultdict(list)
            for path, yt, yp in zip(test_paths, em["y_true"], em["y_pred"]):
                u = user_from_path(path)
                user_y_true[u].append(yt); user_y_pred[u].append(yp)

            user_csv = os.path.join(fold_log_dir, "per_user_accuracy.csv")
            with open(user_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["user", "n_samples", "top1_accuracy(%)"])
                for u in sorted(user_y_true.keys()):
                    yt = np.array(user_y_true[u])
                    yp = np.array(user_y_pred[u])
                    w.writerow([u, len(yt),
                                f"{(yt == yp).mean()*100.0:.2f}"])

            fold_results.append({
                "fold"                  : fold_idx,
                "test_user"             : test_user,
                "cap"                   : cap,
                "train_files"           : stats["total_files"],
                "actual_mean_per_word"  : stats["mean_per_word"],
                "actual_min_per_word"   : stats["min_per_word"],
                "actual_max_per_word"   : stats["max_per_word"],
                "words_at_nominal"     : stats["words_at_nominal"],
                "words_below"           : stats["words_below"],
                "missing_words_in_test" : len(missing_words_per_fold[fold_idx]),
                "top1_acc"              : em["top1_acc"],
                "top5_acc"              : em["top5_acc"],
                "macro_f1"              : em["macro_f1"],
                "weighted_f1"           : em["weighted_f1"],
                "test_loss"             : em["test_loss"],
                "y_true"                : em["y_true"],
                "y_pred"                : em["y_pred"],
                "train_time_s"          : fold_train_time,
            })

            total_elapsed = time.time() - overall_start
            print(f"  ⏱  Fold {fold_idx} train+eval: {fold_train_time/60:.1f} min  "
                  f"|  Cumulative: {total_elapsed/3600:.2f} h")

        # ── Round aggregation ─────────────────────────────────
        top1s    = [r["top1_acc"]    for r in fold_results]
        top5s    = [r["top5_acc"]    for r in fold_results]
        mac_f1s  = [r["macro_f1"]    for r in fold_results]
        wtd_f1s  = [r["weighted_f1"] for r in fold_results]
        act_mean = [r["actual_mean_per_word"] for r in fold_results]
        act_min  = [r["actual_min_per_word"]  for r in fold_results]
        act_max  = [r["actual_max_per_word"]  for r in fold_results]

        mean_top1   = float(np.mean(top1s));    std_top1   = float(np.std(top1s))
        mean_top5   = float(np.mean(top5s))
        mean_mac_f1 = float(np.mean(mac_f1s));  std_mac_f1 = float(np.std(mac_f1s))
        mean_wtd_f1 = float(np.mean(wtd_f1s))
        global_mean = float(np.mean(act_mean))
        global_min  = int(np.min(act_min))
        global_max  = int(np.max(act_max))

        round_time = time.time() - round_start
        print(f"\n  ── Round {round_idx + 1} Summary (cap={cap}) "
              f"— took {round_time/3600:.2f} h ──")
        print(f"  Mean Top-1    : {mean_top1:.2f}% ± {std_top1:.2f}%")
        print(f"  Mean Top-5    : {mean_top5:.2f}%")
        print(f"  Mean Macro F1 : {mean_mac_f1:.2f}%")
        print(f"  Actual mean train vids/word : {global_mean:.1f}")

        print(f"\n  {'Fold':<6} {'Test User':<12} "
              f"{'Top-1':>8} {'Top-5':>8} {'MacroF1':>9}")
        print(f"  {'-'*50}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['test_user']:<12} "
                  f"{r['top1_acc']:7.2f}%  "
                  f"{r['top5_acc']:7.2f}%  "
                  f"{r['macro_f1']:8.2f}%")

        # ── Round CSV ─────────────────────────────────────────
        round_csv = os.path.join(round_dir, "round_summary.csv")
        with open(round_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "fold", "test_user",
                "top1(%)", "top5(%)",
                "macro_f1(%)", "weighted_f1(%)",
                "train_files",
                "actual_mean_per_word",
                "actual_min_per_word",
                "actual_max_per_word",
                "words_at_nominal",
                "words_below",
                "missing_words_in_test",
                "train_time_min",
            ])
            for r in fold_results:
                w.writerow([
                    r["fold"], r["test_user"],
                    f"{r['top1_acc']:.2f}",
                    f"{r['top5_acc']:.2f}",
                    f"{r['macro_f1']:.2f}",
                    f"{r['weighted_f1']:.2f}",
                    r["train_files"],
                    f"{r['actual_mean_per_word']:.1f}",
                    r["actual_min_per_word"],
                    r["actual_max_per_word"],
                    r["words_at_nominal"],
                    r["words_below"],
                    r["missing_words_in_test"],
                    f"{r['train_time_s']/60:.1f}",
                ])
            w.writerow([])
            w.writerow([
                "MEAN", "—",
                f"{mean_top1:.2f}", f"{mean_top5:.2f}",
                f"{mean_mac_f1:.2f}", f"{mean_wtd_f1:.2f}",
                "—", f"{global_mean:.1f}", "—", "—", "—", "—", "—", "—",
            ])
            w.writerow([
                "STD", "—",
                f"{std_top1:.2f}", "—",
                f"{std_mac_f1:.2f}", "—",
                "—", "—", "—", "—", "—", "—", "—", "—",
            ])
        print(f"  Round CSV : {round_csv}")

        # ── Append to global summary ──────────────────────────
        with open(global_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                round_idx + 1, cap,
                nominal_per_word,
                f"{global_mean:.1f}",
                global_min, global_max,
                f"{mean_top1:.2f}", f"{std_top1:.2f}",
                f"{mean_top5:.2f}",
                f"{mean_mac_f1:.2f}", f"{std_mac_f1:.2f}",
                f"{mean_wtd_f1:.2f}",
            ])

        round_summaries.append({
            "round"               : round_idx + 1,
            "cap"                 : cap,
            "nominal_per_word"    : nominal_per_word,
            "actual_mean_per_word": global_mean,
            "mean_top1"           : mean_top1,
            "std_top1"            : std_top1,
            "mean_top5"           : mean_top5,
            "mean_macro_f1"       : mean_mac_f1,
            "mean_weighted_f1"    : mean_wtd_f1,
        })

    # ============================================================
    # FINAL SUMMARY + CURVE
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  MASA RQ2 COMPLETE — FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  {'Cap':<5} {'Nominal':>8} {'Actual':>8}  "
          f"{'Top-1':>9} {'±std':>7}  {'Top-5':>9}  {'MacroF1':>9}")
    print(f"  {'-'*70}")
    for s in round_summaries:
        print(
            f"  {s['cap']:<5} {s['nominal_per_word']:>8} "
            f"{s['actual_mean_per_word']:>8.1f}  "
            f"{s['mean_top1']:8.2f}%  ±{s['std_top1']:5.2f}%  "
            f"{s['mean_top5']:8.2f}%  {s['mean_macro_f1']:8.2f}%"
        )

    if round_summaries:
        plot_rq2_curve(
            round_summaries,
            save_path=os.path.join(args.results_dir, "rq2_accuracy_curve.png"),
        )

    total = time.time() - overall_start
    print(f"\n  Global summary CSV : {global_csv}")
    print(f"  Total wall time    : {total/3600:.2f} h")
    print(f"  All results in     : {args.results_dir}/")
    print(f"  TensorBoard        : tensorboard --logdir={args.results_dir}/runs")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MASA RQ2 — Data Efficiency under LOSO"
    )
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument(
        "--cap_values", type=int, nargs="+", default=CAP_VALUES,
        help="Cap sequence (descending), e.g. --cap_values 10 8 6 4 2 1",
    )
    parser.add_argument(
        "--start_from_round", type=int, default=START_FROM_ROUND,
        help="Resume: skip rounds with index < this (0-based)",
    )
    args = parser.parse_args()
    main(args)