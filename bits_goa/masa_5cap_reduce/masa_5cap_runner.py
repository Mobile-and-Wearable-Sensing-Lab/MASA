"""
masa_5cap_runner.py — MASA RQ2 v2 (MULTI-GPU: GPUs 5, 6, 7)
=========================================================================
Luck-check version of RQ2 with rotating windows.
Runs 3 LOSO folds in parallel on GPUs 5, 6, 7.

Total runs: 5 caps × 3 rotations × 15 folds = 225 runs
Expected time: ~56 hours on 3 GPUs (vs ~168 hours on 1 GPU)
"""

import os
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES globally — workers set it per-process

import sys
import csv
import json
import random
import argparse
import time
import numpy as np
import torch
import multiprocessing as mp
import multiprocessing.pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# NON-DAEMONIC POOL — fixes DataLoader's num_workers issue
# (Pool workers need to spawn children for torch DataLoader)
# ============================================================

class NoDaemonProcess(mp.Process):
    """Process that allows children (non-daemonic)."""
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context("spawn"))):
    Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
    """Pool whose workers can spawn their own children (DataLoader support)."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)

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

DATA_ROOT   = "/home/pdf2024018/masa/Pose"
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
)

CAP_START = 5
CAP_END   = 1
ROTATIONS = 3

# ---- MULTI-GPU CONFIG ----
GPU_IDS = [5, 6, 7]                  # <── Use GPUs 5, 6, 7
NUM_PARALLEL_FOLDS = len(GPU_IDS)    # 3 folds at a time

# Resume support
START_FROM_CAP      = 0
START_FROM_ROTATION = 0

config = {
    "max_frames"      : 150,
    "num_workers"     : 4,           # ↓ lowered (3 processes share CPU)

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
    uwp = defaultdict(lambda: defaultdict(list))
    for p in paths:
        uwp[user_from_path(p)][label_from_filename(p)].append(p)
    # Convert to plain dicts for pickling
    return {u: {w: sorted(v, key=os.path.basename) for w, v in wp.items()}
            for u, wp in uwp.items()}


# ============================================================
# WINDOW-BASED SUBSAMPLING
# ============================================================

def sample_train_paths_windowed(train_users, user_word_paths, cap,
                                rotation_round, fold_idx, base_seed):
    selected        = []
    per_word_counts = defaultdict(int)
    wrap_events     = []

    for user in sorted(train_users):
        for word in sorted(user_word_paths[user].keys()):
            paths = user_word_paths[user][word]
            n     = len(paths)
            if n == 0: continue
            window_start = rotation_round * cap
            window_end   = window_start + cap

            if window_end <= n:
                chosen = paths[window_start:window_end]
            else:
                indices = [(window_start + i) % n for i in range(cap)]
                chosen  = [paths[i] for i in indices]
                wrap_events.append({
                    "user": user, "word": word, "n_available": n,
                    "cap": cap, "rotation": rotation_round,
                    "window_start": window_start, "window_end": window_end,
                    "indices_used": indices,
                })

            selected.extend(chosen)
            per_word_counts[word] += len(chosen)

    counts = list(per_word_counts.values()) if per_word_counts else [0]
    stats = {
        "total_files"     : len(selected),
        "mean_per_word"   : float(np.mean(counts)),
        "min_per_word"    : int(np.min(counts)),
        "max_per_word"    : int(np.max(counts)),
        "n_wrap_events"   : len(wrap_events),
        "words_at_nominal": int(sum(1 for c in counts if c == cap * len(train_users))),
    }
    return selected, stats, wrap_events


# ============================================================
# EVALUATION HELPERS
# ============================================================

@torch.no_grad()
def _collect_masa_preds(model, loader, device, top_k=5, use_amp=True):
    import torch.nn.functional as F
    model.eval()
    y_true, y_pred, y_topk = [], [], []
    total_loss, total_n = 0.0, 0

    for s1, s2, s3, labels, lengths, padding_mask in loader:
        s1 = s1.to(device); s2 = s2.to(device); s3 = s3.to(device)
        labels = labels.to(device); padding_mask = padding_mask.to(device)

        with autocast(enabled=use_amp and torch.cuda.is_available()):
            logits, _ = model(s1, s2, s3, padding_mask)
            loss = F.cross_entropy(logits, labels, reduction="sum")

        preds = logits.argmax(dim=1)
        k = min(top_k, logits.size(1))
        topk = logits.topk(k, dim=1).indices

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_topk.append(topk.cpu().numpy())
        total_loss += loss.item(); total_n += labels.size(0)

    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(y_topk, axis=0), total_loss / max(total_n, 1))


def _top_k_accuracy(y_true, y_topk):
    correct = (y_topk == y_true[:, None]).any(axis=1).sum()
    return 100.0 * correct / max(len(y_true), 1)


def _save_top_confused_pairs(cm, class_names, save_path_png, n=20):
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2]); pairs = pairs[:n]

    csv_path = save_path_png.replace(".png", ".csv")
    with open(csv_path, "w") as f:
        f.write("true_class,predicted_as,count\n")
        for t, p, c in pairs: f.write(f"{t},{p},{c}\n")

    if pairs:
        labels = [f"{t}→{p}" for t, p, _ in pairs]
        counts = [c for _, _, c in pairs]
        plt.figure(figsize=(14, 6))
        plt.barh(labels[::-1], counts[::-1], color="steelblue", edgecolor="black")
        plt.xlabel("Count"); plt.title(f"Top {n} Most Confused Class Pairs")
        plt.tight_layout(); plt.savefig(save_path_png, dpi=100); plt.close()


def _save_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    rows = []
    for i, name in enumerate(class_names):
        mask = (y_true == i); n_samples = int(mask.sum())
        acc = (y_pred[mask] == i).mean() * 100.0 if n_samples > 0 else 0.0
        rows.append((name, n_samples, acc))
    rows.sort(key=lambda x: x[2])

    with open(os.path.join(save_dir, "per_class_accuracy.csv"), "w") as f:
        f.write("class,n_samples,accuracy(%)\n")
        for name, n, acc in rows: f.write(f"{name},{n},{acc:.2f}\n")

    names = [r[0] for r in rows]; accs = [r[2] for r in rows]
    fig_h = max(8, len(names) * 0.18)
    plt.figure(figsize=(12, fig_h))
    colors = ["#d73027" if a < 50 else "#4575b4" for a in accs]
    plt.barh(names, accs, color=colors, edgecolor="none", height=0.7)
    plt.axvline(x=np.mean(accs), color="black", linestyle="--",
                linewidth=1, label=f"Mean={np.mean(accs):.1f}%")
    plt.xlabel("Accuracy (%)"); plt.title("Per-Class Accuracy (sorted)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy_bar.png"), dpi=80)
    plt.close()
    return rows


def _save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names,
                                    digits=4, zero_division=0)
    with open(save_path, "w") as f: f.write(report)
    return report


# ============================================================
# DETAILED FOLD EVALUATION
# ============================================================

def evaluate_fold_masa(fold_idx, fold_log_dir, test_paths,
                       label_encoder, class_names, device, config):
    eval_dir = os.path.join(fold_log_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    test_set = MASADataset(test_paths, label_encoder,
                           n=config["max_frames"], augment=False)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
        collate_fn=collate_fn_masa,
    )

    num_classes = len(class_names)
    model = MASAClassifier(
        feat_dim=FEAT_DIM, num_classes=num_classes,
        model_dim=config["model_dim"], nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"], drop_path_rate=config["drop_path_rate"],
        mask_ratio=config["mask_ratio"], recon_weight=config["recon_weight"],
        decoder_layers=config["decoder_layers"],
    ).to(device)

    ckpt_path = os.path.join(fold_log_dir, "best_model.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No best_model.pt at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state); model.eval()

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
    save_confusion_matrix(cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx}")
    _save_top_confused_pairs(cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"), n=20)
    _save_per_class_accuracy(y_true, y_pred, class_names, eval_dir)
    _save_classification_report(y_true, y_pred, class_names,
        os.path.join(eval_dir, "classification_report.txt"))

    with open(os.path.join(eval_dir, "metrics.txt"), "w") as f:
        f.write(f"Fold        : {fold_idx}\n")
        f.write(f"Top-1       : {top1:.2f}%\n")
        f.write(f"Top-5       : {top5:.2f}%\n")
        f.write(f"Macro P     : {mac_p:.2f}%\n")
        f.write(f"Macro R     : {mac_r:.2f}%\n")
        f.write(f"Macro F1    : {mac_f:.2f}%\n")
        f.write(f"Weighted F1 : {wtd_f:.2f}%\n")
        f.write(f"Test Loss   : {test_loss:.4f}\n")

    del model, test_loader, test_set
    torch.cuda.empty_cache()

    return {
        "top1_acc": top1, "top5_acc": top5,
        "macro_f1": mac_f, "weighted_f1": wtd_f,
        "test_loss": test_loss,
        "y_true": y_true, "y_pred": y_pred,
    }


# ============================================================
# WORKER: train + eval ONE fold on ONE GPU
# ============================================================

def fold_train_and_eval_worker(task):
    """Runs train + eval for ONE (cap, rotation, fold) on ONE GPU."""
    (fold_idx, test_user, train_users, cap, rotation_round,
     user_word_paths, test_paths, label_encoder_classes,
     fold_log_dir, tb_root, worker_config, gpu_id) = task

    # Pin to GPU BEFORE torch CUDA init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from sklearn.preprocessing import LabelEncoder
    from masa_train import train_one_fold

    # Per-run deterministic seed
    seed = worker_config["seed"] + cap * 10000 + rotation_round * 1000 + fold_idx
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    label_encoder = LabelEncoder().fit(label_encoder_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cuda:0 == physical gpu_id (because of CUDA_VISIBLE_DEVICES pinning)

    # Sample training paths
    train_paths, sample_stats, wrap_events = sample_train_paths_windowed(
        train_users=train_users, user_word_paths=user_word_paths,
        cap=cap, rotation_round=rotation_round, fold_idx=fold_idx,
        base_seed=worker_config["seed"],
    )

    print(f"[GPU {gpu_id}] ▶ Cap={cap} Rot={rotation_round} "
          f"Fold={fold_idx} ({test_user}) | "
          f"train={sample_stats['total_files']} test={len(test_paths)} "
          f"wraps={sample_stats['n_wrap_events']}", flush=True)

    # --- TRAIN ---
    t0 = time.time()
    _ = train_one_fold(
        fold_idx=fold_idx, train_paths=train_paths, test_paths=test_paths,
        label_encoder=label_encoder, class_names=list(label_encoder.classes_),
        fold_log_dir=fold_log_dir, device=device, config=worker_config,
        tb_root=tb_root, dry_run=False,
    )
    fold_train_time = time.time() - t0

    # --- EVAL from best_model.pt ---
    em = evaluate_fold_masa(
        fold_idx=fold_idx, fold_log_dir=fold_log_dir, test_paths=test_paths,
        label_encoder=label_encoder, class_names=list(label_encoder.classes_),
        device=device, config=worker_config,
    )

    print(f"[GPU {gpu_id}] ✔ Cap={cap} Rot={rotation_round} "
          f"Fold={fold_idx} done in {fold_train_time/60:.1f} min | "
          f"Top-1={em['top1_acc']:.2f}% Top-5={em['top5_acc']:.2f}% "
          f"MacroF1={em['macro_f1']:.2f}%", flush=True)

    return {
        "cap": cap, "rotation_round": rotation_round,
        "fold_idx": fold_idx, "test_user": test_user,
        "train_files": sample_stats["total_files"],
        "actual_mean_per_word": sample_stats["mean_per_word"],
        "n_wrap_events": sample_stats["n_wrap_events"],
        "wrap_events_detail": wrap_events,
        "top1_acc": em["top1_acc"], "top5_acc": em["top5_acc"],
        "macro_f1": em["macro_f1"], "weighted_f1": em["weighted_f1"],
        "test_loss": em["test_loss"],
        "fold_time_s": fold_train_time,
        "gpu_id": gpu_id,
    }


# ============================================================
# PLOTTING FUNCTIONS (unchanged from original)
# ============================================================

def plot_main_curve(cap_summaries, save_path):
    caps  = [s["cap"]           for s in cap_summaries]
    top1s = [s["mean_top1"]     for s in cap_summaries]
    top5s = [s["mean_top5"]     for s in cap_summaries]
    f1s   = [s["mean_macro_f1"] for s in cap_summaries]
    stds  = [s["std_top1"]      for s in cap_summaries]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(caps, top1s, "o-", color="#2563EB", lw=2.5, ms=8,
            label="Top-1 Accuracy (avg over rotations)")
    ax.fill_between(caps,
        [t - s for t, s in zip(top1s, stds)],
        [t + s for t, s in zip(top1s, stds)],
        color="#2563EB", alpha=0.15,
        label="Top-1 ± 1 std (folds × rotations)")
    ax.plot(caps, top5s, "s-", color="#DC2626", lw=2.5, ms=8, label="Top-5 Accuracy")
    ax.plot(caps, f1s,   "^-", color="#16A34A", lw=2.5, ms=8, label="Macro F1")

    for x, y in zip(caps, top1s):
        ax.annotate(f"{y:.1f}%", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9, color="#2563EB", fontweight="bold")

    ax.set_xlabel("Cap (videos per word per user)", fontsize=12)
    ax.set_ylabel("Accuracy / F1 (%)", fontsize=12)
    ax.set_title(
        "MASA RQ2 v2 — Effect of Training Data Size on LOSO (Luck Check)\n"
        "(Mean over 3 rotation rounds × 15 LOSO folds)", fontsize=13)
    ax.set_xticks(caps); ax.set_ylim(0, 110)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Main curve saved : {save_path}")


def plot_per_rotation_curves(all_rotation_summaries, save_path):
    rotations = sorted(set(s["rotation_round"] for s in all_rotation_summaries))
    caps_all  = sorted(set(s["cap"] for s in all_rotation_summaries), reverse=True)
    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#F59E0B"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for rot_idx, rot in enumerate(rotations):
        rot_data = sorted(
            [s for s in all_rotation_summaries if s["rotation_round"] == rot],
            key=lambda x: x["cap"], reverse=True)
        caps  = [s["cap"]           for s in rot_data]
        top1s = [s["mean_top1"]     for s in rot_data]
        f1s   = [s["mean_macro_f1"] for s in rot_data]
        c = colors[rot_idx % len(colors)]
        axes[0].plot(caps, top1s, "o-", color=c, lw=2, ms=7, label=f"Rotation {rot}")
        axes[1].plot(caps, f1s,   "o-", color=c, lw=2, ms=7, label=f"Rotation {rot}")

    for ax, metric in zip(axes, ["Top-1 Accuracy (%)", "Macro F1 (%)"]):
        ax.set_xlabel("Cap (videos per word per user)", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks(caps_all); ax.set_ylim(0, 110)
        ax.legend(fontsize=10); ax.grid(alpha=0.3)

    axes[0].set_title("Top-1 per Rotation Round", fontsize=12)
    axes[1].set_title("Macro F1 per Rotation Round", fontsize=12)
    fig.suptitle("MASA RQ2 v2 — Per-Rotation Performance\n"
                 "(Each rotation = different non-overlapping video window)",
                 fontsize=13)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Per-rotation curve saved : {save_path}")


def plot_boxplot(all_fold_results, save_path):
    cap_values = sorted(set(r["cap"] for r in all_fold_results), reverse=True)
    data_top1 = [[r["top1_acc"] for r in all_fold_results if r["cap"] == cap] for cap in cap_values]
    data_f1   = [[r["macro_f1"] for r in all_fold_results if r["cap"] == cap] for cap in cap_values]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bp = axes[0].boxplot(data_top1, labels=[str(c) for c in cap_values],
                          patch_artist=True,
                          medianprops=dict(color="black", linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
    for patch, color in zip(bp["boxes"], plt.cm.Blues(np.linspace(0.4, 0.85, len(cap_values)))):
        patch.set_facecolor(color)
    axes[0].set_xlabel("Cap (videos per word per user)", fontsize=11)
    axes[0].set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    axes[0].set_title(f"Top-1 Distribution per Cap\n"
                      f"({len(data_top1[0])} points each: rotations × folds)", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3); axes[0].set_ylim(0, 105)

    bp2 = axes[1].boxplot(data_f1, labels=[str(c) for c in cap_values],
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=2),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
    for patch, color in zip(bp2["boxes"], plt.cm.Greens(np.linspace(0.4, 0.85, len(cap_values)))):
        patch.set_facecolor(color)
    axes[1].set_xlabel("Cap (videos per word per user)", fontsize=11)
    axes[1].set_ylabel("Macro F1 (%)", fontsize=11)
    axes[1].set_title("Macro F1 Distribution per Cap", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3); axes[1].set_ylim(0, 105)

    fig.suptitle("MASA RQ2 v2 — Score Distributions Across Folds & Rotations", fontsize=13)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Boxplot saved : {save_path}")


def plot_heatmap(all_rotation_summaries, save_path):
    caps      = sorted(set(s["cap"] for s in all_rotation_summaries), reverse=True)
    rotations = sorted(set(s["rotation_round"] for s in all_rotation_summaries))
    matrix = np.zeros((len(caps), len(rotations)))
    for i, cap in enumerate(caps):
        for j, rot in enumerate(rotations):
            match = [s for s in all_rotation_summaries
                     if s["cap"] == cap and s["rotation_round"] == rot]
            if match: matrix[i, j] = match[0]["mean_top1"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto",
                   vmin=max(0, matrix.min()-5), vmax=min(100, matrix.max()+5))
    plt.colorbar(im, ax=ax, label="Mean Top-1 Accuracy (%)")
    ax.set_xticks(range(len(rotations)))
    ax.set_xticklabels([f"Rotation {r}" for r in rotations], fontsize=11)
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels([f"Cap={c}" for c in caps], fontsize=11)
    thresh = matrix.min() + (matrix.max() - matrix.min()) * 0.5
    for i in range(len(caps)):
        for j in range(len(rotations)):
            ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center",
                    fontsize=11, color="black" if matrix[i,j] > thresh else "white")
    ax.set_title("MASA RQ2 v2 — Top-1 Heatmap: Cap × Rotation\n"
                 "(Each cell = mean over 15 LOSO folds)", fontsize=13)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Heatmap saved : {save_path}")


def plot_fold_variance(all_fold_results, save_path):
    folds     = sorted(set(r["fold_idx"] for r in all_fold_results))
    cap_values = sorted(set(r["cap"] for r in all_fold_results), reverse=True)
    colors    = plt.cm.tab20(np.linspace(0, 1, len(folds)))

    fig, ax = plt.subplots(figsize=(14, 6))
    for fi, fold in enumerate(folds):
        fold_data = [r for r in all_fold_results if r["fold_idx"] == fold]
        cap_means = []
        for cap in cap_values:
            vals = [r["top1_acc"] for r in fold_data if r["cap"] == cap]
            cap_means.append(np.mean(vals) if vals else float("nan"))
        test_user = next(r["test_user"] for r in fold_data)
        label = f"Fold {fold} ({test_user[:12]})"
        ax.plot(cap_values, cap_means, "o-", color=colors[fi], lw=1.5,
                ms=5, label=label, alpha=0.8)

    ax.set_xlabel("Cap (videos per word per user)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%, avg over rotations)", fontsize=12)
    ax.set_title("MASA RQ2 v2 — Per-Fold Accuracy Across Caps\n"
                 "(Reveals which signers are consistently hard to recognize)",
                 fontsize=13)
    ax.set_xticks(cap_values); ax.set_ylim(0, 110)
    ax.legend(fontsize=7, ncol=3, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Fold variance plot saved : {save_path}")


def plot_improvement_curve(cap_summaries, save_path):
    caps  = [s["cap"]       for s in cap_summaries]
    top1s = [s["mean_top1"] for s in cap_summaries]
    sorted_pairs = sorted(zip(caps, top1s))
    sorted_caps, sorted_top1s = zip(*sorted_pairs)
    deltas = [sorted_top1s[i] - sorted_top1s[i-1] for i in range(1, len(sorted_top1s))]
    delta_caps = sorted_caps[1:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(sorted_caps, sorted_top1s, "o-", color="#2563EB", lw=2.5, ms=8)
    for x, y in zip(sorted_caps, sorted_top1s):
        axes[0].annotate(f"{y:.1f}%", (x, y),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9, color="#2563EB")
    axes[0].set_xlabel("Cap (videos per word per user)", fontsize=11)
    axes[0].set_ylabel("Mean Top-1 Accuracy (%)", fontsize=11)
    axes[0].set_title("Absolute Performance vs Cap", fontsize=12)
    axes[0].set_xticks(sorted_caps); axes[0].set_ylim(0, 110); axes[0].grid(alpha=0.3)

    bar_colors = ["#16A34A" if d >= 0 else "#DC2626" for d in deltas]
    axes[1].bar([str(c) for c in delta_caps], deltas, color=bar_colors,
                edgecolor="black", linewidth=0.8)
    for i, (x, y) in enumerate(zip(delta_caps, deltas)):
        axes[1].annotate(f"{y:+.1f}%", (i, y),
                         textcoords="offset points",
                         xytext=(0, 5 if y >= 0 else -15),
                         ha="center", fontsize=10, fontweight="bold")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Cap (transition from cap-1)", fontsize=11)
    axes[1].set_ylabel("ΔTop-1 Accuracy (%)", fontsize=11)
    axes[1].set_title("Marginal Gain per Additional Video", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("MASA RQ2 v2 — Diminishing Returns Analysis", fontsize=13)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Improvement curve saved : {save_path}")


def plot_wrap_coverage(window_coverage_data, save_path):
    if not window_coverage_data: return
    caps      = sorted(set(d["cap"] for d in window_coverage_data), reverse=True)
    rotations = sorted(set(d["rotation_round"] for d in window_coverage_data))
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    colors = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#F59E0B"]
    x = np.arange(len(caps))

    for ri, rot in enumerate(rotations):
        counts = []
        for cap in caps:
            match = [d for d in window_coverage_data
                     if d["cap"] == cap and d["rotation_round"] == rot]
            counts.append(match[0]["total_wrap_events"] if match else 0)
        ax.bar(x + ri * width, counts, width, label=f"Rotation {rot}",
               color=colors[ri % len(colors)], edgecolor="black", linewidth=0.7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Cap={c}" for c in caps], fontsize=11)
    ax.set_xlabel("Cap", fontsize=11)
    ax.set_ylabel("Total Cyclic-Wrap Events\n(summed over all folds)", fontsize=11)
    ax.set_title("MASA RQ2 v2 — Window Wrap Events per (Cap, Rotation)\n"
                 "(Wrapping = user has fewer videos than window requires)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=130); plt.close()
    print(f"  Wrap coverage plot saved : {save_path}")


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(config["seed"])
    os.makedirs(args.results_dir, exist_ok=True)

    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump({
            **config,
            "CAP_START": args.cap_start, "CAP_END": args.cap_end,
            "ROTATIONS": args.rotations,
            "GPU_IDS": args.gpu_ids,
            "NUM_PARALLEL_FOLDS": len(args.gpu_ids),
            "DATA_ROOT": args.data_root, "RESULTS_DIR": args.results_dir,
            "START_FROM_CAP": args.start_from_cap,
            "START_FROM_ROTATION": args.start_from_rotation,
        }, f, indent=4)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(args.results_dir, f"rq2v2_log_{ts}.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    cap_values = list(range(args.cap_start, args.cap_end - 1, -1))
    total_runs = len(cap_values) * args.rotations * 15

    print("=" * 70)
    print(f"  MASA RQ2 v2 — LUCK CHECK (MULTI-GPU) | {ts}")
    print(f"  Data root    : {args.data_root}")
    print(f"  Results dir  : {args.results_dir}")
    print(f"  Caps         : {cap_values}")
    print(f"  Rotations    : {args.rotations} per cap")
    print(f"  Folds        : 15")
    print(f"  Epochs/run   : {config['epochs']}")
    print(f"  Total runs   : {total_runs}")
    print(f"  GPU IDs      : {args.gpu_ids}  (parallel = {len(args.gpu_ids)})")
    print("=" * 70)

    n_gpus_visible = torch.cuda.device_count()
    print(f"\n  System visible GPUs: {n_gpus_visible}")
    for i in range(n_gpus_visible):
        print(f"    cuda:{i} → {torch.cuda.get_device_name(i)}")

    all_paths = find_npy_files(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .npy files under {args.data_root}")
    print(f"\n  Total .npy files : {len(all_paths)}")

    user_word_paths = group_by_user_and_word(all_paths)
    users     = sorted(user_word_paths.keys())
    num_folds = len(users)
    print(f"  Users ({num_folds}) : {users}")

    print(f"\n  Per-user recording stats:")
    print(f"  {'User':<12} {'Words':>6} {'MinReps':>8} {'MaxReps':>8} {'MeanReps':>9}")
    print(f"  {'-'*50}")
    for u in users:
        counts = [len(v) for v in user_word_paths[u].values()]
        print(f"  {u:<12} {len(counts):>6} {min(counts):>8} "
              f"{max(counts):>8} {np.mean(counts):>9.1f}")

    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))
    num_classes = len(all_classes)
    label_encoder = LabelEncoder().fit(all_classes)
    np.save(os.path.join(args.results_dir, "class_names.npy"),
            np.array(label_encoder.classes_))
    print(f"\n  Word classes : {num_classes}")

    fold_assignments = []
    for i, test_user in enumerate(users):
        fold_assignments.append({
            "fold_idx": i, "test_user": test_user,
            "train_users": [u for u in users if u != test_user],
        })

    test_paths_per_fold = {}
    missing_words_per_fold = {}
    for fa in fold_assignments:
        fi = fa["fold_idx"]
        test_paths_per_fold[fi] = [
            p for word_paths in user_word_paths[fa["test_user"]].values()
            for p in word_paths
        ]
        test_labels = set(label_from_filename(p) for p in test_paths_per_fold[fi])
        missing_words_per_fold[fi] = sorted(set(all_classes) - test_labels)

    cov_csv = os.path.join(args.results_dir, "test_coverage.csv")
    with open(cov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "test_user", "test_files",
                    "n_missing_words", "missing_words"])
        for fa in fold_assignments:
            fi = fa["fold_idx"]
            w.writerow([fi, fa["test_user"], len(test_paths_per_fold[fi]),
                        len(missing_words_per_fold[fi]),
                        "|".join(missing_words_per_fold[fi])])
    print(f"\n  Test coverage saved: {cov_csv}")

    # Global CSVs (create headers once)
    global_csv = os.path.join(args.results_dir, "rq2_summary.csv")
    if not os.path.isfile(global_csv):
        with open(global_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "cap", "rotation_round", "nominal_train_per_word",
                "actual_mean_per_word", "total_wrap_events",
                "mean_top1(%)", "std_top1(%)", "mean_top5(%)",
                "mean_macro_f1(%)", "std_macro_f1(%)", "mean_weighted_f1(%)",
            ])

    cap_avg_csv = os.path.join(args.results_dir, "rq2_cap_averaged_summary.csv")
    if not os.path.isfile(cap_avg_csv):
        with open(cap_avg_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "cap", "nominal_train_per_word", "actual_mean_per_word",
                "mean_top1(%)", "std_top1(%)", "mean_top5(%)",
                "mean_macro_f1(%)", "std_macro_f1(%)",
                "mean_weighted_f1(%)", "mean_wrap_events",
            ])

    window_coverage_csv = os.path.join(args.results_dir, "rq2_window_coverage.csv")
    if not os.path.isfile(window_coverage_csv):
        with open(window_coverage_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "cap", "rotation_round", "fold_idx", "test_user",
                "n_wrap_events", "user", "word",
                "n_available", "window_start", "window_end", "indices_used",
            ])

    # ============================================================
    # MAIN SWEEP — folds parallelized across GPUs within each rotation
    # ============================================================
    all_rotation_summaries = []
    all_fold_results       = []
    window_coverage_data   = []
    overall_start          = time.time()
    gpu_ids_list           = args.gpu_ids
    parallel_n             = len(gpu_ids_list)
    run_counter            = 0

    for cap_idx, cap in enumerate(cap_values):
        if cap_idx < args.start_from_cap:
            print(f"\n  ⏭ Skipping cap={cap} (before START_FROM_CAP).")
            continue

        nominal_per_word = (num_folds - 1) * cap
        cap_dir = os.path.join(args.results_dir, f"cap_{cap}")
        os.makedirs(cap_dir, exist_ok=True)
        cap_rotation_results = []

        print(f"\n\n{'#'*70}")
        print(f"  CAP = {cap}  |  nominal train/word = {nominal_per_word}  |  "
              f"{args.rotations} rotations × 15 folds")
        print(f"  Running {parallel_n} folds in parallel on GPUs {gpu_ids_list}")
        print(f"{'#'*70}")

        for rotation_round in range(args.rotations):
            if cap_idx == args.start_from_cap and rotation_round < args.start_from_rotation:
                print(f"\n  ⏭ Skipping cap={cap} rotation={rotation_round} "
                      f"(before START_FROM_ROTATION).")
                continue

            rot_dir = os.path.join(cap_dir, f"rotation_{rotation_round}")
            os.makedirs(rot_dir, exist_ok=True)

            print(f"\n  ── Cap={cap}, Rotation={rotation_round} "
                  f"(window start = {rotation_round * cap}) ──")

            # Build tasks for all 15 folds in this (cap, rotation)
            tasks = []
            for fa in fold_assignments:
                fi = fa["fold_idx"]
                fold_log_dir = os.path.join(rot_dir, f"fold_{fi}")
                tasks.append((
                    fi, fa["test_user"], fa["train_users"],
                    cap, rotation_round,
                    user_word_paths, test_paths_per_fold[fi],
                    list(label_encoder.classes_),
                    fold_log_dir,
                    os.path.join(args.results_dir, "runs",
                                 f"cap_{cap}", f"rotation_{rotation_round}"),
                    config,
                    None,   # gpu_id — set at dispatch
                ))

            # Dispatch in waves of `parallel_n`
            rot_fold_results = []
            pending = list(tasks)
            wave_idx = 0

            while pending:
                wave = pending[:parallel_n]
                pending = pending[parallel_n:]

                # Assign GPUs round-robin
                wave_with_gpu = []
                for i, t in enumerate(wave):
                    gpu_id = gpu_ids_list[i % parallel_n]
                    wave_with_gpu.append(t[:-1] + (gpu_id,))

                print(f"\n  ▶ Wave {wave_idx + 1} (cap={cap}, rot={rotation_round}): "
                      f"{[(t[0], gpu_ids_list[i % parallel_n]) for i, t in enumerate(wave)]}")
                wave_idx += 1

                with NoDaemonPool(processes=len(wave_with_gpu)) as pool:
                    wave_results = pool.map(fold_train_and_eval_worker, wave_with_gpu)

                rot_fold_results.extend(wave_results)
                run_counter += len(wave_results)

                total_elapsed = time.time() - overall_start
                remaining = total_runs - run_counter
                avg_per_run = total_elapsed / max(run_counter, 1)
                # ETA estimate with parallelism
                eta = (remaining / parallel_n) * avg_per_run
                print(f"  ✔ Wave done. Progress: {run_counter}/{total_runs} runs "
                      f"| elapsed={total_elapsed/3600:.2f} h "
                      f"| ETA={eta/3600:.2f} h")

            # Sort by fold_idx
            rot_fold_results.sort(key=lambda r: r["fold_idx"])

            # Log wrap events to CSV
            total_wrap_this_rot = 0
            for r in rot_fold_results:
                total_wrap_this_rot += r["n_wrap_events"]
                if r["wrap_events_detail"]:
                    with open(window_coverage_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        for we in r["wrap_events_detail"]:
                            w.writerow([
                                cap, rotation_round, r["fold_idx"], r["test_user"],
                                r["n_wrap_events"],
                                we["user"], we["word"], we["n_available"],
                                we["window_start"], we["window_end"],
                                str(we["indices_used"]),
                            ])

            cap_rotation_results.extend(rot_fold_results)
            all_fold_results.extend(rot_fold_results)

            # Rotation summary
            top1s   = [r["top1_acc"]    for r in rot_fold_results]
            top5s   = [r["top5_acc"]    for r in rot_fold_results]
            mac_f1s = [r["macro_f1"]    for r in rot_fold_results]
            wtd_f1s = [r["weighted_f1"] for r in rot_fold_results]
            act_mean = np.mean([r["actual_mean_per_word"] for r in rot_fold_results])

            rot_summary = {
                "cap": cap, "rotation_round": rotation_round,
                "nominal_per_word": nominal_per_word,
                "actual_mean_per_word": float(act_mean),
                "total_wrap_events": total_wrap_this_rot,
                "mean_top1": float(np.mean(top1s)),
                "std_top1":  float(np.std(top1s)),
                "mean_top5": float(np.mean(top5s)),
                "mean_macro_f1": float(np.mean(mac_f1s)),
                "std_macro_f1":  float(np.std(mac_f1s)),
                "mean_weighted_f1": float(np.mean(wtd_f1s)),
            }
            all_rotation_summaries.append(rot_summary)
            window_coverage_data.append({
                "cap": cap, "rotation_round": rotation_round,
                "total_wrap_events": total_wrap_this_rot,
            })

            print(f"\n  ── Rotation {rotation_round} Summary (cap={cap}) ──")
            print(f"  Mean Top-1    : {rot_summary['mean_top1']:.2f}% ± {rot_summary['std_top1']:.2f}%")
            print(f"  Mean Top-5    : {rot_summary['mean_top5']:.2f}%")
            print(f"  Mean Macro F1 : {rot_summary['mean_macro_f1']:.2f}%")
            print(f"  Wrap events   : {total_wrap_this_rot}")

            # Per-rotation CSV
            rot_csv = os.path.join(rot_dir, "rotation_summary.csv")
            with open(rot_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "fold", "test_user", "gpu_id",
                    "top1(%)", "top5(%)", "macro_f1(%)", "weighted_f1(%)",
                    "train_files", "actual_mean_per_word",
                    "n_wrap_events", "fold_time_min",
                ])
                for r in rot_fold_results:
                    w.writerow([
                        r["fold_idx"], r["test_user"], r.get("gpu_id", "-"),
                        f"{r['top1_acc']:.2f}", f"{r['top5_acc']:.2f}",
                        f"{r['macro_f1']:.2f}", f"{r['weighted_f1']:.2f}",
                        r["train_files"], f"{r['actual_mean_per_word']:.1f}",
                        r["n_wrap_events"], f"{r['fold_time_s']/60:.1f}",
                    ])
                w.writerow([])
                w.writerow([
                    "MEAN", "—", "—",
                    f"{rot_summary['mean_top1']:.2f}",
                    f"{rot_summary['mean_top5']:.2f}",
                    f"{rot_summary['mean_macro_f1']:.2f}",
                    f"{rot_summary['mean_weighted_f1']:.2f}",
                    "—", f"{act_mean:.1f}", total_wrap_this_rot, "—",
                ])

            with open(global_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    cap, rotation_round, nominal_per_word,
                    f"{act_mean:.1f}", total_wrap_this_rot,
                    f"{rot_summary['mean_top1']:.2f}",
                    f"{rot_summary['std_top1']:.2f}",
                    f"{rot_summary['mean_top5']:.2f}",
                    f"{rot_summary['mean_macro_f1']:.2f}",
                    f"{rot_summary['std_macro_f1']:.2f}",
                    f"{rot_summary['mean_weighted_f1']:.2f}",
                ])

        # Cap-level aggregation
        if not cap_rotation_results: continue

        all_top1s   = [r["top1_acc"]    for r in cap_rotation_results]
        all_top5s   = [r["top5_acc"]    for r in cap_rotation_results]
        all_mac_f1s = [r["macro_f1"]    for r in cap_rotation_results]
        all_wtd_f1s = [r["weighted_f1"] for r in cap_rotation_results]
        all_acts    = [r["actual_mean_per_word"] for r in cap_rotation_results]
        all_wraps   = sum(r["n_wrap_events"] for r in cap_rotation_results)

        cap_mean_top1   = float(np.mean(all_top1s))
        cap_std_top1    = float(np.std(all_top1s))
        cap_mean_top5   = float(np.mean(all_top5s))
        cap_mean_mac_f1 = float(np.mean(all_mac_f1s))
        cap_std_mac_f1  = float(np.std(all_mac_f1s))
        cap_mean_wtd_f1 = float(np.mean(all_wtd_f1s))
        cap_mean_act    = float(np.mean(all_acts))

        print(f"\n  ══ Cap={cap} OVERALL (all {args.rotations} rotations) ══")
        print(f"  Mean Top-1    : {cap_mean_top1:.2f}% ± {cap_std_top1:.2f}%")
        print(f"  Mean Top-5    : {cap_mean_top5:.2f}%")
        print(f"  Mean Macro F1 : {cap_mean_mac_f1:.2f}%")
        print(f"  Total wraps   : {all_wraps}")

        cap_csv = os.path.join(cap_dir, "cap_summary.csv")
        with open(cap_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "rotation_round", "fold", "test_user", "gpu_id",
                "top1(%)", "top5(%)", "macro_f1(%)", "weighted_f1(%)",
                "train_files", "actual_mean_per_word", "n_wrap_events",
            ])
            for r in cap_rotation_results:
                w.writerow([
                    r["rotation_round"], r["fold_idx"], r["test_user"],
                    r.get("gpu_id", "-"),
                    f"{r['top1_acc']:.2f}", f"{r['top5_acc']:.2f}",
                    f"{r['macro_f1']:.2f}", f"{r['weighted_f1']:.2f}",
                    r["train_files"], f"{r['actual_mean_per_word']:.1f}",
                    r["n_wrap_events"],
                ])
            w.writerow([])
            w.writerow([
                "MEAN(all rotations)", "—", "—", "—",
                f"{cap_mean_top1:.2f}", f"{cap_mean_top5:.2f}",
                f"{cap_mean_mac_f1:.2f}", f"{cap_mean_wtd_f1:.2f}",
                "—", f"{cap_mean_act:.1f}", f"{all_wraps}",
            ])

        with open(cap_avg_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                cap, nominal_per_word,
                f"{cap_mean_act:.1f}",
                f"{cap_mean_top1:.2f}", f"{cap_std_top1:.2f}",
                f"{cap_mean_top5:.2f}",
                f"{cap_mean_mac_f1:.2f}", f"{cap_std_mac_f1:.2f}",
                f"{cap_mean_wtd_f1:.2f}",
                f"{all_wraps / (args.rotations * num_folds):.2f}",
            ])
    # ============================================================
    # FINAL SUMMARY + PLOTS
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  MASA RQ2 v2 COMPLETE — FINAL SUMMARY")
    print("=" * 70)

    cap_summaries = []
    for cap in cap_values:
        cap_results = [r for r in all_fold_results if r["cap"] == cap]
        if not cap_results:
            continue
        top1s   = [r["top1_acc"]    for r in cap_results]
        top5s   = [r["top5_acc"]    for r in cap_results]
        mac_f1s = [r["macro_f1"]    for r in cap_results]
        wtd_f1s = [r["weighted_f1"] for r in cap_results]
        cap_summaries.append({
            "cap"                 : cap,
            "nominal_per_word"    : (num_folds - 1) * cap,
            "actual_mean_per_word": float(np.mean(
                [r["actual_mean_per_word"] for r in cap_results]
            )),
            "mean_top1"       : float(np.mean(top1s)),
            "std_top1"        : float(np.std(top1s)),
            "mean_top5"       : float(np.mean(top5s)),
            "mean_macro_f1"   : float(np.mean(mac_f1s)),
            "std_macro_f1"    : float(np.std(mac_f1s)),
            "mean_weighted_f1": float(np.mean(wtd_f1s)),
        })

    print(f"\n  {'Cap':<5} {'Nominal':>8} {'Top-1':>9} {'±std':>7} "
          f"{'Top-5':>9} {'MacroF1':>9}")
    print(f"  {'-'*58}")
    for s in cap_summaries:
        print(f"  {s['cap']:<5} {s['nominal_per_word']:>8} "
              f"{s['mean_top1']:7.2f}%  ±{s['std_top1']:5.2f}%  "
              f"{s['mean_top5']:7.2f}%  {s['mean_macro_f1']:7.2f}%")

    print("\n  Generating plots...")
    plot_main_curve(
        cap_summaries,
        os.path.join(args.results_dir, "rq2_accuracy_curve.png"),
    )
    plot_per_rotation_curves(
        all_rotation_summaries,
        os.path.join(args.results_dir, "rq2_per_rotation_curves.png"),
    )
    plot_boxplot(
        all_fold_results,
        os.path.join(args.results_dir, "rq2_boxplot.png"),
    )
    plot_heatmap(
        all_rotation_summaries,
        os.path.join(args.results_dir, "rq2_heatmap.png"),
    )
    plot_fold_variance(
        all_fold_results,
        os.path.join(args.results_dir, "rq2_fold_variance.png"),
    )
    plot_improvement_curve(
        cap_summaries,
        os.path.join(args.results_dir, "rq2_improvement_curve.png"),
    )
    plot_wrap_coverage(
        window_coverage_data,
        os.path.join(args.results_dir, "rq2_wrap_coverage.png"),
    )

    total = time.time() - overall_start

    print(f"\n  Global summary CSV      : {global_csv}")
    print(f"  Cap-averaged CSV        : {cap_avg_csv}")
    print(f"  Window coverage CSV     : {window_coverage_csv}")
    print(f"  All results             : {args.results_dir}/")
    print(f"  TensorBoard             : "
          f"tensorboard --logdir={args.results_dir}/runs")
    print(f"  Total wall time         : {total/3600:.2f} h")
    print(f"  GPUs used               : {args.gpu_ids}")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    # CRITICAL: 'spawn' is required for CUDA + multiprocessing
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="MASA RQ2 v2: Luck-check data efficiency (Multi-GPU)"
    )
    parser.add_argument(
        "--data_root", type=str, default=DATA_ROOT,
        help="Root directory of .npy pose files",
    )
    parser.add_argument(
        "--results_dir", type=str, default=RESULTS_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--cap_start", type=int, default=CAP_START,
        help="Starting (highest) cap, e.g. 5",
    )
    parser.add_argument(
        "--cap_end", type=int, default=CAP_END,
        help="Ending (lowest) cap, e.g. 1",
    )
    parser.add_argument(
        "--rotations", type=int, default=ROTATIONS,
        help="Number of non-overlapping window rotations per cap",
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=GPU_IDS,
        help="GPU IDs to use in parallel, e.g. --gpu_ids 5 6 7",
    )
    parser.add_argument(
        "--start_from_cap", type=int, default=START_FROM_CAP,
        help="Resume: skip caps with index < this (0-based).",
    )
    parser.add_argument(
        "--start_from_rotation", type=int, default=START_FROM_ROTATION,
        help="Resume: in the start_from_cap, skip rotations < this",
    )
    args = parser.parse_args()
    main(args)