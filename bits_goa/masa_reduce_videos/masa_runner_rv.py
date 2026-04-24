"""
rq2_runner_masa.py — MASA RQ2: Progressive Training Data Reduction (LOSO)
MULTI-GPU VERSION: Runs 3 folds in parallel across GPUs 2, 3, 4
With detailed per-fold logging (same style as popsign reference).
Includes exception-safe worker: try/finally ensures log file closure.
===========================================================================
"""

import os
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES globally — each worker sets its own

import sys
import csv
import json
import random
import argparse
import time
import traceback
import numpy as np
import torch
import multiprocessing as mp
import multiprocessing.pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# NON-DAEMONIC POOL
# ============================================================

class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context("spawn"))):
    Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
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

CAP_VALUES = [10, 8, 6, 4, 2, 1]

GPU_IDS = [2, 3, 4]
NUM_PARALLEL_FOLDS = len(GPU_IDS)

config = {
    "max_frames"      : 150,
    "num_workers"     : 4,

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
    uwp = defaultdict(lambda: defaultdict(list))
    for p in paths:
        uwp[user_from_path(p)][label_from_filename(p)].append(p)
    return {u: dict(wp) for u, wp in uwp.items()}


# ============================================================
# CAP-BASED SUBSAMPLING
# ============================================================

def sample_train_paths(train_users, user_word_paths, cap,
                       round_idx, fold_idx, base_seed):
    seed = base_seed + round_idx * 1000 + fold_idx
    rng  = np.random.default_rng(seed=seed)

    selected        = []
    per_word_counts = defaultdict(int)

    for user in sorted(train_users):
        for word in sorted(user_word_paths[user].keys()):
            paths = sorted(user_word_paths[user][word])
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
        "mean_per_word"   : float(np.mean(counts)) if counts else 0.0,
        "min_per_word"    : int(np.min(counts))    if counts else 0,
        "max_per_word"    : int(np.max(counts))    if counts else 0,
        "words_at_nominal": int(sum(1 for c in counts if c == cap * len(train_users))),
        "words_below"     : int(sum(1 for c in counts if c <  cap * len(train_users))),
    }
    return selected, stats


# ============================================================
# EVALUATION HELPERS
# ============================================================

@torch.no_grad()
def _collect_masa_preds(model, loader, device, top_k=5, use_amp=True):
    import torch.nn.functional as F
    model.eval()
    y_true, y_pred, y_topk = [], [], []
    total_loss, total_n    = 0.0, 0

    for s1, s2, s3, labels, lengths, padding_mask in loader:
        s1 = s1.to(device); s2 = s2.to(device); s3 = s3.to(device)
        labels = labels.to(device); padding_mask = padding_mask.to(device)

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
        plt.barh(labels[::-1], counts[::-1], color="steelblue", edgecolor="black")
        plt.xlabel("Count"); plt.title(f"Top {n} Most Confused Class Pairs")
        plt.tight_layout(); plt.savefig(save_path_png, dpi=100); plt.close()


def _compute_per_class_accuracy(y_true, y_pred, class_names):
    rows = []
    for i, name in enumerate(class_names):
        mask = (y_true == i)
        n_samples = int(mask.sum())
        acc = (y_pred[mask] == i).mean() * 100.0 if n_samples > 0 else 0.0
        rows.append((name, n_samples, acc))
    return rows


def _save_per_class_accuracy_csv_and_plot(rows, save_dir):
    rows_sorted = sorted(rows, key=lambda x: x[2])
    csv_path = os.path.join(save_dir, "per_class_accuracy.csv")
    with open(csv_path, "w") as f:
        f.write("class,n_samples,accuracy(%)\n")
        for name, n, acc in rows_sorted:
            f.write(f"{name},{n},{acc:.2f}\n")

    names = [r[0] for r in rows_sorted]
    accs  = [r[2] for r in rows_sorted]
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


def _print_worst_best_classes(rows, n=10):
    rows_sorted = sorted(rows, key=lambda x: x[2])
    rows_with_samples = [r for r in rows_sorted if r[1] > 0]

    print(f"\n  --- Bottom {n} classes (worst accuracy) ---")
    for name, n_s, acc in rows_with_samples[:n]:
        print(f"    {name:<25s} n={n_s:4d}  acc={acc:.1f}%")

    print(f"\n  --- Top {n} classes (best accuracy) ---")
    for name, n_s, acc in rows_with_samples[-n:][::-1]:
        print(f"    {name:<25s} n={n_s:4d}  acc={acc:.1f}%")


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
        raise FileNotFoundError(f"No best_model.pt at {ckpt_path}.")
    ckpt = torch.load(ckpt_path, map_location=device)
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

    # ── Print detailed summary ──────────────────────────────
    print(f"\n  --- Final Epoch Evaluation (fold {fold_idx}) ---")
    print(f"  Top-1        : {top1:.2f}%")
    print(f"  Top-5        : {top5:.2f}%")
    print(f"  Macro F1     : {mac_f:.2f}%")
    print(f"  Weighted F1  : {wtd_f:.2f}%")

    rows = _compute_per_class_accuracy(y_true, y_pred, class_names)
    _print_worst_best_classes(rows, n=10)

    # ── Save artifacts ──────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    save_confusion_matrix(
        cm, class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — Fold {fold_idx}",
    )
    _save_top_confused_pairs(cm, class_names,
        os.path.join(eval_dir, "top_confused_pairs.png"), n=20)
    _save_per_class_accuracy_csv_and_plot(rows, eval_dir)
    report_txt = _save_classification_report(y_true, y_pred, class_names,
        os.path.join(eval_dir, "classification_report.txt"))

    # Print report summary (last lines — averages)
    print(f"\n  Classification Report (averages):")
    for line in report_txt.split("\n")[-5:]:
        if line.strip():
            print(f"    {line}")

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
        "macro_p": mac_p, "macro_r": mac_r,
        "macro_f1": mac_f, "weighted_f1": wtd_f,
        "test_loss": test_loss,
        "y_true": y_true, "y_pred": y_pred,
    }


# ============================================================
# WORKER: train + eval ONE fold on ONE GPU
# (Now wrapped in try/finally for safe log cleanup on exceptions)
# ============================================================

def fold_train_and_eval_worker(task):
    """
    Runs train + eval for ONE (cap, fold) on ONE GPU.
    Writes a detailed per-fold training.log capturing everything.
    Uses try/finally to guarantee log file closure and stdout/stderr
    restoration even if training/evaluation crashes.
    """
    (fold_idx, test_user, train_users, cap, round_idx,
     user_word_paths, test_paths, label_encoder_classes,
     fold_log_dir, tb_root, worker_config,
     missing_words_count, nominal_per_word, gpu_id) = task

    # Pin to GPU BEFORE torch CUDA initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from sklearn.preprocessing import LabelEncoder
    from masa_train import train_one_fold, Tee

    # ── Per-fold log file: captures full per-epoch output ────
    os.makedirs(fold_log_dir, exist_ok=True)
    worker_log_path = os.path.join(fold_log_dir, "training.log")
    worker_log_f    = open(worker_log_path, "w", buffering=1)  # line-buffered

    # Save original stdout/stderr (the worker process's native ones) so we
    # can restore them in `finally` even if something crashes.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.__stdout__, worker_log_f)
    sys.stderr = Tee(sys.__stderr__, worker_log_f)

    # ────────────────────────────────────────────────────────────
    # Everything from here until `return result` is wrapped in
    # try/finally so log handle + stdout/stderr always get restored.
    # ────────────────────────────────────────────────────────────
    try:
        # Per-fold deterministic seed
        seed = worker_config["seed"] + round_idx * 1000 + fold_idx
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        label_encoder = LabelEncoder().fit(label_encoder_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Subsample training paths
        train_paths, stats = sample_train_paths(
            train_users=train_users, user_word_paths=user_word_paths,
            cap=cap, round_idx=round_idx, fold_idx=fold_idx,
            base_seed=worker_config["seed"],
        )

        print(f"\n{'='*60}")
        print(f"  FOLD {fold_idx}  |  Cap={cap}  |  Train={stats['total_files']}  Test={len(test_paths)}")
        print(f"  Test user   : {test_user}")
        print(f"  GPU         : {gpu_id}")
        print(f"  TensorBoard : {tb_root}/fold_{fold_idx}")
        print(f"  Log file    : {worker_log_path}")
        print(f"{'='*60}", flush=True)

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

        # --- Per-user accuracy CSV ---
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
                yt = np.array(user_y_true[u]); yp = np.array(user_y_pred[u])
                w.writerow([u, len(yt), f"{(yt == yp).mean()*100.0:.2f}"])

        print(f"\n   Fold {fold_idx} complete in {fold_train_time/60:.1f} min. "
              f"Saved to: {fold_log_dir}", flush=True)

        # Build return dict (SUCCESS path)
        result = {
            "fold": fold_idx, "test_user": test_user, "cap": cap,
            "train_files": stats["total_files"],
            "actual_mean_per_word": stats["mean_per_word"],
            "actual_min_per_word":  stats["min_per_word"],
            "actual_max_per_word":  stats["max_per_word"],
            "words_at_nominal":     stats["words_at_nominal"],
            "words_below":          stats["words_below"],
            "missing_words_in_test": missing_words_count,
            "top1_acc":    em["top1_acc"],
            "top5_acc":    em["top5_acc"],
            "macro_f1":    em["macro_f1"],
            "weighted_f1": em["weighted_f1"],
            "test_loss":   em["test_loss"],
            "y_true":      em["y_true"],
            "y_pred":      em["y_pred"],
            "train_time_s": fold_train_time,
            "gpu_id":       gpu_id,
        }
        return result

    except Exception as e:
        # Log the failure to the per-fold log before the process dies.
        # This ensures users see WHY a fold crashed in its own log.
        print(f"\n{'='*60}", flush=True)
        print(f"[GPU {gpu_id}] ✗ Fold {fold_idx} FAILED: "
              f"{type(e).__name__}: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        traceback.print_exc()
        # Re-raise so the main process's pool.map() sees the failure.
        raise

    finally:
        # ALWAYS restore stdout/stderr so the worker process doesn't end up
        # with a dangling Tee pointing at a closed file handle. Also close
        # the log file cleanly so buffered data is flushed to disk.
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        except Exception:
            pass
        try:
            worker_log_f.flush()
            worker_log_f.close()
        except Exception:
            pass


# ============================================================
# RQ2 CURVE PLOT
# ============================================================

def plot_rq2_curve(round_summaries, save_path):
    if not round_summaries:
        return
    vids  = [s["actual_mean_per_word"] for s in round_summaries]
    top1s = [s["mean_top1"]            for s in round_summaries]
    top5s = [s["mean_top5"]            for s in round_summaries]
    f1s   = [s["mean_macro_f1"]        for s in round_summaries]
    stds  = [s["std_top1"]             for s in round_summaries]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(vids, top1s, marker="o", color="steelblue",
            linewidth=2.5, markersize=9, label="Top-1 Accuracy")
    ax.fill_between(vids,
        [t - s for t, s in zip(top1s, stds)],
        [t + s for t, s in zip(top1s, stds)],
        color="steelblue", alpha=0.15, label="Top-1 ± 1 std (across folds)")
    ax.plot(vids, top5s, marker="s", color="coral",
            linewidth=2.5, markersize=9, label="Top-5 Accuracy")
    ax.plot(vids, f1s, marker="^", color="mediumseagreen",
            linewidth=2.5, markersize=9, label="Macro F1")

    for x, y in zip(vids, top1s):
        ax.annotate(f"{y:.1f}%", (x, y),
            textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=9, color="steelblue", fontweight="bold")

    ax.set_xlabel("Actual Mean Training Videos per Word", fontsize=12)
    ax.set_ylabel("Accuracy / F1  (%)", fontsize=12)
    ax.set_title("MASA RQ2 — Effect of Training Data Size on LOSO Recognition\n"
                 "(Test set fixed per fold: held-out signer's full recordings)",
                 fontsize=13)
    ax.set_xticks(vids)
    ax.set_xticklabels([f"{v:.0f}" for v in vids], fontsize=10)
    ax.set_ylim(0, 110); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    print(f"  RQ2 curve saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main(args):
    set_seed(config["seed"])
    os.makedirs(args.results_dir, exist_ok=True)

    with open(os.path.join(args.results_dir, "config.json"), "w") as f:
        json.dump({
            **config,
            "CAP_VALUES": args.cap_values,
            "GPU_IDS": args.gpu_ids,
            "NUM_PARALLEL_FOLDS": len(args.gpu_ids),
            "DATA_ROOT": args.data_root,
            "RESULTS_DIR": args.results_dir,
        }, f, indent=4)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(args.results_dir, f"rq2_log_{ts}.txt"), "w",
                 buffering=1)
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    print("=" * 70)
    print(f"  MASA RQ2 — Data Efficiency LOSO (MULTI-GPU)  |  {ts}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Cap values  : {args.cap_values}")
    print(f"  GPU IDs     : {args.gpu_ids}   (parallel = {len(args.gpu_ids)})")
    print(f"  Epochs/fold : {config['epochs']}")
    print("=" * 70)

    n_gpus_visible = torch.cuda.device_count()
    print(f"\n  System visible GPUs: {n_gpus_visible}")
    for i in range(n_gpus_visible):
        print(f"    cuda:{i} → {torch.cuda.get_device_name(i)}")

    # ── Discover + group ──
    all_paths = find_npy_files(args.data_root)
    if not all_paths:
        raise RuntimeError(f"No .npy files found under: {args.data_root}")
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

    # ── Label encoder ──
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))
    num_classes = len(all_classes)
    label_encoder = LabelEncoder().fit(all_classes)
    np.save(os.path.join(args.results_dir, "class_names.npy"),
            np.array(label_encoder.classes_))
    print(f"\n  Word classes : {num_classes}")

    # ── LOSO fold assignments ──
    fold_assignments = []
    for i, test_user in enumerate(users):
        fold_assignments.append({
            "fold_idx": i, "test_user": test_user,
            "train_users": [u for u in users if u != test_user],
        })

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
        test_labels = set(label_from_filename(p) for p in test_paths_per_fold[fi])
        missing_words_per_fold[fi] = sorted(set(all_classes) - test_labels)
        print(f"  {fi:<6} {fa['test_user']:<12} "
              f"{len(test_paths_per_fold[fi]):>10}  "
              f"{len(missing_words_per_fold[fi]):>14}")

    # ── Test coverage CSV ──
    cov_csv = os.path.join(args.results_dir, "test_coverage.csv")
    with open(cov_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "test_user", "test_files",
                    "n_missing_words", "missing_words"])
        for fa in fold_assignments:
            fi = fa["fold_idx"]
            w.writerow([fi, fa["test_user"],
                len(test_paths_per_fold[fi]),
                len(missing_words_per_fold[fi]),
                "|".join(missing_words_per_fold[fi])])
    print(f"\n  Test coverage saved: {cov_csv}")

    # ── Global CSV header ──
    global_csv = os.path.join(args.results_dir, "rq2_summary.csv")
    if not os.path.isfile(global_csv):
        with open(global_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "cap", "nominal_train_per_word",
                "actual_mean_per_word", "actual_min_per_word", "actual_max_per_word",
                "mean_top1(%)", "std_top1(%)", "mean_top5(%)",
                "mean_macro_f1(%)", "std_macro_f1(%)", "mean_weighted_f1(%)",
            ])

    # ============================================================
    # CAP SWEEP
    # ============================================================
    round_summaries = []
    overall_start   = time.time()
    gpu_ids_list    = args.gpu_ids
    parallel_n      = len(gpu_ids_list)

    for round_idx, cap in enumerate(args.cap_values):
        if round_idx < args.start_from_round:
            print(f"\n  ⏭  Skipping round {round_idx + 1} (cap={cap}).")
            continue

        nominal_per_word = (num_folds - 1) * cap
        round_dir = os.path.join(args.results_dir, f"cap_{cap}")
        os.makedirs(round_dir, exist_ok=True)

        print(f"\n\n{'#'*70}")
        print(f"  ROUND {round_idx + 1}/{len(args.cap_values)}  "
              f"|  cap={cap}  |  nominal train/word={nominal_per_word}")
        print(f"  Output folder : {round_dir}")
        print(f"  Running {parallel_n} folds in parallel on GPUs {gpu_ids_list}")
        print(f"{'#'*70}")

        # Build tasks
        tasks = []
        for fa in fold_assignments:
            fi = fa["fold_idx"]
            fold_log_dir = os.path.join(round_dir, f"fold_{fi}")
            tasks.append((
                fi, fa["test_user"], fa["train_users"],
                cap, round_idx,
                user_word_paths, test_paths_per_fold[fi],
                list(label_encoder.classes_),
                fold_log_dir,
                os.path.join(args.results_dir, "runs", f"cap_{cap}"),
                config,
                len(missing_words_per_fold[fi]),
                nominal_per_word,
                None,
            ))

        # Dispatch in waves
        fold_results = []
        round_start  = time.time()
        pending      = list(tasks)

        wave_idx = 0
        while pending:
            wave = pending[:parallel_n]
            pending = pending[parallel_n:]

            wave_with_gpu = []
            for i, t in enumerate(wave):
                gpu_id = gpu_ids_list[i % parallel_n]
                wave_with_gpu.append(t[:-1] + (gpu_id,))

            print(f"\n  ▶ Wave {wave_idx + 1}: "
                  f"{[(t[0], gpu_ids_list[i % parallel_n]) for i, t in enumerate(wave)]}",
                  flush=True)
            wave_idx += 1

            with NoDaemonPool(processes=len(wave_with_gpu)) as pool:
                wave_results = pool.map(fold_train_and_eval_worker, wave_with_gpu)

            # ★★★★★ THE CRITICAL FIX: collect results from each wave ★★★★★
            fold_results.extend(wave_results)

            done = len(fold_results)
            elapsed = time.time() - overall_start
            print(f"  ✔ Wave done. Round progress: {done}/{len(tasks)} folds "
                  f"| Cumulative: {elapsed/3600:.2f} h", flush=True)

        # Defensive: skip aggregation if empty
        if not fold_results:
            print(f"  ⚠ No fold results for cap={cap}, skipping aggregation.")
            continue

        fold_results.sort(key=lambda r: r["fold"])

        # ── Round aggregation ──
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
        global_mean = float(np.mean(act_mean)) if act_mean else 0.0
        global_min  = int(np.min(act_min))     if act_min  else 0
        global_max  = int(np.max(act_max))     if act_max  else 0

        round_time = time.time() - round_start
        print(f"\n  ── Round {round_idx + 1} Summary (cap={cap}) "
              f"— took {round_time/3600:.2f} h ──")
        print(f"  Mean Top-1    : {mean_top1:.2f}% ± {std_top1:.2f}%")
        print(f"  Mean Top-5    : {mean_top5:.2f}%")
        print(f"  Mean Macro F1 : {mean_mac_f1:.2f}%")
        print(f"  Actual mean train vids/word : {global_mean:.1f}")

        print(f"\n  {'Fold':<6} {'Test User':<12} {'GPU':>4} "
              f"{'Top-1':>8} {'Top-5':>8} {'MacroF1':>9}")
        print(f"  {'-'*55}")
        for r in fold_results:
            print(f"  {r['fold']:<6} {r['test_user']:<12} "
                  f"{r.get('gpu_id','-'):>4} "
                  f"{r['top1_acc']:7.2f}%  "
                  f"{r['top5_acc']:7.2f}%  "
                  f"{r['macro_f1']:8.2f}%")

        # ── Round CSV ──
        round_csv = os.path.join(round_dir, "round_summary.csv")
        with open(round_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "fold", "test_user", "gpu_id",
                "top1(%)", "top5(%)", "macro_f1(%)", "weighted_f1(%)",
                "train_files", "actual_mean_per_word",
                "actual_min_per_word", "actual_max_per_word",
                "words_at_nominal", "words_below",
                "missing_words_in_test", "train_time_min",
            ])
            for r in fold_results:
                w.writerow([
                    r["fold"], r["test_user"], r.get("gpu_id", "-"),
                    f"{r['top1_acc']:.2f}", f"{r['top5_acc']:.2f}",
                    f"{r['macro_f1']:.2f}", f"{r['weighted_f1']:.2f}",
                    r["train_files"], f"{r['actual_mean_per_word']:.1f}",
                    r["actual_min_per_word"], r["actual_max_per_word"],
                    r["words_at_nominal"], r["words_below"],
                    r["missing_words_in_test"],
                    f"{r['train_time_s']/60:.1f}",
                ])
            w.writerow([])
            w.writerow(["MEAN", "—", "—",
                f"{mean_top1:.2f}", f"{mean_top5:.2f}",
                f"{mean_mac_f1:.2f}", f"{mean_wtd_f1:.2f}",
                "—", f"{global_mean:.1f}", "—", "—", "—", "—", "—", "—"])
            w.writerow(["STD", "—", "—",
                f"{std_top1:.2f}", "—", f"{std_mac_f1:.2f}", "—",
                "—", "—", "—", "—", "—", "—", "—", "—"])
        print(f"  Round CSV : {round_csv}")

        # ── Append to global summary ──
        with open(global_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                round_idx + 1, cap, nominal_per_word,
                f"{global_mean:.1f}", global_min, global_max,
                f"{mean_top1:.2f}", f"{std_top1:.2f}", f"{mean_top5:.2f}",
                f"{mean_mac_f1:.2f}", f"{std_mac_f1:.2f}", f"{mean_wtd_f1:.2f}",
            ])

        round_summaries.append({
            "round": round_idx + 1, "cap": cap,
            "nominal_per_word": nominal_per_word,
            "actual_mean_per_word": global_mean,
            "mean_top1": mean_top1, "std_top1": std_top1,
            "mean_top5": mean_top5, "mean_macro_f1": mean_mac_f1,
            "mean_weighted_f1": mean_wtd_f1,
        })

    # ============================================================
    # FINAL SUMMARY + CURVE
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  FINAL K-FOLD SUMMARY")
    print("=" * 70)
    print(f"\n  {'Cap':<5} {'Nominal':>8} {'Actual':>8}  "
          f"{'Top-1':>9} {'±std':>7}  {'Top-5':>9}  {'MacroF1':>9}")
    print(f"  {'-'*70}")
    for s in round_summaries:
        print(f"  {s['cap']:<5} {s['nominal_per_word']:>8} "
              f"{s['actual_mean_per_word']:>8.1f}  "
              f"{s['mean_top1']:8.2f}%  ±{s['std_top1']:5.2f}%  "
              f"{s['mean_top5']:8.2f}%  {s['mean_macro_f1']:8.2f}%")

    plot_rq2_curve(round_summaries,
        save_path=os.path.join(args.results_dir, "rq2_accuracy_curve.png"))

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
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="MASA RQ2 — Data Efficiency under LOSO (Multi-GPU)")
    parser.add_argument("--data_root",   type=str, default=DATA_ROOT)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--cap_values", type=int, nargs="+", default=CAP_VALUES,
        help="Cap sequence (descending), e.g. --cap_values 10 8 6 4 2 1")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=GPU_IDS,
        help="GPU IDs to use in parallel, e.g. --gpu_ids 2 3 4")
    parser.add_argument("--start_from_round", type=int, default=START_FROM_ROUND,
        help="Resume: skip rounds with index < this (0-based)")
    args = parser.parse_args()
    main(args)