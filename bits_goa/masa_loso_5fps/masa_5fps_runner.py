"""
masa_runner.py — Full LOSO K-Fold Training (MASA 5 FPS variant)
MULTI-GPU VERSION: Runs folds in parallel across GPU 0 and GPU 1
================================================================
"""

import os
# DO NOT set CUDA_VISIBLE_DEVICES globally — workers set it per-process

import sys
import time
import random
import numpy as np
import torch
import csv
import multiprocessing as mp
import multiprocessing.pool
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from masa_dataset import find_npy_files, label_from_filename, user_from_path
from masa_train   import train_one_fold, Tee, save_confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# NON-DAEMONIC POOL — fixes DataLoader's num_workers issue
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


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT   = "/home/pdf2024018/masa/Pose"
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "loso_results_5fps",
)

START_FROM_FOLD = 0
GPU_IDS = [0, 1]
NUM_PARALLEL_FOLDS = len(GPU_IDS)

config = {
    "max_frames"     : 30,
    "fps_stride"     : 6,
    "num_workers"    : 4,

    "batch_size"     : 64,
    "epochs"         : 60,
    "warmup_epochs"  : 3,
    "min_lr_ratio"   : 0.05,
    "lr"             : 2e-4,
    "weight_decay"   : 5e-4,
    "label_smoothing": 0.1,
    "use_mixup"      : True,
    "mixup_alpha"    : 0.2,
    "use_amp"        : True,
    "seed"           : 42,

    "model_dim"      : 256,
    "nhead"          : 8,
    "num_layers"     : 4,
    "dim_feedforward": 512,
    "dropout"        : 0.2,
    "drop_path_rate" : 0.1,

    "mask_ratio"     : 0.4,
    "recon_weight"   : 0.1,
    "decoder_layers" : 2,
}


# ============================================================
# WORKER: runs one fold on one GPU
# ============================================================

def fold_worker(task):
    (fold_idx, assignment, user_to_paths, label_encoder_classes,
     fold_dir, config, tb_root, gpu_id) = task

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from sklearn.preprocessing import LabelEncoder
    from masa_train import train_one_fold

    seed = config["seed"] + fold_idx
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    label_encoder = LabelEncoder().fit(label_encoder_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_user   = assignment["test_user"]
    train_users = assignment["train_users"]
    train_paths = [p for u in train_users for p in user_to_paths[u]]
    test_paths  = user_to_paths[test_user]

    print(f"[GPU {gpu_id}] ▶ Fold {fold_idx} ({test_user}) started "
          f"| train={len(train_paths)} test={len(test_paths)}", flush=True)

    t0 = time.time()
    result = train_one_fold(
        fold_idx      = fold_idx,
        train_paths   = train_paths,
        test_paths    = test_paths,
        label_encoder = label_encoder,
        class_names   = list(label_encoder.classes_),
        fold_log_dir  = fold_dir,
        device        = device,
        config        = config,
        tb_root       = tb_root,
        dry_run       = False,
    )
    fold_time = time.time() - t0

    result["test_user"]   = test_user
    result["train_users"] = train_users
    result["fold_time_s"] = fold_time
    result["gpu_id"]      = gpu_id

    print(f"[GPU {gpu_id}] ✔ Fold {fold_idx} ({test_user}) done in "
          f"{fold_time/60:.1f} min | Test Top-1 = {result['top1_acc']:.2f}%",
          flush=True)

    return result


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ============================================================
# MAIN (all original module-level code moved inside)
# ============================================================

def main():
    set_seed(config["seed"])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(RESULTS_DIR, f"kfold_summary_{ts}.log"), "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    print("=" * 60)
    print(f"  MASA LOSO 5-FPS Training (MULTI-GPU)  |  {ts}")
    print(f"  DATA_ROOT   : {DATA_ROOT}")
    print(f"  RESULTS_DIR : {RESULTS_DIR}")
    print(f"  GPUs        : {GPU_IDS}  (parallel folds = {NUM_PARALLEL_FOLDS})")
    print(f"  fps_stride  : {config['fps_stride']}  (30fps → 5fps)")
    print(f"  max_frames  : {config['max_frames']}  (= {config['max_frames']/5:.1f} sec)")
    print(f"  START_FROM  : fold {START_FROM_FOLD}")
    print("=" * 60)

    n_gpus_visible = torch.cuda.device_count()
    print(f"\n  Visible GPUs: {n_gpus_visible}")
    for i in range(n_gpus_visible):
        print(f"    cuda:{i} → {torch.cuda.get_device_name(i)}")

    # ── Discover ────────────────────────────────────────────
    all_paths = find_npy_files(DATA_ROOT)
    if not all_paths:
        raise RuntimeError(f"No .npy files under {DATA_ROOT}")

    print(f"\n  Total .npy files : {len(all_paths)}")

    user_to_paths = defaultdict(list)
    for p in all_paths:
        user_to_paths[user_from_path(p)].append(p)
    user_to_paths = dict(user_to_paths)

    users     = sorted(user_to_paths.keys())
    num_folds = len(users)
    print(f"  Users ({num_folds}): {users}")
    for u in users:
        print(f"    {u}: {len(user_to_paths[u])} files")

    # ── Label encoder ───────────────────────────────────────
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))
    num_classes = len(all_classes)

    label_encoder = LabelEncoder().fit(all_classes)
    np.save(os.path.join(RESULTS_DIR, "class_names.npy"),
            np.array(label_encoder.classes_))

    print(f"\n  Total word classes : {num_classes}")

    # ── Fold assignments ────────────────────────────────────
    print(f"\n  {'Fold':<6} {'Test':<15} {'Train # files':>14}")
    print(f"  {'-'*50}")
    fold_assignments = []
    for i, test_user in enumerate(users):
        train_users = [u for u in users if u != test_user]
        n_train = sum(len(user_to_paths[u]) for u in train_users)
        fold_assignments.append({"test_user": test_user, "train_users": train_users})
        print(f"  {i:<6} {test_user:<15} {n_train:>14}")
    print()

    # ── Build task list ─────────────────────────────────────
    tasks = []
    for fold_idx, assignment in enumerate(fold_assignments):
        if fold_idx < START_FROM_FOLD:
            print(f"  Skipping fold {fold_idx}.")
            continue
        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx}")
        tasks.append((
            fold_idx,
            assignment,
            user_to_paths,
            list(label_encoder.classes_),
            fold_dir,
            config,
            os.path.join(RESULTS_DIR, "runs"),
            None,
        ))

    print(f"\n  Total folds to run: {len(tasks)}")
    print(f"  Parallel workers  : {NUM_PARALLEL_FOLDS}\n")

    # ── Launch waves ────────────────────────────────────────
    fold_results  = []
    overall_start = time.time()

    pending = list(tasks)
    while pending:
        wave = pending[:NUM_PARALLEL_FOLDS]
        pending = pending[NUM_PARALLEL_FOLDS:]

        wave_with_gpu = []
        for i, t in enumerate(wave):
            gpu_id = GPU_IDS[i % len(GPU_IDS)]
            wave_with_gpu.append(t[:-1] + (gpu_id,))

        print(f"  ▶ Launching wave: "
              f"{[(t[0], GPU_IDS[i % len(GPU_IDS)]) for i, t in enumerate(wave)]}")

        with NoDaemonPool(processes=len(wave_with_gpu)) as pool:
            wave_results = pool.map(fold_worker, wave_with_gpu)

        fold_results.extend(wave_results)

        done = len(fold_results)
        elapsed = time.time() - overall_start
        print(f"  ✔ Wave done. Completed {done}/{len(tasks)} folds | "
              f"elapsed={elapsed/60:.1f} min\n")

    fold_results.sort(key=lambda r: r["fold"])

    # ============================================================
    # AGGREGATE
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  FINAL MASA 5-FPS LOSO K-FOLD SUMMARY  (MULTI-GPU)")
    print("=" * 70)

    print(
        f"\n  {'Fold':<6} {'Test':<15} {'GPU':>4} {'TrTop1':>7} {'TrTop5':>7} "
        f"{'TeTop1':>7} {'TeTop5':>7} {'MacroF1':>8} {'WtdF1':>7}"
    )
    print(f"  {'-'*82}")
    for r in fold_results:
        print(
            f"  {r['fold']:<6} {r['test_user']:<15} "
            f"{r.get('gpu_id','-'):>4} "
            f"{r['train_top1_acc']:6.2f}%  "
            f"{r['train_top5_acc']:6.2f}%  "
            f"{r['top1_acc']:6.2f}%  "
            f"{r['top5_acc']:6.2f}%  "
            f"{r['macro_f1']:7.2f}%  "
            f"{r['weighted_f1']:6.2f}%"
        )
    print(f"  {'-'*82}")

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

    # ── Summary CSV ─────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "kfold_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fold","test_user","train_users","gpu_id",
            "train_top1(%)","train_top5(%)",
            "test_top1(%)","test_top5(%)",
            "macro_f1(%)","weighted_f1(%)",
            "test_loss","fold_time_min",
        ])
        for r in fold_results:
            w.writerow([
                r["fold"], r["test_user"], "+".join(r["train_users"]),
                r.get("gpu_id", "-"),
                f"{r['train_top1_acc']:.2f}", f"{r['train_top5_acc']:.2f}",
                f"{r['top1_acc']:.2f}",       f"{r['top5_acc']:.2f}",
                f"{r['macro_f1']:.2f}",       f"{r['weighted_f1']:.2f}",
                f"{r['test_loss']:.4f}",
                f"{r.get('fold_time_s',0)/60:.2f}",
            ])
        for stat, fn in [("MEAN", np.mean), ("STD", np.std)]:
            w.writerow(
                [stat, "—", "—", "—"] +
                [f"{fn([r[k] for r in fold_results]):.2f}" for k, _ in metric_keys] +
                ["—", "—"]
            )
    print(f"\n  Summary CSV : {csv_path}")

    # ── Aggregated Confusion Matrix ─────────────────────────
    if fold_results:
        all_y_true = np.concatenate([r["y_true"] for r in fold_results])
        all_y_pred = np.concatenate([r["y_pred"] for r in fold_results])
        agg_cm = confusion_matrix(all_y_true, all_y_pred)
        save_confusion_matrix(
            agg_cm, list(label_encoder.classes_),
            os.path.join(RESULTS_DIR, "confusion_matrix_aggregated.png"),
            title="MASA 5-FPS LOSO — Aggregated Confusion Matrix",
        )
        print("  Aggregated confusion matrix saved.")

    # ── Mean Curves ─────────────────────────────────────────
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

        def save_curve(data_list, title, ylabel, path):
            fig, ax = plt.subplots(figsize=(10, 5))
            for color, lbl, data in data_list:
                ax.plot(data, color=color, linewidth=2, label=lbl)
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.legend(); ax.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()
            print(f"  Saved: {os.path.basename(path)}")

        save_curve(
            [("steelblue","Mean Train CE",mean_train_loss),
             ("coral","Mean Test CE",mean_test_loss)],
            "Mean Cross-Entropy Loss", "Loss",
            os.path.join(RESULTS_DIR, "mean_loss_curve.png"))
        save_curve(
            [("steelblue","Train Top-1",mean_train_top1),
             ("coral","Test Top-1",mean_test_top1)],
            "Mean Top-1 Accuracy", "Accuracy (%)",
            os.path.join(RESULTS_DIR, "mean_top1_curve.png"))
        save_curve(
            [("steelblue","Train Top-5",mean_train_top5),
             ("coral","Test Top-5",mean_test_top5)],
            "Mean Top-5 Accuracy", "Accuracy (%)",
            os.path.join(RESULTS_DIR, "mean_top5_curve.png"))
        save_curve(
            [("mediumseagreen","Mean Recon Loss (×λ)",mean_recon)],
            "Mean Reconstruction Loss", "Loss",
            os.path.join(RESULTS_DIR, "mean_recon_curve.png"))

    # ── Bar Charts ──────────────────────────────────────────
    if fold_results:
        x     = np.arange(len(fold_results))
        xlbls = [f"Fold {r['fold']}\n({r['test_user']})" for r in fold_results]

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # CRITICAL: 'spawn' is required for CUDA + multiprocessing
    mp.set_start_method("spawn", force=True)
    main()