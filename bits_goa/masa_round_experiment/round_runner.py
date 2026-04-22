"""
vocab_partition_runner_masa.py — MASA Vocabulary-Partitioned LOSO
==================================================================
Experiment 3 for MASA:
    Reproduces popsign's vocab-partition experiment using IDENTICAL
    word→bucket assignments (read from popsign's round_*_buckets.csv).

For each round and each bucket:
    1. Filter MASA .npy files to only those whose class label is in the bucket.
    2. Build a label encoder on just the bucket's ~98 classes.
    3. Run 15-fold LOSO using the MASA training stack (unchanged).

This guarantees the 25 MASA experiments use the same vocabulary
partitions as popsign → direct comparison.

Protocol:
    NUM_ROUNDS  : 5
    NUM_BUCKETS : 5
    LOSO folds  : 15
    Epochs      : 60
    Total runs  : 375
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # pin to GPU 0

import sys
import csv
import json
import random
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Local MASA modules (copies in this folder)
from masa_dataset import find_npy_files, label_from_filename, user_from_path
from masa_train   import train_one_fold, Tee, save_confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

# MASA data (NPY, MMPose COCO-WholeBody)
DATA_ROOT   = "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/MASA/Data/ISL_GOA/Pose"

# Where popsign wrote the authoritative bucket assignments
POPSIGN_BUCKET_DIR = (
    "/mnt/9a528fe4-4fe8-4dff-9a0c-8b1a3cf3d7ba/popsign/"
    "both_hands/kfold/code_30fps/round_experiment/results_vocab_partition"
)

# Results go inside this experiment's subfolder
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results_vocab_partition_masa",
)

NUM_BUCKETS = 5
NUM_ROUNDS  = 5

config = {
    # Data
    "max_frames"      : 150,
    "num_workers"     : 6,

    # Training — MASA-LOSO-proven
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

# Resume support — skip (round,bucket) pairs already completed on disk.
# Set > 0 if a run crashed and you want to pick up where you left off.
START_FROM_ROUND  = 0
START_FROM_BUCKET = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ============================================================
# READ POPSIGN BUCKET CSVs
# ============================================================

def load_popsign_buckets(round_idx, popsign_dir):
    """
    Read round_<round_idx>_buckets.csv and return list of 5 word-lists.
    Format:
        bucket_id,num_words,word_0,word_1,...,word_N
        bucket_0,98,apple,banana,...
        bucket_1,98,...
        ...
    """
    csv_path = os.path.join(popsign_dir, f"round_{round_idx}_buckets.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing popsign bucket file: {csv_path}")

    buckets = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)   # skip header row
        for row in reader:
            if not row or len(row) < 2:
                continue
            bucket_id = row[0]
            num_words = int(row[1])
            words     = [w for w in row[2:] if w != ""]
            if len(words) != num_words:
                print(f"  ⚠ Round {round_idx} {bucket_id}: "
                      f"header says {num_words} but got {len(words)} words")
            buckets.append(words)

    return buckets


# ============================================================
# CURVE HELPERS
# ============================================================

def pad_to(lst, length):
    return lst + [lst[-1]] * (length - len(lst))


def save_mean_curve(fold_results, key_train, key_test,
                    ylabel, title, save_path):
    max_ep = max(len(r[key_train]) for r in fold_results)
    mean_train = np.mean([pad_to(r[key_train], max_ep) for r in fold_results], axis=0)
    mean_test  = np.mean([pad_to(r[key_test],  max_ep) for r in fold_results], axis=0)

    plt.figure(figsize=(10, 5))
    for r in fold_results:
        ep = list(range(len(r[key_train])))
        plt.plot(ep, r[key_train], alpha=0.15, color="steelblue")
        plt.plot(ep, r[key_test],  alpha=0.15, color="coral")

    plt.plot(mean_train, color="steelblue", linewidth=2, label="Mean Train")
    plt.plot(mean_test,  color="coral",     linewidth=2, label="Mean Test")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=100); plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    # ── Save config ───────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, "config.json"), "w") as f:
        json.dump({**config,
                   "NUM_BUCKETS"       : NUM_BUCKETS,
                   "NUM_ROUNDS"        : NUM_ROUNDS,
                   "DATA_ROOT"         : DATA_ROOT,
                   "POPSIGN_BUCKET_DIR": POPSIGN_BUCKET_DIR,
                   "START_FROM_ROUND"  : START_FROM_ROUND,
                   "START_FROM_BUCKET" : START_FROM_BUCKET},
                  f, indent=4)

    # ── Logging ───────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = open(os.path.join(RESULTS_DIR, f"log_{ts}.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"  MASA Vocabulary-Partitioned LOSO  |  {ts}")
    print(f"  Rounds            : {NUM_ROUNDS}")
    print(f"  Buckets per round : {NUM_BUCKETS}")
    print(f"  Epochs per fold   : {config['epochs']}")
    print(f"  Data root         : {DATA_ROOT}")
    print(f"  Popsign buckets   : {POPSIGN_BUCKET_DIR}")
    print(f"  Results dir       : {RESULTS_DIR}")
    print(f"  Device            : {device}")
    if device.type == "cuda":
        print(f"  GPU               : {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── Discover MASA files ──────────────────────────────────
    all_paths = find_npy_files(DATA_ROOT)
    if not all_paths:
        raise RuntimeError(f"No .npy files found under: {DATA_ROOT}")

    user_to_paths = defaultdict(list)
    for p in all_paths:
        user_to_paths[user_from_path(p)].append(p)

    users       = sorted(user_to_paths.keys())
    all_labels  = [label_from_filename(p) for p in all_paths]
    all_classes = sorted(set(all_labels))

    print(f"\n  Total .npy files : {len(all_paths)}")
    print(f"  Total classes    : {len(all_classes)}")
    print(f"  Total users      : {len(users)}  →  {users}")

    # ── Pre-load & verify ALL rounds' bucket files ───────────
    print(f"\n  Loading popsign bucket assignments …")
    all_rounds_buckets = []
    for r in range(NUM_ROUNDS):
        bs = load_popsign_buckets(r, POPSIGN_BUCKET_DIR)
        if len(bs) != NUM_BUCKETS:
            raise ValueError(
                f"Round {r}: expected {NUM_BUCKETS} buckets, got {len(bs)}"
            )
        all_rounds_buckets.append(bs)

        sizes      = [len(b) for b in bs]
        total_bkt  = sum(sizes)
        words_here = set(w for b in bs for w in b)

        missing    = words_here - set(all_classes)
        unassigned = set(all_classes) - words_here

        print(f"    Round {r}: bucket sizes={sizes}  total={total_bkt}")

        if missing:
            print(f"      ⚠ Words in popsign buckets but not in MASA data "
                  f"({len(missing)}): {sorted(missing)[:10]}"
                  + ("..." if len(missing) > 10 else ""))
        if unassigned:
            print(f"      ⚠ MASA classes NOT assigned to any bucket "
                  f"({len(unassigned)}): {sorted(unassigned)[:10]}"
                  + ("..." if len(unassigned) > 10 else ""))
        if not missing and not unassigned:
            print(f"      ✓ Buckets cover MASA vocabulary exactly.")

    # Copy popsign bucket CSVs into our results dir for provenance
    import shutil
    for r in range(NUM_ROUNDS):
        src = os.path.join(POPSIGN_BUCKET_DIR, f"round_{r}_buckets.csv")
        dst = os.path.join(RESULTS_DIR,        f"round_{r}_buckets.csv")
        shutil.copy2(src, dst)
    print(f"\n  Copied popsign bucket CSVs → {RESULTS_DIR}")

    # ── CSV headers ───────────────────────────────────────────
    global_csv = os.path.join(RESULTS_DIR, "global_summary.csv")
    if not os.path.isfile(global_csv):
        with open(global_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "bucket", "fold", "test_user",
                "train_top1(%)", "train_top5(%)",
                "test_top1(%)",  "test_top5(%)",
                "macro_f1(%)", "weighted_f1(%)", "test_loss",
            ])

    round_csv = os.path.join(RESULTS_DIR, "round_summary.csv")
    if not os.path.isfile(round_csv):
        with open(round_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round",
                "mean_top1(%)", "mean_top5(%)", "mean_macro_f1(%)",
                "std_top1(%)",  "std_top5(%)",  "std_macro_f1(%)",
            ])

    # ── Per-word tracking across all rounds ─────────────────────
    word_metrics = defaultdict(list)  # word → list of 0/1 correct
    round_results_summary = []

    overall_start = time.time()

    # ============================================================
    # ROUND LOOP
    # ============================================================
    for r_idx in range(NUM_ROUNDS):
        if r_idx < START_FROM_ROUND:
            print(f"\n  ⏭ Skipping round {r_idx} (before START_FROM_ROUND).")
            continue

        round_seed = config["seed"] + r_idx
        set_seed(round_seed)

        print(f"\n\n{'='*70}")
        print(f"  ROUND {r_idx + 1}/{NUM_ROUNDS}  |  seed={round_seed}")
        print(f"{'='*70}")

        buckets = all_rounds_buckets[r_idx]
        print(f"  Bucket sizes (from popsign): {[len(b) for b in buckets]}")

        bucket_top1s, bucket_top5s, bucket_f1s = [], [], []

        # ============================================================
        # BUCKET LOOP
        # ============================================================
        for b_idx, bucket in enumerate(buckets):

            # Resume — skip buckets in previous rounds, OR
            # buckets earlier than START_FROM_BUCKET in the starting round
            if r_idx == START_FROM_ROUND and b_idx < START_FROM_BUCKET:
                print(f"\n  ⏭ Skipping round {r_idx} bucket {b_idx} "
                      f"(before START_FROM_BUCKET).")
                continue

            bucket_set = set(bucket)
            print(f"\n  --- Round {r_idx}  Bucket {b_idx}  "
                  f"({len(bucket)} words) ---")

            # Filter MASA paths to this bucket's vocabulary
            filtered = [
                p for p in all_paths
                if label_from_filename(p) in bucket_set
            ]
            if not filtered:
                print(f"   ⚠ No MASA files for bucket {b_idx}. Skipping.")
                continue

            b_user_map = defaultdict(list)
            for p in filtered:
                b_user_map[user_from_path(p)].append(p)

            # Bucket-specific label encoder
            le = LabelEncoder().fit(sorted(bucket))

            fold_results = []

            print(f"\n  {'Fold':<5} {'User':<12} {'TrTop1':>7} {'TrTop5':>7} "
                  f"{'TeTop1':>7} {'TeTop5':>7} {'MacroF1':>8} {'WtdF1':>7}")
            print(f"  {'-'*65}")

            # ========================================================
            # FOLD LOOP (LOSO)
            # ========================================================
            for f_idx, test_user in enumerate(users):
                train_users = [u for u in users if u != test_user]
                train_paths = [p for u in train_users for p in b_user_map[u]]
                test_paths  = b_user_map[test_user]

                if not train_paths or not test_paths:
                    print(f"  {f_idx:<5} {test_user:<12} SKIPPED "
                          f"(train={len(train_paths)} test={len(test_paths)})")
                    continue

                fold_log_dir = os.path.join(
                    RESULTS_DIR, f"r{r_idx}_b{b_idx}_f{f_idx}"
                )

                fold_start = time.time()
                result = train_one_fold(
                    fold_idx      = f_idx,
                    train_paths   = train_paths,
                    test_paths    = test_paths,
                    label_encoder = le,
                    class_names   = list(le.classes_),
                    fold_log_dir  = fold_log_dir,
                    device        = device,
                    config        = config,
                    tb_root       = os.path.join(RESULTS_DIR, "runs"),
                    dry_run       = False,
                )
                fold_elapsed = time.time() - fold_start

                result["test_user"]   = test_user
                result["train_users"] = train_users
                fold_results.append(result)

                print(
                    f"  {f_idx:<5} {test_user:<12} "
                    f"{result['train_top1_acc']:6.2f}%  "
                    f"{result['train_top5_acc']:6.2f}%  "
                    f"{result['top1_acc']:6.2f}%  "
                    f"{result['top5_acc']:6.2f}%  "
                    f"{result['macro_f1']:7.2f}%  "
                    f"{result['weighted_f1']:6.2f}%  "
                    f"[{fold_elapsed/60:.1f} min]"
                )

                # Append to global CSV
                with open(global_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        r_idx, b_idx, f_idx, test_user,
                        f"{result['train_top1_acc']:.2f}",
                        f"{result['train_top5_acc']:.2f}",
                        f"{result['top1_acc']:.2f}",
                        f"{result['top5_acc']:.2f}",
                        f"{result['macro_f1']:.2f}",
                        f"{result['weighted_f1']:.2f}",
                        f"{result['test_loss']:.4f}",
                    ])

                # Per-word accuracy tracking
                for y_t, y_p in zip(result["y_true"], result["y_pred"]):
                    word = le.inverse_transform([y_t])[0]
                    word_metrics[word].append(int(y_t == y_p))

            if not fold_results:
                print(f"\n  ⚠ No fold results for bucket {b_idx}. Skipping aggregates.")
                continue

            # ── Aggregated confusion matrix for this bucket ───
            y_true_all = np.concatenate([r["y_true"] for r in fold_results])
            y_pred_all = np.concatenate([r["y_pred"] for r in fold_results])

            cm = confusion_matrix(
                y_true_all, y_pred_all,
                labels=list(range(len(le.classes_))),
            )
            save_confusion_matrix(
                cm, list(le.classes_),
                os.path.join(RESULTS_DIR, f"cm_r{r_idx}_b{b_idx}.png"),
                title=f"Confusion Matrix — Round {r_idx}  Bucket {b_idx}",
            )

            # ── Mean curves ──────────────────────────────────
            save_mean_curve(
                fold_results, "train_losses", "test_losses",
                "Loss",
                f"Mean Loss — R{r_idx} B{b_idx}",
                os.path.join(RESULTS_DIR, f"loss_r{r_idx}_b{b_idx}.png"),
            )
            save_mean_curve(
                fold_results, "train_top1s", "test_top1s",
                "Top-1 Accuracy (%)",
                f"Mean Top-1 — R{r_idx} B{b_idx}",
                os.path.join(RESULTS_DIR, f"top1_r{r_idx}_b{b_idx}.png"),
            )
            save_mean_curve(
                fold_results, "train_top5s", "test_top5s",
                "Top-5 Accuracy (%)",
                f"Mean Top-5 — R{r_idx} B{b_idx}",
                os.path.join(RESULTS_DIR, f"top5_r{r_idx}_b{b_idx}.png"),
            )

            # ── Bucket summary ───────────────────────────────
            b_top1 = np.mean([r["top1_acc"]  for r in fold_results])
            b_top5 = np.mean([r["top5_acc"]  for r in fold_results])
            b_f1   = np.mean([r["macro_f1"]  for r in fold_results])
            bucket_top1s.append(b_top1)
            bucket_top5s.append(b_top5)
            bucket_f1s.append(b_f1)

            elapsed = time.time() - overall_start
            print(f"\n  Bucket {b_idx} summary → "
                  f"Top1={b_top1:.2f}%  Top5={b_top5:.2f}%  F1={b_f1:.2f}%  "
                  f"| cumulative elapsed: {elapsed/3600:.2f} h")

        # ── Round aggregate ─────────────────────────────────────
        if not bucket_top1s:
            print(f"\n  ⚠ Round {r_idx} had no valid buckets.")
            continue

        r_top1, r_top5, r_f1 = np.mean(bucket_top1s), np.mean(bucket_top5s), np.mean(bucket_f1s)
        r_std_top1, r_std_top5, r_std_f1 = np.std(bucket_top1s), np.std(bucket_top5s), np.std(bucket_f1s)

        round_results_summary.append((r_top1, r_top5, r_f1))

        with open(round_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                r_idx,
                f"{r_top1:.2f}", f"{r_top5:.2f}", f"{r_f1:.2f}",
                f"{r_std_top1:.2f}", f"{r_std_top5:.2f}", f"{r_std_f1:.2f}",
            ])

        print(f"\n  Round {r_idx} summary → "
              f"Top1={r_top1:.2f}±{r_std_top1:.2f}%  "
              f"Top5={r_top5:.2f}±{r_std_top5:.2f}%  "
              f"F1={r_f1:.2f}±{r_std_f1:.2f}%")

    # ============================================================
    # FINAL AGGREGATE
    # ============================================================
    if not round_results_summary:
        print("\n  ⚠ No round results to aggregate.")
        return

    final_top1 = np.mean([r[0] for r in round_results_summary])
    final_top5 = np.mean([r[1] for r in round_results_summary])
    final_f1   = np.mean([r[2] for r in round_results_summary])
    final_std_top1 = np.std([r[0] for r in round_results_summary])
    final_std_top5 = np.std([r[1] for r in round_results_summary])
    final_std_f1   = np.std([r[2] for r in round_results_summary])

    print("\n\n" + "=" * 70)
    print("  MASA VOCAB-PARTITION FINAL RESULTS (aggregated across rounds)")
    print("=" * 70)
    print(f"  Top-1 Accuracy : {final_top1:.2f}% ± {final_std_top1:.2f}%")
    print(f"  Top-5 Accuracy : {final_top5:.2f}% ± {final_std_top5:.2f}%")
    print(f"  Macro F1       : {final_f1:.2f}% ± {final_std_f1:.2f}%")

    with open(os.path.join(RESULTS_DIR, "final_summary.txt"), "w") as f:
        f.write("MASA Vocabulary-Partitioned LOSO — Final Results\n")
        f.write(f"Rounds  : {NUM_ROUNDS}\n")
        f.write(f"Buckets : {NUM_BUCKETS}\n")
        f.write(f"Folds   : {len(users)}\n")
        f.write(f"Epochs  : {config['epochs']}\n\n")
        f.write(f"Top-1 Accuracy : {final_top1:.2f}% ± {final_std_top1:.2f}%\n")
        f.write(f"Top-5 Accuracy : {final_top5:.2f}% ± {final_std_top5:.2f}%\n")
        f.write(f"Macro F1       : {final_f1:.2f}% ± {final_std_f1:.2f}%\n")

    # ── Difficult words (sorted hardest → easiest) ───────────
    difficult_csv = os.path.join(RESULTS_DIR, "difficult_words.csv")
    with open(difficult_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "accuracy(%)", "num_evaluations"])
        for word, vals in sorted(
            word_metrics.items(),
            key=lambda x: np.mean(x[1])
        ):
            w.writerow([word, f"{np.mean(vals) * 100:.2f}", len(vals)])

    total = time.time() - overall_start
    print(f"\n  Difficult words CSV : {difficult_csv}")
    print(f"  Total wall time     : {total/3600:.2f} h")
    print(f"  All results         : {RESULTS_DIR}")
    print(f"  TensorBoard         : tensorboard --logdir={RESULTS_DIR}/runs")
    print("=" * 70)


if __name__ == "__main__":
    main()
