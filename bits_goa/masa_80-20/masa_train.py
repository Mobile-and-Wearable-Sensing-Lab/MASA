"""
masa_train.py — Train one LOSO fold (MASA experiment)
=====================================================
Defensive version with:
  • Loss-component logging (CE + recon separately)
  • Per-batch NaN guard (skips bad batches, logs count)
  • Optional `dry_run` mode  → uses config["epochs"] or 3 by default
  • PyTorch 2.1.0 — torch.cuda.amp API (stable, deprecation warnings ignorable)
  • Gradient norm tracking
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# TEE
# ============================================================
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# MIXUP (3-scale aware)
# ============================================================
def mixup_data(s1, s2, s3, y, alpha=0.2):
    if alpha <= 0:
        return s1, s2, s3, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    B   = s1.size(0)
    idx = torch.randperm(B, device=s1.device)
    return (
        lam * s1 + (1 - lam) * s1[idx],
        lam * s2 + (1 - lam) * s2[idx],
        lam * s3 + (1 - lam) * s3[idx],
        y, y[idx], lam,
    )


def mixup_criterion(crit, logits, ya, yb, lam):
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)


# ============================================================
# LR SCHEDULE
# ============================================================
def cosine_lr(opt, epoch, warmup, total, base_lr, min_ratio):
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(1, warmup)
    else:
        p  = (epoch - warmup) / max(1, total - warmup)
        lr = base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + np.cos(np.pi * p)))
    for g in opt.param_groups: g["lr"] = lr
    return lr


# ============================================================
# TOP-K ACCURACY
# ============================================================
@torch.no_grad()
def topk_accuracy(logits, targets, k):
    topk    = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1).expand_as(topk)).any(dim=1)
    return correct.float().mean().item() * 100.0


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================
def save_confusion_matrix(cm, class_names, path, title="Confusion Matrix"):
    n = len(class_names)
    fig_size = max(12, n // 3)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n), yticks=np.arange(n),
        xticklabels=class_names, yticklabels=class_names,
        title=title, ylabel="True label", xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             fontsize=max(4, 8 - n // 20))
    plt.setp(ax.get_yticklabels(), fontsize=max(4, 8 - n // 20))
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=4,
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    fold_idx,
    train_paths,
    test_paths,
    label_encoder,
    class_names,
    fold_log_dir,
    device,
    config,
    tb_root,
    dry_run=False,
):
    """
    dry_run=True → train a subset with fewer epochs. For pipeline health checks.
                   If config["epochs"] is supplied it is still respected.
    """
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

    from masa_dataset import (
        MASADataset, collate_fn_masa, compute_normalization_stats, FEAT_DIM
    )
    from masa_model import MASAClassifier

    os.makedirs(fold_log_dir, exist_ok=True)
    tb_dir = os.path.join(tb_root, f"fold_{fold_idx}" + ("_dry" if dry_run else ""))
    writer = SummaryWriter(log_dir=tb_dir)

    # ---- epochs & subset for dry_run ----
    epochs = int(config["epochs"])
    if dry_run:
        sub_train = max(500, len(train_paths) // 10)
        sub_test  = max(200, len(test_paths)  // 5)
        import random as _r
        _r.seed(0)
        train_paths = _r.sample(train_paths, min(sub_train, len(train_paths)))
        test_paths  = _r.sample(test_paths,  min(sub_test,  len(test_paths)))

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}"
          f"{'  [DRY RUN]' if dry_run else ''}  "
          f"|  Test: {len(test_paths)}  Train: {len(train_paths)}"
          f"  |  Epochs: {epochs}")
    print(f"{'='*60}")

    # ---- normalization stats (logging only now) ----
    print("  Normalization diagnostic …")
    _ = compute_normalization_stats(train_paths, sample_limit=500)

    # ---- datasets ----
    train_ds = MASADataset(train_paths, label_encoder,
                           n=config["max_frames"], augment=True)
    test_ds  = MASADataset(test_paths,  label_encoder,
                           n=config["max_frames"], augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
        collate_fn=collate_fn_masa, drop_last=True,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=4 if config["num_workers"] > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
        collate_fn=collate_fn_masa,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=4 if config["num_workers"] > 0 else None,
    )

    # ---- model ----
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

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_p:,}")

    # ---- loss / opt / amp ----
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    use_amp = bool(config.get("use_amp", True)) and torch.cuda.is_available()
    scaler  = GradScaler(enabled=use_amp)
    print(f"  AMP enabled: {use_amp}")

    # ---- history ----
    train_losses, test_losses  = [], []
    train_top1s,  test_top1s   = [], []
    train_top5s,  test_top5s   = [], []
    recon_losses               = []

    best_top1    = 0.0
    best_weights = None
    bad_batches  = 0

    # ==========================
    # TRAINING LOOP
    # ==========================
    for epoch in range(epochs):
        lr = cosine_lr(
            optimizer, epoch,
            config["warmup_epochs"], epochs,
            config["lr"], config["min_lr_ratio"],
        )

        # ---- TRAIN ----
        model.train()
        ep_ce, ep_recon   = 0.0, 0.0
        ep_top1, ep_top5  = 0.0, 0.0
        ep_gnorm          = 0.0
        n_batches         = 0

        for bi, (s1, s2, s3, labels, lengths, pmask) in enumerate(train_loader):
            s1    = s1.to(device, non_blocking=True)
            s2    = s2.to(device, non_blocking=True)
            s3    = s3.to(device, non_blocking=True)
            y     = labels.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)

            # MixUp
            if config.get("use_mixup", True):
                s1, s2, s3, ya, yb, lam = mixup_data(
                    s1, s2, s3, y, alpha=config.get("mixup_alpha", 0.2)
                )
            else:
                ya, yb, lam = y, y, 1.0

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                logits, recon_loss = model(s1, s2, s3, pmask)
                ce_loss = mixup_criterion(criterion, logits, ya, yb, lam)
                loss    = ce_loss + recon_loss

            # ---- NaN guard ----
            if not torch.isfinite(loss):
                bad_batches += 1
                if bad_batches <= 3:
                    print(f"    ⚠️  NaN/Inf loss at epoch {epoch+1} batch {bi} "
                          f"(CE={ce_loss.item()}, Recon={recon_loss.item()}) — skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            scaler.step(optimizer)
            scaler.update()

            ep_ce     += ce_loss.item()
            ep_recon  += recon_loss.item() if torch.is_tensor(recon_loss) else 0.0
            ep_top1   += topk_accuracy(logits, y, 1)
            ep_top5   += topk_accuracy(logits, y, min(5, num_classes))
            ep_gnorm  += gnorm
            n_batches += 1

            # Verbose diagnostic on very first batch of training
            if epoch == 0 and bi == 0:
                print(f"    [first batch] CE={ce_loss.item():.3f}  "
                      f"Recon={recon_loss.item():.3f}  "
                      f"gradnorm={gnorm:.2f}  "
                      f"logits[min,max]=[{logits.min().item():.2f}, "
                      f"{logits.max().item():.2f}]")

        if n_batches == 0:
            print(f"    ❌  All batches failed in epoch {epoch+1}. Abort fold.")
            break

        ep_ce    /= n_batches
        ep_recon /= n_batches
        ep_top1  /= n_batches
        ep_top5  /= n_batches
        ep_gnorm /= n_batches

        train_losses.append(ep_ce)
        recon_losses.append(ep_recon)
        train_top1s.append(ep_top1)
        train_top5s.append(ep_top5)

        # ---- EVAL ----
        model.eval()
        te_loss, te_top1, te_top5 = 0.0, 0.0, 0.0
        all_pred, all_true         = [], []
        n_te                       = 0

        with torch.no_grad():
            for s1, s2, s3, labels, lengths, pmask in test_loader:
                s1    = s1.to(device, non_blocking=True)
                s2    = s2.to(device, non_blocking=True)
                s3    = s3.to(device, non_blocking=True)
                y     = labels.to(device, non_blocking=True)
                pmask = pmask.to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    logits, _ = model(s1, s2, s3, pmask)
                    loss      = criterion(logits, y)

                if not torch.isfinite(loss):
                    continue
                te_loss += loss.item()
                te_top1 += topk_accuracy(logits, y, 1)
                te_top5 += topk_accuracy(logits, y, min(5, num_classes))
                n_te    += 1

                all_pred.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(y.cpu().numpy())

        n_te     = max(1, n_te)
        te_loss /= n_te
        te_top1 /= n_te
        te_top5 /= n_te

        test_losses.append(te_loss)
        test_top1s.append(te_top1)
        test_top5s.append(te_top5)

        if te_top1 > best_top1:
            best_top1    = te_top1
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # TensorBoard
        writer.add_scalar("Loss/train_ce",    ep_ce,    epoch)
        writer.add_scalar("Loss/train_recon", ep_recon, epoch)
        writer.add_scalar("Loss/test_ce",     te_loss,  epoch)
        writer.add_scalar("Top1/train",       ep_top1,  epoch)
        writer.add_scalar("Top1/test",        te_top1,  epoch)
        writer.add_scalar("Top5/train",       ep_top5,  epoch)
        writer.add_scalar("Top5/test",        te_top5,  epoch)
        writer.add_scalar("Grad/norm",        ep_gnorm, epoch)
        writer.add_scalar("LR",               lr,       epoch)

        # Print every epoch in dry_run, every 5 otherwise
        if dry_run or (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Fold {fold_idx} | Ep {epoch+1:>3}/{epochs} | "
                f"LR {lr:.2e} | "
                f"CE {ep_ce:.3f}  Recon {ep_recon:.3f} | "
                f"Te {te_loss:.3f} | "
                f"Top1 {ep_top1:.1f}%/{te_top1:.1f}% | "
                f"Top5 {ep_top5:.1f}%/{te_top5:.1f}% | "
                f"|g| {ep_gnorm:.2f}"
            )

    writer.close()

    if bad_batches > 0:
        print(f"  ⚠️  Total bad batches skipped: {bad_batches}")

    # ---- Final metrics on best weights ----
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for s1, s2, s3, labels, lengths, pmask in test_loader:
            s1    = s1.to(device); s2 = s2.to(device); s3 = s3.to(device)
            pmask = pmask.to(device)
            with autocast(enabled=use_amp):
                logits, _ = model(s1, s2, s3, pmask)
            all_pred.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(labels.numpy())

    all_pred = np.array(all_pred); all_true = np.array(all_true)
    macro_f1    = f1_score(all_true, all_pred, average="macro",    zero_division=0) * 100
    weighted_f1 = f1_score(all_true, all_pred, average="weighted", zero_division=0) * 100

    train_top1_acc = train_top1s[-1] if train_top1s else 0.0
    train_top5_acc = train_top5s[-1] if train_top5s else 0.0

    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    save_confusion_matrix(
        cm, class_names,
        os.path.join(fold_log_dir, "confusion_matrix.png"),
        title=f"Fold {fold_idx} Confusion Matrix",
    )

    torch.save(
        {"model_state": best_weights, "config": config},
        os.path.join(fold_log_dir, "best_model.pt"),
    )

    final_top1 = test_top1s[-1] if test_top1s else 0.0
    final_top5 = test_top5s[-1] if test_top5s else 0.0
    final_loss = test_losses[-1] if test_losses else float("nan")

    print(f"\n  Fold {fold_idx} DONE | "
          f"Best Test Top-1: {best_top1:.2f}% | "
          f"Last Top-1: {final_top1:.2f}% | "
          f"Macro F1: {macro_f1:.2f}%")

    del model, optimizer, scaler, train_loader, test_loader, train_ds, test_ds
    if best_weights is not None: del best_weights
    torch.cuda.empty_cache(); gc.collect()

    return {
        "fold"           : fold_idx,
        "train_top1_acc" : train_top1_acc,
        "train_top5_acc" : train_top5_acc,
        "top1_acc"       : final_top1,
        "top5_acc"       : final_top5,
        "macro_f1"       : macro_f1,
        "weighted_f1"    : weighted_f1,
        "test_loss"      : final_loss,
        "train_losses"   : train_losses,
        "test_losses"    : test_losses,
        "train_top1s"    : train_top1s,
        "test_top1s"     : test_top1s,
        "train_top5s"    : train_top5s,
        "test_top5s"     : test_top5s,
        "recon_losses"   : recon_losses,
        "y_true"         : all_true,
        "y_pred"         : all_pred,
    }