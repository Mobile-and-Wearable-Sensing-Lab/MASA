"""
masa_dataset.py — Dataset for MMPose .npy whole-body keypoints (MASA experiment)
================================================================================
5-FPS VARIANT. Accepts a `fps_stride` parameter to downsample incoming
30-fps NPY files on-the-fly.

To get 5 fps from 30-fps NPY files, use fps_stride=6.
"""

import os
import math
import random
import numpy as np
import torch


# ============================================================
# FILE / LABEL UTILS
# ============================================================

def find_npy_files(root_dir):
    paths = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                paths.append(os.path.join(r, f))
    return sorted(paths)


LABEL_MERGE_MAP = {
    "byte": "bite", "glove": "gloves", "lady": "woman",
    "female": "woman", "few": "some", "after": "next",
    "over": "finish", "large": "big", "alot": "big",
    "this": "it", "that": "it", "photgraph": "photograph",
}


def label_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0].lower()

    if "__" in stem:
        first = stem.split("__", 1)[0]
        if first.startswith("_"):
            parts = [p for p in first.split("_") if p]
            label = parts[-1]
        else:
            label = first
    elif "_" in stem:
        label = stem.split("_", 1)[0]
    else:
        label = stem

    return LABEL_MERGE_MAP.get(label, label)


def user_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part.lower().startswith("isl_data_user"):
            return part
    for part in parts:
        if part.lower().startswith("user"):
            return part
    return os.path.basename(os.path.dirname(path))


# ============================================================
# JOINT LAYOUT
# ============================================================
# COCO-WholeBody 133: body[0:7] + left_hand[91:112] + right_hand[112:133]
# After slicing, new indices:
#   body  : 0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear, 5=L_sh, 6=R_sh
#   lhand : 7:28  (wrist at 7)
#   rhand : 28:49 (wrist at 28)
BODY_JOINTS  = list(range(0, 7))
LHAND_JOINTS = list(range(91, 112))
RHAND_JOINTS = list(range(112, 133))
MASA_JOINT_INDICES = BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS

NUM_JOINTS  = 49
CONF_THRESH = 0.3
MAX_ABS     = 10.0

# Symmetric body keypoint pairs for horizontal mirroring (left↔right)
BODY_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6)]

FEAT_DIM = NUM_JOINTS * 2 + NUM_JOINTS + 2 + NUM_JOINTS * 2   # = 247


# ============================================================
# LOADING + PREPROCESSING
# ============================================================

def load_npy(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)

    if data.dtype == object:
        data = data.item() if data.ndim == 0 else data[0]
    if isinstance(data, dict):
        for key in ("keypoints", "pose", "joints", "data"):
            if key in data:
                data = data[key]
                break

    data = np.array(data, dtype=np.float32)

    if data.ndim == 2:
        data = data[np.newaxis]
    if data.ndim == 4:
        data = data[:, 0]

    T, N, C = data.shape

    if N >= 133:
        data = data[:, MASA_JOINT_INDICES, :]
    elif N != 49:
        if N < 49:
            pad = np.zeros((T, 49 - N, C), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)
        else:
            data = data[:, :49, :]

    if C < 3:
        score = np.ones((T, NUM_JOINTS, 1), dtype=np.float32)
        data  = np.concatenate([data[:, :, :2], score], axis=2)
    elif C > 3:
        data = data[:, :, :3]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    conf = data[:, :, 2:3]
    mask = (conf < CONF_THRESH).astype(np.float32)
    data[:, :, :2] = data[:, :, :2] * (1.0 - mask)

    return data


def trim_to_active(data: np.ndarray, min_hand_energy: float = 1.0) -> np.ndarray:
    hands = data[:, 7:, :2]
    energy = np.abs(hands).sum(axis=(1, 2))
    active = np.where(energy > min_hand_energy)[0]
    if len(active) == 0:
        return data
    return data[active[0]: active[-1] + 1]


def normalize_body_relative(data: np.ndarray) -> np.ndarray:
    data = data.copy().astype(np.float32)
    T = data.shape[0]

    ls = data[:, 5, :2]
    rs = data[:, 6, :2]
    mid_shoulder   = (ls + rs) / 2.0
    shoulder_width = np.linalg.norm(ls - rs, axis=1)

    valid = shoulder_width > 20.0

    if valid.any():
        sw_fallback  = float(np.median(shoulder_width[valid]))
        mid_fallback = np.median(mid_shoulder[valid], axis=0).astype(np.float32)
    else:
        sw_fallback  = 100.0
        mid_fallback = np.array([0.0, 0.0], dtype=np.float32)

    sw  = np.where(valid, shoulder_width, sw_fallback).astype(np.float32)
    sw  = np.maximum(sw, 20.0)
    mid = np.where(valid[:, None], mid_shoulder, mid_fallback).astype(np.float32)

    data[:, :7, :2] = (data[:, :7, :2] - mid[:, None]) / sw[:, None, None]

    lhand_energy = np.abs(data[:, 7:28, :2]).sum(axis=(1, 2))
    active = lhand_energy > 1e-3
    if active.any():
        lw = data[active, 7:8, :2]
        data[active, 7:28, :2] = (data[active, 7:28, :2] - lw) / sw[active, None, None]

    rhand_energy = np.abs(data[:, 28:, :2]).sum(axis=(1, 2))
    active = rhand_energy > 1e-3
    if active.any():
        rw = data[active, 28:29, :2]
        data[active, 28:, :2] = (data[active, 28:, :2] - rw) / sw[active, None, None]

    data[:, :, :2] = np.clip(data[:, :, :2], -MAX_ABS, MAX_ABS)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def compute_hand_flags(data: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    left  = (np.abs(data[:, 7:28, :2]).sum(axis=(1, 2)) > threshold).astype(np.float32)
    right = (np.abs(data[:, 28:,  :2]).sum(axis=(1, 2)) > threshold).astype(np.float32)
    return np.stack([left, right], axis=1)


def compute_normalization_stats(paths, sample_limit=500, fps_stride=1):
    import random as _r
    all_xy = []
    sampled = paths if len(paths) <= sample_limit else _r.sample(paths, sample_limit)
    n_ok = 0
    for path in sampled:
        try:
            d = load_npy(path)
            if d.shape[0] == 0:
                continue
            if fps_stride > 1:
                d = d[::fps_stride]
            if d.shape[0] == 0:
                continue
            d = trim_to_active(d)
            if d.shape[0] == 0:
                continue
            d = normalize_body_relative(d)
            xy = d[:, :, :2].reshape(d.shape[0], -1)
            all_xy.append(xy)
            n_ok += 1
        except Exception:
            continue

    if all_xy:
        concat = np.concatenate(all_xy, axis=0)
        concat = concat[np.isfinite(concat).all(axis=1)]
        print(f"[NormStats] OK files: {n_ok}/{len(sampled)} | "
              f"frames: {concat.shape[0]} | "
              f"xy range: [{concat.min():.3f}, {concat.max():.3f}] | "
              f"mean {concat.mean():.3f} std {concat.std():.3f}")

    D = NUM_JOINTS * 2
    return {"mean": np.zeros(D, dtype=np.float32),
            "std":  np.ones(D,  dtype=np.float32)}


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def downsample(seq: np.ndarray, stride: int) -> np.ndarray:
    return seq if stride <= 1 else seq[::stride]


def pad_or_trim(seq: np.ndarray, target_len: int):
    T = seq.shape[0]
    orig_len = min(T, target_len)
    if T > target_len:
        start = (T - target_len) // 2
        seq   = seq[start: start + target_len]
    elif T < target_len:
        pad = np.zeros((target_len - T,) + seq.shape[1:], dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    return seq, orig_len


def build_feature(xy, conf, flags, residuals, n):
    xy_flat   = xy.reshape(n, -1)
    conf_flat = conf.reshape(n, -1)
    res_flat  = residuals.reshape(n, -1)
    return np.concatenate([xy_flat, conf_flat, flags, res_flat], axis=1)


# ============================================================
# DATASET
# ============================================================

class MASADataset(torch.utils.data.Dataset):

    FEAT_DIM = FEAT_DIM

    def __init__(
        self,
        file_paths,
        label_encoder,
        n=30,                             # default reduced to 30 for 5fps
        normalize_stats=None,
        augment=False,
        hand_threshold=1e-3,
        fps_stride=1,                     # 1=no change, 6=30→5 fps
        mirror_prob=0.3,                  # horizontal-flip probability (0 to disable)
    ):
        self.paths          = list(file_paths)
        self.n              = n
        self.n2             = max(n // 2, 4)
        self.n3             = max(n // 4, 4)
        self.augment        = augment
        self.hand_threshold = hand_threshold
        self.fps_stride     = max(1, int(fps_stride))
        self.mirror_prob    = float(mirror_prob)

        if not self.paths:
            raise ValueError("No .npy paths given.")

        self.labels_str     = [label_from_filename(p) for p in self.paths]
        self.encoded_labels = label_encoder.transform(self.labels_str)

    def __len__(self):
        return len(self.paths)

    def _empty_sample(self, label):
        s1 = torch.zeros(self.n,  self.FEAT_DIM, dtype=torch.float32)
        s2 = torch.zeros(self.n2, self.FEAT_DIM, dtype=torch.float32)
        s3 = torch.zeros(self.n3, self.FEAT_DIM, dtype=torch.float32)
        return (s1, s2, s3,
                torch.tensor(label,   dtype=torch.long),
                torch.tensor(0,       dtype=torch.long))

    def _build_scale(self, data, target_n):
        T = data.shape[0]
        xy    = data[:, :, :2].copy()
        conf  = data[:, :, 2:3].copy()
        flags = compute_hand_flags(data, self.hand_threshold)

        res = np.zeros_like(xy)
        if T > 1:
            res[:-1] = xy[1:] - xy[:-1]

        xy,    orig_len = pad_or_trim(xy,    target_n)
        conf,  _        = pad_or_trim(conf,  target_n)
        flags, _        = pad_or_trim(flags, target_n)
        res,   _        = pad_or_trim(res,   target_n)

        feat = build_feature(xy, conf, flags, res, target_n)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat = np.clip(feat, -MAX_ABS, MAX_ABS)
        return feat.astype(np.float32), orig_len

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = int(self.encoded_labels[idx])

        try:
            data = load_npy(path)
        except Exception as e:
            print(f"[MASADataset] load error {path}: {e}")
            return self._empty_sample(label)

        if data.shape[0] == 0 or not np.isfinite(data).all():
            return self._empty_sample(label)

        # FPS reduction
        if self.fps_stride > 1:
            data = data[::self.fps_stride]
            if data.shape[0] == 0:
                return self._empty_sample(label)

        data = trim_to_active(data)
        if data.shape[0] == 0:
            return self._empty_sample(label)

        data = normalize_body_relative(data)
        if not np.isfinite(data).all():
            return self._empty_sample(label)

        if self.augment:
            data = self._augment(data)

        s1_feat, orig_len = self._build_scale(data,                 self.n)
        s2_feat, _        = self._build_scale(downsample(data, 2),  self.n2)
        s3_feat, _        = self._build_scale(downsample(data, 4),  self.n3)

        return (
            torch.from_numpy(s1_feat),
            torch.from_numpy(s2_feat),
            torch.from_numpy(s3_feat),
            torch.tensor(label,    dtype=torch.long),
            torch.tensor(orig_len, dtype=torch.long),
        )

    # ---- augmentation helpers ----
    def _mirror(self, data: np.ndarray) -> np.ndarray:
        """
        Horizontal flip:
          - negate x (coords already centered at 0 after normalize_body_relative)
          - swap symmetric body keypoints (eyes, ears, shoulders)
          - swap entire left-hand and right-hand blocks
        """
        data = data.copy()

        # 1) Negate x (body centered on mid-shoulder; each hand centered on its wrist)
        data[:, :, 0] *= -1.0

        # 2) Swap symmetric body pairs (x, y, conf together)
        for l, r in BODY_FLIP_PAIRS:
            tmp = data[:, l, :].copy()
            data[:, l, :] = data[:, r, :]
            data[:, r, :] = tmp

        # 3) Swap left hand (7:28) and right hand (28:49) blocks
        lhand = data[:, 7:28,  :].copy()
        rhand = data[:, 28:49, :].copy()
        data[:, 7:28,  :] = rhand
        data[:, 28:49, :] = lhand

        return data

    def _augment(self, data: np.ndarray) -> np.ndarray:
        T = data.shape[0]

        # Temporal resample
        if random.random() < 0.7 and T > 4:
            factor  = random.uniform(0.8, 1.25)
            new_len = max(4, int(T * factor))
            idx     = np.linspace(0, T - 1, new_len).astype(int)
            data    = data[idx]
            T       = new_len

        # Horizontal mirror (left/right flip)
        if self.mirror_prob > 0.0 and random.random() < self.mirror_prob:
            data = self._mirror(data)

        # Spatial scale + rotation + shift on xy
        if random.random() < 0.8:
            xy    = data[:, :, :2].copy()
            scale = random.uniform(0.9, 1.1)
            angle = random.uniform(-10, 10) * math.pi / 180
            c, s  = math.cos(angle), math.sin(angle)
            R     = np.array([[c, -s], [s, c]], dtype=np.float32)
            xy   *= scale
            xy    = xy @ R.T
            xy   += np.random.uniform(-0.05, 0.05, (1, 1, 2)).astype(np.float32)
            data  = data.copy()
            data[:, :, :2] = xy

        # Gaussian noise (small — units are shoulder-widths)
        if random.random() < 0.5:
            n = np.random.normal(0, 0.01, data[:, :, :2].shape).astype(np.float32)
            data = data.copy()
            data[:, :, :2] += n

        # Temporal mask — tuned for short 5fps sequences
        if random.random() < 0.3 and T > 4:
            ml    = random.randint(1, max(1, T // 4))
            start = random.randint(0, T - ml)
            data  = data.copy()
            data[start: start + ml, :, :2] = 0.0
            data[start: start + ml, :, 2]  = 0.0

        data[:, :, :2] = np.clip(data[:, :, :2], -MAX_ABS, MAX_ABS)
        return data


# ============================================================
# COLLATE
# ============================================================

def collate_fn_masa(batch):
    s1s, s2s, s3s, labels, lengths = zip(*batch)

    s1 = torch.stack(s1s)
    s2 = torch.stack(s2s)
    s3 = torch.stack(s3s)
    labels  = torch.stack(labels).long()
    lengths = torch.stack(lengths).long()

    s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
    s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
    s3 = torch.nan_to_num(s3, nan=0.0, posinf=0.0, neginf=0.0)

    B, T1, _ = s1.shape
    padding_mask = torch.zeros(B, T1, dtype=torch.bool)
    for i, L in enumerate(lengths):
        L = int(L.item())
        if L < T1:
            padding_mask[i, L:] = True

    return s1, s2, s3, labels, lengths, padding_mask