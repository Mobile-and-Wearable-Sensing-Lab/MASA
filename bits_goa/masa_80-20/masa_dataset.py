"""
masa_dataset.py — Dataset for MMPose .npy whole-body keypoints (MASA experiment)
================================================================================
Input: .npy files produced by MMPose COCO-WholeBody (133 joints, 3 channels).
       Shape per file: (T, 133, 3)  where last dim = (x, y, score)

Pipeline per sample:
    load → slice to 49 joints → zero low-conf kpts → trim silence
    → normalize to shoulder-width units (body and hands)
    → build 3 temporal scales → pack features → clip → return

Feature dim per frame (FEAT_DIM):
    49 joints × 2 (xy)  +  49 joints × 1 (conf)  +  2 (hand flags)
    + 49 joints × 2 (motion residuals)            =  247
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
# COCO-WholeBody 133: we use body[0:7] + left_hand[91:112] + right_hand[112:133]
# = 7 + 21 + 21 = 49 joints.
BODY_JOINTS  = list(range(0, 7))
LHAND_JOINTS = list(range(91, 112))
RHAND_JOINTS = list(range(112, 133))
MASA_JOINT_INDICES = BODY_JOINTS + LHAND_JOINTS + RHAND_JOINTS

NUM_JOINTS  = 49
CONF_THRESH = 0.3        # MMPose hand scores are often low; don't be too strict
MAX_ABS     = 10.0       # shoulder-widths; hard clip guard

# Slice positions after re-indexing:
#   body  : 0:7   (0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear, 5=L_sh, 6=R_sh)
#   lhand : 7:28  (wrist at idx 7)
#   rhand : 28:49 (wrist at idx 28)

# Symmetric body keypoint pairs for horizontal mirroring
BODY_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6)]

# Final per-frame feature dim
FEAT_DIM = NUM_JOINTS * 2 + NUM_JOINTS + 2 + NUM_JOINTS * 2   # 98 + 49 + 2 + 98 = 247


# ============================================================
# LOADING + PREPROCESSING
# ============================================================

def load_npy(path: str) -> np.ndarray:
    """Load .npy → float32 (T, 49, 3). Robust to shape variants."""
    data = np.load(path, allow_pickle=True)

    if data.dtype == object:
        data = data.item() if data.ndim == 0 else data[0]
    if isinstance(data, dict):
        for key in ("keypoints", "pose", "joints", "data"):
            if key in data:
                data = data[key]
                break

    data = np.array(data, dtype=np.float32)

    if data.ndim == 2:                      # (N, C) single frame
        data = data[np.newaxis]
    if data.ndim == 4:                      # (T, 1, N, C)
        data = data[:, 0]

    T, N, C = data.shape

    # Slice to 49 joints
    if N >= 133:
        data = data[:, MASA_JOINT_INDICES, :]
    elif N != 49:
        if N < 49:
            pad = np.zeros((T, 49 - N, C), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)
        else:
            data = data[:, :49, :]

    # Ensure 3 channels (x, y, score)
    if C < 3:
        score = np.ones((T, NUM_JOINTS, 1), dtype=np.float32)
        data  = np.concatenate([data[:, :, :2], score], axis=2)
    elif C > 3:
        data = data[:, :, :3]

    # Replace NaN/Inf from upstream
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero out low-confidence keypoints (x,y only — keep the conf value)
    conf = data[:, :, 2:3]
    mask = (conf < CONF_THRESH).astype(np.float32)
    data[:, :, :2] = data[:, :, :2] * (1.0 - mask)

    return data  # (T, 49, 3)


def trim_to_active(data: np.ndarray, min_hand_energy: float = 1.0) -> np.ndarray:
    """Remove leading/trailing frames where no hand is visible."""
    hands = data[:, 7:, :2]                  # (T, 42, 2), pixel coords
    energy = np.abs(hands).sum(axis=(1, 2))  # (T,)
    active = np.where(energy > min_hand_energy)[0]
    if len(active) == 0:
        return data
    return data[active[0]: active[-1] + 1]


def normalize_body_relative(data: np.ndarray) -> np.ndarray:
    """
    Convert everything to **shoulder-width units** centered on mid-shoulder.
    Body   : (xy - mid_shoulder) / shoulder_width
    Hands  : (xy - wrist)        / shoulder_width
    Robust to frames where shoulders or hands are missing.
    """
    data = data.copy().astype(np.float32)
    T = data.shape[0]

    ls = data[:, 5, :2]    # left shoulder
    rs = data[:, 6, :2]    # right shoulder
    mid_shoulder   = (ls + rs) / 2.0
    shoulder_width = np.linalg.norm(ls - rs, axis=1)   # (T,)

    # Per-frame validity: both shoulders present and apart enough
    valid = shoulder_width > 20.0     # at least 20 px between shoulders

    # Sequence-wide fallback from valid frames
    if valid.any():
        sw_fallback  = float(np.median(shoulder_width[valid]))
        mid_fallback = np.median(mid_shoulder[valid], axis=0).astype(np.float32)
    else:
        sw_fallback  = 100.0
        mid_fallback = np.array([0.0, 0.0], dtype=np.float32)

    sw  = np.where(valid, shoulder_width, sw_fallback).astype(np.float32)
    sw  = np.maximum(sw, 20.0)
    mid = np.where(valid[:, None], mid_shoulder, mid_fallback).astype(np.float32)

    # --- Body ---
    data[:, :7, :2] = (data[:, :7, :2] - mid[:, None]) / sw[:, None, None]

    # --- Left hand (joints 7:28, wrist at 7) ---
    lhand_energy = np.abs(data[:, 7:28, :2]).sum(axis=(1, 2))
    active = lhand_energy > 1e-3
    if active.any():
        lw = data[active, 7:8, :2]
        data[active, 7:28, :2] = (data[active, 7:28, :2] - lw) / sw[active, None, None]

    # --- Right hand (joints 28:49, wrist at 28) ---
    rhand_energy = np.abs(data[:, 28:, :2]).sum(axis=(1, 2))
    active = rhand_energy > 1e-3
    if active.any():
        rw = data[active, 28:29, :2]
        data[active, 28:, :2] = (data[active, 28:, :2] - rw) / sw[active, None, None]

    # Final hard clamp — nothing in pose space should exceed ±10 shoulder-widths
    data[:, :, :2] = np.clip(data[:, :, :2], -MAX_ABS, MAX_ABS)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def compute_hand_flags(data: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Per-frame binary flags: (left_hand_present, right_hand_present)."""
    left  = (np.abs(data[:, 7:28, :2]).sum(axis=(1, 2)) > threshold).astype(np.float32)
    right = (np.abs(data[:, 28:,  :2]).sum(axis=(1, 2)) > threshold).astype(np.float32)
    return np.stack([left, right], axis=1)


# ============================================================
# NORMALIZATION STATS
# ============================================================

def compute_normalization_stats(paths, sample_limit=500):
    """
    Stats are NOT used for division anymore — just for logging so you can
    see the distribution is sane. We return an identity transform.
    """
    import random as _r
    all_xy = []
    sampled = paths if len(paths) <= sample_limit else _r.sample(paths, sample_limit)
    n_ok = 0
    for path in sampled:
        try:
            d = load_npy(path)
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
    """Pack (n, FEAT_DIM)."""
    xy_flat   = xy.reshape(n, -1)           # (n, 98)
    conf_flat = conf.reshape(n, -1)         # (n, 49)
    res_flat  = residuals.reshape(n, -1)    # (n, 98)
    return np.concatenate([xy_flat, conf_flat, flags, res_flat], axis=1)


# ============================================================
# DATASET
# ============================================================

class MASADataset(torch.utils.data.Dataset):
    """
    Returns 3 temporal scales per sample:
        s1: n       frames (full)
        s2: n // 2  frames (stride 2)
        s3: n // 4  frames (stride 4)
    Each shape: (scale_len, FEAT_DIM)
    """

    FEAT_DIM = FEAT_DIM

    def __init__(
        self,
        file_paths,
        label_encoder,
        n=150,
        normalize_stats=None,   # accepted but unused (kept for API compat)
        augment=False,
        hand_threshold=1e-3,
        mirror_prob=0.3,        # horizontal-flip probability (set 0 to disable)
    ):
        self.paths          = list(file_paths)
        self.n              = n
        self.n2             = max(n // 2, 4)
        self.n3             = max(n // 4, 4)
        self.augment        = augment
        self.hand_threshold = hand_threshold
        self.mirror_prob    = mirror_prob

        if not self.paths:
            raise ValueError("No .npy paths given.")

        self.labels_str     = [label_from_filename(p) for p in self.paths]
        self.encoded_labels = label_encoder.transform(self.labels_str)

    def __len__(self):
        return len(self.paths)

    # ---- helpers ----
    def _empty_sample(self, label):
        s1 = torch.zeros(self.n,  self.FEAT_DIM, dtype=torch.float32)
        s2 = torch.zeros(self.n2, self.FEAT_DIM, dtype=torch.float32)
        s3 = torch.zeros(self.n3, self.FEAT_DIM, dtype=torch.float32)
        return (s1, s2, s3,
                torch.tensor(label,   dtype=torch.long),
                torch.tensor(0,       dtype=torch.long))

    def _build_scale(self, data, target_n):
        """data: (T, 49, 3) already normalized. Returns (target_n, FEAT_DIM)."""
        T = data.shape[0]
        xy    = data[:, :, :2].copy()                             # (T, 49, 2)
        conf  = data[:, :, 2:3].copy()                            # (T, 49, 1)
        flags = compute_hand_flags(data, self.hand_threshold)     # (T, 2)

        # Residuals on normalized xy (AMP-safe)
        res = np.zeros_like(xy)
        if T > 1:
            res[:-1] = xy[1:] - xy[:-1]

        # Pad/trim
        xy,    orig_len = pad_or_trim(xy,    target_n)
        conf,  _        = pad_or_trim(conf,  target_n)
        flags, _        = pad_or_trim(flags, target_n)
        res,   _        = pad_or_trim(res,   target_n)

        feat = build_feature(xy, conf, flags, res, target_n)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        feat = np.clip(feat, -MAX_ABS, MAX_ABS)
        return feat.astype(np.float32), orig_len

    # ---- main ----
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

    # ---- augmentation ----
    def _mirror(self, data: np.ndarray) -> np.ndarray:
        """
        Horizontal flip:
          - negate x (coords already centered at 0 after normalize_body_relative)
          - swap symmetric body keypoints (eyes, ears, shoulders)
          - swap entire left-hand and right-hand blocks
        Confidence scores are swapped along with the keypoints they belong to.
        """
        data = data.copy()

        # 1) Negate x (valid because body is centered on mid-shoulder,
        #    and each hand is centered on its own wrist → all xy around 0)
        data[:, :, 0] *= -1.0

        # 2) Swap symmetric body pairs (x, y, conf all together)
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

        # Temporal mask
        if random.random() < 0.3 and T > 8:
            ml    = random.randint(3, min(10, T // 3))
            start = random.randint(0, T - ml)
            data  = data.copy()
            data[start: start + ml, :, :2] = 0.0
            data[start: start + ml, :, 2]  = 0.0

        # Clip after augment
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

    # Final safety — any NaN produced anywhere → zero
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
