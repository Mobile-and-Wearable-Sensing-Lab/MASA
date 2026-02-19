import os
import numpy as np
import torch
import math
from torch.utils.data import Dataset

class ISL_GOA(Dataset):
    def __init__(self, data_root, split="train", list_file=None, pose_subdir="Pose", score_thr=0.10):
        self.data_root = data_root
        self.split = split
        self.score_thr = score_thr
        self.pose_root = os.path.join(self.data_root, pose_subdir)

        # Indices for COCO-WholeBody (133 joints)
        self.LH = slice(91, 112)
        self.RH = slice(112, 133)
        self.BODY_7 = [0, 5, 6, 7, 8, 9, 10]

        if list_file is None:
            list_file = os.path.join(self.data_root, "Annotations", "pretrain_list.txt")

        self.samples = []
        with open(list_file, "r") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    self.samples.append((parts[0], int(parts[1]) if len(parts) > 1 else 0))

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx):
        rel_path, label = self.samples[idx]
        pose_path = os.path.join(self.pose_root, rel_path)
        
        if not os.path.exists(pose_path):
            return self.get_sample(0)

        # Load [T, 133, 3] -> (x, y, score)
        arr = np.load(pose_path).astype(np.float32)
        
        # --- DYNAMIC RESOLUTION LOGIC ---
        # If your extraction script didn't save resolution, 
        # use the max coordinates in the skeleton as a proxy for resolution.
        # Otherwise, if you saved (W, H) in the filename or a sidecar file, load it here.
        valid_coords = arr[..., :2][arr[..., 2] > self.score_thr]
        if len(valid_coords) > 0:
            # Estimate resolution from the furthest points found in the video
            current_width = np.max(valid_coords[:, 0])
            current_height = np.max(valid_coords[:, 1])
        else:
            current_width, current_height = 1920, 1080 # Fallback
            
        T = arr.shape[0]

        # 1. Body Processing
        body_raw = arr[:, self.BODY_7, :2]
        body_sc  = arr[:, self.BODY_7, 2]
        
        body_xy = np.zeros_like(body_raw)
        body_xy[..., 0] = body_raw[..., 0] / current_width
        body_xy[..., 1] = body_raw[..., 1] / current_height
        
        # 2. Hand Processing
        right_data = self.process_part(arr[:, self.RH, :], current_width, current_height)
        left_data  = self.process_part(arr[:, self.LH, :], current_width, current_height)

        return {
            "right": right_data,
            "left":  left_data,
            "body": {
                "body_pose": torch.from_numpy(body_xy).float(),
                "body_pose_gt": torch.from_numpy(body_xy * 256.0).float(),
                "body_pose_conf": (torch.from_numpy(body_sc) > self.score_thr).float().unsqueeze(-1),
                "root_mask": torch.zeros((T, 7, 1)) 
            },
            "label": torch.tensor(label, dtype=torch.long)
        }

    def process_part(self, raw_part, w, h):
        xy = raw_part[..., :2]
        sc = raw_part[..., 2]
        
        norm_xy = np.zeros_like(xy)
        norm_xy[..., 0] = xy[..., 0] / w
        norm_xy[..., 1] = xy[..., 1] / h
        
        return {
            "kp2d": torch.from_numpy(norm_xy).float(),
            "gts": torch.from_numpy(norm_xy * 256.0).float(), 
            "flag_2d": (torch.from_numpy(sc) > self.score_thr).float().unsqueeze(-1),
            "mask": torch.zeros((xy.shape[0], 21, 1))
        }
