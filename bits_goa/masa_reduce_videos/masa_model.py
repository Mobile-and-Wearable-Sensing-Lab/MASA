"""
masa_model.py — Multi-Scale Sign Language Recognition Model (MASA-inspired)
===========================================================================
NOTE: This is NOT a literal reproduction of the MASA paper. It is a
       multi-scale Transformer classifier with a lightweight motion-
       reconstruction auxiliary task borrowed from MASA's philosophy.

Architecture:
  1. Three independent ScaleEncoders (150, 75, 37 frames)
  2. Attention-weighted temporal pooling per scale
  3. Learned-gate fusion of the three pooled embeddings
  4. Classifier head
  5. Auxiliary motion-residual reconstruction decoder on scale-1 (train only)

Input feature layout per frame (FEAT_DIM = 247):
  [ xy (98) | conf (49) | hand_flags (2) | motion_residuals (98) ]
Reconstruction target = last 98 dims of s1 (the residuals).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# Stochastic Depth (DropPath)
# ==========================================

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


# ==========================================
# Pre-LN Transformer Layer
# ==========================================

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x, key_padding_mask=None):
        n      = self.norm1(x)
        # If a row is entirely padding, attention would emit NaN.
        # Guarantee at least one non-padded position per row.
        if key_padding_mask is not None:
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
        a, _   = self.attn(n, n, n, key_padding_mask=key_padding_mask,
                           need_weights=False)
        x      = x + self.drop_path(a)
        x      = x + self.drop_path(self.ff(self.norm2(x)))
        return x


# ==========================================
# Sinusoidal Positional Encoding
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ==========================================
# Single-Scale Encoder
# ==========================================

class ScaleEncoder(nn.Module):
    """Transformer encoder for one temporal scale. (B, T, feat_dim) → (B, T, d_model)."""
    def __init__(self, feat_dim, d_model, nhead, num_layers,
                 dim_ff, dropout, drop_path_rate, max_len=300):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        dpr = [drop_path_rate * i / max(1, num_layers - 1) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_ff, dropout, dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


# ==========================================
# Attention-Weighted Temporal Pooling
# ==========================================

class AttentionPool(nn.Module):
    """Per-frame importance score → softmax → weighted sum."""
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x, padding_mask=None):
        # x: (B, T, D); padding_mask: (B, T) True=pad
        w = self.score(x).squeeze(-1)                     # (B, T)
        if padding_mask is not None:
            # If a row is fully padded, keep the first position alive
            safe_mask = padding_mask.clone()
            all_pad   = safe_mask.all(dim=1)
            if all_pad.any():
                safe_mask[all_pad, 0] = False
            w = w.masked_fill(safe_mask, float("-inf"))
        w = torch.softmax(w, dim=-1)                      # (B, T)
        return (w.unsqueeze(-1) * x).sum(dim=1)           # (B, D)


# ==========================================
# Multi-Scale Gated Fusion
# ==========================================

class GatedFusion(nn.Module):
    def __init__(self, d_model, num_scales=3):
        super().__init__()
        self.gate = nn.Linear(d_model * num_scales, num_scales)
        # No final projection — gated weighted-sum already returns dim=D.

    def forward(self, embeddings):
        cat     = torch.cat(embeddings, dim=-1)          # (B, D*S)
        g       = torch.softmax(self.gate(cat), dim=-1)  # (B, S)
        stacked = torch.stack(embeddings, dim=1)         # (B, S, D)
        fused   = (g.unsqueeze(-1) * stacked).sum(dim=1) # (B, D)
        return fused


# ==========================================
# Motion Decoder (AE auxiliary)
# ==========================================

class MotionDecoder(nn.Module):
    """Reconstructs motion residuals at masked positions."""
    def __init__(self, d_model, dim_ff, nhead, num_layers, out_dim, dropout=0.1):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dim_ff, dropout, 0.0)
            for _ in range(num_layers)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, memory, masked_pos):
        """
        memory     : (B, T, D)  encoder output (all frames)
        masked_pos : (B, T) bool
        """
        B, T, D = memory.shape
        x = memory.clone()
        # Replace masked positions with the learnable mask token
        x = torch.where(
            masked_pos.unsqueeze(-1),
            self.mask_token.expand(B, T, D),
            x,
        )
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.out_proj(x)


# ==========================================
# MASA Classifier (full model)
# ==========================================

class MASAClassifier(nn.Module):

    FEAT_DIM = 247   # must match masa_dataset.FEAT_DIM

    def __init__(
        self,
        feat_dim        = 247,
        num_classes     = 489,
        model_dim       = 256,
        nhead           = 8,
        num_layers      = 4,
        dim_feedforward = 512,
        dropout         = 0.3,
        drop_path_rate  = 0.1,
        mask_ratio      = 0.4,
        recon_weight    = 0.1,
        decoder_layers  = 2,
    ):
        super().__init__()

        self.mask_ratio   = float(mask_ratio)
        self.recon_weight = float(recon_weight)
        RECON_DIM = 49 * 2   # 98 (xy residuals)

        enc_kwargs = dict(
            feat_dim        = feat_dim,
            d_model         = model_dim,
            nhead           = nhead,
            num_layers      = num_layers,
            dim_ff          = dim_feedforward,
            dropout         = dropout,
            drop_path_rate  = drop_path_rate,
        )
        # INDEPENDENT encoders per scale (not shared)
        self.encoder1 = ScaleEncoder(**enc_kwargs, max_len=200)
        self.encoder2 = ScaleEncoder(**enc_kwargs, max_len=120)
        self.encoder3 = ScaleEncoder(**enc_kwargs, max_len=60)

        self.pool1 = AttentionPool(model_dim)
        self.pool2 = AttentionPool(model_dim)
        self.pool3 = AttentionPool(model_dim)

        self.fusion = GatedFusion(model_dim, num_scales=3)

        self.decoder = MotionDecoder(
            d_model   = model_dim,
            dim_ff    = dim_feedforward,
            nhead     = nhead,
            num_layers= decoder_layers,
            out_dim   = RECON_DIM,
            dropout   = dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(0.3),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(model_dim // 2, num_classes),
        )

        self._init_weights()

    # ---- weight init ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- random masking ----
    def _random_mask(self, B, T, device):
        num_mask = max(1, int(T * self.mask_ratio))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.randperm(T, device=device)[:num_mask]
            mask[i, idx] = True
        return mask

    # ---- forward ----
    def forward(self, s1, s2, s3, padding_mask=None):
        """
        s1: (B, n,    FEAT_DIM)
        s2: (B, n/2,  FEAT_DIM)
        s3: (B, n/4,  FEAT_DIM)
        padding_mask: (B, n) True=pad  (for scale 1 only)
        """
        B = s1.size(0)

        h1 = self.encoder1(s1, key_padding_mask=padding_mask)
        h2 = self.encoder2(s2)
        h3 = self.encoder3(s3)

        z1 = self.pool1(h1, padding_mask)
        z2 = self.pool2(h2)
        z3 = self.pool3(h3)

        z      = self.fusion([z1, z2, z3])
        logits = self.classifier(z)

        # Auxiliary motion reconstruction (train only)
        recon_loss = s1.new_zeros(())
        if self.training and self.recon_weight > 0.0:
            T1         = s1.size(1)
            masked_pos = self._random_mask(B, T1, s1.device)    # (B, T)

            # If frames are padded, do NOT mask them (target is zero anyway)
            if padding_mask is not None:
                masked_pos = masked_pos & (~padding_mask)

            pred_res   = self.decoder(h1, masked_pos)           # (B, T, 98)
            gt_res     = s1[:, :, -98:]                         # last 98 dims

            # MSE over masked positions only
            mask_f     = masked_pos.unsqueeze(-1).float()       # (B, T, 1)
            denom      = mask_f.sum().clamp(min=1.0) * pred_res.size(-1)
            recon_loss = ((pred_res - gt_res) ** 2 * mask_f).sum() / denom
            recon_loss = self.recon_weight * recon_loss

        return logits, recon_loss

    # ---- embedding extraction (optional) ----
    @torch.no_grad()
    def get_embeddings(self, s1, s2, s3, padding_mask=None):
        self.eval()
        h1 = self.encoder1(s1, key_padding_mask=padding_mask)
        h2 = self.encoder2(s2)
        h3 = self.encoder3(s3)
        z1 = self.pool1(h1, padding_mask)
        z2 = self.pool2(h2)
        z3 = self.pool3(h3)
        return self.fusion([z1, z2, z3])
