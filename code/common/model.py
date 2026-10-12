"""Shared detector head used by every modality.

encoder embedding  ─┐
                    ├─ concat ─ fuse MLP ─┬─ projection head ─> z   (contrastive loss)
handcrafted feats  ─┘                     └─ classifier ──────> logit (BCE loss)
"""
import torch
import torch.nn as nn


class DetectorHead(nn.Module):
    def __init__(self, enc_dim, feat_dim, hidden=256, proj_dim=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.fuse = nn.Sequential(
            nn.Linear(enc_dim + feat_dim, hidden), nn.GELU(), nn.Dropout(0.1)
        )
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, proj_dim))
        self.cls = nn.Linear(hidden, 1)

    def forward(self, enc, feats=None):
        x = enc if feats is None or self.feat_dim == 0 else torch.cat([enc, feats], dim=1)
        h = self.fuse(x)
        return self.cls(h).squeeze(1), self.proj(h)
