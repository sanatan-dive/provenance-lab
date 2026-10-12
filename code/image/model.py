"""Dual-branch image detector.

Branch A: ViT on the RGB image (32x32 upsampled to 224) -> CLS embedding (768-d).
Branch B: small CNN on the 32x32 FFT log-magnitude spectrum -> 128-d embedding.
[A | B] is the encoder embedding; radial-profile / residual scalars are the handcrafted feats.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

from common.model import DetectorHead

from .features import N_RADIAL

# ViT-small (22M params, 384-d CLS): ViT-base at batch 32 / 224px exceeds the 4.1GB MPS cap we
# run under (OOM in smoke test); small keeps the batch at 32, which SupCon needs for positives.
VIT_NAME = "WinKawaks/vit-small-patch16-224"
FEAT_DIM = N_RADIAL + 3


class FreqCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        layers, c_in = [], 1
        for c_out in (32, 64, 128):  # 32 -> 16 -> 8 -> 4
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                       nn.MaxPool2d(2)]
            c_in = c_out
        layers += [nn.Conv2d(c_in, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(),
                   nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ImageDetector(nn.Module):
    def __init__(self, use_handcrafted=True, use_freq=True, vit_name=VIT_NAME):
        super().__init__()
        self.use_handcrafted, self.use_freq = use_handcrafted, use_freq
        self.vit = ViTModel.from_pretrained(vit_name, add_pooling_layer=False)
        self.freq = FreqCNN() if use_freq else None
        enc_dim = self.vit.config.hidden_size + (128 if use_freq else 0)
        self.head = DetectorHead(enc_dim, FEAT_DIM if use_handcrafted else 0)

    def forward(self, batch):
        x = batch["pixels"].float() / 255.0
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5  # google/vit-* image processor mean = std = 0.5
        enc = self.vit(pixel_values=x).last_hidden_state[:, 0]
        if self.use_freq:
            enc = torch.cat([enc, self.freq(batch["spectrum"])], dim=1)
        return self.head(enc, batch["feats"] if self.use_handcrafted else None)
