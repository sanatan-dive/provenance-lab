"""Small from-scratch CNN on log-mel spectrograms + handcrafted features -> shared DetectorHead."""
import torch
import torch.nn as nn
import torchaudio.transforms as T

from common.model import DetectorHead


# Log-mel statistics of the training set (computed once); a fixed scale instead of a BatchNorm input layer.
MEL_MEAN, MEL_STD = -7.1, 4.1


def block(cin, cout):
    # GroupNorm keeps no running statistics, so eval behaves like training. BatchNorm's running stats,
    # collected on SpecAugment-masked inputs, made eval accuracy swing between epochs.
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class AudioCNN(nn.Module):
    def __init__(self, feat_dim, use_handcrafted=True, emb_dim=256, spec_augment=True):
        super().__init__()
        self.use_handcrafted = use_handcrafted
        self.spec_augment = spec_augment
        self.conv = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, emb_dim))
        self.drop = nn.Dropout(0.3)
        self.fmask = T.FrequencyMasking(freq_mask_param=12, iid_masks=True)
        self.tmask = T.TimeMasking(time_mask_param=40, iid_masks=True)
        self.head = DetectorHead(emb_dim, feat_dim if use_handcrafted else 0)

    def forward(self, batch):
        x = (batch["mel"] - MEL_MEAN) / MEL_STD  # [B, 1, 80, 401]
        if self.training and self.spec_augment:  # SpecAugment, train only (val/test see clean input)
            x = self.tmask(self.fmask(self.fmask(x)))
        x = self.conv(x)
        emb = self.drop(x.mean(dim=(2, 3)))  # global average pool -> [B, emb_dim]
        return self.head(emb, batch["feats"] if self.use_handcrafted else None)
