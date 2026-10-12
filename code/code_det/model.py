"""CodeBERT encoder (mean-pooled) + handcrafted code features -> DetectorHead."""
import torch.nn as nn
from transformers import AutoModel

from common.model import DetectorHead

from .features import FEATURE_NAMES

ENCODER = "microsoft/codebert-base"


class CodeDetector(nn.Module):
    def __init__(self, use_handcrafted=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER)
        self.encoder.gradient_checkpointing_enable()  # fits batch 16 x 256 tokens under the MPS cap
        self.use_handcrafted = use_handcrafted
        feat_dim = len(FEATURE_NAMES) if use_handcrafted else 0
        self.head = DetectorHead(self.encoder.config.hidden_size, feat_dim)

    def forward(self, batch):
        h = self.encoder(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"]).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        enc = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(enc, batch["feats"] if self.use_handcrafted else None)
