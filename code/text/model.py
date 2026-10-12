import torch.nn as nn
from transformers import AutoModel

from common.model import DetectorHead

ENCODER_NAME = "bert-base-uncased"


class TextDetector(nn.Module):
    """BERT [CLS] embedding (+ optional handcrafted features) -> DetectorHead."""

    def __init__(self, feat_dim=8, use_handcrafted=True):
        super().__init__()
        self.use_handcrafted = use_handcrafted
        self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
        self.encoder.gradient_checkpointing_enable()  # fits batch 16 x 256 under the MPS memory cap
        self.head = DetectorHead(self.encoder.config.hidden_size, feat_dim if use_handcrafted else 0)

    def forward(self, batch):
        out = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        cls = out.last_hidden_state[:, 0]
        return self.head(cls, batch["feats"] if self.use_handcrafted else None)
