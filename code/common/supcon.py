"""Supervised contrastive loss (Khosla et al., 2020).

Pulls embeddings with the same label (human/human, AI/AI) together and pushes
human vs AI embeddings apart.
"""
import torch
import torch.nn.functional as F


def supcon_loss(z, labels, temperature=0.1):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    n = z.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_count = pos.sum(1)
    has_pos = pos_count > 0
    loss = -(log_prob * pos).sum(1)[has_pos] / pos_count[has_pos]
    return loss.mean()
