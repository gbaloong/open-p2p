"""OFFLINE-only latent-action learning and conservative skill segmentation.

Future frames are allowed ONLY as offline supervision, never as online inputs.
This small VQ residual predictor is a new baseline inspired by latent-action work,
not a reproduction or pretrained implementation of LAPA/V-JEPA.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LatentActionVQ(nn.Module):
    def __init__(self, feature_dim: int, latent_dim=32, codes=64, commitment=0.25):
        super().__init__()
        if min(feature_dim, latent_dim, codes) < 1 or commitment < 0:
            raise ValueError("Invalid latent model dimensions")
        self.feature_dim, self.commitment = feature_dim, commitment
        self.encoder = nn.Sequential(nn.Linear(2*feature_dim, 128), nn.SiLU(), nn.Linear(128, latent_dim))
        self.codebook = nn.Embedding(codes, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(feature_dim+latent_dim, 128), nn.SiLU(), nn.Linear(128, feature_dim))

    def forward(self, current, future):
        if current.shape != future.shape or current.shape[-1] != self.feature_dim:
            raise ValueError("Expected matching current/future feature arrays")
        if not torch.isfinite(current).all() or not torch.isfinite(future).all():
            raise ValueError("Non-finite video features")
        z = self.encoder(torch.cat((current, future-current), dim=-1))
        distance = ((z.unsqueeze(-2)-self.codebook.weight)**2).sum(-1)
        indices = distance.argmin(-1)
        q = self.codebook(indices)
        straight_through = z+(q-z).detach()
        predicted = self.decoder(torch.cat((current, straight_through), -1))
        reconstruction = F.mse_loss(predicted, future-current)
        codebook_loss = F.mse_loss(q, z.detach())
        commitment_loss = F.mse_loss(z, q.detach())
        return {"indices": indices, "prediction": predicted,
                "loss": reconstruction+codebook_loss+self.commitment*commitment_loss,
                "reconstruction_loss": reconstruction}


def skill_segments(prediction_errors, *, minimum_length=8, threshold_mad=3.0):
    """A heuristic baseline. Mask edits/cuts/loading screens BEFORE calling it."""
    errors = np.asarray(prediction_errors, dtype=float)
    if errors.ndim != 1 or not np.isfinite(errors).all() or minimum_length < 1 or threshold_mad < 0:
        raise ValueError("Invalid error series or segmentation settings")
    if not len(errors):
        return []
    median = np.median(errors)
    mad = np.median(np.abs(errors-median))
    threshold = median+threshold_mad*max(mad, 1e-8)
    boundaries = [0]
    for i, error in enumerate(errors):
        if error > threshold and i-boundaries[-1] >= minimum_length and len(errors)-i >= minimum_length:
            boundaries.append(i)
    boundaries.append(len(errors))
    return list(zip(boundaries[:-1], boundaries[1:]))
