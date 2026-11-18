"""Location (L1) + confidence (BCE with hard negative mining) loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .utils import gen_defaults, voxel_interval


class LocationAndConfidenceLoss(nn.Module):
    """
    Parameters
    ----------
    grid : number of voxels along (x, y, z), see :func:`vwr_smlm.utils.grid_size`
    default_centering : regress offsets from the voxel center instead of its minimum corner
    negapos_ratio : number of hard negatives used per positive voxel in the confidence loss
    """

    def __init__(
        self, grid: tuple[int, int, int], default_centering: bool = False, negapos_ratio: int = 3
    ) -> None:
        super().__init__()
        self.register_buffer("defaults", gen_defaults(grid, centering=default_centering))
        self.register_buffer("default_interval", voxel_interval(grid))
        self.register_buffer("num_voxels", torch.tensor(grid, dtype=torch.long))
        self.register_buffer(
            "index_multiplier", torch.tensor([1, grid[0], grid[0] * grid[1]], dtype=torch.long)
        )
        self.default_centering = default_centering
        self.negapos_ratio = negapos_ratio

    def voxel_index(self, normalized_coords: torch.Tensor) -> torch.Tensor:
        """Flattened index of the voxel that contains each normalized coordinate."""
        idx = (normalized_coords * self.num_voxels).long()
        idx = torch.minimum(idx.clamp_min(0), self.num_voxels - 1)
        return (idx * self.index_multiplier).sum(-1)

    def forward(
        self, predictions: torch.Tensor, targets: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        predictions : (B, N_voxels, 4) network output
        targets : list of (n_i, 4) normalized molecule coordinates (+ confidence) per image

        Returns
        -------
        location_loss, confidence_loss (both averaged over the batch)
        """
        batch_size = predictions.shape[0]
        device = predictions.device

        batch_idx = torch.cat(
            [
                torch.full((len(t),), b, dtype=torch.long, device=device)
                for b, t in enumerate(targets)
            ]
        )
        coords = torch.cat(targets)[:, :3]
        voxels = self.voxel_index(coords)

        # offset of each molecule from its voxel's default position
        scale = self.default_interval / 2 if self.default_centering else self.default_interval
        target_offsets = (coords - self.defaults[voxels]) / scale
        location_loss = F.l1_loss(
            predictions[batch_idx, voxels, :3], target_offsets, reduction="sum"
        )

        # --- confidence loss with hard negative mining ---
        confidences = predictions[..., 3]
        target_conf = torch.zeros_like(confidences)
        target_conf[batch_idx, voxels] = 1
        positive_mask = target_conf == 1

        with torch.no_grad():
            neg_loss = F.binary_cross_entropy(confidences, target_conf, reduction="none")
            neg_loss[positive_mask] = 0
            # rank of every voxel when sorted by loss in descending order
            rank = neg_loss.argsort(dim=1, descending=True).argsort(dim=1)
            num_negative = torch.clamp(
                positive_mask.sum(1) * self.negapos_ratio, max=positive_mask.shape[1]
            )
            negative_mask = rank < num_negative.unsqueeze(1)

        mask = positive_mask | negative_mask
        confidence_loss = F.binary_cross_entropy(
            confidences[mask], target_conf[mask], reduction="sum"
        )

        return location_loss / batch_size, confidence_loss / batch_size
