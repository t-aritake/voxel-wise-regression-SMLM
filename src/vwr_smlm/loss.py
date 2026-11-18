"""Location (L1) + confidence (BCE with hard negative mining) loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .utils import decode_offset, gen_defaults, voxel_interval


class LocationAndConfidenceLoss(nn.Module):
    """
    Parameters
    ----------
    grid : number of voxels along (x, y, z), see :func:`vwr_smlm.utils.grid_size`
    offset_margin : see :func:`vwr_smlm.utils.decode_offset` (``None`` = legacy corner offsets)
    negapos_ratio : number of hard negatives used per positive voxel in the confidence loss
    neighbor_margin : if set (``m``), neighboring voxels whose region extended by ``m`` voxels
        contains a molecule are (a) excluded from the confidence loss, so they are not punished
        for firing on a molecule near the boundary, and (b) trained to regress that molecule's
        location (offset in [-m, 1 + m]; requires ``offset_margin >= m``).  The voxel that
        contains the molecule remains the only positive.  Use NMS at inference
        (:func:`vwr_smlm.utils.nms`) to merge duplicate detections.
    """

    def __init__(
        self,
        grid: tuple[int, int, int],
        offset_margin: float | None = None,
        negapos_ratio: int = 3,
        neighbor_margin: float | None = None,
    ) -> None:
        super().__init__()
        if neighbor_margin is not None and (offset_margin or 0.0) < neighbor_margin:
            raise ValueError(
                f"neighbor_margin={neighbor_margin} requires offset_margin >= neighbor_margin "
                f"(got offset_margin={offset_margin})"
            )
        self.register_buffer("defaults", gen_defaults(grid))
        self.register_buffer("default_interval", voxel_interval(grid))
        self.register_buffer("num_voxels", torch.tensor(grid, dtype=torch.long))
        self.register_buffer(
            "index_multiplier", torch.tensor([1, grid[0], grid[0] * grid[1]], dtype=torch.long)
        )
        self.offset_margin = offset_margin
        self.negapos_ratio = negapos_ratio
        self.neighbor_margin = neighbor_margin
        shifts = torch.cartesian_prod(*[torch.tensor([-1, 0, 1])] * 3)
        self.register_buffer("neighbor_shifts", shifts[(shifts != 0).any(1)])  # (26, 3)

    def neighbor_assignments(
        self, batch_idx: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Neighboring voxels whose extended region contains a molecule.

        Returns ``(batch, voxel, molecule, offset)`` for every (molecule, neighbor) pair, where
        ``offset`` is the molecule position relative to the neighbor's minimum corner in voxel
        units (within [-m, 1 + m]).
        """
        m = self.neighbor_margin
        u = coords * self.num_voxels  # continuous voxel coordinates
        base = torch.minimum(u.floor().long().clamp_min(0), self.num_voxels - 1)
        frac = (u - base)[:, None, :]  # (M, 1, 3)
        s = self.neighbor_shifts[None]  # (1, 26, 3)
        cand = base[:, None, :] + s  # (M, 26, 3)
        ok = ((s == 0) | ((s == -1) & (frac < m)) | ((s == 1) & (frac > 1 - m))).all(-1)
        ok &= ((cand >= 0) & (cand < self.num_voxels)).all(-1)
        mol, k = ok.nonzero(as_tuple=True)
        cand = cand[mol, k]
        voxel = (cand * self.index_multiplier).sum(-1)
        offset = u[mol] - cand.to(u.dtype)
        return batch_idx[mol], voxel, mol, offset

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

        # offset of each molecule from the minimum corner of its voxel, in voxel units ([0, 1))
        target_offsets = (coords - self.defaults[voxels]) / self.default_interval
        pred_offsets = decode_offset(predictions[batch_idx, voxels, :3], self.offset_margin)
        location_loss = F.l1_loss(pred_offsets, target_offsets, reduction="sum")

        confidences = predictions[..., 3]
        target_conf = torch.zeros_like(confidences)
        target_conf[batch_idx, voxels] = 1
        positive_mask = target_conf == 1

        # --- neighbors of boundary molecules: location only, no confidence target ---
        ignore_mask = torch.zeros_like(positive_mask)
        if self.neighbor_margin is not None and len(coords) > 0:
            nb_batch, nb_voxel, _, nb_offset = self.neighbor_assignments(batch_idx, coords)
            free = ~positive_mask[nb_batch, nb_voxel]  # positives keep their own molecule
            nb_batch, nb_voxel, nb_offset = nb_batch[free], nb_voxel[free], nb_offset[free]
            ignore_mask[nb_batch, nb_voxel] = True
            if len(nb_voxel) > 0:
                # a voxel next to several molecules regresses the one closest to its center
                dist = ((nb_offset - 0.5) ** 2).sum(-1)
                order = dist.argsort()
                key = (nb_batch * confidences.shape[1] + nb_voxel)[order]
                order = order[key.argsort(stable=True)]
                key = (nb_batch * confidences.shape[1] + nb_voxel)[order]
                first = torch.ones_like(key, dtype=torch.bool)
                first[1:] = key[1:] != key[:-1]
                sel = order[first]
                nb_pred = decode_offset(
                    predictions[nb_batch[sel], nb_voxel[sel], :3], self.offset_margin
                )
                location_loss = location_loss + F.l1_loss(nb_pred, nb_offset[sel], reduction="sum")

        # --- confidence loss with hard negative mining ---
        with torch.no_grad():
            neg_loss = F.binary_cross_entropy(confidences, target_conf, reduction="none")
            neg_loss[positive_mask | ignore_mask] = 0
            # rank of every voxel when sorted by loss in descending order
            rank = neg_loss.argsort(dim=1, descending=True).argsort(dim=1)
            num_negative = torch.clamp(
                positive_mask.sum(1) * self.negapos_ratio, max=positive_mask.shape[1]
            )
            negative_mask = (rank < num_negative.unsqueeze(1)) & ~ignore_mask

        mask = positive_mask | negative_mask
        confidence_loss = F.binary_cross_entropy(
            confidences[mask], target_conf[mask], reduction="sum"
        )

        return location_loss / batch_size, confidence_loss / batch_size
