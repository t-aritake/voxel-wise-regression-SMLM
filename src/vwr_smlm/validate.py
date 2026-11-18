"""Evaluate a trained model on freshly simulated data.

Reports the MMD (Laplacian kernel) between estimated and true molecules and the Jaccard
index on a super-resolved voxel grid, and saves the offset from every estimate to the
closest true molecule (``closest_log.npy``) in the Hydra run directory.

Simulation parameters are taken from the trained model; the ``validate`` section of
``configs/config.yaml`` controls the rest::

    uv run vwr-validate validate.model_dir=data/learned_models/220308_1357
"""

from __future__ import annotations

import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .datasets import RandomDataset, collate_fn
from .utils import (
    CONFIG_DIR,
    build_augmentation,
    gen_defaults,
    get_device,
    grid_size,
    load_model,
    nms,
    nms_radius,
    offset_margin,
    run_dir,
    to_absolute,
    voxel_interval,
)


def laplacian_kernel_mmd(
    coord1: torch.Tensor,
    weights1: torch.Tensor,
    coord2: torch.Tensor,
    weights2: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Squared MMD with kernel ``exp(-||x - y||_1 / sigma)`` between two weighted point sets.

    ``coord*``: (B, n, 3), ``weights*``: (B, n). Returns (B,).
    """
    points = torch.cat((coord1, coord2), dim=1)
    K = torch.exp(-torch.cdist(points, points, p=1) / sigma)
    w = torch.cat((weights1, -weights2), dim=1).unsqueeze(-1)
    return (w.transpose(1, 2) @ K @ w).flatten()


def flat_voxel_ids(
    coords: torch.Tensor, minimum: torch.Tensor, voxel: torch.Tensor, n: torch.Tensor
) -> np.ndarray:
    idx = torch.div(coords - minimum, voxel, rounding_mode="floor")
    return (idx[:, 0] + idx[:, 1] * n[0] + idx[:, 2] * n[0] * n[1]).cpu().numpy()


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    vcfg = cfg.validate
    print(OmegaConf.to_yaml(vcfg))
    np.random.seed(vcfg.seed)
    torch.manual_seed(vcfg.seed)
    device = get_device(cfg.device)

    model, model_cfg = load_model(vcfg.model_dir, device)
    sim = model_cfg.simulation
    if vcfg.patchsize is not None:
        sim = OmegaConf.merge(sim, {"low_patchsize": vcfg.patchsize})

    aug_cfg = OmegaConf.merge(model_cfg.augmentation, {"enabled": vcfg.augmentation})
    dataset = RandomDataset(
        **OmegaConf.to_container(sim),
        num_particles=vcfg.num_particles,
        data_size=vcfg.num_data,
        augmentation=build_augmentation(sim, aug_cfg),
    )
    loader = DataLoader(dataset, batch_size=vcfg.batch_size, shuffle=False, collate_fn=collate_fn)

    grid = grid_size(sim.low_depth, sim.low_patchsize, sim.low_patchsize)
    defaults = gen_defaults(grid, device)
    interval = voxel_interval(grid, device)
    space_min = dataset.space.minimum.to(device)
    space_size = dataset.space.size.to(device)
    radius = nms_radius(cfg, device)
    eval_voxel = torch.tensor(list(vcfg.eval_voxel), dtype=torch.float64, device=device)
    eval_grid = torch.ceil(space_size / eval_voxel)

    closest_log: list[np.ndarray] = []
    mmd_log: list[float] = []
    intersect = 0
    union = 0

    print("start")
    start = time.perf_counter()
    for data, targets in loader:
        data = data.to(device)
        predictions = model(data).double()
        coords = to_absolute(
            predictions, defaults, interval, dataset.space, offset_margin(model_cfg)
        )

        for pred, coord, target in zip(predictions, coords, targets, strict=True):
            true_coord = target[:, :3].to(device).double() * space_size + space_min
            true_voxels = flat_voxel_ids(true_coord, space_min, eval_voxel, eval_grid)

            detected = pred[:, 3] > vcfg.threshold
            if not detected.any():
                union += len(np.unique(true_voxels))
                continue
            est_coord = coord[detected]
            est_coord = est_coord[nms(est_coord, pred[detected, 3], radius)]

            # offset to the closest true molecule from each estimate
            diff = (true_coord[None, :, :] - est_coord[:, None, :]).cpu().numpy()
            nearest = (diff**2).sum(2).argmin(1)
            closest_log.append(diff[np.arange(len(est_coord)), nearest])

            # Jaccard index on the super-resolved voxel grid
            est_voxels = flat_voxel_ids(est_coord, space_min, eval_voxel, eval_grid)
            union += len(np.union1d(est_voxels, true_voxels))
            intersect += len(np.intersect1d(est_voxels, true_voxels))

            mmd = laplacian_kernel_mmd(
                est_coord[None].float(), torch.ones(1, len(est_coord), device=device),
                true_coord[None].float(), torch.ones(1, len(true_coord), device=device),
                vcfg.mmd_sigma,
            )  # fmt: skip
            mmd_log.append(mmd.item())

    elapsed = time.perf_counter() - start
    print(f"elapsed time: {elapsed:.2f} s")
    print(f"fps={len(dataset) / elapsed:.1f}")

    if closest_log:
        out = run_dir() / "closest_log.npy"
        np.save(out, np.concatenate(closest_log))
        print(f"saved to {out}")
    print(
        f"MMD: {np.mean(mmd_log):.4g} +/- {np.std(mmd_log):.4g}, "
        f"Jaccard: {intersect / max(union, 1):.4f}"
    )


if __name__ == "__main__":
    main()
