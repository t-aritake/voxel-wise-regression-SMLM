"""Localize molecules in real multi-focal-plane data (e.g. tubulin, 2019-12-16).

Writes ``x, y, z, confidence`` [nm] of every voxel whose confidence exceeds the threshold
to ``coord_<threshold>.txt`` in the Hydra run directory::

    uv run vwr-localize localize.model_dir=data/learned_models/220308_1357
"""

from __future__ import annotations

import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .datasets import MultiCameraTiffDataset
from .utils import (
    CONFIG_DIR,
    TargetSpace,
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


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    lcfg = cfg.localize
    print(OmegaConf.to_yaml(lcfg))
    device = get_device(cfg.device)
    model, model_cfg = load_model(lcfg.model_dir, device)
    sim = model_cfg.simulation

    dataset = MultiCameraTiffDataset(
        lcfg.data_dir, cameras=list(lcfg.cameras), prefix="masked_" if lcfg.masked else ""
    )
    loader = DataLoader(
        dataset,
        batch_size=lcfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    depth, height, width = dataset[0].shape
    grid = grid_size(depth, height, width)
    defaults = gen_defaults(grid, device)
    interval = voxel_interval(grid, device)
    # same physical convention as the simulated training data, extended to the full field of view
    space = TargetSpace.from_params(
        height, width, depth, sim.resolution_xy_low, sim.resolution_depth_low, sim.scale_depth
    )

    radius = nms_radius(cfg, device)
    out_path = run_dir() / f"coord_{lcfg.threshold:.2f}.txt"
    start = time.perf_counter()
    with open(out_path, "w") as f:
        for i, data in enumerate(loader, start=1):
            predictions = model(data.to(device, non_blocking=True)).double()
            predictions[..., :3] = to_absolute(
                predictions, defaults, interval, space, offset_margin(model_cfg)
            )
            for pred in predictions:
                det = pred[pred[:, 3] > lcfg.threshold]
                det = det[nms(det[:, :3], det[:, 3], radius)]
                np.savetxt(f, det.cpu().numpy(), delimiter=",")
            if i % 10 == 0:
                elapsed = time.perf_counter() - start
                fps = i * lcfg.batch_size / elapsed
                print(f"{i}/{len(loader)}  elapsed: {elapsed:.1f} s  fps={fps:.1f}")

    elapsed = time.perf_counter() - start
    print(f"elapsed time: {elapsed:.2f} s")
    print(f"fps={len(dataset) / elapsed:.1f}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
