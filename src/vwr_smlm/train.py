"""Train the voxel-wise regression network on simulated data.

Options are read from ``configs/config.yaml`` (sections ``simulation``, ``augmentation``,
``train``) and can be overridden on the command line::

    uv run vwr-train train.epochs=5 train.num_data_train=10000
"""

from __future__ import annotations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .datasets import RandomDataset, collate_fn
from .loss import LocationAndConfidenceLoss
from .models import Network
from .utils import (
    CONFIG_DIR,
    WEIGHTS_NAME,
    build_augmentation,
    get_device,
    grid_size,
    offset_margin,
    run_dir,
    save_run,
)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: LocationAndConfidenceLoss,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    running_count = 0
    for batch_idx, (data, targets) in enumerate(loader, start=1):
        data = data.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        optimizer.zero_grad(set_to_none=True)
        location_loss, confidence_loss = loss_fn(model(data), targets)
        loss = location_loss + confidence_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        running_count += 1
        if batch_idx % log_interval == 0:
            progress = f"{batch_idx * len(data)}/{len(loader.dataset)}"
            percent = 100.0 * batch_idx / len(loader)
            print(
                f"Train Epoch: {epoch} [{progress} ({percent:.0f}%)]"
                f"\tLoss: {running_loss / running_count:.4e}"
            )
            running_loss = 0.0
            running_count = 0
    return total_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: LocationAndConfidenceLoss,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for data, targets in loader:
        data = data.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]
        location_loss, confidence_loss = loss_fn(model(data), targets)
        total += (location_loss + confidence_loss).item()
    total /= len(loader)
    print(f"\nTest set: Average loss: {total:.3g}\n" + "-" * 20)
    return total


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    sim, tcfg = cfg.simulation, cfg.train
    print(
        OmegaConf.to_yaml(
            OmegaConf.masked_copy(cfg, ["simulation", "augmentation", "model", "train"])
        )
    )
    device = get_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    loader_kwargs: dict = {"collate_fn": collate_fn, "num_workers": cfg.num_workers}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True

    sim_kwargs = OmegaConf.to_container(sim)
    # lateral drift is applied to the training set only
    trainset = RandomDataset(
        **sim_kwargs,
        num_particles=tcfg.num_particle_train,
        data_size=tcfg.num_data_train,
        augmentation=build_augmentation(sim, cfg.augmentation),
    )
    testset = RandomDataset(
        **sim_kwargs, num_particles=tcfg.num_particle_test, data_size=tcfg.num_data_test
    )
    train_loader = DataLoader(trainset, batch_size=tcfg.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(
        testset, batch_size=tcfg.test_batch_size, shuffle=False, **loader_kwargs
    )

    model = Network(input_depth=sim.low_depth).to(device)
    grid = grid_size(sim.low_depth, sim.low_patchsize, sim.low_patchsize)
    loss_fn = LocationAndConfidenceLoss(
        grid,
        offset_margin=offset_margin(cfg),
        negapos_ratio=tcfg.negapos_ratio,
        neighbor_margin=tcfg.neighbor_margin,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)

    train_history: list[float] = []
    test_history: list[float] = []
    for epoch in range(1, tcfg.epochs + 1):
        train_history.append(
            train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, epoch, tcfg.log_interval
            )
        )
        test_history.append(evaluate(model, test_loader, loss_fn, device))

    if tcfg.save_model:
        savedir = run_dir()
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state, savedir / WEIGHTS_NAME)
        save_run(savedir, cfg, train_loss=train_history, test_loss=test_history)
        print(f"saved to {savedir}")


if __name__ == "__main__":
    main()
