from pathlib import Path

import numpy as np
import pytest
import torch

from vwr_smlm.datasets import RandomDataset, collate_fn
from vwr_smlm.loss import LocationAndConfidenceLoss
from vwr_smlm.models import Network
from vwr_smlm.utils import gen_defaults, grid_size, load_model

MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "learned_models"


def make_dataset(n=1, **kw):
    params = dict(
        low_patchsize=16, low_depth=4, resolution_xy_low=192, resolution_depth_low=400,
        scale_xy=8, scale_depth=8, num_particles=n, data_size=2,
    )  # fmt: skip
    params.update(kw)
    return RandomDataset(**params)


def test_pixel_and_voxel_index_agree():
    """A molecule's brightest pixel must be the voxel the loss assigns it to."""
    np.random.seed(0)
    ds = make_dataset(lmbd=0, sigma=0)
    ds.particle_coordinates[:] = [[2.5 * 192, 10.5 * 192, 400]]
    image, target = ds[0]
    d, h, w = np.unravel_index(image.numpy().argmax(), image.shape)

    grid = grid_size(*image.shape)
    loss = LocationAndConfidenceLoss(grid)
    voxel = loss.voxel_index(target[:, :3]).item()
    H, W = image.shape[1:]
    assert voxel == h + w * H + d * H * W
    # and the voxel's default coordinate lies just below the molecule
    assert torch.all(gen_defaults(grid)[voxel] <= target[0, :3])


def test_training_step_runs():
    np.random.seed(0)
    ds = make_dataset(n=5)
    images, targets = collate_fn([ds[0], ds[1]])
    model = Network()
    pred = model(images)
    assert pred.shape == (2, 16 * 16 * 4, 4)
    loc, conf = LocationAndConfidenceLoss(grid_size(4, 16, 16))(pred, targets)
    (loc + conf).backward()
    assert torch.isfinite(loc) and torch.isfinite(conf)


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="no learned models")
@pytest.mark.parametrize("run", sorted(p.name for p in MODELS_DIR.glob("*") if p.is_dir()))
def test_legacy_checkpoint_loads(run):
    model, cfg = load_model(MODELS_DIR / run, "cpu")
    assert cfg.simulation.low_depth == 4
    with torch.no_grad():
        assert model(torch.zeros(1, 4, 16, 16)).shape == (1, 16 * 16 * 4, 4)
