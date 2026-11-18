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


@pytest.mark.parametrize("margin", [None, 0.25, 0.5])
def test_offset_margin_roundtrip(margin):
    """The ideal network output gives zero location loss and decodes to the true coordinates."""
    from vwr_smlm.utils import TargetSpace, to_absolute, voxel_interval

    grid = (8, 8, 4)
    loss_fn = LocationAndConfidenceLoss(grid, offset_margin=margin)
    torch.manual_seed(0)
    target = torch.rand(20, 4)
    target[:, 3] = 1
    voxels = loss_fn.voxel_index(target[:, :3])
    interval = voxel_interval(grid)
    offset = (target[:, :3] - loss_fn.defaults[voxels]) / interval  # in [0, 1)
    m = 0.0 if margin is None else margin
    raw = 0.5 + (offset - 0.5) / (1 + 2 * m)  # inverse of decode_offset

    pred = torch.full((1, 8 * 8 * 4, 4), 1e-4)
    pred[0, voxels, :3] = raw
    pred[0, voxels, 3] = 1 - 1e-4
    loc, _ = loss_fn(pred, [target])
    assert loc.item() < 1e-4

    space = TargetSpace(torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
    decoded = to_absolute(pred[0, voxels], loss_fn.defaults[voxels], interval, space, margin)
    torch.testing.assert_close(decoded.float(), target[:, :3], atol=1e-5, rtol=0)
    if margin is not None:
        # boundaries no longer require a saturated sigmoid
        assert raw.min() > 0.5 - 0.5 / (1 + 2 * m) - 1e-6
        assert raw.max() < 0.5 + 0.5 / (1 + 2 * m) + 1e-6


def test_neighbor_assignments():
    grid = (8, 8, 4)
    loss_fn = LocationAndConfidenceLoss(grid, offset_margin=0.25, neighbor_margin=0.25)
    n = torch.tensor(grid, dtype=torch.float32)
    # voxel (3, 3, 1); x near the lower boundary only / x and y near boundaries / center
    pos = torch.tensor([[3.1, 3.5, 1.5], [3.1, 3.9, 1.5], [3.5, 3.5, 1.5]])
    batch, voxel, mol, offset = loss_fn.neighbor_assignments(
        torch.zeros(3, dtype=torch.long), pos / n
    )

    mult = torch.tensor([1, 8, 64])
    got = {(m.item(), v.item()): o for m, v, o in zip(mol, voxel, offset, strict=True)}
    expected = {
        (0, (torch.tensor([2, 3, 1]) * mult).sum().item()),
        (1, (torch.tensor([2, 3, 1]) * mult).sum().item()),
        (1, (torch.tensor([3, 4, 1]) * mult).sum().item()),
        (1, (torch.tensor([2, 4, 1]) * mult).sum().item()),
    }
    assert set(got) == expected  # the molecule at the center has no neighbors
    torch.testing.assert_close(
        got[(0, (torch.tensor([2, 3, 1]) * mult).sum().item())], torch.tensor([1.1, 0.5, 0.5])
    )


def test_neighbor_loss_runs_and_requires_offset_margin():
    with pytest.raises(ValueError):
        LocationAndConfidenceLoss((8, 8, 4), offset_margin=None, neighbor_margin=0.25)
    np.random.seed(0)
    ds = make_dataset(n=20)
    images, targets = collate_fn([ds[0], ds[1]])
    pred = Network()(images)
    grid = grid_size(4, 16, 16)
    base = LocationAndConfidenceLoss(grid, offset_margin=0.25)(pred, targets)
    nb = LocationAndConfidenceLoss(grid, offset_margin=0.25, neighbor_margin=0.25)(pred, targets)
    assert nb[0] > base[0]  # extra location terms for the neighbors
    (nb[0] + nb[1]).backward()


def test_nms():
    from vwr_smlm.utils import nms

    coords = torch.tensor([[0.0, 0, 0], [10, 0, 0], [100, 0, 0], [0, 0, 80]])
    conf = torch.tensor([0.6, 0.9, 0.8, 0.7])
    keep = nms(coords, conf, torch.tensor([50.0, 50.0, 100.0]))
    # 0 is suppressed by 1; 3 is within the z radius of 1 (|dx|=10, |dz|=80 -> 0.04+0.64 < 1)
    assert keep.tolist() == [False, True, True, False]
    assert nms(coords, conf, None).all()
