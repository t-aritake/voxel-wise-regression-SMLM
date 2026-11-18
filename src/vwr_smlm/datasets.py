"""Datasets: simulated multi-focal-plane images and real (tubulin) TIFF stacks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import TargetSpace


def psf(observe_coordinates: np.ndarray, molecule_coordinates: np.ndarray) -> np.ndarray:
    """Astigmatism-free defocus PSF model.

    Parameters
    ----------
    observe_coordinates : (n_obs, 3) observation coordinates [nm]
    molecule_coordinates : (n_mol, 3) molecule coordinates [nm]

    Returns
    -------
    (n_mol, n_obs) intensity of each molecule at each observation point
    """
    w0 = 133.0525
    d = 302.3763
    A = 0.000737
    B = 0.122484
    adash = 5e7

    diff = observe_coordinates[None, :, :] - molecule_coordinates[:, None, :]
    z_diff = diff[:, :, 2] / d
    w = w0 * np.sqrt(1 + z_diff**2 + A * z_diff**3 + B * z_diff**4)
    a = adash / (2 * np.pi * w**2)
    return a * np.exp(-np.sum((diff[:, :, :2] / w[:, :, None]) ** 2 / 2, axis=2))


class RandomDataset(Dataset):
    """Molecules uniformly distributed in the target space, imaged on ``low_depth`` focal planes.

    Each item is ``(image, target)`` where ``image`` has shape ``(low_depth, P, P)`` and
    ``target`` has shape ``(n_molecules, 4)``: coordinates normalized to [0, 1) within the
    target space and a confidence of 1.

    Parameters
    ----------
    low_patchsize : image size (P) of each focal plane
    low_depth : number of focal planes
    resolution_xy_low : nm / pixel in the lateral direction
    resolution_depth_low : nm between focal planes
    scale_xy, scale_depth : super-resolution factors
    min_weight, max_weight : range of the molecule peak intensity
    xymin, zmin : minimum (starting) coordinates
    num_particles : number of molecules per image (int or array of length ``data_size``)
    data_size : number of images
    augmentation : callable applied to the observation coordinates (e.g. lateral drift)
    lmbd : mean of the Poisson shot noise
    sigma : std of the Gaussian read noise
    """

    def __init__(
        self,
        low_patchsize: int,
        low_depth: int,
        resolution_xy_low: int,
        resolution_depth_low: int,
        scale_xy: int,
        scale_depth: int,
        min_weight: float = 1.0,
        max_weight: float = 1.0,
        xymin: float = 0,
        zmin: float = 0,
        num_particles: int | np.ndarray = 3,
        data_size: int = 10000,
        augmentation: Callable[[np.ndarray], np.ndarray] | None = None,
        lmbd: float = 10,
        sigma: float = 3,
    ) -> None:
        super().__init__()
        self.low_patchsize = low_patchsize
        self.low_depth = low_depth
        self.resolution_xy_low = resolution_xy_low
        self.resolution_depth_low = resolution_depth_low
        self.scale_xy = scale_xy
        self.scale_depth = scale_depth
        self.augmentation = augmentation
        self.lmbd = lmbd
        self.sigma = sigma
        self.data_size = data_size

        self.space = TargetSpace.from_params(
            low_patchsize, low_patchsize, low_depth,
            resolution_xy_low, resolution_depth_low, scale_depth, xymin, zmin,
        )  # fmt: skip
        lo = self.space.minimum.numpy().astype(float)
        hi = lo + self.space.size.numpy().astype(float)

        if np.isscalar(num_particles):
            num_particles = np.full(data_size, int(num_particles))
        self.num_particles = np.asarray(num_particles, dtype=int)
        # offsets[i]:offsets[i+1] are the molecules of image i
        self._offsets = np.concatenate([[0], np.cumsum(self.num_particles)])
        total = int(self._offsets[-1])

        # observation coordinates are shared by all images
        self.observe_coordinates = self._gen_observation_coordinates()
        self.particle_coordinates = np.stack(
            [
                np.random.uniform(lo[0], hi[0], size=total),
                np.random.uniform(lo[1], hi[1], size=total),
                np.random.uniform(lo[2], hi[2], size=total),
            ],
            axis=1,
        )
        self.weight = np.random.uniform(min_weight, max_weight, size=total)
        self._lo, self._extent = lo, hi - lo

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sl = slice(self._offsets[index], self._offsets[index + 1])
        molecules = self.particle_coordinates[sl]

        observe = self.observe_coordinates
        if self.augmentation is not None:
            observe = self.augmentation(observe)

        shape = (self.low_patchsize, self.low_patchsize, self.low_depth)
        x = (psf(observe, molecules) * self.weight[sl, None]).sum(0).reshape(shape, order="F")
        # shot noise + read noise
        x += np.random.poisson(lam=self.lmbd, size=x.shape)
        x += np.random.normal(size=x.shape) * self.sigma
        image = torch.from_numpy(x.transpose(2, 0, 1)).float()

        target = np.ones((len(molecules), 4))
        target[:, :3] = (molecules - self._lo) / self._extent
        return image, torch.from_numpy(target).float()

    def _gen_observation_coordinates(self) -> np.ndarray:
        observe_xy = (np.arange(self.low_patchsize) + 0.5) * self.resolution_xy_low
        observe_z = np.arange(self.low_depth) * self.resolution_depth_low
        grid = np.stack(np.meshgrid(observe_xy, observe_xy, observe_z), axis=-1)
        # (z, y, x, 3): x changes fastest, then y, then z
        return grid.transpose(2, 0, 1, 3).reshape(-1, 3)


def collate_fn(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Stack images; keep targets as a list since the number of molecules varies."""
    images, targets = zip(*batch, strict=False)
    return torch.stack(images, dim=0), list(targets)


class MultiCameraTiffDataset(Dataset):
    """Real multi-focal-plane data stored as ``root/<camera>/<index:05d>.tiff``.

    ``cameras`` gives the directory of each focal plane in z order.
    """

    def __init__(
        self,
        root: str | Path,
        cameras: Sequence[str] = ("cam2", "cam1", "cam3", "cam4"),
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dirs = [Path(root) / f"{prefix}{cam}" for cam in cameras]
        for d in self.dirs:
            if not d.is_dir():
                raise FileNotFoundError(d)
        self._length = min(len(list(d.glob("*.tiff"))) for d in self.dirs)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        planes = [np.asarray(Image.open(d / f"{index:05d}.tiff")) for d in self.dirs]
        return torch.from_numpy(np.stack(planes).astype(np.float32))


class PreGeneratedData(Dataset):
    """Load ``(observe, target)`` pairs saved as ``dirpath/data_<i>.pt``."""

    def __init__(self, dirpath: str | Path) -> None:
        self.dirpath = Path(dirpath)
        self._data_size = len(list(self.dirpath.glob("data_*.pt")))

    def __len__(self) -> int:
        return self._data_size

    def __getitem__(self, index: int):
        return torch.load(self.dirpath / f"data_{index}.pt", weights_only=True)
