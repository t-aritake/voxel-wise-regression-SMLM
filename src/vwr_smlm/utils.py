"""Shared helpers: voxel grids, coordinate rescaling, run directories and configs."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

CONFIG_NAME = "config.yaml"
HISTORY_NAME = "history.json"
# configs/ at the repository root (the package is installed in editable mode by uv)
CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")
WEIGHTS_NAME = "cnn_microscopy.pth"
LEGACY_ARGS_NAME = "args.pkl"


# ---------------------------------------------------------------------------
# Voxel grid
# ---------------------------------------------------------------------------
def grid_size(depth: int, height: int, width: int) -> tuple[int, int, int]:
    """Number of voxels along (x, y, z) for an input image of shape ``(D, H, W)``.

    The image row axis (H) corresponds to x and the column axis (W) to y, so that the
    flattened voxel index ``ix + iy * nx + iz * nx * ny`` matches :class:`~vwr_smlm.models.Network`.
    """
    return height, width, depth


def gen_defaults(
    grid: tuple[int, int, int], device: torch.device | str | None = None, centering: bool = False
) -> torch.Tensor:
    """Normalized minimum (or center) coordinate of every voxel, shape ``(nx * ny * nz, 3)``.

    Ordered so that x changes fastest, then y, then z.
    """
    axes = [torch.arange(n, device=device, dtype=torch.float32) / n for n in grid]
    coord = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (nx, ny, nz, 3)
    coord = coord.permute(2, 1, 0, 3).reshape(-1, 3)
    if centering:
        coord = coord + 0.5 / torch.tensor(grid, device=device, dtype=torch.float32)
    return coord


def voxel_interval(
    grid: tuple[int, int, int], device: torch.device | str | None = None
) -> torch.Tensor:
    """Normalized edge length of a voxel along (x, y, z)."""
    return 1.0 / torch.tensor(grid, device=device, dtype=torch.float32)


def decode_offset(raw: torch.Tensor, margin: float | None = None) -> torch.Tensor:
    """Map the network output (sigmoid, in [0, 1]) to an offset from the voxel's minimum corner,
    in voxel units.

    ``margin=None`` (legacy): offset = raw, i.e. the voxel is exactly [0, 1].  A molecule near a
    voxel boundary then needs a saturated sigmoid.

    ``margin=m``: the output is centered on the voxel center and its range is extended by ``m``
    voxels on each side, offset = 0.5 + (1 + 2m)(raw - 0.5) in [-m, 1 + m].  Boundaries (0, 1)
    then correspond to raw = 0.5 -/+ 0.5 / (1 + 2m), away from saturation.  Each molecule is still
    assigned to the voxel that contains it (= the voxel with the nearest center).
    """
    if margin is None:
        return raw
    return 0.5 + (1.0 + 2.0 * margin) * (raw - 0.5)


def to_absolute(
    predictions: torch.Tensor,
    defaults: torch.Tensor,
    interval: torch.Tensor,
    space: TargetSpace,
    margin: float | None = None,
) -> torch.Tensor:
    """Convert per-voxel predictions ``predictions[..., :3]`` to coordinates in nm.

    ``defaults`` are the minimum corners of the voxels (``gen_defaults(..., centering=False)``).
    """
    normalized = decode_offset(predictions[..., :3], margin) * interval + defaults
    return normalized * space.size.to(predictions) + space.minimum.to(predictions)


def nms(
    coords: torch.Tensor, confidence: torch.Tensor, radius: torch.Tensor | None
) -> torch.Tensor:
    """Merge duplicate detections of one molecule (e.g. from neighboring voxels).

    A detection is dropped if a detection with higher confidence lies within the ellipsoid
    ``sum(((a - b) / radius) ** 2) < 1`` (``radius`` per axis, in the units of ``coords``).
    Single-pass ("fast") NMS: suppression is not transitive.  Returns a boolean keep mask.
    """
    n = len(coords)
    if radius is None or n < 2:
        return torch.ones(n, dtype=torch.bool, device=coords.device)
    d = ((coords[:, None, :] - coords[None, :, :]) / radius.to(coords)).pow(2).sum(-1)
    close = d < 1
    order = confidence.argsort(descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n, device=coords.device)
    higher = rank[:, None] < rank[None, :]  # [i, j]: i ranks above j
    return ~(close & higher).any(0)


# ---------------------------------------------------------------------------
# Target space (physical extent covered by the normalized [0, 1)^3 cube)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetSpace:
    minimum: torch.Tensor  # (3,) nm
    size: torch.Tensor  # (3,) nm

    @classmethod
    def from_params(
        cls,
        patch_height: int,
        patch_width: int,
        depth: int,
        resolution_xy: float,
        resolution_depth: float,
        scale_depth: int,
        xymin: float = 0.0,
        zmin: float = 0.0,
    ) -> TargetSpace:
        """Same convention as :class:`vwr_smlm.datasets.RandomDataset`.

        z is extended below the first and above the last focal plane by half of the
        (high-resolution) depth range of a voxel.
        """
        res_depth_high = resolution_depth // scale_depth
        z_lo = zmin - res_depth_high * (scale_depth // 2)
        z_hi = (
            zmin + resolution_depth * (depth - 1) + res_depth_high * int((scale_depth - 0.1) // 2)
        )
        minimum = torch.tensor([xymin, xymin, z_lo], dtype=torch.float64)
        size = torch.tensor(
            [resolution_xy * patch_height, resolution_xy * patch_width, max(z_hi - z_lo, 1)],
            dtype=torch.float64,
        )
        return cls(minimum, size)


# ---------------------------------------------------------------------------
# Run directories / configs
# ---------------------------------------------------------------------------
def run_dir() -> Path:
    """Output directory of the current Hydra run."""
    from hydra.core.hydra_config import HydraConfig

    return Path(HydraConfig.get().runtime.output_dir)


def save_run(dirpath: str | Path, cfg: DictConfig, **history: Any) -> None:
    """Save the resolved config (``config.yaml``) and loss history next to the weights."""
    dirpath = Path(dirpath)
    to_save = OmegaConf.masked_copy(cfg, ["seed", "simulation", "augmentation", "model", "train"])
    OmegaConf.save(to_save, dirpath / CONFIG_NAME, resolve=True)
    with open(dirpath / HISTORY_NAME, "w") as f:
        json.dump(history, f, indent=2)


class _Opaque:
    """Placeholder for any object in a legacy pickle other than ``argparse.Namespace``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


class _LegacyUnpickler(pickle.Unpickler):
    # The old ``args.pkl`` also contains pickled DataLoaders / datasets, which are not
    # loadable with recent PyTorch.  Only the argparse.Namespace is reconstructed.
    def find_class(self, module: str, name: str) -> Any:
        if module == "argparse" and name == "Namespace":
            return argparse.Namespace
        if module == "builtins" and name in {"list", "dict", "tuple", "set", "float", "int"}:
            return super().find_class(module, name)
        return _Opaque


_SIMULATION_KEYS = (
    "low_patchsize", "low_depth", "resolution_xy_low", "resolution_depth_low",
    "scale_xy", "scale_depth", "min_weight", "max_weight",
)  # fmt: skip


def load_config(dirpath: str | Path) -> DictConfig:
    """Load the training config of a model directory.

    Reads ``config.yaml`` written by ``vwr-train``; falls back to the legacy ``args.pkl``
    of the original implementation (converted to the same structure).
    """
    dirpath = Path(dirpath)
    if (dirpath / CONFIG_NAME).exists():
        return OmegaConf.load(dirpath / CONFIG_NAME)
    if (dirpath / LEGACY_ARGS_NAME).exists():
        with open(dirpath / LEGACY_ARGS_NAME, "rb") as f:
            args = _LegacyUnpickler(f).load()[0]
        simulation = {k: getattr(args, k) for k in _SIMULATION_KEYS}
        simulation.update(lmbd=10, sigma=3)  # defaults of the original RandomDataset
        return OmegaConf.create(
            {
                "seed": args.seed,
                "simulation": simulation,
                "augmentation": {"enabled": True, "max_translation": 2, "base_plane": 2},
                "model": {"offset_margin": None},
                "train": {k: v for k, v in vars(args).items() if k not in _SIMULATION_KEYS},
            }
        )
    raise FileNotFoundError(f"neither {CONFIG_NAME} nor {LEGACY_ARGS_NAME} found in {dirpath}")


def load_model(
    dirpath: str | Path, device: torch.device | str
) -> tuple[torch.nn.Module, DictConfig]:
    """Load a trained :class:`~vwr_smlm.models.Network` in eval mode together with its config."""
    from .models import Network

    cfg = load_config(dirpath)
    model = Network(input_depth=cfg.simulation.low_depth)
    state = torch.load(Path(dirpath) / WEIGHTS_NAME, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def nms_radius(cfg: DictConfig, device: torch.device | str | None = None) -> torch.Tensor | None:
    """``inference.nms_radius`` as a tensor (``None`` = no NMS)."""
    r = OmegaConf.select(cfg, "inference.nms_radius", default=None)
    return None if r is None else torch.tensor(list(r), dtype=torch.float64, device=device)


def offset_margin(cfg: DictConfig) -> float | None:
    """``model.offset_margin`` of a config (``None`` for models without the setting)."""
    return OmegaConf.select(cfg, "model.offset_margin", default=None)


def get_device(name: str = "auto") -> torch.device:
    """``auto`` selects CUDA when available."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def build_augmentation(simulation: DictConfig, augmentation: DictConfig):
    """Random lateral drift between focal planes, or ``None`` if disabled."""
    import numpy as np

    from .augmentation import RandomTranslation

    if not augmentation.enabled:
        return None
    z_planes = np.arange(simulation.low_depth) * simulation.resolution_depth_low
    return RandomTranslation(
        augmentation.max_translation,
        augmentation.base_plane,
        z_planes,
        simulation.resolution_xy_low / simulation.scale_xy,
    )
