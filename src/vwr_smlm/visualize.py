"""Render localization results (``coord_*.txt`` from ``vwr-localize``).

Produces ``tubulin_scatter<suffix>.png`` (3D scatter) and ``localization<suffix>.png``
(xy projection colored by z) in the Hydra run directory.

Example::

    uv run vwr-visualize visualize.coord_file=data/results/tubuling/coord_0.90.txt
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from .utils import CONFIG_DIR, run_dir  # noqa: E402


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    opts = cfg.visualize
    print(OmegaConf.to_yaml(opts))
    result_path = Path(opts.coord_file)
    savedir = run_dir()
    suffix = result_path.stem.removeprefix("coord")
    voxel = np.asarray(opts.voxel_size)

    coords = np.loadtxt(result_path, delimiter=",", ndmin=2)
    inside = (
        np.all((coords[:, :2] >= 0) & (coords[:, :2] < opts.xy_extent), axis=1)
        & (coords[:, 2] >= opts.zmin)
        & (coords[:, 2] < opts.zmax)
    )
    coords = coords[inside]

    cmap = matplotlib.colormaps["viridis"]
    znorm = Normalize(vmin=opts.zmin, vmax=opts.zmax)

    # --- 3D scatter ---
    plt.rcParams["grid.color"] = "white"
    fig = plt.figure(figsize=(24, 24))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=cmap(znorm(coords[:, 2])), s=0.01)
    ax.view_init(60, 45)
    ax.set_xlim(0, opts.xy_extent)
    ax.set_ylim(0, opts.xy_extent)
    ax.set_zlim(opts.zmin, opts.zmax)
    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), "xyz", strict=False):
        axis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        axis.labelpad = 30.0
        getattr(ax, f"set_{name}label")(f"{name} [nm]", fontsize=30)
    ax.tick_params(labelsize=20)
    fig.savefig(savedir / f"tubulin_scatter{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    # --- xy projection on the super-resolved grid, colored by z ---
    n_xy = int(np.ceil(opts.xy_extent / voxel[0]))
    indices = np.unique(np.floor(coords[:, :3] / voxel).astype(int), axis=0)
    # draw shallow planes first so that deeper ones end up on top (same as the original)
    indices = indices[np.argsort(indices[:, 2], kind="stable")]
    image = np.zeros((n_xy, n_xy, 3))
    image[indices[:, 0], indices[:, 1]] = cmap(znorm(indices[:, 2] * voxel[2]))[:, :3]

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image)
    ax.axis("off")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(ScalarMappable(norm=znorm, cmap=cmap), cax=cax)
    cbar.set_label("z [nm]", size=24)
    cbar.ax.tick_params(labelsize=20)
    fig.savefig(savedir / f"localization{suffix}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
