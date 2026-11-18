# voxel-wise-regression-SMLM
Implementation code of the paper "Single-Molecule Localization by Voxel-Wise Regression Using Convolutional Neural Network", Results in Optics, 2020

The real data used in the expriments are not contained in this repositry.
For real dataset, please contact the author.

## Directories

```
.
|-configs/config.yaml   all options (Hydra)
|-src/vwr_smlm/         package (models, datasets, loss, train, validate, localize, visualize)
|-tests/
|-data
| |-datasets            directory to put data for datasets
| |-learned_models      trained models (one timestamped directory per run)
| |-results             outputs of validate / localize / visualize
```

## Prerequisites

* [uv](https://docs.astral.sh/uv/)
* Python 3.10+ / PyTorch 2.4+ (installed automatically by `uv sync`)

```
uv sync
```

## Usage

All options live in `configs/config.yaml` and can be overridden on the command line
with Hydra syntax (`section.key=value`). Each run writes into its own timestamped
directory under `output_dirs.*` (the resolved config is saved alongside it in `.hydra/`).

**Training** (the model is saved to `data/learned_models/<timestamp>/`)

``CUDA_VISIBLE_DEVICES=0 uv run vwr-train``

``uv run vwr-train train.epochs=5 train.num_data_train=10000 simulation.low_patchsize=32``

**Validate** on simulated data

``uv run vwr-validate validate.model_dir=data/learned_models/<run>``

**Localize** molecules in real data (`data/datasets/191216/cam*/%05d.tiff`)

``uv run vwr-localize localize.model_dir=data/learned_models/<run> localize.threshold=0.9``

**Visualize** localization results

``uv run vwr-visualize visualize.coord_file=data/results/localize/<run>/coord_0.90.txt``

Models trained with the original (2020) code, which stored `args.pkl`, can still be
loaded by `vwr-validate` / `vwr-localize`.

**Tests**

``uv run pytest``

## Citation
If you find our research useful, please cite the paper:
```
@article{ARITAKE2020100019,
title = "Single-Molecule Localization by Voxel-Wise Regression Using Convolutional Neural Network",
journal = "Results in Optics",
pages = "100019",
year = "2020",
issn = "2666-9501",
doi = "https://doi.org/10.1016/j.rio.2020.100019",
url = "http://www.sciencedirect.com/science/article/pii/S2666950120300195",
author = "Toshimitsu Aritake and Hideitsu Hino and Shigeyuki Namiki and Daisuke Asanuma and Kenzo Hirose and Noboru Murata",
keywords = "3D single-molecule localization microscopy, Multi-focal plane microscopy, Convolutional neural network, Regression-based method",
}

```

## Contact
toshimitsu.aritake[at]ruri.waseda.jp
