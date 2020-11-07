# voxel-wise-regression-SMLM
Implementation code of the paper "Single-Molecule Localization by Voxel-Wise Regression Using Convolutional Neural Network", Results in Optics, 2020

The real data used in the expriments are not contained in this repositry.
For real dataset, please contact the author.

## Directories

```
.                       main codes
|-data
| |-datasets            directory to put data for datasets
| |-learned_models      directory to save parameters of a learned model
```

## Prerequisites

* PyTorch 1.3.1+
* Python 3.7.x


## Usage

**Training**

``CUDA_VISIBLE_DEVICES=0 python run.py --low-patchsize=64 --num-particle-train=50 --num-particle-test=50 --low-depth=4 --epochs=20 --num-data-train=100000 --num-data-test=10000 --batch-size=50 --log-interval=100 --lr=1e-3 --min-weight=0.3 --save-model``

**Validate**

``CUDA_VISIBLE_DEVICES=0 python validate.py directory_of_learned_model``

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
