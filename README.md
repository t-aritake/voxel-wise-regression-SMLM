# voxel-wise-regression-SMLM
## Directories

.
|-data
| |-datasets            directory to put data for datasets
| |-learned_models      directory to save parameters of a learned model

## Usage

**Training**
``CUDA_VISIBLE_DEVICES=1 python run.py --low-patchsize=64 --num-particle-train=50 --num-particle-test=50 --low-depth=4 --epochs=20 --num-data-train=100000 --num-data-test=10000 --batch-size=50 --log-interval=100 --lr=1e-3 --min-weight=0.3 --save-model``

**Validate**
``CUDA_VISIBLE_DEVICES=1 python validate.py directory_of_learned_model``

**Run with real data**
``CUDA_VISIBLE_DEVICES=1 python run_tubulin.py directory_of_learned_model directory_to_save_results``
