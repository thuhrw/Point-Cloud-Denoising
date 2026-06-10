# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a baseline solution for a point cloud denoising competition. It uses **Jittor** as the deep learning framework (not PyTorch). The architecture follows a config-driven design with three main component types: data, model, and system.

## Common Commands

### Training
```bash
python run.py --task configs/task/train_vm.yaml
```
- Checkpoints are saved to `experiments/vm/checkpoint_{epoch}.pkl`

### Prediction (Inference)
Edit `configs/task/predict_vm.yaml` to set `load_ckpt` to your checkpoint path, then:
```bash
python run.py --task configs/task/predict_vm.yaml
```
- Results are saved to `tmp_predict/` (configurable via writer config)

### Evaluation (requires ground truth data - competition use only)
```bash
python evaluate.py \
    --pred_dir ./results/dataset_test_noisy \
    --gt_dir ./test_gt \
    --noisy_dir ./dataset_test_noisy \
    --mesh_dir ./dataset_train \
    --workers 8
```

### Packing submission
```bash
cd results/dataset_test_noisy
zip -r ../../result.zip shapenet/
```

## Architecture

### Config-Driven Design
The `run.py` entry point uses a hierarchical config system:
- **Task configs** (`configs/task/*.yaml`) - define the mode and component references
- **Component configs** - reference sub-configs for data, model, system, transform
- Each component is instantiated via factory functions (`get_model`, `get_system`)

### Key Components

1. **Model** (`src/model/`)
   - `VelocityModule` - the main denoising model using velocity prediction
   - Uses `FeatureExtraction` (encoder) and `Decoder` networks
   - Implements patch-based Langevin dynamics for denoising during inference
   - `process_fn` converts `Asset` objects to model input tensors

2. **System** (`src/system/`)
   - `VMSystem` - orchestrates training/validation/prediction loops
   - `VMWriter` - saves predictions to `.npy` files
   - Extends `DummySystem` base class with hooks (on_*_epoch_start/end, etc.)

3. **Data Pipeline** (`src/data/`)
   - `Asset` - data container with vertices, faces, sampled points
   - `PCDatasetModule` - manages train/validate/predict dataloaders
   - `Transform` - applies a sequence of data augmentations
   - `Augment` subclasses - `sample`, `normalize_pc`, `add_noise`, `linear`, `patch`

### Data Flow During Training
1. Raw mesh → `Asset` (vertices/faces loaded)
2. `Transform.apply()` runs augmentations sequentially:
   - `sample` - samples points from mesh faces
   - `normalize_pc` - normalizes to unit sphere
   - `add_noise` - adds Laplacian noise for training
   - `linear` - optional rotation/scaling augmentation
   - `patch` - extracts local patches, creates noisy/clean/mix triplets
3. Model's `process_fn` converts `Asset` to tensor dict
4. Collate function batches tensors for training

### Data Flow During Prediction
1. Load `.npy` noisy point clouds directly
2. Apply transform (sample → normalize → add_noise → patch)
3. Patch-based denoising via `denoise_langevin_dynamics`
4. `VMWriter` saves denoised point clouds

### Adding New Models
To add a new model:
1. Create class in `src/model/` extending `ModelSpec`
2. Implement `process_fn()` and `training_step()`
3. Add to `src/model/parse.py` MAP
4. Create config in `configs/model/*.yaml`
5. Reference in task config

### Important Notes
- Uses Jittor, not PyTorch. Syntax is similar but check Jittor docs for differences
- All transforms are applied to `Asset` objects in-place
- The `patch` augment stores data in `asset.meta['pc_noisy/clean/mix']` for training
- Prediction uses patch-based processing with farthest point sampling for seed selection
