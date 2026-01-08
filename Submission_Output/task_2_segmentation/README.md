# Task 2: Skull Boundary Segmentation & Geometric Analysis

This package implements a research pipeline for fetal skull segmentation and geometric measurement (BPD/OFD) from ultrasound images.

## Structure

- `pythonscripts/`: Contains all source code.
    - `dataset.py`: Data loading, preprocessing (CLAHE, Denoising), and augmentation.
    - `model.py`: Standard U-Net architecture.
    - `geometry.py`: Geometric analysis (Ellipse fitting, Landmark extraction).
    - `trainer.py`: Training loop with Combined Loss (BCE + Soft Dice).
    - `inference.py`: Inference pipeline with Post-processing (Hypothesis 4).
    - `train.py`: CLI for training models.
    - `test.py`: CLI for evaluation and visualization.
- `models/`: Directory for saved model checkpoints.
- `report/`: Contains the original research notebook (`task2.ipynb`).

## Usage

Run these commands from the `Submission_Output` directory.

### Training

Train the baseline model:
```bash
python -m task_2_segmentation.pythonscripts.train --data_dir datasets/task_2/images --mask_dir datasets/task_2/masks --hypothesis baseline
```

Train Hypothesis 1 (High Resolution 512x512):
```bash
python -m task_2_segmentation.pythonscripts.train --data_dir datasets/task_2/images --mask_dir datasets/task_2/masks --hypothesis hyp1
```

Train Hypothesis 2 (Thicker Boundary 3px):
```bash
python -m task_2_segmentation.pythonscripts.train --data_dir datasets/task_2/images --mask_dir datasets/task_2/masks --hypothesis hyp2
```

Train Hypothesis 3 (Augmentation):
```bash
python -m task_2_segmentation.pythonscripts.train --data_dir datasets/task_2/images --mask_dir datasets/task_2/masks --hypothesis hyp3
```

### Evaluation

Evaluate a trained model:
```bash
python -m task_2_segmentation.pythonscripts.test --data_dir datasets/task_2/images --model_path task_2_segmentation/models/baseline_boundary_unet.pt --hypothesis baseline
```

Evaluate with Post-processing (Hypothesis 4) and Visualization:
```bash
python -m task_2_segmentation.pythonscripts.test --data_dir datasets/task_2/images --model_path task_2_segmentation/models/baseline_boundary_unet.pt --use_postprocessing --visualize
```

## Hypotheses Implemented

1.  **Baseline**: 256x256 resolution, 1px boundary target.
2.  **Hypothesis 1**: 512x512 resolution for finer detail.
3.  **Hypothesis 2**: 3px boundary target to handle annotation uncertainty.
4.  **Hypothesis 3**: Geometric augmentation (Rotation, Translation) for robustness.
5.  **Hypothesis 4**: Post-processing (Morphological Closing + Largest Component) to remove artifacts.
