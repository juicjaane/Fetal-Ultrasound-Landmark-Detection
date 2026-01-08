# Task 1: Fetal Ultrasound Landmark Detection

## Overview
This repository contains the code for training and evaluating a deep learning model to detect BPD and OFD landmarks in fetal ultrasound images.

## Folder Structure
- `Model_Weights/`: Stores trained model checkpoints.
- `Python_Script/`: Contains all source code.
  - `trainer.py`: Script for training the model.
  - `tester.py`: Script for evaluating the model.
  - `dataset.py`: Dataset loading and augmentation logic.
  - `models.py`: U-Net architecture definitions.
  - `losses.py`: Custom loss functions (Heatmap MSE + Angle Consistency).
  - `utils.py`: Utility functions.
- `Report/`: Project documentation.
- `Assets/`: Helper scripts and EDA tools.

## Usage

### Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- Pandas
- NumPy

### Training
To train the model, run:
```bash
python Python_Script/trainer.py --data_csv /path/to/csv --image_dir /path/to/images --output_dir Model_Weights --epochs 20
```

### Evaluation
To evaluate a trained model:
```bash
python Python_Script/tester.py --data_csv /path/to/csv --image_dir /path/to/images --weights_path Model_Weights/single_head_best.pth
```

## Key Features
- **Label Correction**: Automatically handles inverted BPD/OFD labels in the provided dataset.
- **Angle Consistency Loss**: Enforces geometric validity of predictions.
- **Robust Preprocessing**: Uses Z-score normalization and affine augmentations.
