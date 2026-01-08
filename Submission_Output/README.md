# Origin Medical Submission

This repository contains the solution for the Origin Medical challenge, divided into two tasks.

## Directory Structure

The submission is organized as follows:

```
Submission_Output/
├── datasets/                   # Contains datasets for both tasks
│   ├── task_1/                 # Dataset for Task 1 (Images + CSV)
│   └── task_2/                 # Dataset for Task 2 (Images + Masks)
├── task_1_landmark_detection/  # Solution for Task 1
│   ├── models/                 # Saved model weights
│   ├── pythonscripts/          # Source code
│   ├── report/                 # Analysis and reports
│   └── README.md               # Task-specific documentation
├── task_2_segmentation/        # Solution for Task 2
│   ├── models/                 # Saved model weights
│   ├── pythonscripts/          # Source code
│   ├── report/                 # Analysis and reports (Notebook)
│   └── README.md               # Task-specific documentation
└── README.md                   # This file
```

## Task 1: Landmark Detection

Please refer to `task_1_landmark_detection/README.md` for detailed instructions on training and evaluating the landmark detection model.

## Task 2: Segmentation & Geometric Analysis

Please refer to `task_2_segmentation/README.md` for detailed instructions on training and evaluating the segmentation model.

## Datasets

The datasets are stored in the `datasets/` directory.
- Task 1 dataset is located in `datasets/task_1/`.
- Task 2 dataset is located in `datasets/task_2/`.
- Ensure that the data paths in the training/testing commands point to the correct location in `datasets/`.

## GitHub Repository

For more details and version history, please visit the GitHub repository:
[https://github.com/juicjaane/Fetal-Ultrasound-Landmark-Detection](https://github.com/juicjaane/Fetal-Ultrasound-Landmark-Detection)
