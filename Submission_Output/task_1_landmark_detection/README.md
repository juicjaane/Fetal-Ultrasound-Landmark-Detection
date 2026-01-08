# Task 1: Landmark Detection

This folder contains the solution for the Landmark Detection task.

## Structure

- `pythonscripts/`: Contains the source code for the model and training.
- `models/`: Contains the trained model weights.
- `report/`: Contains the analysis report.

## Usage

Run these commands from the `Submission_Output` directory.

### Training

```bash
python -m task_1_landmark_detection.pythonscripts.trainer --data_csv datasets/task_1/role_challenge_dataset_ground_truth.csv --image_dir datasets/task_1/images
```

### Evaluation

```bash
python -m task_1_landmark_detection.pythonscripts.tester --data_csv datasets/task_1/role_challenge_dataset_ground_truth.csv --image_dir datasets/task_1/images --weights_path task_1_landmark_detection/models/your_model.pt
```
