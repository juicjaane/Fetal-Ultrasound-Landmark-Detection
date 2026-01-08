# Archived Model Outputs

This directory contains the full output folders for all model training runs performed for Task 1 (Landmark Detection).

## Best Models Extracted
The following models were identified as the best performing in their respective categories and have been copied to the parent `models` directory:

*   **Best Multi-Head Model:** `STAGE3_FULL__multi_head_P1_A1` (Renamed to `best_multi_head_model.pth`)
    *   Landmark Mean Error: 14.3956
*   **Best Single-Head Model:** `single_head_P0_A0` (Renamed to `best_single_head_model.pth`)
    *   Landmark Mean Error: 20.0526
*   **Best High-Resolution Model:** `STAGE1_BASELINE__high_res_P0_A0` (Renamed to `best_high_res_model.pth`)
    *   Landmark Mean Error: 26.3405

## Contents
Each folder in this archive corresponds to a specific training configuration and contains:
*   `best_model.pth`: The model checkpoint.
*   `eval_metrics.json`: Detailed evaluation metrics.
*   `per_image_metrics.json`: Metrics for each image in the validation set.
*   `visualization.png`: Visual examples of the model's predictions.

For a full comparison of all models, please refer to `model_results.csv` in the parent directory.
