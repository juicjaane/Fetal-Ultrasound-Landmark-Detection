import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import UltrasoundHeatmapDataset, PreprocessZScore
from .models import UNetSingleHead, UNetMultiHead
from .losses import soft_argmax_2d
from .utils import euclidean, compute_biometry


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Landmark Detection Model")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument(
        "--model_type",
        type=str,
        default="single_head",
        choices=["single_head", "multi_head"],
        help="Model architecture"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation_results.csv",
        help="Path to save results"
    )
    return parser.parse_args()


def load_and_correct_df(csv_path):
    df = pd.read_csv(csv_path)

    # Correct label semantics (Original CSV is inverted)
    df["BPD_1_x"] = df["ofd_1_x"]
    df["BPD_1_y"] = df["ofd_1_y"]
    df["BPD_2_x"] = df["ofd_2_x"]
    df["BPD_2_y"] = df["ofd_2_y"]

    df["OFD_1_x"] = df["bpd_1_x"]
    df["OFD_1_y"] = df["bpd_1_y"]
    df["OFD_2_x"] = df["bpd_2_x"]
    df["OFD_2_y"] = df["bpd_2_y"]

    return df


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    df = load_and_correct_df(args.data_csv)

    # Filter missing images
    df['exists'] = df['image_name'].apply(
        lambda x: os.path.isfile(os.path.join(args.image_dir, x))
    )
    df = df[df['exists']].reset_index(drop=True)
    print(f"Loaded {len(df)} samples for evaluation.")

    # Dataset
    preprocess = PreprocessZScore()
    dataset = UltrasoundHeatmapDataset(
        df=df,
        image_dir=args.image_dir,
        preprocess=preprocess,
        augment_policy=None,
        augment=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    in_channels = 1
    if args.model_type == "single_head":
        model = UNetSingleHead(in_channels=in_channels).to(device)
    elif args.model_type == "multi_head":
        model = UNetMultiHead(in_channels=in_channels).to(device)
    else:
        raise ValueError("Unknown model type")

    # Load Weights
    print(f"Loading weights from {args.weights_path}")
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for img, gt_hm, name in tqdm(loader, desc="Evaluating"):
            img = img.to(device)
            gt_hm = gt_hm.to(device)

            pred_hm = model(img)

            # Extract coordinates
            # Using soft_argmax with high beta for precision
            gt_coords = soft_argmax_2d(gt_hm, beta=100.0)[0].cpu().numpy()
            pred_coords = soft_argmax_2d(pred_hm, beta=100.0)[0].cpu().numpy()

            # Compute Errors
            lm_errors = [euclidean(gt_coords[i], pred_coords[i]) for i in range(4)]
            mean_lm_error = float(np.mean(lm_errors))

            # Biometry
            gt_bpd, gt_ofd = compute_biometry(gt_coords)
            pred_bpd, pred_ofd = compute_biometry(pred_coords)

            results.append({
                "image_name": name[0],
                "BPD1_err": lm_errors[0],
                "BPD2_err": lm_errors[1],
                "OFD1_err": lm_errors[2],
                "OFD2_err": lm_errors[3],
                "mean_err": mean_lm_error,
                "gt_bpd": gt_bpd,
                "pred_bpd": pred_bpd,
                "bpd_err": abs(gt_bpd - pred_bpd),
                "gt_ofd": gt_ofd,
                "pred_ofd": pred_ofd,
                "ofd_err": abs(gt_ofd - pred_ofd)
            })

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

    # Summary
    print("\nEvaluation Summary:")
    print(f"Mean Landmark Error: {res_df['mean_err'].mean():.2f} px")
    print(f"Mean BPD Error: {res_df['bpd_err'].mean():.2f} px")
    print(f"Mean OFD Error: {res_df['ofd_err'].mean():.2f} px")


if __name__ == "__main__":
    main()
