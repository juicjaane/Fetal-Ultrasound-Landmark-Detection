import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import set_seed
from .dataset import UltrasoundHeatmapDataset, PreprocessZScore, AUG_POLICIES
from .models import UNetSingleHead, UNetMultiHead
from .losses import WeightedHeatmapMSELoss, HeatmapWithAngleLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Landmark Detection Model")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to ground truth CSV")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument(
        "--model_type",
        type=str,
        default="single_head",
        choices=["single_head", "multi_head"],
        help="Model architecture"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_angle_loss", action="store_true", help="Use Angle Consistency Loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_and_correct_df(csv_path):
    df = pd.read_csv(csv_path)

    # Correct label semantics (Original CSV is inverted)
    # As per taska-3.ipynb
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
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    df = load_and_correct_df(args.data_csv)

    # Filter missing images
    df['exists'] = df['image_name'].apply(
        lambda x: os.path.isfile(os.path.join(args.image_dir, x))
    )
    df = df[df['exists']].reset_index(drop=True)
    print(f"Loaded {len(df)} valid samples.")

    # Split
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    val_split = 0.15
    val_size = int(len(indices) * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Datasets
    preprocess = PreprocessZScore()
    augment_policy = AUG_POLICIES["A2_standard"]

    train_dataset = UltrasoundHeatmapDataset(
        df=train_df,
        image_dir=args.image_dir,
        preprocess=preprocess,
        augment_policy=augment_policy,
        augment=True
    )

    val_dataset = UltrasoundHeatmapDataset(
        df=val_df,
        image_dir=args.image_dir,
        preprocess=preprocess,
        augment_policy=None,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Model
    in_channels = 1  # Z-score only
    if args.model_type == "single_head":
        model = UNetSingleHead(in_channels=in_channels).to(device)
    elif args.model_type == "multi_head":
        model = UNetMultiHead(in_channels=in_channels).to(device)
    else:
        raise ValueError("Unknown model type")

    # Loss
    # Weights: [BPD1, BPD2, OFD1, OFD2]
    weights = torch.tensor([1.0, 1.0, 2.0, 2.0], device=device)

    base_loss = WeightedHeatmapMSELoss(weights)

    if args.use_angle_loss:
        criterion = HeatmapWithAngleLoss(base_loss, lambda_angle=0.01, warmup_epochs=5)
    else:
        def criterion(p, t):
            return base_loss(p, t), {}

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        model.train()
        if isinstance(criterion, HeatmapWithAngleLoss):
            criterion.set_epoch(epoch)

        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for imgs, hms, _ in pbar:
            imgs = imgs.to(device)
            hms = hms.to(device)

            preds = model(imgs)
            loss, parts = criterion(preds, hms)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Val
        model.eval()
        val_losses = []

        with torch.no_grad():
            for imgs, hms, _ in val_loader:
                imgs = imgs.to(device)
                hms = hms.to(device)

                preds = model(imgs)
                loss, _ = criterion(preds, hms)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = os.path.join(args.output_dir, f"{args.model_type}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
