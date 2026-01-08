import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataset import SkullBoundaryDataset, SkullBoundaryAugmentedDataset
from .model import UNet
from .trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Skull Boundary Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to masks directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="baseline",
        choices=["baseline", "hyp1", "hyp2", "hyp3"],
        help="Experiment hypothesis"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration based on hypothesis
    image_size = 256
    boundary_thickness = 1
    use_augmentation = False

    if args.hypothesis == "hyp1":
        image_size = 512
    elif args.hypothesis == "hyp2":
        boundary_thickness = 3
    elif args.hypothesis == "hyp3":
        use_augmentation = True

    print(
        f"Configuration: Hypothesis={args.hypothesis}, "
        f"Size={image_size}, Thickness={boundary_thickness}, "
        f"Augmentation={use_augmentation}"
    )

    # Data Preparation
    image_files = sorted(glob.glob(os.path.join(args.data_dir, "*")))
    train_files, val_files = train_test_split(
        image_files, test_size=0.2, random_state=42, shuffle=True
    )

    if use_augmentation:
        train_dataset = SkullBoundaryAugmentedDataset(
            train_files, args.mask_dir, image_size, boundary_thickness
        )
    else:
        train_dataset = SkullBoundaryDataset(
            train_files, args.mask_dir, image_size, boundary_thickness
        )

    val_dataset = SkullBoundaryDataset(
        val_files, args.mask_dir, image_size, boundary_thickness
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Calculate positive weight
    print("Calculating positive weight...")
    total_pixels = 0
    foreground_pixels = 0

    # Iterate the train_dataset for a few batches to approximate
    for i in range(min(len(train_dataset), 50)):
        _, mask = train_dataset[i]
        total_pixels += mask.numel()
        foreground_pixels += mask.sum().item()

    if foreground_pixels > 0:
        pos_weight_val = (total_pixels - foreground_pixels) / foreground_pixels
    else:
        pos_weight_val = 1.0

    pos_weight = torch.tensor(pos_weight_val, dtype=torch.float32).to(device)
    print(f"Positive weight: {pos_weight.item()}")

    # Model
    model = UNet().to(device)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, device, args.lr, pos_weight)
    trainer.train(
        args.epochs,
        os.path.join(args.output_dir, f"model_{args.hypothesis}.pth")
    )

    model_path = os.path.join(args.output_dir, f"{args.hypothesis}_boundary_unet.pt")
    trainer.train(args.epochs, model_path)

if __name__ == "__main__":
    main()
