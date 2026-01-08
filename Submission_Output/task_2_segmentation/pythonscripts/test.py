import argparse
import os
import glob
from collections import defaultdict
import torch
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from .model import UNet
from .inference import run_geometry_pipeline
from .geometry import extract_bpd_ofd_landmarks, angle_between


def main():
    parser = argparse.ArgumentParser(description="Evaluate Skull Boundary Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="baseline",
        choices=["baseline", "hyp1", "hyp2", "hyp3"],
        help="Experiment hypothesis"
    )
    parser.add_argument(
        "--use_postprocessing",
        action="store_true",
        help="Apply post-processing (Hypothesis 4)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization of random samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save visualizations"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    image_size = 512 if args.hypothesis == "hyp1" else 256

    # Load Model
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Data (Validation set only)
    image_files = sorted(glob.glob(os.path.join(args.data_dir, "*")))
    _, val_files = train_test_split(image_files, test_size=0.2, random_state=42, shuffle=True)

    print(f"Evaluating on {len(val_files)} validation samples...")

    stats = defaultdict(int)

    # Visualization setup
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        vis_samples = np.random.choice(val_files, min(5, len(val_files)), replace=False)
    else:
        vis_samples = []

    for img_path in val_files:
        img, binary, contour, ellipse = run_geometry_pipeline(
            model, img_path, image_size, device,
            threshold=0.5, use_postprocessing=args.use_postprocessing
        )

        # Visualization
        if args.visualize and img_path in vis_samples:
            canvas = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if contour is not None:
                cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 1)
            if ellipse is not None:
                cv2.ellipse(canvas, ellipse, (0, 0, 255), 2)
                bpd_pts, ofd_pts = extract_bpd_ofd_landmarks(ellipse)
                cv2.line(
                    canvas,
                    tuple(bpd_pts[0].astype(int)),
                    tuple(bpd_pts[1].astype(int)),
                    (0, 255, 0),
                    2
                )
                cv2.line(
                    canvas,
                    tuple(ofd_pts[0].astype(int)),
                    tuple(ofd_pts[1].astype(int)),
                    (255, 0, 0),
                    2
                )

            base_name = os.path.basename(img_path)
            save_path = os.path.join(args.output_dir, f"vis_{base_name}")
            cv2.imwrite(save_path, canvas)
            print(f"Saved visualization to {save_path}")

        if ellipse is None:
            continue

        stats["valid_ellipse"] += 1

        bpd_pts, ofd_pts = extract_bpd_ofd_landmarks(ellipse)

        bpd_len = np.linalg.norm(bpd_pts[0] - bpd_pts[1])
        ofd_len = np.linalg.norm(ofd_pts[0] - ofd_pts[1])

        # Check 1: BPD < OFD
        if bpd_len < ofd_len:
            stats["bpd_less_than_ofd"] += 1

        # Check 2: Perpendicularity
        v_bpd = bpd_pts[1] - bpd_pts[0]
        v_ofd = ofd_pts[1] - ofd_pts[0]
        angle = angle_between(v_bpd, v_ofd)

        if abs(angle - 90) < 15:
            stats["axes_perpendicular"] += 1

        stats["total"] += 1

    print("\nEvaluation Results:")
    for k, v in stats.items():
        if k != "total":
            print(f"{k}: {v} / {stats['total']} ({v/stats['total']*100:.1f}%)")

            
    print(f"Total processed: {stats['total']}")

if __name__ == "__main__":
    main()
