import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import zscore, percentile_clip


def generate_gaussian_heatmap(x, y, size, sigma):
    """
    x, y: float coordinates in resized image space
    """
    xs = np.arange(size)
    ys = np.arange(size)
    xx, yy = np.meshgrid(xs, ys)

    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)


class PreprocessZScore:
    name = "P0_zscore"

    def __call__(self, img: np.ndarray):
        # img: uint8 or float, shape (H, W)
        img = img.astype(np.float32) / 255.0
        img = zscore(img)
        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


class PreprocessLogZScore:
    name = "P1_log_zscore"

    def __call__(self, img: np.ndarray):
        img = img.astype(np.float32) / 255.0

        # Log compression (physics aware)
        img = np.log(img + 1e-3)

        # Percentile clipping
        img = percentile_clip(img, 1.0, 99.0)

        # Z-score
        img = zscore(img)

        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


class PreprocessMultiChannel:
    name = "P2_multichannel"

    def __call__(self, img: np.ndarray):
        img = img.astype(np.float32) / 255.0

        # Channel 1: z-scored raw
        raw = zscore(img)

        # Channel 2: gradient magnitude
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        grad = zscore(grad)

        stacked = np.stack([raw, grad], axis=0)  # (2, H, W)
        return torch.from_numpy(stacked)


def sample_affine_params(max_rotation, max_translation, scale_range, do_flip):
    angle = random.uniform(-max_rotation, max_rotation)
    scale = random.uniform(scale_range[0], scale_range[1])

    tx = random.uniform(-max_translation, max_translation)
    ty = random.uniform(-max_translation, max_translation)

    flip = do_flip and random.random() < 0.5

    return angle, scale, tx, ty, flip


def apply_affine(img, points, angle, scale, tx, ty, flip):
    """
    img: (H, W)
    points: (N, 2) in pixel coordinates
    """
    h, w = img.shape
    cx, cy = w / 2, h / 2

    # Rotation + scale
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

    # Translation (fractional)
    M[0, 2] += tx * w
    M[1, 2] += ty * h

    # Warp image
    warped = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Transform points
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    pts_warped = (M @ pts_h.T).T

    # Optional horizontal flip
    if flip:
        warped = cv2.flip(warped, 1)
        pts_warped[:, 0] = w - pts_warped[:, 0]

    return warped, pts_warped


class AugmentationPolicy:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __call__(self, img, points):
        if self.params is None:
            return img, points

        angle, scale, tx, ty, flip = sample_affine_params(**self.params)
        return apply_affine(img, points, angle, scale, tx, ty, flip)


AUG_POLICIES = {
    "A0_none": AugmentationPolicy("A0_none", None),

    "A1_light": AugmentationPolicy("A1_light", dict(
        max_rotation=5,
        max_translation=0.03,
        scale_range=(0.95, 1.05),
        do_flip=False
    )),

    "A2_standard": AugmentationPolicy("A2_standard", dict(
        max_rotation=10,
        max_translation=0.05,
        scale_range=(0.9, 1.1),
        do_flip=True
    )),

    "A3_strong": AugmentationPolicy("A3_strong", dict(
        max_rotation=15,
        max_translation=0.08,
        scale_range=(0.85, 1.15),
        do_flip=True
    )),
}


class UltrasoundHeatmapDataset(Dataset):
    def __init__(
        self,
        df,
        image_dir,
        preprocess,
        augment_policy,
        img_size=256,
        sigma=3.0,
        augment=False
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.augment_policy = augment_policy
        self.img_size = img_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ------------------------------
        # Load image
        # ------------------------------
        img_path = os.path.join(self.image_dir, row.image_name)
        img = cv2.imread(img_path, 0)
        if img is None:
            raise FileNotFoundError(f"Failed to load {row.image_name} at {img_path}")

        # ------------------------------
        # Load landmarks (native space)
        # ------------------------------
        # Note: Assuming DF has corrected columns BPD_1_x etc.
        points = np.array([
            [row.BPD_1_x, row.BPD_1_y],
            [row.BPD_2_x, row.BPD_2_y],
            [row.OFD_1_x, row.OFD_1_y],
            [row.OFD_2_x, row.OFD_2_y],
        ], dtype=np.float32)

        # ------------------------------
        # Augmentation (optional)
        # ------------------------------
        if self.augment and self.augment_policy is not None:
            img, points = self.augment_policy(img, points)

        # Safety check
        # assert np.isfinite(points).all(), "Non-finite landmarks after augmentation"

        # ------------------------------
        # Resize image and scale landmarks
        # ------------------------------
        h, w = img.shape
        img_resized = cv2.resize(img, (self.img_size, self.img_size))

        sx = self.img_size / w
        sy = self.img_size / h

        points_resized = points.copy()
        points_resized[:, 0] *= sx
        points_resized[:, 1] *= sy

        # Safety checks
        # assert np.isfinite(points_resized).all(), "Non-finite landmarks after resize"

        # ------------------------------
        # Preprocessing (image only)
        # ------------------------------
        img_tensor = self.preprocess(img_resized)

        # ------------------------------
        # Heatmap generation
        # ------------------------------
        heatmaps = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)

        for i, (x, y) in enumerate(points_resized):
            # Clamp to image bounds
            x = np.clip(x, 0, self.img_size - 1)
            y = np.clip(y, 0, self.img_size - 1)

            heatmaps[i] = generate_gaussian_heatmap(
                x, y, self.img_size, self.sigma
            )

        heatmaps = torch.from_numpy(heatmaps)

        return img_tensor, heatmaps, row.image_name
