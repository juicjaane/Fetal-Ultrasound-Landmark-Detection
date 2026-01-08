import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_image(img):
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)
    return img


def denoise_image(img):
    img_uint8 = (img * 255).astype(np.uint8)
    img_denoised = cv2.medianBlur(img_uint8, 5)
    img_denoised = img_denoised.astype(np.float32) / 255.0
    return img_denoised


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE on normalized image
    """
    img_uint8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    img_clahe = clahe.apply(img_uint8)
    img_clahe = img_clahe.astype(np.float32) / 255.0
    return img_clahe


def preprocess_image_minimal(img):
    img = normalize_image(img)
    img = denoise_image(img)
    return img


def resize_image(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask, size):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def mask_to_boundary(mask, thickness=1):
    """
    Convert a filled binary mask to a boundary-only mask.
    """
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    boundary = np.zeros_like(mask)
    for cnt in contours:
        cv2.drawContours(
            boundary,
            [cnt],
            contourIdx=-1,
            color=1,
            thickness=thickness
        )
    return boundary


def image_to_mask_path(image_path, mask_dir):
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    return os.path.join(mask_dir, f"{name}_Annotation.png")


def preprocess_pair_boundary(img_path, mask_path, size, boundary_thickness=1):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img = preprocess_image_minimal(img)
    # Basic mask preprocessing (thresholding)
    mask = (mask > 0).astype(np.uint8)

    img = resize_image(img, size)
    mask = resize_mask(mask, size)

    boundary = mask_to_boundary(mask, thickness=boundary_thickness)

    return img, boundary


def random_affine(img, mask, max_rotate=10, max_translate=0.05):
    h, w = img.shape
    angle = random.uniform(-max_rotate, max_rotate)
    tx = random.uniform(-max_translate, max_translate) * w
    ty = random.uniform(-max_translate, max_translate) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += [tx, ty]

    img_aug = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    mask_aug = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101
    )

    if random.random() < 0.5:
        img_aug = np.fliplr(img_aug)
        mask_aug = np.fliplr(mask_aug)

    return img_aug, mask_aug


class SkullBoundaryDataset(Dataset):
    def __init__(self, image_paths, mask_dir, size, boundary_thickness=1):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.size = size
        self.boundary_thickness = boundary_thickness

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = image_to_mask_path(img_path, self.mask_dir)

        img, boundary = preprocess_pair_boundary(
            img_path, mask_path, self.size, self.boundary_thickness
        )

        img = np.expand_dims(img, axis=0)
        boundary = np.expand_dims(boundary, axis=0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(boundary, dtype=torch.float32)
        )


class SkullBoundaryAugmentedDataset(Dataset):
    def __init__(self, image_paths, mask_dir, size, boundary_thickness=1):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.size = size
        self.boundary_thickness = boundary_thickness

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = image_to_mask_path(img_path, self.mask_dir)

        img, boundary = preprocess_pair_boundary(
            img_path, mask_path, self.size, self.boundary_thickness
        )

        img, boundary = random_affine(img, boundary)

        img = torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0)
        boundary = torch.tensor(
            boundary.copy(), dtype=torch.float32
        ).unsqueeze(0)

        return img, boundary
