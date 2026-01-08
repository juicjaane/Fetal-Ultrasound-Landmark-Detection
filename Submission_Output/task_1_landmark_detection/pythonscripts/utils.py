import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore(img: np.ndarray):
    """
    Apply Z-score normalization to an image.
    """
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std() + 1e-6
    return (img - mean) / std


def percentile_clip(img: np.ndarray, low=1.0, high=99.0):
    """
    Clip image intensities to specific percentiles.
    """
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    img = np.clip(img, lo, hi)
    return img


def euclidean(p1, p2):
    """
    Compute Euclidean distance between two points.
    """
    return np.sqrt(((p1 - p2) ** 2).sum())


def compute_biometry(coords):
    """
    Compute BPD and OFD lengths from coordinates.
    Assumes coords order: [BPD1, BPD2, OFD1, OFD2]
    """
    bpd = euclidean(coords[0], coords[1])
    ofd = euclidean(coords[2], coords[3])
    return bpd, ofd


def heatmap_entropy(hm):
    """
    Compute entropy of a heatmap distribution.
    """
    p = hm / (hm.sum() + 1e-8)
    return -(p * np.log(p + 1e-8)).sum()
