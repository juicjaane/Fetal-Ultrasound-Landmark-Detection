import cv2
import numpy as np
import torch
from .dataset import preprocess_image_minimal, resize_image
from .geometry import get_largest_contour, fit_ellipse_to_contour


def postprocess_boundary(binary_mask, closing_kernel_size=5):
    """
    Postprocess predicted boundary mask for geometric stability.
    Steps:
    - Morphological closing
    - Largest connected component selection
    """
    # Ensure binary uint8
    mask = (binary_mask > 0).astype(np.uint8)

    # Morphological closing to fill gaps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8
    )

    if num_labels <= 1:
        return closed

    # Ignore background label 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    largest_component = (labels == largest_label).astype(np.uint8)

    return largest_component


def predict_boundary(model, img_np, device):
    """
    img_np: (H, W) normalized image
    returns: (H, W) probability map
    """
    img_tensor = torch.tensor(
        img_np, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)

    return probs.squeeze().cpu().numpy()


def run_geometry_pipeline(
    model, img_path, image_size, device, threshold=0.5, use_postprocessing=False
):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image_minimal(img)
    img = resize_image(img, image_size)

    prob = predict_boundary(model, img, device)
    binary = (prob > threshold).astype(np.uint8)

    if use_postprocessing:
        binary = postprocess_boundary(binary)

    contour = get_largest_contour(binary)
    ellipse = fit_ellipse_to_contour(contour)

    return img, binary, contour, ellipse

