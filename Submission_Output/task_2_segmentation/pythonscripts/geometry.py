import cv2
import numpy as np


def get_largest_contour(binary_mask):
    """
    Extract the largest contour from a binary mask.
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_ellipse_to_contour(contour):
    """
    Fit an ellipse to the given contour.
    Requires at least 5 points.
    """
    if contour is None or len(contour) < 5:
        return None
    try:
        ellipse = cv2.fitEllipse(contour)
        return ellipse
    except Exception:
        return None


def extract_bpd_ofd_landmarks(ellipse):
    """
    ellipse: output of cv2.fitEllipse
    returns:
        BPD points (2, 2)
        OFD points (2, 2)
    """
    (cx, cy), (width, height), angle = ellipse

    # Major and minor radii
    if width >= height:
        major_radius = width / 2.0
        minor_radius = height / 2.0
        major_angle = angle
    else:
        major_radius = height / 2.0
        minor_radius = width / 2.0
        major_angle = angle + 90.0

    # Convert to radians
    theta_major = np.deg2rad(major_angle)
    theta_minor = theta_major + np.pi / 2.0

    # OFD endpoints (major axis)
    ofd_p1 = (
        cx + major_radius * np.cos(theta_major),
        cy + major_radius * np.sin(theta_major)
    )
    ofd_p2 = (
        cx - major_radius * np.cos(theta_major),
        cy - major_radius * np.sin(theta_major)
    )

    # BPD endpoints (minor axis)
    bpd_p1 = (
        cx + minor_radius * np.cos(theta_minor),
        cy + minor_radius * np.sin(theta_minor)
    )
    bpd_p2 = (
        cx - minor_radius * np.cos(theta_minor),
        cy - minor_radius * np.sin(theta_minor)
    )

    return np.array([bpd_p1, bpd_p2]), np.array([ofd_p1, ofd_p2])


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

