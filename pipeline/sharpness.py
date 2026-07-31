"""Frame sharpness scoring via variance of the Laplacian."""

from __future__ import annotations

from pathlib import Path

import cv2


def variance_of_laplacian(image_path: Path) -> float:
    """Score an image's sharpness as the variance of its Laplacian.

    Higher scores indicate sharper (more high-frequency detail) images;
    heavily blurred images score close to zero.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())
