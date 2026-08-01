import cv2
import numpy as np

from moaview.quality import calculate_quality


def test_constant_image_has_expected_brightness_and_zero_contrast() -> None:
    image = np.full((80, 100, 3), 120, dtype=np.uint8)

    metrics = calculate_quality(image)

    assert metrics.brightness == 120.0
    assert metrics.contrast == 0.0
    assert metrics.sharpness == 0.0


def test_edge_image_is_sharper_than_blurred_version() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, 60:] = 255
    blurred = cv2.GaussianBlur(image, (15, 15), sigmaX=4)

    sharp_metrics = calculate_quality(image)
    blurred_metrics = calculate_quality(blurred)

    assert sharp_metrics.sharpness > blurred_metrics.sharpness

