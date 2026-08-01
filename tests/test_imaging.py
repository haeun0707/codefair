import cv2
import numpy as np
import pytest

from moaview.imaging import InvalidImageError, crop_detection, decode_image
from moaview.models import Detection


def test_decode_png_bytes() -> None:
    source = np.full((20, 30, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", source)
    assert ok

    decoded = decode_image(encoded.tobytes())

    assert decoded.shape == source.shape
    assert np.array_equal(decoded, source)


def test_decode_rejects_empty_bytes() -> None:
    with pytest.raises(InvalidImageError, match="비어"):
        decode_image(b"")


def test_crop_clamps_box_to_image() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    detection = Detection((-5, -3, 8, 6), 0.9, "car")

    crop = crop_detection(image, detection)

    assert crop.shape == (6, 8, 3)

