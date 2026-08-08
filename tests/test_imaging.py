import cv2
import numpy as np
import pytest

from dasimanna.demo import create_demo_pet_image
from dasimanna.imaging import (
    InvalidImageError,
    aggregate_signatures,
    compare_signatures,
    decode_image,
    extract_image_signature,
    measure_image_quality,
)


def test_decode_image_rejects_invalid_bytes() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"not-an-image")


def test_decode_image_accepts_png() -> None:
    image = create_demo_pet_image("target")
    ok, encoded = cv2.imencode(".png", image)

    assert ok
    decoded = decode_image(encoded.tobytes())
    assert decoded.shape == image.shape


def test_blur_reduces_sharpness_and_reliability() -> None:
    image = create_demo_pet_image("target")
    blurred = cv2.GaussianBlur(image, (25, 25), 7.0)

    clear_quality = measure_image_quality(image)
    blurred_quality = measure_image_quality(blurred)

    assert clear_quality.sharpness > blurred_quality.sharpness
    assert clear_quality.reliability > blurred_quality.reliability


def test_target_signature_is_closer_than_distractor() -> None:
    reference = extract_image_signature(create_demo_pet_image("target", 0))
    second_reference = extract_image_signature(create_demo_pet_image("target", 1))
    aggregate = aggregate_signatures([reference, second_reference])
    similar = extract_image_signature(create_demo_pet_image("target", 1))
    distractor = extract_image_signature(create_demo_pet_image("distractor", 0))

    similar_score = compare_signatures(aggregate, similar)["overall"]
    distractor_score = compare_signatures(aggregate, distractor)["overall"]

    assert 0.0 <= distractor_score <= 1.0
    assert similar_score > distractor_score
    assert aggregate.dominant_colors


def test_measure_image_quality_requires_color_image() -> None:
    with pytest.raises(InvalidImageError):
        measure_image_quality(np.zeros((10, 10), dtype=np.uint8))
