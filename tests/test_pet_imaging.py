import cv2
import numpy as np

from dasimanna.imaging import (
    aggregate_signatures,
    compare_signatures,
    extract_image_signature,
    measure_image_quality,
)


def _pattern(color: tuple[int, int, int]) -> np.ndarray:
    image = np.full((120, 160, 3), (225, 225, 225), dtype=np.uint8)
    cv2.ellipse(image, (80, 67), (52, 34), 0, 0, 360, color, -1)
    cv2.rectangle(image, (72, 28), (88, 86), (245, 245, 245), -1)
    return image


def test_same_pet_under_brightness_change_scores_above_different_color() -> None:
    reference = _pattern((70, 120, 185))
    brighter = cv2.convertScaleAbs(reference, alpha=0.88, beta=18)
    different = _pattern((115, 115, 115))

    reference_signature = extract_image_signature(reference)
    same_score = compare_signatures(reference_signature, extract_image_signature(brighter))
    different_score = compare_signatures(reference_signature, extract_image_signature(different))

    assert same_score["overall"] > different_score["overall"]
    assert same_score["color"] > different_score["color"]


def test_multiple_reference_signatures_can_be_aggregated() -> None:
    first = extract_image_signature(_pattern((70, 120, 185)))
    second = extract_image_signature(cv2.convertScaleAbs(_pattern((70, 120, 185)), alpha=0.8))

    aggregate = aggregate_signatures([first, second])

    assert aggregate.color_histogram.shape == first.color_histogram.shape
    assert aggregate.perceptual_bits.shape == first.perceptual_bits.shape
    assert aggregate.dominant_colors


def test_blur_reduces_measured_sharpness() -> None:
    clear = _pattern((70, 120, 185))
    blurred = cv2.GaussianBlur(clear, (21, 21), 5.0)

    assert measure_image_quality(clear).sharpness > measure_image_quality(blurred).sharpness
