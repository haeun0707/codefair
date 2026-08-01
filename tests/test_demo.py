import numpy as np

from moaview.demo import create_demo_image
from moaview.detection import MockVehicleDetector
from moaview.pipeline import analyze_image


def test_three_demo_images_complete_mock_flow() -> None:
    detector = MockVehicleDetector()
    sharpness_values: list[float] = []

    for index in range(3):
        image = create_demo_image(index)
        candidates = analyze_image(image, detector)
        assert image.dtype == np.uint8
        assert len(candidates) == 2
        sharpness_values.append(candidates[0].metrics.sharpness)

    assert sharpness_values[2] > sharpness_values[0]

