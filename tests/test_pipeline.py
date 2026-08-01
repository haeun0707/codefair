import numpy as np

from moaview.models import Detection
from moaview.pipeline import analyze_image


class FakeDetector:
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return [
            Detection((10, 12, 80, 70), 0.88, "test car", 2),
            Detection((40, 30, 40, 50), 0.75, "invalid box", 2),
        ]


def test_pipeline_uses_injected_detector_and_skips_invalid_box() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[12:70, 10:80] = 160

    candidates = analyze_image(image, FakeDetector())

    assert len(candidates) == 1
    assert candidates[0].detection.label == "test car"
    assert candidates[0].crop_bgr.shape == (58, 70, 3)
    assert candidates[0].enhanced_bgr.shape == candidates[0].crop_bgr.shape

