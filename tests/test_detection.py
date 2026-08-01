import numpy as np

from moaview.detection import MockVehicleDetector, YoloVehicleDetector


def test_mock_detector_returns_valid_candidates() -> None:
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    detections = MockVehicleDetector().detect(image)

    assert len(detections) == 2
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        assert 0 <= x1 < x2 <= image.shape[1]
        assert 0 <= y1 < y2 <= image.shape[0]
        assert 0.0 <= detection.confidence <= 1.0
        assert "mock" in detection.label


def test_yolo_is_lazy_and_does_not_import_or_download_on_init(tmp_path) -> None:
    detector = YoloVehicleDetector(tmp_path / "missing.pt")

    assert detector._model is None


class FakeBox:
    def __init__(self, class_id: int, confidence: float, bbox: list[float]):
        self.cls = np.array([class_id])
        self.conf = np.array([confidence])
        self.xyxy = np.array([bbox])


class FakeBoxesResult:
    names = {0: "person", 2: "car"}
    boxes = [
        FakeBox(2, 0.93, [10.2, 20.4, 80.1, 90.8]),
        FakeBox(0, 0.99, [1, 2, 30, 40]),
    ]


class FakeYoloModel:
    def predict(self, **kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["verbose"] is False
        return [FakeBoxesResult()]


def test_yolo_adapter_parses_vehicle_boxes_without_real_model() -> None:
    detector = YoloVehicleDetector("not-used-in-this-test.pt")
    detector._model = FakeYoloModel()

    detections = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "car"
    assert detections[0].bbox == (10, 20, 80, 91)
    assert detections[0].confidence == 0.93
