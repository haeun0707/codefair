"""교체 가능한 mock 및 Ultralytics YOLO 차량 detector."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .models import Detection


VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})  # COCO: car, motorcycle, bus, truck


class Detector(Protocol):
    """UI와 모델 구현을 분리하는 최소 detector 인터페이스."""

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """이미지에서 차량 후보를 반환한다."""


class DetectorUnavailableError(RuntimeError):
    """모델이나 실행 환경을 준비하지 못했을 때 발생한다."""


class MockVehicleDetector:
    """네트워크나 AI 모델 없이 UI 흐름을 확인하는 가짜 detector."""

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
            raise ValueError("Mock 탐지에 사용할 이미지가 올바르지 않습니다.")

        height, width = image_bgr.shape[:2]
        if width < 2 or height < 2:
            return []

        first = Detection(
            bbox=(
                max(0, round(width * 0.07)),
                max(0, round(height * 0.30)),
                max(1, round(width * 0.58)),
                max(1, round(height * 0.84)),
            ),
            confidence=0.91,
            label="car (mock)",
            class_id=2,
        )
        if width < 160:
            return [first]

        second = Detection(
            bbox=(
                round(width * 0.55),
                round(height * 0.38),
                max(round(width * 0.96), 1),
                max(round(height * 0.80), 1),
            ),
            confidence=0.84,
            label="truck (mock)",
            class_id=7,
        )
        return [first, second]


class YoloVehicleDetector:
    """로컬 가중치를 필요할 때만 여는 CPU용 YOLO detector."""

    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.25):
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise DetectorUnavailableError(
                f"YOLO 모델 파일을 찾지 못했습니다: {self.model_path}. "
                "경량 모델 파일을 models 폴더에 넣거나 Mock 모드로 전환해 주세요."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DetectorUnavailableError(
                "Ultralytics를 불러올 수 없습니다. "
                "pip install -r requirements.txt를 실행한 뒤 다시 시도해 주세요."
            ) from exc

        try:
            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            raise DetectorUnavailableError(
                "YOLO 모델을 열지 못했습니다. 모델 파일과 Ultralytics 버전을 확인해 주세요."
            ) from exc
        return self._model

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
            raise ValueError("차량 탐지에 사용할 이미지가 올바르지 않습니다.")

        model = self._load_model()
        try:
            results = model.predict(
                source=image_bgr,
                conf=self.confidence_threshold,
                classes=sorted(VEHICLE_CLASS_IDS),
                device="cpu",
                verbose=False,
            )
        except Exception as exc:
            raise DetectorUnavailableError(
                "YOLO 차량 탐지에 실패했습니다. 이미지 형식과 모델 파일을 확인하거나 "
                "Mock 모드로 흐름을 먼저 확인해 주세요."
            ) from exc

        detections: list[Detection] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
                confidence = float(box.conf[0].item())
                if isinstance(names, dict):
                    label = str(names.get(class_id, f"class {class_id}"))
                else:
                    label = str(names[class_id])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        label=label,
                        class_id=class_id,
                    )
                )
        return detections

