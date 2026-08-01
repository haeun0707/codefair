"""이미지 디코딩, crop, 표시용 주석 함수."""

from __future__ import annotations

import cv2
import numpy as np

from .models import Detection


class InvalidImageError(ValueError):
    """업로드 파일을 안전한 이미지로 읽을 수 없을 때 발생한다."""


def decode_image(data: bytes) -> np.ndarray:
    """JPG/PNG 바이트를 BGR 이미지로 디코딩한다."""

    if not data:
        raise InvalidImageError("파일이 비어 있습니다. 다른 JPG 또는 PNG 파일을 선택해 주세요.")

    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(
            "이미지를 읽을 수 없습니다. 손상되지 않은 JPG 또는 PNG 파일인지 확인해 주세요."
        )
    return image


def crop_detection(image_bgr: np.ndarray, detection: Detection) -> np.ndarray:
    """bbox를 이미지 경계 안으로 제한해 안전하게 crop한다."""

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
        raise InvalidImageError("차량 영역을 자를 원본 이미지가 올바르지 않습니다.")

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    x1 = max(0, min(int(x1), width))
    x2 = max(0, min(int(x2), width))
    y1 = max(0, min(int(y1), height))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        raise InvalidImageError("탐지된 차량 영역의 크기가 올바르지 않습니다.")
    return image_bgr[y1:y2, x1:x2].copy()


def annotate_detections(
    image_bgr: np.ndarray, detections: list[Detection] | tuple[Detection, ...]
) -> np.ndarray:
    """후보 번호와 탐지 상자를 그린 표시용 사본을 만든다."""

    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    thickness = max(1, round(min(height, width) / 300))
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.bbox
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
        color = (52, 211, 153) if index % 2 else (251, 191, 36)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"{index}: {detection.label} {detection.confidence:.0%}"
        text_y = max(18, y1 - 7)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(1, thickness),
            cv2.LINE_AA,
        )
    return annotated

