"""원본 crop에서 직접 계산하는 화질 지표."""

from __future__ import annotations

import cv2
import numpy as np

from .models import QualityMetrics


def calculate_quality(image_bgr: np.ndarray) -> QualityMetrics:
    """선명도, 평균 밝기, 대비를 계산한다.

    선명도는 grayscale Laplacian 분산, 밝기는 grayscale 평균, 대비는
    grayscale 표준편차다. 세 값 모두 확률이 아니며 서로 다른 크기의 이미지
    사이에서 절대적인 품질 보증으로 사용하지 않는다.
    """

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
        raise ValueError("화질을 계산할 이미지가 비어 있거나 올바르지 않습니다.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return QualityMetrics(
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
    )

