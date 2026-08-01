"""MoaView에서 공통으로 사용하는 값 객체."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """한 차량 후보의 탐지 결과.

    bbox는 OpenCV 좌표계의 ``(x1, y1, x2, y2)``이며 오른쪽/아래 끝은
    crop에 포함하지 않는다.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    class_id: int | None = None


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """한 crop에서 계산한 설명 가능한 화질 지표."""

    sharpness: float
    brightness: float
    contrast: float


@dataclass(slots=True)
class CandidateAnalysis:
    """탐지 후보와 사람이 검토할 파생 자료."""

    detection: Detection
    crop_bgr: np.ndarray
    enhanced_bgr: np.ndarray
    metrics: QualityMetrics

