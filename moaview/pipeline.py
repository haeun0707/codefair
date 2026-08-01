"""탐지부터 crop·화질·보정까지 연결하는 UI 독립 파이프라인."""

from __future__ import annotations

import numpy as np

from .detection import Detector
from .enhancement import enhance_reference
from .imaging import InvalidImageError, crop_detection
from .models import CandidateAnalysis
from .quality import calculate_quality


def analyze_image(image_bgr: np.ndarray, detector: Detector) -> list[CandidateAnalysis]:
    """한 이미지의 모든 유효 차량 후보를 분석한다."""

    analyses: list[CandidateAnalysis] = []
    for detection in detector.detect(image_bgr):
        try:
            crop = crop_detection(image_bgr, detection)
        except InvalidImageError:
            # 모델이 낸 잘못된 단일 bbox만 제외하고 다른 후보는 계속 검토한다.
            continue
        analyses.append(
            CandidateAnalysis(
                detection=detection,
                crop_bgr=crop,
                enhanced_bgr=enhance_reference(crop),
                metrics=calculate_quality(crop),
            )
        )
    return analyses

