"""원본 정보를 확인하기 쉽게 만드는 보수적인 OpenCV 보정."""

from __future__ import annotations

import cv2
import numpy as np


def enhance_reference(image_bgr: np.ndarray) -> np.ndarray:
    """국소 대비와 가장자리를 약하게 보정한 참고 영상을 반환한다.

    이 함수는 새로운 문양이나 문자를 생성하지 않으며, 결과를 복원 영상으로
    표현해서는 안 된다.
    """

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.size == 0:
        raise ValueError("보정할 이미지가 비어 있거나 올바르지 않습니다.")

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    adjusted_lightness = clahe.apply(lightness)
    contrast_adjusted = cv2.cvtColor(
        cv2.merge((adjusted_lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR
    )

    blurred = cv2.GaussianBlur(contrast_adjusted, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(contrast_adjusted, 1.25, blurred, -0.25, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

