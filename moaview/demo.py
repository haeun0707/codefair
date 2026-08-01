"""개인정보 없는 mock 시연용 합성 이미지."""

from __future__ import annotations

import cv2
import numpy as np


def create_demo_image(index: int, width: int = 720, height: int = 420) -> np.ndarray:
    """서로 다른 흐림·밝기를 가진 가상 도로 장면을 만든다."""

    if index not in (0, 1, 2):
        raise ValueError("데모 이미지 번호는 0, 1, 2 중 하나여야 합니다.")

    sky = np.full((height, width, 3), (205, 180, 145), dtype=np.uint8)
    cv2.rectangle(sky, (0, round(height * 0.52)), (width, height), (72, 72, 78), -1)
    cv2.line(sky, (0, round(height * 0.78)), (width, round(height * 0.78)), (220, 220, 220), 5)

    shift = index * 35
    # 가상 차량 1
    cv2.rectangle(sky, (75 + shift, 170), (400 + shift, 330), (175, 70, 45), -1)
    cv2.rectangle(sky, (150 + shift, 120), (335 + shift, 200), (120, 155, 175), -1)
    cv2.circle(sky, (145 + shift, 330), 34, (28, 28, 30), -1)
    cv2.circle(sky, (335 + shift, 330), 34, (28, 28, 30), -1)
    cv2.rectangle(sky, (235 + shift, 270), (325 + shift, 306), (245, 245, 245), -1)
    cv2.putText(sky, "MOA-26", (241 + shift, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

    # 후보 선택을 보여 주기 위한 작은 가상 차량 2
    cv2.rectangle(sky, (500, 220), (680, 315), (65, 125, 80), -1)
    cv2.circle(sky, (540, 315), 24, (25, 25, 28), -1)
    cv2.circle(sky, (640, 315), 24, (25, 25, 28), -1)

    if index == 0:
        sky = cv2.GaussianBlur(sky, (15, 15), sigmaX=5)
        sky = cv2.convertScaleAbs(sky, alpha=0.68, beta=0)
    elif index == 1:
        sky = cv2.GaussianBlur(sky, (7, 7), sigmaX=2)
        sky = cv2.convertScaleAbs(sky, alpha=0.88, beta=5)
    return sky

