"""개인정보 없이 전체 흐름을 시험하는 가상 반려동물 자료."""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np

from .models import LocationPoint, PetTraits


def create_demo_pet_image(kind: str, variant: int = 0) -> np.ndarray:
    """단순 도형으로 만든 가상 강아지 사진을 반환한다."""

    image = np.full((420, 560, 3), (236, 240, 232), dtype=np.uint8)
    cv2.rectangle(image, (0, 300), (560, 420), (178, 202, 165), -1)

    if kind == "target":
        body_color = (80, 135, 190)
        marking_color = (235, 235, 235)
        shift = variant * 7
        cv2.ellipse(image, (305 + shift, 260), (145, 85), 0, 0, 360, body_color, -1)
        cv2.circle(image, (175 + shift, 190), 78, body_color, -1)
        ear_points = np.array([[120 + shift, 142], [145 + shift, 63], [178 + shift, 140]])
        cv2.fillConvexPoly(image, ear_points, (45, 72, 112))
        cv2.ellipse(image, (200 + shift, 194), (29, 58), 12, 0, 360, marking_color, -1)
        cv2.circle(image, (148 + shift, 180), 8, (20, 20, 20), -1)
        cv2.circle(image, (217 + shift, 180), 8, (20, 20, 20), -1)
        cv2.ellipse(image, (183 + shift, 222), (16, 10), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(image, (447 + shift, 215), (68, 22), -42, 180, 420, body_color, 20)
        for x in (230, 330, 390):
            cv2.rectangle(image, (x + shift, 285), (x + 25 + shift, 365), body_color, -1)
    else:
        body_color = (115, 115, 120)
        cv2.ellipse(image, (315, 265), (150, 80), 0, 0, 360, body_color, -1)
        cv2.circle(image, (175, 195), 75, body_color, -1)
        cv2.fillConvexPoly(image, np.array([[120, 150], [135, 85], [165, 145]]), body_color)
        cv2.fillConvexPoly(image, np.array([[185, 145], [220, 85], [230, 160]]), body_color)
        cv2.circle(image, (150, 190), 8, (20, 20, 20), -1)
        cv2.circle(image, (205, 190), 8, (20, 20, 20), -1)
        cv2.line(image, (455, 260), (520, 280), body_color, 24)

    if variant == 1:
        image = cv2.convertScaleAbs(image, alpha=0.86, beta=5)
    elif variant == 2:
        image = cv2.GaussianBlur(image, (13, 13), 3.0)
    return image


def build_demo_case() -> dict[str, object]:
    """기준 사진, 마지막 위치, 네 건의 가상 목격 제보를 만든다."""

    reference_traits = PetTraits(
        species="강아지",
        fur_colors=("갈색", "흰색"),
        face_marking="이마 중앙에 흰 줄",
        ear_shape="한쪽 귀가 접힘",
        body_shape="보통",
        tail_shape="위로 말린 꼬리",
    )
    origin = LocationPoint("마지막 확인", datetime(2026, 8, 7, 15, 0), 37.56650, 126.97800)
    sightings = [
        {
            "report_id": "제보 A",
            "place": "공원 동쪽 입구",
            "point": LocationPoint("제보 A", datetime(2026, 8, 7, 15, 12), 37.56720, 126.97900),
            "image": create_demo_pet_image("target", 1),
            "visibility": "대부분 보임",
            "traits": reference_traits,
            "note": "갈색 강아지가 화단 옆으로 이동함",
        },
        {
            "report_id": "제보 B",
            "place": "도서관 골목",
            "point": LocationPoint("제보 B", datetime(2026, 8, 7, 15, 28), 37.56810, 126.98030),
            "image": create_demo_pet_image("target", 2),
            "visibility": "일부만 보임",
            "traits": PetTraits(
                "강아지", ("갈색", "흰색"), "확인 불가", "한쪽 귀가 접힘", "보통", "위로 말린 꼬리"
            ),
            "note": "멀리서 촬영되어 얼굴 무늬는 확인하기 어려움",
        },
        {
            "report_id": "제보 C",
            "place": "외곽 체육공원",
            "point": LocationPoint("제보 C", datetime(2026, 8, 7, 15, 20), 37.64000, 127.09000),
            "image": create_demo_pet_image("target", 0),
            "visibility": "전체가 잘 보임",
            "traits": reference_traits,
            "note": "모습은 비슷하지만 짧은 시간에 이동하기 어려운 거리",
        },
        {
            "report_id": "제보 D",
            "place": "시장 주차장",
            "point": LocationPoint("제보 D", datetime(2026, 8, 7, 15, 35), 37.56870, 126.98110),
            "image": create_demo_pet_image("distractor", 0),
            "visibility": "전체가 잘 보임",
            "traits": PetTraits(
                "강아지", ("회색",), "무늬 없음", "양쪽 귀가 섬", "마름", "아래로 긴 꼬리"
            ),
            "note": "비슷한 크기지만 색과 귀·꼬리 특징이 다름",
        },
    ]
    return {
        "pet_name": "별이",
        "reference_images": [create_demo_pet_image("target", 0), create_demo_pet_image("target", 1)],
        "reference_traits": reference_traits,
        "origin": origin,
        "origin_place": "시청 앞 잔디광장",
        "sightings": sightings,
    }
