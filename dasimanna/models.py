"""다시만나 AI에서 공통으로 사용하는 데이터 구조."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class PetTraits:
    """보호자 또는 제보자가 사람이 보고 기록한 반려동물 특징."""

    species: str
    fur_colors: tuple[str, ...]
    face_marking: str
    ear_shape: str
    body_shape: str
    tail_shape: str


@dataclass(frozen=True, slots=True)
class ImageQuality:
    """원본 사진에서 측정한 참고용 화질 지표."""

    sharpness: float
    brightness: float
    contrast: float
    reliability: float


@dataclass(slots=True)
class ImageSignature:
    """사진에서 추출한 설명 가능한 시각 단서."""

    color_histogram: np.ndarray
    edge_histogram: np.ndarray
    perceptual_bits: np.ndarray
    dominant_colors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """한 가지 비교 단서의 일치 여부와 설명."""

    category: str
    status: str
    detail: str
    score: float | None


@dataclass(frozen=True, slots=True)
class MovementCheck:
    """두 시공간 지점 사이의 이동 가능성 참고 결과."""

    distance_km: float
    elapsed_minutes: float
    required_speed_kmh: float | None
    feasible: bool | None
    detail: str


@dataclass(frozen=True, slots=True)
class MatchAssessment:
    """한 목격 제보의 검토 우선순위와 근거."""

    report_id: str
    priority: int
    decision: str
    image_similarity: float
    clue_similarity: float
    reliability: float
    evidence: tuple[EvidenceItem, ...]
    requests: tuple[str, ...]
    movement: MovementCheck | None


@dataclass(frozen=True, slots=True)
class LocationPoint:
    """경로 계산에 사용하는 시간·위치 한 점."""

    point_id: str
    observed_at: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class SearchPrediction:
    """확인된 목격을 바탕으로 만든 보수적인 수색 영역 참고값."""

    latitude: float
    longitude: float
    radius_km: float
    basis: str
    caution: str
