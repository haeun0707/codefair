"""반려동물 기준 자료와 목격 제보의 설명 가능한 비교 로직."""

from __future__ import annotations

from .imaging import compare_signatures
from .models import (
    EvidenceItem,
    ImageQuality,
    ImageSignature,
    MatchAssessment,
    MovementCheck,
    PetTraits,
)


UNKNOWN_VALUES = {"", "확인 불가", "모름", "미확인", "선택 안 함"}
VISIBILITY_FACTORS = {
    "전체가 잘 보임": 1.00,
    "대부분 보임": 0.82,
    "일부만 보임": 0.52,
    "거의 보이지 않음": 0.22,
}
TRAIT_LABELS = {
    "species": "동물 종류",
    "fur_colors": "털 색",
    "face_marking": "얼굴 무늬",
    "ear_shape": "귀 모양",
    "body_shape": "체형",
    "tail_shape": "꼬리",
}
TRAIT_WEIGHTS = {
    "species": 0.12,
    "fur_colors": 0.24,
    "face_marking": 0.22,
    "ear_shape": 0.14,
    "body_shape": 0.12,
    "tail_shape": 0.16,
}
MIN_SHARPNESS_FOR_REVIEW = 60.0
MIN_RELIABILITY_FOR_REVIEW = 0.40
LIMITED_VISIBILITY = {"일부만 보임", "거의 보이지 않음"}


def assess_sighting(
    report_id: str,
    reference_signature: ImageSignature,
    candidate_signature: ImageSignature,
    reference_traits: PetTraits,
    observed_traits: PetTraits,
    image_quality: ImageQuality,
    visibility: str,
    movement: MovementCheck | None = None,
) -> MatchAssessment:
    """한 제보의 검토 우선순위와 일치·불일치·미확인 근거를 만든다."""

    if visibility not in VISIBILITY_FACTORS:
        raise ValueError("지원하지 않는 가시성 값입니다.")

    similarities = compare_signatures(reference_signature, candidate_signature)
    reliability = image_quality.reliability * VISIBILITY_FACTORS[visibility]
    should_hold = (
        reliability < MIN_RELIABILITY_FOR_REVIEW
        or image_quality.sharpness < MIN_SHARPNESS_FOR_REVIEW
        or visibility in LIMITED_VISIBILITY
    )
    if should_hold:
        hold_detail = (
            "사진이 흐리거나 동물이 충분히 보이지 않아 이 사진 단서는 비교하지 않았습니다. "
            "더 가까운 사진이나 다른 각도의 사진이 필요합니다."
        )
        evidence: list[EvidenceItem] = [
            EvidenceItem("사진 색 분포", "확인 불가", hold_detail, None),
            EvidenceItem("윤곽·질감", "확인 불가", hold_detail, None),
            EvidenceItem("저해상도 시각 패턴", "확인 불가", hold_detail, None),
        ]
    else:
        evidence = [
            _image_evidence("사진 색 분포", similarities["color"], 0.66),
            _image_evidence("윤곽·질감", similarities["texture"], 0.58),
            _image_evidence("저해상도 시각 패턴", similarities["visual"], 0.62),
        ]

    trait_scores: list[tuple[float, float]] = []
    mismatch_count = 0
    for field_name in TRAIT_LABELS:
        item = _compare_trait(field_name, reference_traits, observed_traits)
        evidence.append(item)
        if item.score is not None:
            weight = TRAIT_WEIGHTS[field_name]
            trait_scores.append((item.score, weight))
            if item.status == "불일치":
                mismatch_count += 1

    if movement is not None:
        if movement.feasible is True:
            evidence.append(EvidenceItem("시간·위치", "가능 범위", movement.detail, 1.0))
        elif movement.feasible is False:
            evidence.append(EvidenceItem("시간·위치", "재확인", movement.detail, 0.0))
        else:
            evidence.append(EvidenceItem("시간·위치", "확인 불가", movement.detail, None))

    if trait_scores:
        weighted_total = sum(score * weight for score, weight in trait_scores)
        weight_total = sum(weight for _, weight in trait_scores)
        trait_score = weighted_total / weight_total
        combined = 0.48 * similarities["overall"] + 0.52 * trait_score
    else:
        combined = similarities["overall"]

    clue_similarity = float(max(0.0, min(1.0, combined)))
    priority = round(clue_similarity * 100 * (0.78 + 0.22 * reliability))
    if movement is not None and movement.feasible is True:
        priority = min(100, priority + 4)
    elif movement is not None and movement.feasible is False:
        # 직선거리와 사용자 가정 속도만 사용하므로 강한 배제 근거로 쓰지 않는다.
        priority = max(0, priority - 10)

    requests: list[str] = []
    if image_quality.sharpness < MIN_SHARPNESS_FOR_REVIEW:
        requests.append("흔들림이 적고 더 가까운 사진을 요청하세요.")
    if image_quality.brightness < 35 or image_quality.brightness > 225:
        requests.append("밝기가 다른 장소나 각도에서 다시 촬영해 달라고 요청하세요.")
    if visibility in LIMITED_VISIBILITY:
        requests.append("얼굴·귀·꼬리가 보이는 다른 각도의 사진을 요청하세요.")
    known_trait_count = len(trait_scores)
    if known_trait_count < 3:
        requests.append("털 색 외에 얼굴 무늬·귀·꼬리 특징을 추가로 확인하세요.")

    if should_hold:
        decision = "판단 보류"
    elif movement is not None and movement.feasible is False:
        decision = "이동 경로 재확인"
    elif mismatch_count >= 2 or priority < 42:
        decision = "우선순위 낮음"
    elif priority >= 70 and mismatch_count == 0:
        decision = "우선 확인"
    else:
        decision = "추가 확인"

    return MatchAssessment(
        report_id=report_id,
        priority=int(max(0, min(100, priority))),
        decision=decision,
        image_similarity=similarities["overall"],
        clue_similarity=clue_similarity,
        reliability=float(max(0.0, min(1.0, reliability))),
        evidence=tuple(evidence),
        requests=tuple(dict.fromkeys(requests)),
        movement=movement,
    )


def rank_assessments(assessments: list[MatchAssessment]) -> list[MatchAssessment]:
    """검토 우선도 순으로 정렬하되 판단 보류를 자동 확정처럼 올리지 않는다."""

    decision_order = {
        "우선 확인": 0,
        "추가 확인": 1,
        "이동 경로 재확인": 2,
        "판단 보류": 3,
        "우선순위 낮음": 4,
    }
    return sorted(
        assessments,
        key=lambda item: (decision_order.get(item.decision, 9), -item.priority, item.report_id),
    )


def _image_evidence(category: str, score: float, match_threshold: float) -> EvidenceItem:
    if score >= match_threshold:
        status = "일치 단서"
        detail = f"기준 사진과 ‘{category}’ 단서가 비교적 비슷합니다."
    elif score < match_threshold - 0.18:
        status = "불일치 단서"
        detail = f"기준 사진과 ‘{category}’ 단서의 차이가 큽니다. 각도·조명의 영향도 확인하세요."
    else:
        status = "확인 필요"
        detail = f"‘{category}’ 단서만으로는 분명하게 구분하기 어렵습니다."
    return EvidenceItem(category, status, detail, score)


def _compare_trait(
    field_name: str, reference_traits: PetTraits, observed_traits: PetTraits
) -> EvidenceItem:
    label = TRAIT_LABELS[field_name]
    reference = getattr(reference_traits, field_name)
    observed = getattr(observed_traits, field_name)

    if field_name == "fur_colors":
        reference_set = {
            _normalize_trait_text(item) for item in reference if _normalize_trait_text(item) not in UNKNOWN_VALUES
        }
        observed_set = {
            _normalize_trait_text(item) for item in observed if _normalize_trait_text(item) not in UNKNOWN_VALUES
        }
        if not reference_set or not observed_set:
            return EvidenceItem(label, "확인 불가", "기준 또는 제보의 털 색 정보가 부족합니다.", None)
        overlap = len(reference_set & observed_set) / len(reference_set | observed_set)
        if overlap >= 0.5:
            return EvidenceItem(
                label,
                "일치",
                f"공통 색: {', '.join(sorted(reference_set & observed_set))}",
                overlap,
            )
        return EvidenceItem(
            label,
            "불일치",
            f"기준 {', '.join(sorted(reference_set))} / 제보 {', '.join(sorted(observed_set))}",
            overlap,
        )

    normalized_reference = _normalize_trait_text(str(reference))
    normalized_observed = _normalize_trait_text(str(observed))
    if normalized_reference in UNKNOWN_VALUES or normalized_observed in UNKNOWN_VALUES:
        return EvidenceItem(label, "확인 불가", f"{label} 정보가 충분하지 않습니다.", None)
    if normalized_reference == normalized_observed:
        return EvidenceItem(label, "일치", f"둘 다 ‘{reference}’로 기록되었습니다.", 1.0)
    return EvidenceItem(label, "불일치", f"기준 ‘{reference}’ / 제보 ‘{observed}’", 0.0)


def _normalize_trait_text(value: str) -> str:
    """직접 입력값의 앞뒤·연속 공백과 영문 대소문자를 정리한다."""

    return " ".join(str(value).strip().casefold().split())
