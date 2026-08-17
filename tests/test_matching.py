from dasimanna.demo import create_demo_pet_image
from dasimanna.imaging import extract_image_signature, measure_image_quality
from dasimanna.matching import assess_sighting, rank_assessments
from dasimanna.models import ImageQuality, MovementCheck, PetTraits


REFERENCE_TRAITS = PetTraits(
    "강아지", ("갈색", "흰색"), "이마 중앙에 흰 줄", "한쪽 귀가 접힘", "보통", "위로 말린 꼬리"
)


def test_matching_traits_create_priority_and_reasons() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)
    quality = measure_image_quality(image)
    movement = MovementCheck(0.3, 10, 1.8, True, "가능 범위")

    assessment = assess_sighting(
        "A",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        quality,
        "전체가 잘 보임",
        movement,
    )

    assert assessment.decision == "우선 확인"
    assert assessment.priority >= 70
    assert any(item.category == "꼬리" and item.status == "일치" for item in assessment.evidence)


def test_low_quality_forces_hold_even_when_traits_match() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)
    low_quality = ImageQuality(1.0, 10.0, 2.0, 0.12)

    assessment = assess_sighting(
        "B",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        low_quality,
        "일부만 보임",
    )

    assert assessment.decision == "판단 보류"
    assert assessment.requests
    assert all(
        item.status == "확인 불가" and item.score is None
        for item in assessment.evidence
        if item.category in {"사진 색 분포", "윤곽·질감", "저해상도 시각 패턴"}
    )


def test_multiple_trait_mismatches_lower_priority() -> None:
    target = create_demo_pet_image("target")
    distractor = create_demo_pet_image("distractor")
    mismatched_traits = PetTraits(
        "강아지", ("회색",), "무늬 없음", "양쪽 귀가 섬", "마름", "아래로 긴 꼬리"
    )

    assessment = assess_sighting(
        "D",
        extract_image_signature(target),
        extract_image_signature(distractor),
        REFERENCE_TRAITS,
        mismatched_traits,
        measure_image_quality(distractor),
        "전체가 잘 보임",
    )

    assert assessment.decision == "우선순위 낮음"
    assert sum(item.status == "불일치" for item in assessment.evidence) >= 2


def test_clue_similarity_combines_photo_and_observed_traits() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)
    quality = measure_image_quality(image)
    mismatched_traits = PetTraits(
        "강아지", ("회색",), "무늬 없음", "양쪽 귀가 섬", "마름", "아래로 긴 꼬리"
    )

    matched = assess_sighting(
        "matched",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        quality,
        "전체가 잘 보임",
    )
    mismatched = assess_sighting(
        "mismatched",
        signature,
        signature,
        REFERENCE_TRAITS,
        mismatched_traits,
        quality,
        "전체가 잘 보임",
    )

    assert matched.image_similarity == mismatched.image_similarity
    assert matched.clue_similarity > mismatched.clue_similarity
    assert 0.0 <= mismatched.clue_similarity <= 1.0


def test_impossible_movement_is_not_treated_as_match_confirmation() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)
    movement = MovementCheck(20.0, 10.0, 120.0, False, "속도 범위 초과")

    assessment = assess_sighting(
        "C",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        measure_image_quality(image),
        "전체가 잘 보임",
        movement,
    )

    assert assessment.decision == "이동 경로 재확인"
    assert any(item.category == "시간·위치" and item.status == "재확인" for item in assessment.evidence)


def test_ranking_keeps_hold_below_actionable_reports() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)
    good = assess_sighting(
        "good",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        measure_image_quality(image),
        "전체가 잘 보임",
    )
    held = assess_sighting(
        "held",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        ImageQuality(1, 20, 2, 0.1),
        "거의 보이지 않음",
    )

    assert [item.report_id for item in rank_assessments([held, good])] == ["good", "held"]


def test_partially_hidden_pet_is_held_even_with_clear_matching_photo() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)

    assessment = assess_sighting(
        "가려진 제보",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        measure_image_quality(image),
        "일부만 보임",
    )

    assert assessment.decision == "판단 보류"
    assert any("다른 각도" in request for request in assessment.requests)


def test_blurry_pet_is_held_and_requests_closer_photo() -> None:
    image = create_demo_pet_image("target")
    signature = extract_image_signature(image)
    blurry_quality = ImageQuality(20.0, 128.0, 45.0, 0.75)

    assessment = assess_sighting(
        "흐린 제보",
        signature,
        signature,
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        blurry_quality,
        "전체가 잘 보임",
    )

    assert assessment.decision == "판단 보류"
    assert any("더 가까운 사진" in request for request in assessment.requests)
    assert all(
        item.status == "확인 불가" and item.score is None
        for item in assessment.evidence
        if item.category in {"사진 색 분포", "윤곽·질감", "저해상도 시각 패턴"}
    )
