import numpy as np

from dasimanna.matching import assess_sighting, rank_assessments
from dasimanna.models import ImageQuality, ImageSignature, PetTraits


def _signature(bit: int = 1) -> ImageSignature:
    return ImageSignature(
        color_histogram=np.array([1.0, 0.0], dtype=np.float32),
        edge_histogram=np.array([0.8, 0.2], dtype=np.float32),
        perceptual_bits=np.full(16, bit, dtype=np.uint8),
        dominant_colors=("갈색", "흰색"),
    )


REFERENCE_TRAITS = PetTraits(
    "강아지", ("갈색", "흰색"), "이마 중앙에 흰 줄", "한쪽 귀가 접힘", "보통", "위로 말린 꼬리"
)


def test_low_quality_or_occluded_sighting_is_held_even_when_traits_match() -> None:
    assessment = assess_sighting(
        "흐린 제보",
        _signature(),
        _signature(),
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        ImageQuality(sharpness=2.0, brightness=18.0, contrast=4.0, reliability=0.08),
        "일부만 보임",
    )

    assert assessment.decision == "판단 보류"
    assert any("가까운 사진" in request for request in assessment.requests)
    assert any("다른 각도" in request for request in assessment.requests)


def test_unknown_observed_traits_are_not_counted_as_matches() -> None:
    unknown = PetTraits("확인 불가", (), "확인 불가", "확인 불가", "확인 불가", "확인 불가")

    assessment = assess_sighting(
        "정보 부족",
        _signature(),
        _signature(),
        REFERENCE_TRAITS,
        unknown,
        ImageQuality(sharpness=100.0, brightness=128.0, contrast=45.0, reliability=0.9),
        "대부분 보임",
    )

    trait_evidence = assessment.evidence[3:9]
    assert all(item.status == "확인 불가" for item in trait_evidence)
    assert any("특징을 추가로 확인" in request for request in assessment.requests)


def test_ranking_keeps_review_decisions_and_does_not_claim_identity() -> None:
    good = assess_sighting(
        "우선 제보",
        _signature(),
        _signature(),
        REFERENCE_TRAITS,
        REFERENCE_TRAITS,
        ImageQuality(100.0, 128.0, 45.0, 0.9),
        "전체가 잘 보임",
    )
    different = PetTraits("고양이", ("회색",), "무늬 없음", "양쪽 귀가 섬", "마름", "아래로 긴 꼬리")
    low = assess_sighting(
        "낮은 제보",
        _signature(),
        _signature(0),
        REFERENCE_TRAITS,
        different,
        ImageQuality(100.0, 128.0, 45.0, 0.9),
        "전체가 잘 보임",
    )

    ranked = rank_assessments([low, good])

    assert ranked[0].report_id == "우선 제보"
    assert ranked[0].decision == "우선 확인"
    assert all(item.decision != "동일 동물 확정" for item in ranked)


def test_custom_trait_text_ignores_extra_spaces_and_case() -> None:
    reference = PetTraits("Rabbit", ("연한  황토색",), "코 옆 점", "왼쪽 귀 끝이 접힘", "작고 둥근 편", "짧음")
    observed = PetTraits(" rabbit ", (" 연한 황토색 ",), "코 옆  점", "왼쪽 귀 끝이 접힘", "작고 둥근 편", "짧음")

    assessment = assess_sighting(
        "직접 입력 제보",
        _signature(),
        _signature(),
        reference,
        observed,
        ImageQuality(100.0, 128.0, 45.0, 0.9),
        "전체가 잘 보임",
    )

    trait_evidence = assessment.evidence[3:9]
    assert all(item.status == "일치" for item in trait_evidence)
