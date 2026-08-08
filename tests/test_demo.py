import numpy as np

from dasimanna.demo import build_demo_case, create_demo_pet_image


def test_demo_images_are_valid_and_different() -> None:
    target = create_demo_pet_image("target", 0)
    distractor = create_demo_pet_image("distractor", 0)

    assert target.shape == (420, 560, 3)
    assert target.dtype == np.uint8
    assert not np.array_equal(target, distractor)


def test_demo_case_has_expected_scenarios() -> None:
    case = build_demo_case()

    assert len(case["reference_images"]) == 2
    assert len(case["sightings"]) == 4
    assert case["reference_traits"].species == "강아지"
    assert {item["report_id"] for item in case["sightings"]} == {
        "제보 A",
        "제보 B",
        "제보 C",
        "제보 D",
    }
