import cv2
import numpy as np
from streamlit.testing.v1 import AppTest


def test_demo_completes_review_flow_without_auto_confirmation() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.toggle[0].set_value(True).run()

    assert not app.exception
    assert any("다시만나 AI" in title.value for title in app.title)
    assert len(app.button) == 1

    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.checkbox) == 4
    assert all(not checkbox.value for checkbox in app.checkbox)
    assert len(app.metric) >= 20
    assert any("검토 우선도는 동일 동물일 확률이 아닙니다" in item.value for item in app.markdown)
    assert any(metric.label == "단서 유사도" and metric.value.endswith("%") for metric in app.metric)
    assert any("단서 유사도는 사진과 관찰 특징이 닮은 정도" in item.value for item in app.caption)
    assert len(app.get("vega_lite_chart")) == 1
    assert any("원본을 확인한 제보" in message.value for message in app.info)


def test_manual_jpg_png_flow_runs_without_location_coordinates() -> None:
    image = np.full((100, 140, 3), (90, 135, 185), dtype=np.uint8)
    cv2.circle(image, (70, 50), 28, (235, 235, 235), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    upload = ("pet.png", encoded.tobytes(), "image/png")

    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert app.toggle[0].value is False
    next(item for item in app.number_input if item.label == "제보 수").set_value(3).run()
    assert len(app.file_uploader) == 4
    app.file_uploader[0].set_value(upload)
    for uploader in app.file_uploader[1:]:
        uploader.set_value(upload)
    app.run(timeout=30)

    assert not app.exception
    assert not app.button[0].disabled
    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.checkbox) == 3
    assert all(not checkbox.value for checkbox in app.checkbox)
    assert any("마지막 확인에서 약" in item.value for item in app.markdown)

    app.checkbox[0].set_value(True).run(timeout=30)
    assert not app.exception
    assert any("위도·경도를 쓰지 않아도" in message.value for message in app.info)
    assert any("시간순 장소 경로" in item.value for item in app.markdown)


def test_latest_human_confirmed_report_is_used_even_with_movement_warning() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.toggle[0].set_value(True).run()
    app.button[0].click().run(timeout=30)

    next(item for item in app.checkbox if item.key and item.key.endswith("A")).set_value(True).run()
    next(item for item in app.checkbox if item.key and item.key.endswith("C")).set_value(True).run()

    assert not app.exception
    assert any(
        "시청 앞 잔디광장 → 공원 동쪽 입구 → 외곽 체육공원" in item.value
        for item in app.markdown
    )
    assert any(
        metric.label == "다음 수색 기준 장소" and metric.value == "외곽 체육공원"
        for metric in app.metric
    )
    assert any("보호자가 원본을 확인했으므로" in item.value for item in app.warning)


def test_registration_and_report_traits_allow_custom_text() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert app.toggle[0].value is False
    direct_labels = {item.label for item in app.text_input}
    assert {
        "동물 종류 직접 입력",
        "털 색 직접 입력",
        "얼굴 무늬 직접 입력",
        "귀 모양 직접 입력",
        "체형 직접 입력",
        "꼬리 모양 직접 입력",
    } <= direct_labels

    app.selectbox[0].set_value("직접 입력").run()
    custom_species_inputs = [item for item in app.text_input if item.label == "동물 종류 직접 입력"]
    assert custom_species_inputs
    custom_species_inputs[0].set_value("토끼").run()

    assert not app.exception
    assert any(item.value == "토끼" for item in app.text_input if item.label == "동물 종류 직접 입력")


def test_report_custom_text_survives_temporary_report_removal() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(item for item in app.number_input if item.label == "제보 수").set_value(1).run()

    report_species_inputs = [item for item in app.text_input if item.label == "동물 종류 직접 입력"]
    assert len(report_species_inputs) == 2
    report_species_inputs[1].set_value("소형 토끼").run()

    next(item for item in app.number_input if item.label == "제보 수").set_value(0).run()
    next(item for item in app.number_input if item.label == "제보 수").set_value(1).run()

    report_species_inputs = [item for item in app.text_input if item.label == "동물 종류 직접 입력"]
    assert len(report_species_inputs) == 2
    assert report_species_inputs[1].value == "소형 토끼"


def test_zero_sightings_can_save_reference_information_without_error() -> None:
    image = np.full((100, 140, 3), (90, 135, 185), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    report_count = next(item for item in app.number_input if item.label == "제보 수")
    assert report_count.value == 0
    assert len(app.file_uploader) == 1

    app.file_uploader[0].set_value(("pet.png", encoded.tobytes(), "image/png"))
    app.run(timeout=30)
    assert not app.button[0].disabled
    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert any("아직 등록된 목격 제보가 없습니다" in message.value for message in app.info)
