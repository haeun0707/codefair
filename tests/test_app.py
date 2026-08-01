from streamlit.testing.v1 import AppTest


def test_default_mock_demo_completes_full_ui_flow() -> None:
    app = AppTest.from_file("app.py", default_timeout=20).run()

    assert not app.exception
    assert len(app.button) == 1

    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.metric) == 12
    assert any("가장 선명한 근거 영상" in message.value for message in app.success)
    assert any("복원하지 않습니다" in message.value for message in app.warning)
