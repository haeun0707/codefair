"""MoaView Streamlit 이미지 MVP."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

from moaview.demo import create_demo_image
from moaview.detection import (
    DetectorUnavailableError,
    MockVehicleDetector,
    YoloVehicleDetector,
)
from moaview.imaging import InvalidImageError, annotate_detections, decode_image
from moaview.pipeline import analyze_image


st.set_page_config(page_title="MoaView", page_icon="🚙", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.75rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_yolo_detector(model_path: str, confidence: float) -> YoloVehicleDetector:
    """같은 설정의 detector를 재사용하되 실제 모델은 첫 탐지 때 연다."""

    return YoloVehicleDetector(model_path, confidence)


def _upload_signature(files: list[Any], use_demo: bool) -> str:
    digest = hashlib.sha256(b"demo" if use_demo else b"uploads")
    for file in files:
        digest.update(file.name.encode("utf-8", errors="replace"))
        digest.update(file.getvalue())
    return digest.hexdigest()


def _source_labels(files: list[Any], use_demo: bool) -> list[str]:
    if use_demo:
        return ["가상 CCTV 1", "가상 CCTV 2", "가상 CCTV 3"]
    return [file.name for file in files]


def _metadata_form(labels: list[str]) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    st.subheader("1. 자료 정보")
    for index, label in enumerate(labels):
        with st.expander(f"CCTV {index + 1} · {label}", expanded=index == 0):
            col_a, col_b, col_c = st.columns(3)
            camera = col_a.text_input(
                "카메라 이름",
                value=f"CCTV {index + 1}",
                key=f"camera_{index}_{label}",
            )
            default_time = datetime(2026, 8, 1, 14, 20) + timedelta(minutes=index * 2)
            captured_at = col_b.text_input(
                "촬영 시각",
                value=default_time.strftime("%Y-%m-%d %H:%M"),
                key=f"time_{index}_{label}",
                help="예: 2026-08-01 14:20",
            )
            direction = col_c.selectbox(
                "이동 방향",
                ("알 수 없음", "왼쪽 → 오른쪽", "오른쪽 → 왼쪽", "접근", "멀어짐"),
                key=f"direction_{index}_{label}",
            )
            metadata.append(
                {"camera": camera.strip() or f"CCTV {index + 1}", "captured_at": captured_at, "direction": direction}
            )
    return metadata


def _run_analysis(files: list[Any], use_demo: bool, detector: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(3 if use_demo else len(files)):
        source_name = f"가상 CCTV {index + 1}" if use_demo else files[index].name
        try:
            image = create_demo_image(index) if use_demo else decode_image(files[index].getvalue())
            candidates = analyze_image(image, detector)
            results.append(
                {"source_name": source_name, "image": image, "candidates": candidates, "error": None}
            )
        except (InvalidImageError, DetectorUnavailableError, ValueError) as exc:
            results.append(
                {"source_name": source_name, "image": None, "candidates": [], "error": str(exc)}
            )
        except Exception as exc:
            results.append(
                {
                    "source_name": source_name,
                    "image": None,
                    "candidates": [],
                    "error": (
                        "예상하지 못한 처리 오류가 발생했습니다. 파일 형식을 확인하고 Mock 모드로 "
                        f"다시 시도해 주세요. ({type(exc).__name__})"
                    ),
                }
            )
    return results


st.title("MoaView · 차량 근거 영상 검토 도우미")
st.caption("차량 후보를 사람이 선택하고, 원본 화질과 보정 참고 영상을 비교하는 교육용 MVP")
st.info(
    "AI 탐지는 검토를 돕는 후보 제안입니다. 차량을 확정하거나 보이지 않는 번호판 문자를 복원하지 않습니다.",
    icon="ℹ️",
)

with st.sidebar:
    st.header("실행 설정")
    mode = st.radio("탐지 모드", ("Mock", "로컬 YOLO"), horizontal=True)
    use_demo = st.checkbox(
        "가상 데모 이미지 3장 사용",
        value=True,
        disabled=mode != "Mock",
        help="실제 이미지나 모델 없이 전체 화면 흐름을 확인합니다.",
    )
    model_path = "models/yolo11n.pt"
    confidence = 0.25
    if mode == "로컬 YOLO":
        use_demo = False
        model_path = st.text_input("로컬 모델 경로", value=model_path)
        confidence = st.slider("최소 탐지 확신도", 0.05, 0.90, 0.25, 0.05)
        st.caption("모델을 자동 다운로드하지 않습니다. `.pt` 파일을 로컬에 준비해 주세요.")
    st.divider()
    st.caption("처리는 현재 세션에서만 수행하며 업로드 이미지를 저장소에 저장하지 않습니다.")

uploaded_files = st.file_uploader(
    "JPG/PNG 이미지 1~3개를 올려 주세요",
    type=("jpg", "jpeg", "png"),
    accept_multiple_files=True,
    disabled=use_demo,
)
files = list(uploaded_files or [])
if len(files) > 3:
    st.error("이미지는 최대 3개까지 처리할 수 있습니다. 3개만 남기고 다시 선택해 주세요.")

valid_input = use_demo or 1 <= len(files) <= 3
labels = _source_labels(files, use_demo) if valid_input else []
metadata = _metadata_form(labels) if labels else []

signature = _upload_signature(files, use_demo)
if st.session_state.get("input_signature") != signature:
    st.session_state.input_signature = signature
    st.session_state.analysis_results = None

if mode == "Mock":
    detector = MockVehicleDetector()
    st.warning(
        "Mock 모드의 상자·종류·확신도는 화면 시험용 가짜 값이며 실제 AI 탐지 결과가 아닙니다.",
        icon="🧪",
    )
else:
    detector = get_yolo_detector(str(Path(model_path)), confidence)

run_clicked = st.button(
    "2. 차량 탐지 시작",
    type="primary",
    disabled=not valid_input or len(files) > 3,
    width="stretch",
)
if run_clicked:
    with st.spinner("차량 후보와 화질 지표를 계산하고 있습니다..."):
        st.session_state.analysis_results = _run_analysis(files, use_demo, detector)

results = st.session_state.get("analysis_results")
if results:
    st.subheader("3. 목표 차량 선택")
    selected_evidence: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        current_meta = metadata[index] if index < len(metadata) else {
            "camera": f"CCTV {index + 1}", "captured_at": "", "direction": "알 수 없음"
        }
        st.markdown(f"### {current_meta['camera']}")
        st.caption(f"촬영 시각: {current_meta['captured_at']} · 이동 방향: {current_meta['direction']}")
        if result["error"]:
            st.error(result["error"])
            st.info("입력 파일과 실행 설정을 확인한 뒤 ‘차량 탐지 시작’을 다시 눌러 주세요.")
            continue

        candidates = result["candidates"]
        st.image(
            annotate_detections(result["image"], [item.detection for item in candidates]),
            caption="탐지 후보 표시",
            channels="BGR",
            width="stretch",
        )
        if not candidates:
            st.warning("탐지된 차량 후보가 없습니다. 다른 이미지 또는 더 낮은 탐지 기준을 사용해 보세요.")
            continue

        card_columns = st.columns(len(candidates))
        for candidate_index, (column, candidate) in enumerate(zip(card_columns, candidates), start=1):
            with column:
                st.image(
                    candidate.crop_bgr,
                    caption=f"후보 {candidate_index}",
                    channels="BGR",
                    width="stretch",
                )
                st.caption(
                    f"{candidate.detection.label} · 탐지 확신도 {candidate.detection.confidence:.0%}"
                )

        selected_index = st.radio(
            "검토할 목표 차량을 직접 선택하세요",
            options=range(len(candidates)),
            format_func=lambda value, items=candidates: (
                f"후보 {value + 1} · {items[value].detection.label} "
                f"({items[value].detection.confidence:.0%})"
            ),
            horizontal=True,
            key=f"candidate_{signature}_{index}",
        )
        selected = candidates[selected_index]
        selected_evidence.append({"metadata": current_meta, "candidate": selected})

        st.markdown("#### 4. 화질과 보정 참고 영상")
        metric_columns = st.columns(4)
        metric_columns[0].metric("객체탐지 확신도", f"{selected.detection.confidence:.0%}")
        metric_columns[1].metric("선명도", f"{selected.metrics.sharpness:.1f}")
        metric_columns[2].metric("평균 밝기", f"{selected.metrics.brightness:.1f}")
        metric_columns[3].metric("대비", f"{selected.metrics.contrast:.1f}")
        original_column, enhanced_column = st.columns(2)
        original_column.image(
            selected.crop_bgr,
            caption="원본 차량 영상",
            channels="BGR",
            width="stretch",
        )
        enhanced_column.image(
            selected.enhanced_bgr,
            caption="보정 참고 영상",
            channels="BGR",
            width="stretch",
        )
        st.caption("지표는 원본 crop에서 계산합니다. 선명도 값은 확률이나 동일 차량 점수가 아닙니다.")
        st.divider()

    if selected_evidence:
        best = max(
            selected_evidence,
            key=lambda item: (
                item["candidate"].metrics.sharpness,
                item["candidate"].detection.confidence,
            ),
        )
        st.subheader("5. 초기 검토 결과")
        st.success(f"가장 선명한 근거 영상: {best['metadata']['camera']}")
        st.markdown(
            f"""
            선택 근거:

            - 선택된 차량 crop 중 선명도 점수가 가장 높음: **{best['candidate'].metrics.sharpness:.1f}**
            - 객체탐지 확신도: **{best['candidate'].detection.confidence:.0%}**
            """
        )
        st.warning(
            "보정 영상은 확인을 돕기 위한 참고자료이며, 원본에 존재하지 않는 정보를 복원하지 않습니다. "
            "최종 판단은 반드시 사람이 원본 자료와 함께 내려야 합니다.",
            icon="⚠️",
        )
elif valid_input:
    st.caption("자료 정보를 확인한 뒤 ‘차량 탐지 시작’을 눌러 주세요.")
else:
    st.info("가상 데모 이미지를 사용하거나 JPG/PNG 이미지 1~3개를 업로드해 주세요.")
