"""다시만나 AI — 실종 반려동물 목격 제보 검토 Streamlit 앱."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import streamlit as st

from dasimanna.demo import build_demo_case
from dasimanna.imaging import (
    InvalidImageError,
    aggregate_signatures,
    decode_image,
    extract_image_signature,
    measure_image_quality,
)
from dasimanna.matching import assess_sighting, rank_assessments
from dasimanna.models import LocationPoint, PetTraits
from dasimanna.movement import check_movement, check_movement_by_distance, predict_search_area


st.set_page_config(page_title="다시만나 AI", page_icon="🐾", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 4rem;}
    .hero {padding: 1.45rem 1.6rem; border-radius: 22px; color: white;
           background: linear-gradient(120deg, #155e75 0%, #0f766e 55%, #65a30d 125%);
           margin-bottom: 1.1rem; box-shadow: 0 12px 32px rgba(15,118,110,.16);}
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.15rem;}
    .hero p {margin: 0; opacity: .94; font-size: 1.02rem;}
    [data-testid="stMetric"] {border: 1px solid #dfe9e7; border-radius: 14px; padding: .8rem; background: #fbfefd;}
    .evidence-note {border-left: 4px solid #0f766e; padding: .65rem .9rem; background: #f0fdfa;
                    border-radius: 0 10px 10px 0; margin: .4rem 0 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


CUSTOM_OPTION = "직접 입력"
COLOR_OPTIONS = ("검정", "흰색", "회색", "갈색", "크림", "주황", "노랑", CUSTOM_OPTION)
FACE_OPTIONS = (
    "확인 불가",
    "무늬 없음",
    "이마 중앙에 흰 줄",
    "얼굴 한쪽만 다른 색",
    "눈 주변 무늬",
    "코 주변 흰색",
    CUSTOM_OPTION,
)
EAR_OPTIONS = (
    "확인 불가",
    "양쪽 귀가 섬",
    "양쪽 귀가 접힘",
    "한쪽 귀가 접힘",
    "긴 처진 귀",
    CUSTOM_OPTION,
)
BODY_OPTIONS = ("확인 불가", "마름", "보통", "통통함", "다리가 짧음", "몸통이 긺", CUSTOM_OPTION)
TAIL_OPTIONS = (
    "확인 불가",
    "위로 말린 꼬리",
    "아래로 긴 꼬리",
    "짧은 꼬리",
    "풍성한 꼬리",
    CUSTOM_OPTION,
)


def _save_text_widget(widget_key: str, storage_key: str) -> None:
    """직접 입력 위젯 값을 화면 수명과 분리된 현재 세션 상태에 복사한다."""

    st.session_state[storage_key] = st.session_state.get(widget_key, "")


def _persistent_text_input(
    container: Any,
    label: str,
    *,
    key: str,
    initial_value: str = "",
    placeholder: str | None = None,
    help_text: str | None = None,
) -> str:
    """조건부 화면에서 사라져도 현재 세션 동안 값을 유지하는 입력 칸."""

    storage_key = f"{key}_session_value"
    if storage_key not in st.session_state:
        st.session_state[storage_key] = st.session_state.get(key, initial_value)
    if key not in st.session_state:
        st.session_state[key] = st.session_state[storage_key]
    return container.text_input(
        label,
        key=key,
        placeholder=placeholder,
        help=help_text,
        on_change=_save_text_widget,
        args=(key, storage_key),
    )


def _select_or_custom(
    container: Any,
    label: str,
    options: tuple[str, ...],
    value: str,
    key: str,
) -> str:
    """미리 정한 선택지 또는 사용자가 직접 쓴 값을 반환한다."""

    is_custom_default = value not in options
    selected_value = CUSTOM_OPTION if is_custom_default else value
    selected = container.selectbox(
        label,
        options,
        index=_option_index(options, selected_value),
        key=key,
    )
    custom = _persistent_text_input(
        container,
        f"{label} 직접 입력",
        key=f"{key}_custom",
        initial_value=value if is_custom_default else "",
        placeholder=f"예: {label}을 자유롭게 적어 주세요",
        help_text="내용을 쓰면 위에서 고른 선택지보다 직접 입력 내용이 우선 적용됩니다.",
    )
    if custom.strip():
        return custom.strip()
    return selected if selected != CUSTOM_OPTION else "확인 불가"


def _trait_form(
    prefix: str,
    defaults: PetTraits | None = None,
    *,
    unknown_by_default: bool = False,
) -> PetTraits:
    species_options = ("확인 불가", "강아지", "고양이", CUSTOM_OPTION)
    species_value = defaults.species if defaults else ("확인 불가" if unknown_by_default else "강아지")
    species = _select_or_custom(
        st,
        "동물 종류",
        species_options,
        species_value,
        f"{prefix}_species",
    )
    default_colors = (
        list(defaults.fur_colors)
        if defaults
        else ([] if unknown_by_default else ["갈색", "흰색"])
    )
    preset_default_colors = [color for color in default_colors if color in COLOR_OPTIONS]
    custom_default_colors = [color for color in default_colors if color not in COLOR_OPTIONS]
    if custom_default_colors:
        preset_default_colors.append(CUSTOM_OPTION)
    selected_colors = st.multiselect(
        "털 색(복수 선택)",
        COLOR_OPTIONS,
        default=preset_default_colors,
        key=f"{prefix}_colors",
    )
    custom_color_text = _persistent_text_input(
        st,
        "털 색 직접 입력",
        key=f"{prefix}_colors_custom",
        initial_value=", ".join(custom_default_colors),
        placeholder="예: 연한 황토색, 등 쪽은 검은색",
        help_text="여러 색은 쉼표(,)로 나누어 적어 주세요. 입력 내용은 현재 세션 동안 유지됩니다.",
    )
    custom_colors = [item.strip() for item in custom_color_text.split(",") if item.strip()]
    fur_colors = tuple(color for color in selected_colors if color != CUSTOM_OPTION) + tuple(custom_colors)
    left, right = st.columns(2)
    face_marking = _select_or_custom(
        left,
        "얼굴 무늬",
        FACE_OPTIONS,
        defaults.face_marking
        if defaults
        else ("확인 불가" if unknown_by_default else "이마 중앙에 흰 줄"),
        f"{prefix}_face",
    )
    ear_shape = _select_or_custom(
        right,
        "귀 모양",
        EAR_OPTIONS,
        defaults.ear_shape
        if defaults
        else ("확인 불가" if unknown_by_default else "한쪽 귀가 접힘"),
        f"{prefix}_ear",
    )
    left, right = st.columns(2)
    body_shape = _select_or_custom(
        left,
        "체형",
        BODY_OPTIONS,
        defaults.body_shape if defaults else ("확인 불가" if unknown_by_default else "보통"),
        f"{prefix}_body",
    )
    tail_shape = _select_or_custom(
        right,
        "꼬리 모양",
        TAIL_OPTIONS,
        defaults.tail_shape
        if defaults
        else ("확인 불가" if unknown_by_default else "위로 말린 꼬리"),
        f"{prefix}_tail",
    )
    return PetTraits(species, fur_colors, face_marking, ear_shape, body_shape, tail_shape)


def _option_index(options: tuple[str, ...], value: str) -> int:
    return options.index(value) if value in options else 0


def _combine_date_time(day: date, clock: time) -> datetime:
    return datetime.combine(day, clock).replace(second=0, microsecond=0)


def _build_manual_case() -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    st.subheader("1. 실종 반려동물 등록")
    st.caption("서로 다른 각도에서 촬영하고, 동물이 화면의 대부분을 차지하도록 자른 사진이 좋습니다.")
    st.caption(
        "모든 세부사항에는 항상 보이는 직접 입력 칸이 있습니다. 내용을 쓰면 선택값보다 우선하며, "
        "현재 앱 세션 동안 임시 보관됩니다."
    )
    pet_name = st.text_input("반려동물 이름", value="우리 아이")
    reference_files = list(
        st.file_uploader(
            "기준 사진 1~5장",
            type=("jpg", "jpeg", "png"),
            accept_multiple_files=True,
            key="reference_files",
        )
        or []
    )
    if len(reference_files) > 5:
        errors.append("기준 사진은 최대 5장까지 올릴 수 있습니다.")
    reference_images = []
    for file in reference_files[:5]:
        try:
            reference_images.append(decode_image(file.getvalue()))
        except InvalidImageError as exc:
            errors.append(f"기준 사진 {file.name}: {exc}")
    if reference_images:
        st.markdown("**선택한 기준 사진 미리보기**")
        preview_columns = st.columns(min(len(reference_images), 5))
        for column, image, file in zip(preview_columns, reference_images, reference_files):
            with column:
                st.image(image, channels="BGR", caption=file.name, use_container_width=True)
    st.caption("큰 원본은 처리 시간이 늘어날 수 있으며, 동물이 크게 보이도록 자른 사진이 가장 좋습니다.")
    reference_traits = _trait_form("reference")

    st.subheader("2. 마지막으로 확인한 시각과 위치")
    st.caption(
        "장소 이름과 시각만 먼저 입력하면 됩니다. 지도 표시가 필요할 때만 아래 선택 항목에 좌표를 추가하세요."
    )
    origin_place = st.text_input("장소 이름", value="마지막 확인 장소")
    c1, c2 = st.columns(2)
    origin_date = c1.date_input("날짜", value=date.today(), key="origin_date")
    origin_time = c2.time_input("시각", value=time(15, 0), key="origin_time")
    origin_observed_at = _combine_date_time(origin_date, origin_time)
    with st.expander("선택: 지도에 표시할 좌표 추가"):
        c3, c4 = st.columns(2)
        origin_lat = c3.number_input(
            "위도(선택)", value=None, min_value=-90.0, max_value=90.0, format="%.6f", key="origin_lat"
        )
        origin_lon = c4.number_input(
            "경도(선택)", value=None, min_value=-180.0, max_value=180.0, format="%.6f", key="origin_lon"
        )
    if (origin_lat is None) != (origin_lon is None):
        errors.append("마지막 확인 위치의 위도와 경도는 둘 다 입력하거나 둘 다 비워 주세요.")
    origin = (
        LocationPoint(
            "마지막 확인",
            origin_observed_at,
            float(origin_lat),
            float(origin_lon),
        )
        if origin_lat is not None and origin_lon is not None
        else None
    )

    st.subheader("3. 시민 목격 제보 등록")
    report_count = int(
        st.number_input(
            "제보 수",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
            help="아직 제보가 없다면 0으로 두고 반려동물 정보만 먼저 등록할 수 있습니다.",
        )
    )
    if report_count == 0:
        st.info("아직 목격 제보가 없다면 0으로 두어도 됩니다. 새 제보가 들어오면 수를 늘려 사진과 정보를 추가하세요.")
    sightings: list[dict[str, Any]] = []
    for index in range(report_count):
        report_id = f"제보 {index + 1}"
        with st.expander(report_id, expanded=index == 0):
            uploaded = st.file_uploader(
                "목격 사진 1장",
                type=("jpg", "jpeg", "png"),
                key=f"report_file_{index}",
            )
            if uploaded is not None:
                try:
                    image = decode_image(uploaded.getvalue())
                    st.image(
                        image,
                        channels="BGR",
                        caption=f"{report_id} 사진 미리보기 · {uploaded.name}",
                        width=360,
                    )
                except InvalidImageError as exc:
                    errors.append(f"{report_id}: {exc}")
                    image = None
            else:
                image = None
            place = st.text_input("목격 장소", value=f"목격 장소 {index + 1}", key=f"place_{index}")
            c1, c2, c3 = st.columns(3)
            observed_date = c1.date_input("촬영 날짜", value=date.today(), key=f"date_{index}")
            default_observed_at = datetime.combine(date.today(), time(15, 10)) + timedelta(
                minutes=index * 10
            )
            observed_time = c2.time_input(
                "촬영 시각", value=default_observed_at.time(), key=f"time_{index}"
            )
            distance_from_origin_km = float(
                c3.number_input(
                    "마지막 확인 장소와의 대략 거리(km)",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.5,
                    step=0.1,
                    format="%.1f",
                    key=f"distance_{index}",
                    help="정확한 직선거리를 몰라도 지도나 평소 이동 경험을 참고해 대략 입력할 수 있습니다.",
                )
            )
            direction_from_origin = _persistent_text_input(
                st,
                "마지막 확인 장소에서 본 이동 방향",
                key=f"direction_{index}",
                initial_value="확인 불가",
                placeholder="예: 동쪽 공원 방향, 학교 뒤 골목 방향",
                help_text="좌표가 없어도 시간순 장소 경로와 다음 수색 방향을 정리하는 데 사용합니다.",
            )
            with st.expander("선택: 이 제보를 지도에 표시할 좌표 추가"):
                c4, c5 = st.columns(2)
                latitude = c4.number_input(
                    "위도(선택)",
                    value=None,
                    min_value=-90.0,
                    max_value=90.0,
                    format="%.6f",
                    key=f"lat_{index}",
                )
                longitude = c5.number_input(
                    "경도(선택)",
                    value=None,
                    min_value=-180.0,
                    max_value=180.0,
                    format="%.6f",
                    key=f"lon_{index}",
                )
            visibility = st.select_slider(
                "사진에서 동물이 보이는 정도",
                options=("거의 보이지 않음", "일부만 보임", "대부분 보임", "전체가 잘 보임"),
                value="대부분 보임",
                key=f"visibility_{index}",
            )
            st.caption(
                "사진에서 실제로 확인되는 특징만 기록하세요. 직접 입력 칸에 쓴 내용은 선택값보다 우선하고, "
                "보이지 않으면 입력칸을 비운 채 ‘확인 불가’로 두세요."
            )
            observed_traits = _trait_form(f"report_{index}", unknown_by_default=True)
            note = st.text_area("제보자 설명(선택)", key=f"note_{index}")
            observed_at = _combine_date_time(observed_date, observed_time)
            if (latitude is None) != (longitude is None):
                errors.append(f"{report_id}: 위도와 경도는 둘 다 입력하거나 둘 다 비워 주세요.")
            point = (
                LocationPoint(
                    report_id,
                    observed_at,
                    float(latitude),
                    float(longitude),
                )
                if latitude is not None and longitude is not None
                else None
            )
            sightings.append(
                {
                    "report_id": report_id,
                    "place": place,
                    "observed_at": observed_at,
                    "point": point,
                    "distance_from_origin_km": distance_from_origin_km,
                    "direction_from_origin": direction_from_origin,
                    "image": image,
                    "visibility": visibility,
                    "traits": observed_traits,
                    "note": note,
                }
            )

    if not reference_images:
        errors.append("기준 사진을 한 장 이상 올려 주세요.")
    if any(sighting["image"] is None for sighting in sightings):
        errors.append("모든 제보에 목격 사진을 한 장씩 올려 주세요.")
    if not reference_traits.fur_colors:
        errors.append("기준 반려동물의 털 색을 한 가지 이상 선택해 주세요.")

    case = {
        "pet_name": pet_name.strip() or "반려동물",
        "reference_images": reference_images,
        "reference_traits": reference_traits,
        "origin": origin,
        "origin_observed_at": origin_observed_at,
        "origin_place": origin_place,
        "sightings": sightings,
    }
    return case, errors


def _analyze_case(case: dict[str, Any], max_speed_kmh: float) -> list[dict[str, Any]]:
    reference_images = case["reference_images"]
    reference_signature = aggregate_signatures(
        [extract_image_signature(image) for image in reference_images]
    )
    analyzed: list[dict[str, Any]] = []
    origin: LocationPoint | None = case["origin"]
    origin_observed_at: datetime | None = case.get("origin_observed_at")
    if origin_observed_at is None and origin is not None:
        origin_observed_at = origin.observed_at
    for sighting in case["sightings"]:
        image = sighting["image"]
        quality = measure_image_quality(image)
        candidate_signature = extract_image_signature(image)
        point: LocationPoint | None = sighting["point"]
        if origin is not None and point is not None:
            movement = check_movement(origin, point, max_speed_kmh)
        elif origin_observed_at is not None and sighting.get("distance_from_origin_km") is not None:
            movement = check_movement_by_distance(
                float(sighting["distance_from_origin_km"]),
                origin_observed_at,
                sighting["observed_at"],
                max_speed_kmh,
            )
        else:
            movement = None
        assessment = assess_sighting(
            sighting["report_id"],
            reference_signature,
            candidate_signature,
            case["reference_traits"],
            sighting["traits"],
            quality,
            sighting["visibility"],
            movement,
        )
        analyzed.append({**sighting, "quality": quality, "assessment": assessment})

    ranked = rank_assessments([item["assessment"] for item in analyzed])
    rank_by_id = {assessment.report_id: index for index, assessment in enumerate(ranked)}
    return sorted(analyzed, key=lambda item: rank_by_id[item["report_id"]])


def _render_decision(decision: str, priority: int) -> None:
    message = f"{decision} · 검토 우선도 {priority}/100"
    if decision == "우선 확인":
        st.success(message)
    elif decision in {"추가 확인", "이동 경로 재확인"}:
        st.warning(message)
    elif decision == "판단 보류":
        st.info(message)
    else:
        st.error(message)


def _render_results(case: dict[str, Any], analyzed: list[dict[str, Any]], max_speed_kmh: float) -> None:
    st.divider()
    st.subheader("4. 먼저 확인할 제보 순서")
    st.markdown(
        '<div class="evidence-note">검토 우선도는 동일 동물일 확률이 아닙니다. '
        "AI가 제보 순서를 정리한 참고값이며 보호자·구조 담당자가 원본과 현장을 최종 확인해야 합니다.</div>",
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("전체 제보", f"{len(analyzed)}건")
    metric_columns[1].metric(
        "우선 확인", f"{sum(item['assessment'].decision == '우선 확인' for item in analyzed)}건"
    )
    metric_columns[2].metric(
        "판단 보류", f"{sum(item['assessment'].decision == '판단 보류' for item in analyzed)}건"
    )
    metric_columns[3].metric("등록 기준 사진", f"{len(case['reference_images'])}장")

    if not analyzed:
        st.info(
            "아직 등록된 목격 제보가 없습니다. 반려동물 기준 정보는 준비되었으며, "
            "새 제보가 들어오면 ‘제보 수’를 늘려 사진과 정보를 추가해 주세요."
        )
        return

    st.markdown("**제보별 단서 비교 그래프**")
    st.caption(
        "단서 유사도는 사진과 관찰 특징이 닮은 정도이며 동일 동물일 확률이 아닙니다. "
        "분석 신뢰도가 낮거나 동물이 가려진 제보는 그래프 값과 관계없이 판단 보류될 수 있습니다."
    )
    chart_rows = [
        {
            "제보": f"{rank}순위 · {item['report_id']}",
            "단서 유사도(%)": round(item["assessment"].clue_similarity * 100),
            "검토 우선도(점)": item["assessment"].priority,
            "분석 신뢰도(%)": round(item["assessment"].reliability * 100),
        }
        for rank, item in enumerate(analyzed, start=1)
    ]
    st.bar_chart(
        chart_rows,
        x="제보",
        y=["단서 유사도(%)", "검토 우선도(점)", "분석 신뢰도(%)"],
        y_label="점수 (0~100)",
        sort=False,
        stack=False,
        height=320,
    )

    confirmed_reports: list[dict[str, Any]] = []
    for rank, item in enumerate(analyzed, start=1):
        assessment = item["assessment"]
        with st.expander(
            f"{rank}순위 · {item['report_id']} · {item['place']} · {assessment.decision}",
            expanded=rank <= 2,
        ):
            image_column, detail_column = st.columns((1.05, 1.35))
            with image_column:
                st.image(item["image"], channels="BGR", caption=f"{item['report_id']} 원본 목격 사진")
                st.caption(f"제보자 설명: {item['note'] or '없음'}")
            with detail_column:
                _render_decision(assessment.decision, assessment.priority)
                point: LocationPoint | None = item["point"]
                observed_at = item.get("observed_at") or (point.observed_at if point else None)
                observed_text = observed_at.strftime("%Y-%m-%d %H:%M") if observed_at else "확인 불가"
                location_text = (
                    f"{item['place']} · 좌표 {point.latitude:.5f}, {point.longitude:.5f}"
                    if point is not None
                    else (
                        f"{item['place']} · 마지막 확인에서 약 "
                        f"{item.get('distance_from_origin_km', 0.0):.1f}km · "
                        f"{item.get('direction_from_origin', '방향 확인 불가')}"
                    )
                )
                st.write(f"**촬영:** {observed_text} · **위치:** {location_text}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("단서 유사도", f"{assessment.clue_similarity * 100:.0f}%")
                m2.metric("사진 유사도", f"{assessment.image_similarity * 100:.0f}%")
                m3.metric("분석 신뢰도", f"{assessment.reliability * 100:.0f}%")
                m4.metric("선명도", f"{item['quality'].sharpness:.1f}")
                st.progress(assessment.priority)

            st.markdown("**일치·불일치 근거**")
            st.dataframe(
                [
                    {
                        "단서": evidence.category,
                        "결과": evidence.status,
                        "설명": evidence.detail,
                        "참고 점수": "—" if evidence.score is None else f"{evidence.score * 100:.0f}",
                    }
                    for evidence in assessment.evidence
                ],
                hide_index=True,
                use_container_width=True,
            )
            if assessment.requests:
                st.markdown("**추가로 요청할 자료**")
                for request in assessment.requests:
                    st.markdown(f"- {request}")

            confirmed = st.checkbox(
                "보호자가 원본을 확인했으며 경로 분석에 사용",
                value=False,
                key=f"confirmed_{item['report_id']}",
            )
            if confirmed:
                confirmed_reports.append(item)

    _render_route(case, confirmed_reports, max_speed_kmh)


def _render_route(case: dict[str, Any], confirmed_reports: list[dict[str, Any]], max_speed_kmh: float) -> None:
    st.divider()
    st.subheader("5. 확인된 제보로 경로와 다음 수색 범위 보기")
    if not confirmed_reports:
        st.info("원본을 확인한 제보를 한 건 이상 선택하면 경로와 예상 수색 범위를 표시합니다.")
        return

    def report_time(item: dict[str, Any]) -> datetime:
        point: LocationPoint | None = item.get("point")
        return item.get("observed_at") or (point.observed_at if point else datetime.min)

    sorted_reports = sorted(confirmed_reports, key=report_time)
    origin: LocationPoint | None = case.get("origin")
    origin_observed_at: datetime | None = case.get("origin_observed_at")
    if origin_observed_at is None and origin is not None:
        origin_observed_at = origin.observed_at
    origin_place = str(case.get("origin_place") or "마지막 확인 장소")

    ordered_places = [origin_place] + [str(item["place"]) for item in sorted_reports]
    st.markdown(f"**시간순 장소 경로:** {' → '.join(ordered_places)}")
    st.caption(
        "좌표가 없어도 장소 이름·촬영 시각·마지막 확인 장소와의 대략 거리로 정리한 경로입니다. "
        "대략 거리는 실제 도로와 다를 수 있습니다."
    )

    route_rows: list[dict[str, str]] = []
    for sequence, item in enumerate(sorted_reports, start=1):
        movement = item["assessment"].movement
        direction = item.get("direction_from_origin") or (
            "좌표 기반" if item.get("point") is not None else "확인 불가"
        )
        route_rows.append(
            {
                "순서": str(sequence),
                "촬영 시각": report_time(item).strftime("%Y-%m-%d %H:%M"),
                "장소": str(item["place"]),
                "마지막 확인 기준 거리": (
                    "확인 불가" if movement is None else f"약 {movement.distance_km:.2f}km"
                ),
                "방향 단서": str(direction),
                "필요 속도": (
                    "계산 불가"
                    if movement is None or movement.required_speed_kmh is None
                    else f"{movement.required_speed_kmh:.1f}km/h"
                ),
                "이동 검토": (
                    "거리 정보 부족"
                    if movement is None
                    else (
                        "가능 범위"
                        if movement.feasible is True
                        else "촬영 시각·장소 재확인"
                    )
                ),
            }
        )

    st.dataframe(route_rows, hide_index=True, use_container_width=True)
    plausible_reports = [
        item
        for item in sorted_reports
        if item["assessment"].movement is None or item["assessment"].movement.feasible is not False
    ]
    latest_report = plausible_reports[-1] if plausible_reports else sorted_reports[-1]
    latest_movement = latest_report["assessment"].movement
    latest_direction = latest_report.get("direction_from_origin") or "현장에서 확인한 이동 방향"
    horizon_minutes = st.slider("마지막 목격 이후 몇 분 범위를 살펴볼까요?", 15, 120, 45, 15)
    reference_speed = (
        min(latest_movement.required_speed_kmh, max_speed_kmh)
        if latest_movement is not None and latest_movement.required_speed_kmh is not None
        else max_speed_kmh * 0.5
    )
    search_radius_km = max(0.2, reference_speed * horizon_minutes / 60.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("다음 수색 기준 장소", str(latest_report["place"]))
    c2.metric("우선 확인 방향", str(latest_direction))
    c3.metric("이동 참고 반경", f"약 {search_radius_km:.2f}km")
    st.warning(
        "이 방향과 반경은 확인한 제보의 시간·대략 거리로 만든 수색 참고값입니다. "
        "도로·하천·울타리·동물 행동을 반영한 정확한 위치 예측이 아닙니다."
    )

    confirmed_with_location = [item for item in sorted_reports if item.get("point") is not None]
    if origin is not None and confirmed_with_location:
        st.markdown("**선택 좌표가 있는 제보의 지도 참고**")
        feasible_points = [origin]
        previous = origin
        for item in confirmed_with_location:
            point: LocationPoint = item["point"]
            movement = check_movement(previous, point, max_speed_kmh)
            if movement.feasible is True:
                feasible_points.append(point)
                previous = point

        prediction = predict_search_area(feasible_points, horizon_minutes, max_speed_kmh)
        map_points = [
            {"latitude": point.latitude, "longitude": point.longitude}
            for point in feasible_points
        ] + [{"latitude": prediction.latitude, "longitude": prediction.longitude}]
        st.map(map_points, latitude="latitude", longitude="longitude", zoom=14, use_container_width=True)
        st.caption(
            "지도 마지막 점은 선택 입력한 좌표를 단순 직선으로 연장한 참고 중심입니다. "
            "지도 연결이 없는 환경에서는 위 장소 경로표를 사용하세요."
        )
    else:
        st.info(
            "위도·경도를 쓰지 않아도 위 장소 경로표와 수색 방향을 사용할 수 있습니다. "
            "좌표를 선택 입력하면 지도도 함께 표시됩니다."
        )

    st.markdown("**주변 수색 순서 제안**")
    st.markdown(
        "1. 마지막으로 확인된 지점 주변의 숨을 만한 낮은 공간과 조용한 길을 먼저 확인합니다.\n"
        "2. 반려동물이 익숙한 이름·소리를 사용하되 차량 통행 지역에서는 성인이 안전을 확인합니다.\n"
        "3. 현장 사진과 시간을 기록하고, 새 제보가 오면 기존 경로와 다시 비교합니다.\n"
        "4. 연락처·정확한 집 주소는 공개 게시물에 그대로 노출하지 않습니다."
    )


st.title("🐾 다시만나 AI")
st.markdown(
    """
    <div class="hero">
      <h1>목격 제보를 수색 순서로</h1>
      <p>실종 반려동물 목격 사진을 근거별로 비교하고, 먼저 확인할 제보와 다음 수색 범위를 정리합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "AI는 실종 동물을 확정하지 않습니다. 사진이 흐리거나 가려졌다면 ‘판단 보류’하며, "
    "보호자와 구조 담당자의 검토를 돕는 보조 도구입니다.",
    icon="🛡️",
)

with st.sidebar:
    st.header("시제품 설정")
    demo_mode = st.toggle(
        "가상 데모 자료 사용",
        value=False,
        help="끄면 보호자와 제보자가 준비한 JPG/JPEG/PNG 사진을 직접 올릴 수 있습니다.",
    )
    max_speed_kmh = st.slider(
        "이동 가능성 시나리오 속도",
        min_value=2.0,
        max_value=40.0,
        value=15.0,
        step=1.0,
        help=(
            "실제 최대 속도를 판정하는 값이 아닙니다. 사용자가 가정한 평균 이동 속도로 "
            "제보 시각·좌표 오류를 재확인하는 데만 사용합니다."
        ),
    )
    st.divider()
    st.caption(
        "업로드 사진과 위치는 현재 앱 세션에서만 분석하며 저장소에 쓰지 않습니다. "
        "대회 시연에는 가상 동물과 가상 위치를 사용하세요."
    )

if demo_mode:
    case = build_demo_case()
    errors: list[str] = []
    st.subheader("1. 가상 실종 반려동물")
    left, right = st.columns((1.15, 1))
    with left:
        st.image(
            case["reference_images"],
            channels="BGR",
            caption=["기준 사진 · 정면", "기준 사진 · 다른 조명"],
        )
    with right:
        traits: PetTraits = case["reference_traits"]
        st.markdown(f"### {case['pet_name']}")
        st.write(f"**종류:** {traits.species}")
        st.write(f"**털 색:** {', '.join(traits.fur_colors)}")
        st.write(f"**얼굴 무늬:** {traits.face_marking}")
        st.write(f"**귀·체형·꼬리:** {traits.ear_shape} · {traits.body_shape} · {traits.tail_shape}")
        st.write(
            f"**마지막 확인:** {case['origin_place']} · {case['origin'].observed_at:%Y-%m-%d %H:%M}"
        )
    st.subheader("2. 가상 목격 제보")
    st.caption("닮은 제보, 흐린 제보, 시간상 이동이 어려운 제보, 특징이 다른 제보가 섞여 있습니다.")
    preview_columns = st.columns(len(case["sightings"]))
    for column, sighting in zip(preview_columns, case["sightings"]):
        with column:
            st.image(sighting["image"], channels="BGR", caption=sighting["report_id"])
            st.caption(f"{sighting['place']} · {sighting['point'].observed_at:%H:%M}")
else:
    st.success("직접 사진 업로드 모드입니다. 기준 사진과 목격 사진을 아래에서 선택해 주세요.")
    case, errors = _build_manual_case()

for error in dict.fromkeys(errors):
    st.error(error)

analyze_clicked = st.button(
    "제보 비교 및 경로 분석",
    type="primary",
    use_container_width=True,
    disabled=bool(errors) or case is None,
)
if analyze_clicked and case is not None:
    try:
        with st.spinner("사진 단서와 관찰 특징, 시간·위치 이동 가능성을 비교하고 있습니다..."):
            st.session_state.analysis_bundle = {
                "mode": "demo" if demo_mode else "manual",
                "case": case,
                "analyzed": _analyze_case(case, max_speed_kmh),
                "max_speed": max_speed_kmh,
            }
    except (InvalidImageError, ValueError) as exc:
        st.error(f"분석을 완료하지 못했습니다: {exc}")
    except Exception as exc:
        st.error(
            "예상하지 못한 오류가 발생했습니다. 사진 형식과 입력값을 확인한 뒤 다시 시도해 주세요. "
            f"({type(exc).__name__})"
        )

bundle = st.session_state.get("analysis_bundle")
current_mode = "demo" if demo_mode else "manual"
if bundle and bundle.get("mode") == current_mode:
    _render_results(bundle["case"], bundle["analyzed"], bundle["max_speed"])
