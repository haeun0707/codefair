"""목격 시각·위치의 이동 가능성 확인과 단순 수색 영역 예측."""

from __future__ import annotations

import math
from datetime import datetime

from .models import LocationPoint, MovementCheck, SearchPrediction


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 사이의 대권거리(km)를 계산한다."""

    _validate_coordinates(lat1, lon1)
    _validate_coordinates(lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def check_movement(start: LocationPoint, end: LocationPoint, max_speed_kmh: float) -> MovementCheck:
    """직선거리 기준으로 해당 시간 안의 이동 가능성을 보수적으로 확인한다."""

    distance = haversine_km(start.latitude, start.longitude, end.latitude, end.longitude)
    return check_movement_by_distance(
        distance,
        start.observed_at,
        end.observed_at,
        max_speed_kmh,
        distance_source="좌표 직선거리",
    )


def check_movement_by_distance(
    distance_km: float,
    start_time: datetime,
    end_time: datetime,
    max_speed_kmh: float,
    *,
    distance_source: str = "입력한 대략 거리",
) -> MovementCheck:
    """좌표 없이도 거리 추정값과 촬영 시각으로 이동 가능성을 확인한다.

    사용자가 입력한 대략 거리는 실제 보행 경로와 다를 수 있으므로 결과는 제보의
    시각·장소를 다시 확인하는 보조 단서로만 사용한다.
    """

    if distance_km < 0:
        raise ValueError("거리는 0km 이상이어야 합니다.")
    if max_speed_kmh <= 0:
        raise ValueError("최대 이동 속도는 0보다 커야 합니다.")
    elapsed_minutes = (end_time - start_time).total_seconds() / 60.0
    if elapsed_minutes <= 0:
        return MovementCheck(
            distance_km,
            elapsed_minutes,
            None,
            None,
            "이전 위치보다 촬영 시각이 빠르거나 같아 이동 가능성을 계산할 수 없습니다.",
        )

    required_speed = distance_km / (elapsed_minutes / 60.0)
    feasible = required_speed <= max_speed_kmh
    if feasible:
        detail = (
            f"{distance_source} {distance_km:.2f}km, 필요한 평균 속도 {required_speed:.1f}km/h로 "
            "설정한 범위 안입니다. 실제 길·장애물과 거리 오차는 별도로 확인해야 합니다."
        )
    else:
        detail = (
            f"{distance_source} {distance_km:.2f}km를 이동하려면 평균 {required_speed:.1f}km/h가 필요해 "
            f"설정값 {max_speed_kmh:.1f}km/h를 넘습니다. 촬영 시각·장소·거리 추정값을 다시 확인하세요."
        )
    return MovementCheck(distance_km, elapsed_minutes, required_speed, feasible, detail)


def predict_search_area(
    confirmed_points: list[LocationPoint],
    horizon_minutes: int,
    max_speed_kmh: float,
    base_uncertainty_km: float = 0.25,
) -> SearchPrediction:
    """마지막 두 확인 지점의 방향을 짧게 연장해 수색 중심과 반경을 제안한다.

    도로망, 지형, 동물 행동을 사용하지 않는 직선 기반 참고값이며 정확한 위치 예측이 아니다.
    """

    if not confirmed_points:
        raise ValueError("확인된 위치가 한 곳 이상 필요합니다.")
    if horizon_minutes <= 0 or max_speed_kmh <= 0 or base_uncertainty_km < 0:
        raise ValueError("예측 시간·속도는 양수이고 기본 오차는 0 이상이어야 합니다.")

    points = sorted(confirmed_points, key=lambda point: point.observed_at)
    latest = points[-1]
    max_projection_km = max_speed_kmh * horizon_minutes / 60.0

    if len(points) == 1:
        radius = base_uncertainty_km + max_projection_km
        return SearchPrediction(
            latest.latitude,
            latest.longitude,
            radius,
            "확인된 목격이 한 곳뿐이어서 마지막 위치를 중심으로 설정했습니다.",
            "원형 반경은 가능한 수색 범위 참고값이며 실제 위치를 뜻하지 않습니다.",
        )

    previous = points[-2]
    elapsed_hours = (latest.observed_at - previous.observed_at).total_seconds() / 3600.0
    if elapsed_hours <= 0:
        radius = base_uncertainty_km + max_projection_km
        return SearchPrediction(
            latest.latitude,
            latest.longitude,
            radius,
            "최근 두 목격의 시각 순서를 확인할 수 없어 마지막 위치를 중심으로 설정했습니다.",
            "촬영 시각을 수정한 뒤 다시 계산하세요.",
        )

    recent_distance = haversine_km(
        previous.latitude, previous.longitude, latest.latitude, latest.longitude
    )
    observed_speed = min(recent_distance / elapsed_hours, max_speed_kmh)
    projection_km = min(observed_speed * horizon_minutes / 60.0, max_projection_km)

    lat_delta = latest.latitude - previous.latitude
    lon_delta = latest.longitude - previous.longitude
    segment_length = math.hypot(lat_delta, lon_delta)
    if segment_length <= 1e-12 or recent_distance <= 1e-8:
        predicted_lat, predicted_lon = latest.latitude, latest.longitude
    else:
        scale = projection_km / recent_distance
        predicted_lat = latest.latitude + lat_delta * scale
        predicted_lon = latest.longitude + lon_delta * scale

    radius = base_uncertainty_km + max(0.20, projection_km * 0.65)
    return SearchPrediction(
        predicted_lat,
        predicted_lon,
        radius,
        f"최근 두 확인 지점의 이동 방향을 {horizon_minutes}분만큼 직선으로 연장했습니다.",
        "도로·하천·울타리·동물 행동을 반영하지 않은 수색 우선순위 참고값입니다.",
    )


def _validate_coordinates(latitude: float, longitude: float) -> None:
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("위도는 -90~90, 경도는 -180~180 범위여야 합니다.")
