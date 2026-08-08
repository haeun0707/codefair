from datetime import datetime, timedelta

import pytest

from dasimanna.models import LocationPoint
from dasimanna.movement import (
    check_movement,
    check_movement_by_distance,
    haversine_km,
    predict_search_area,
)


def test_haversine_known_short_distance() -> None:
    distance = haversine_km(37.5665, 126.9780, 37.5675, 126.9780)
    assert distance == pytest.approx(0.111, abs=0.002)


def test_check_movement_flags_unrealistic_speed() -> None:
    start = LocationPoint("start", datetime(2026, 8, 7, 15, 0), 37.5665, 126.9780)
    end = LocationPoint("end", datetime(2026, 8, 7, 15, 10), 37.6500, 127.1000)

    result = check_movement(start, end, max_speed_kmh=8.0)

    assert result.feasible is False
    assert result.required_speed_kmh is not None
    assert result.required_speed_kmh > 8.0


def test_check_movement_rejects_reverse_time_for_calculation() -> None:
    start = LocationPoint("start", datetime(2026, 8, 7, 15, 10), 37.5665, 126.9780)
    end = LocationPoint("end", datetime(2026, 8, 7, 15, 0), 37.5666, 126.9781)

    result = check_movement(start, end, max_speed_kmh=8.0)

    assert result.feasible is None
    assert result.required_speed_kmh is None


def test_prediction_extends_recent_direction_and_returns_radius() -> None:
    first = LocationPoint("A", datetime(2026, 8, 7, 15, 0), 37.5665, 126.9780)
    second = LocationPoint("B", datetime(2026, 8, 7, 15, 15), 37.5675, 126.9790)

    prediction = predict_search_area([first, second], 30, max_speed_kmh=8.0)

    assert prediction.latitude > second.latitude
    assert prediction.longitude > second.longitude
    assert prediction.radius_km >= 0.25
    assert "직선" in prediction.basis


def test_prediction_with_one_point_stays_centered() -> None:
    point = LocationPoint("A", datetime.now(), 37.5, 127.0)
    prediction = predict_search_area([point], 30, max_speed_kmh=6.0)

    assert prediction.latitude == point.latitude
    assert prediction.longitude == point.longitude
    assert prediction.radius_km == pytest.approx(3.25)


def test_invalid_coordinate_is_rejected() -> None:
    with pytest.raises(ValueError):
        haversine_km(95.0, 0.0, 0.0, 0.0)


def test_movement_can_be_checked_with_estimated_distance_without_coordinates() -> None:
    start = datetime(2026, 8, 7, 15, 0)
    end = datetime(2026, 8, 7, 15, 20)

    result = check_movement_by_distance(1.2, start, end, max_speed_kmh=8.0)

    assert result.feasible is True
    assert result.required_speed_kmh == pytest.approx(3.6)
    assert "입력한 대략 거리" in result.detail


def test_estimated_distance_flags_time_distance_conflict() -> None:
    start = datetime(2026, 8, 7, 15, 0)
    end = datetime(2026, 8, 7, 15, 5)

    result = check_movement_by_distance(3.0, start, end, max_speed_kmh=8.0)

    assert result.feasible is False
    assert result.required_speed_kmh == pytest.approx(36.0)
