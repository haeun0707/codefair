from datetime import datetime, timedelta

import pytest

from dasimanna.models import LocationPoint
from dasimanna.movement import check_movement, haversine_km, predict_search_area


def test_haversine_returns_expected_short_city_distance() -> None:
    distance = haversine_km(37.5665, 126.9780, 37.5675, 126.9790)

    assert 0.1 < distance < 0.2


def test_movement_flags_implausible_scenario_without_claiming_impossibility() -> None:
    start = LocationPoint("마지막 확인", datetime(2026, 8, 7, 15, 0), 37.5665, 126.9780)
    far = LocationPoint("먼 제보", datetime(2026, 8, 7, 15, 10), 37.6665, 127.0780)

    result = check_movement(start, far, max_speed_kmh=15.0)

    assert result.feasible is False
    assert "다시 확인" in result.detail
    assert result.required_speed_kmh is not None


def test_non_forward_time_returns_unknown_movement() -> None:
    now = datetime(2026, 8, 7, 15, 0)
    start = LocationPoint("시작", now, 37.5665, 126.9780)
    earlier = LocationPoint("제보", now - timedelta(minutes=1), 37.5670, 126.9785)

    result = check_movement(start, earlier, max_speed_kmh=15.0)

    assert result.feasible is None
    assert result.required_speed_kmh is None


def test_search_prediction_requires_a_confirmed_point() -> None:
    with pytest.raises(ValueError, match="한 곳 이상"):
        predict_search_area([], horizon_minutes=30, max_speed_kmh=15.0)
