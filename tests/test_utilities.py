"""Tests for shared utility functions."""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from ales_nrm.utilities import (
    _compute_mean_mjd,
    _get_block_mean_time_seconds,
    _get_block_times,
    _parse_time_string_to_seconds,
    _parse_timestamps_to_seconds,
    ensure_odd,
    rotate_points_2d,
)


class TestEnsureOdd:
    """Tests for the ensure_odd utility."""

    def test_odd_unchanged(self):
        """Odd input is returned unchanged."""
        assert ensure_odd(67) == 67

    def test_even_incremented(self):
        """Even input is incremented by one."""
        assert ensure_odd(64) == 65

    def test_one(self):
        """One is already odd."""
        assert ensure_odd(1) == 1

    def test_two(self):
        """Two becomes three."""
        assert ensure_odd(2) == 3

    def test_large_even(self):
        """Large even number is incremented."""
        assert ensure_odd(512) == 513

    def test_large_odd(self):
        """Large odd number is unchanged."""
        assert ensure_odd(501) == 501

    def test_zero(self):
        """Zero becomes one."""
        assert ensure_odd(0) == 1


class TestRotatePoints2D:
    """Tests for the rotate_points_2d utility."""

    def test_zero_rotation_unchanged(self):
        """Zero angle returns identical points."""
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        center = (0.0, 0.0)
        result = rotate_points_2d(pts, center, 0.0)
        np.testing.assert_allclose(result, pts)

    def test_360_rotation_returns_original(self):
        """Full rotation returns original points."""
        pts = np.array([[1.0, 2.0], [-3.0, 5.0]])
        center = (1.0, 1.0)
        result = rotate_points_2d(pts, center, 360.0)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_180_rotation_reflects_through_center(self):
        """180 degrees reflects points through center."""
        center = (0.0, 0.0)
        pts = np.array([[1.0, 0.0]])
        result = rotate_points_2d(pts, center, 180.0)
        expected = np.array([[-1.0, 0.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_90_ccw(self):
        """90 degree CCW rotation."""
        center = (0.0, 0.0)
        pts = np.array([[1.0, 0.0]])
        result = rotate_points_2d(pts, center, 90.0)
        expected = np.array([[0.0, 1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_center_invariant(self):
        """Point at center is unchanged by any rotation."""
        center = (5.0, 3.0)
        pts = np.array([[5.0, 3.0]])
        result = rotate_points_2d(pts, center, 47.0)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_multiple_points(self):
        """Handles arrays with multiple points."""
        center = (0.0, 0.0)
        pts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        result = rotate_points_2d(pts, center, 90.0)
        expected = np.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestParseTimeStringToSeconds:
    """Tests for _parse_time_string_to_seconds helper."""

    def test_midnight(self):
        """Midnight parses to 0 seconds."""
        assert _parse_time_string_to_seconds("00:00:00.000") == 0.0

    def test_noon(self):
        """Noon parses to 43200 seconds."""
        assert _parse_time_string_to_seconds("12:00:00.000") == 43200.0

    def test_fractional_seconds(self):
        """Fractional seconds are preserved."""
        result = _parse_time_string_to_seconds("01:02:03.456")
        expected = 3600 + 120 + 3.456
        assert result == pytest.approx(expected)

    def test_without_fractional(self):
        """Works without fractional seconds."""
        result = _parse_time_string_to_seconds("08:30:00")
        assert result == pytest.approx(8 * 3600 + 30 * 60)

    def test_leading_whitespace(self):
        """Leading/trailing whitespace is handled."""
        result = _parse_time_string_to_seconds("  12:00:00.000  ")
        assert result == pytest.approx(43200.0)


class TestParseTimestampsToSeconds:
    """Tests for _parse_timestamps_to_seconds helper."""

    def test_simple_sequence(self):
        """Normal sequence without midnight crossing."""
        stamps = np.array(["08:00:00.000", "08:15:00.000", "08:30:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        expected = np.array([28800.0, 29700.0, 30600.0])
        np.testing.assert_allclose(result, expected)

    def test_midnight_crossing(self):
        """Timestamps spanning midnight get unwrapped."""
        stamps = np.array(["23:50:00.000", "23:55:00.000", "00:05:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        # Last timestamp should be unwrapped to > 86400
        assert result[2] > 86400.0
        # Should be monotonically increasing
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_midnight_crossing_value(self):
        """Midnight-crossing unwrap gives correct value."""
        stamps = np.array(["23:59:00.000", "00:01:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        expected_second = 86400.0 + 60.0  # 00:01 = 86460 seconds
        assert result[1] == pytest.approx(expected_second)

    def test_no_crossing_near_midnight(self):
        """Sequence near midnight but not crossing is unmodified."""
        stamps = np.array(["23:50:00.000", "23:55:00.000", "23:59:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        # All should be < 86400
        assert np.all(result < 86400.0)

    def test_single_timestamp(self):
        """Single timestamp works."""
        stamps = np.array(["12:00:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        assert result[0] == pytest.approx(43200.0)


class TestComputeMeanMjd:
    """Tests for _compute_mean_mjd helper."""

    def test_none_date_returns_none(self):
        """Returns None when observation_date is None."""
        stamps = np.array(["08:00:00.000", "09:00:00.000"])
        assert _compute_mean_mjd(None, stamps) is None

    def test_known_date(self):
        """Computes reasonable MJD for known date and time."""
        date = datetime.date(2023, 1, 1)
        stamps = np.array(["12:00:00.000"])
        mjd = _compute_mean_mjd(date, stamps)
        # 2023-01-01 12:00 UT should be MJD ~59945.5
        assert mjd is not None
        assert mjd == pytest.approx(59945.5, abs=0.01)

    def test_mean_of_timestamps(self):
        """MJD reflects mean of multiple timestamps."""
        date = datetime.date(2023, 1, 1)
        stamps_early = np.array(["06:00:00.000"])
        stamps_late = np.array(["18:00:00.000"])
        stamps_both = np.array(["06:00:00.000", "18:00:00.000"])

        mjd_early = _compute_mean_mjd(date, stamps_early)
        mjd_late = _compute_mean_mjd(date, stamps_late)
        mjd_both = _compute_mean_mjd(date, stamps_both)

        # Mean should be between early and late
        assert mjd_early < mjd_both < mjd_late

    def test_midnight_crossing_mjd(self):
        """MJD handles midnight crossing correctly."""
        date = datetime.date(2023, 1, 1)
        stamps = np.array(["23:59:00.000", "00:01:00.000"])
        mjd = _compute_mean_mjd(date, stamps)
        # Mean is midnight = start of Jan 2
        # observation_date is Jan 1 (date before midnight)
        assert mjd is not None
        # Should be very close to MJD of 2023-01-02 00:00
        assert mjd == pytest.approx(59946.0, abs=0.01)

    def test_type_is_float(self):
        """Return type is float."""
        date = datetime.date(2023, 6, 15)
        stamps = np.array(["12:00:00.000"])
        mjd = _compute_mean_mjd(date, stamps)
        assert isinstance(mjd, float)


class TestGetBlockMeanTimeSeconds:
    """Tests for _get_block_mean_time_seconds."""

    def test_simple_timestamps(self):
        """Computes correct mean for simple timestamps."""
        block = MagicMock()
        block.timestamps = np.array(
            ["08:00:00.000", "08:10:00.000"],
            dtype=object,
        )
        result = _get_block_mean_time_seconds(block)
        expected = 8 * 3600 + 5 * 60
        assert result == pytest.approx(expected)

    def test_midnight_crossing(self):
        """Handles midnight crossing correctly."""
        block = MagicMock()
        block.timestamps = np.array(
            ["23:55:00.000", "00:05:00.000"],
            dtype=object,
        )
        result = _get_block_mean_time_seconds(block)
        assert result == pytest.approx(86400.0)

    def test_empty_timestamps_raises(self):
        """Raises ValueError if no valid timestamps."""
        block = MagicMock()
        block.target = "test"
        block.timestamps = np.array([], dtype=object)
        with pytest.raises(ValueError, match="no valid timestamps"):
            _get_block_mean_time_seconds(block)


class TestGetBlockTimes:
    """Tests for _get_block_times."""

    def test_multiple_blocks(self):
        """Returns correct times for multiple blocks."""
        block1 = MagicMock()
        block1.timestamps = np.array(
            ["07:50:00.000", "07:55:00.000"],
            dtype=object,
        )
        block2 = MagicMock()
        block2.timestamps = np.array(
            ["08:00:00.000", "08:10:00.000", "08:20:00.000"],
            dtype=object,
        )
        times = _get_block_times([block1, block2])
        assert times.shape == (2,)
        expected_1 = 7 * 3600 + 52 * 60 + 30
        expected_2 = 8 * 3600 + 10 * 60
        assert times[0] == pytest.approx(expected_1)
        assert times[1] == pytest.approx(expected_2)

    def test_single_block(self):
        """Returns correct time for a single block."""
        block = MagicMock()
        block.timestamps = np.array(
            ["12:00:00.000"],
            dtype=object,
        )
        times = _get_block_times([block])
        assert times.shape == (1,)
        assert times[0] == pytest.approx(43200.0)
