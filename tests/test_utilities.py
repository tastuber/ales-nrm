"""Tests for shared utility functions."""

import numpy as np

from ales_nrm.utilities import ensure_odd, rotate_points_2d


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
