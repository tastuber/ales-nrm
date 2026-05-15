"""Tests for shared utility functions."""

from ales_nrm.utilities import ensure_odd


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
