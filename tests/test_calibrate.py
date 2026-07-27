"""Tests for ales_nrm.calibrate module — arithmetic helpers."""

import numpy as np
import pytest

from ales_nrm.calibrate import (
    _calibrate_division,
    _calibrate_subtraction,
    _get_calibration_values_constant,
    _get_calibration_values_linear,
)


class TestCalibrateSubtraction:
    """Tests for _calibrate_subtraction."""

    def test_correct_arithmetic(self):
        """Subtraction produces sci - cal."""
        sci = np.array([[10.0, 20.0], [30.0, 40.0]])
        cal = np.array([[3.0, 5.0], [7.0, 9.0]])
        sci_err = np.ones_like(sci)
        cal_err = np.ones_like(cal)
        result, _ = _calibrate_subtraction(sci, sci_err, cal, cal_err)
        expected = np.array([[7.0, 15.0], [23.0, 31.0]])
        np.testing.assert_allclose(result, expected)

    def test_uncertainty_propagation(self):
        """Error is sqrt(sigma_sci^2 + sigma_cal^2)."""
        sci = np.array([[10.0]])
        cal = np.array([[5.0]])
        sci_err = np.array([[3.0]])
        cal_err = np.array([[4.0]])
        _, sigma = _calibrate_subtraction(sci, sci_err, cal, cal_err)
        expected = np.sqrt(9.0 + 16.0)
        np.testing.assert_allclose(sigma, [[expected]])

    def test_zero_uncertainties(self):
        """Zero uncertainties propagate correctly."""
        sci = np.array([[10.0]])
        cal = np.array([[3.0]])
        sci_err = np.array([[0.0]])
        cal_err = np.array([[0.0]])
        result, sigma = _calibrate_subtraction(sci, sci_err, cal, cal_err)
        np.testing.assert_allclose(result, [[7.0]])
        np.testing.assert_allclose(sigma, [[0.0]])

    def test_nan_propagation(self):
        """NaN in input produces NaN in output."""
        sci = np.array([[np.nan, 10.0]])
        cal = np.array([[5.0, np.nan]])
        sci_err = np.array([[1.0, 1.0]])
        cal_err = np.array([[1.0, 1.0]])
        result, sigma = _calibrate_subtraction(sci, sci_err, cal, cal_err)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        shape = (6, 10)
        sci = np.ones(shape) * 20.0
        cal = np.ones(shape) * 5.0
        sci_err = np.ones(shape) * 2.0
        cal_err = np.ones(shape) * 3.0
        result, sigma = _calibrate_subtraction(sci, sci_err, cal, cal_err)
        assert result.shape == shape
        assert sigma.shape == shape


class TestCalibrateDivision:
    """Tests for _calibrate_division."""

    def test_correct_arithmetic(self):
        """Division produces sci / cal."""
        sci = np.array([[0.8, 0.6]])
        cal = np.array([[0.95, 1.0]])
        sci_err = np.array([[0.01, 0.01]])
        cal_err = np.array([[0.01, 0.01]])
        result, _ = _calibrate_division(sci, sci_err, cal, cal_err)
        expected = np.array([[0.8 / 0.95, 0.6 / 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_uncertainty_propagation(self):
        """Error propagation for f = a/b is correct."""
        a = np.array([[0.8]])
        b = np.array([[0.95]])
        sa = np.array([[0.02]])
        sb = np.array([[0.03]])
        _, sigma = _calibrate_division(a, sa, b, sb)
        # sigma_f = sqrt((sa/b)^2 + (a*sb/b^2)^2)
        expected = np.sqrt((0.02 / 0.95) ** 2 + (0.8 * 0.03 / 0.95**2) ** 2)
        np.testing.assert_allclose(sigma, [[expected]])

    def test_sci_zero_gives_zero(self):
        """sci_value=0 gives calibrated=0, sigma=|sa/b|."""
        sci = np.array([[0.0]])
        cal = np.array([[0.95]])
        sci_err = np.array([[0.02]])
        cal_err = np.array([[0.03]])
        result, sigma = _calibrate_division(sci, sci_err, cal, cal_err)
        np.testing.assert_allclose(result, [[0.0]])
        # sigma = sqrt((0.02/0.95)^2 + (0*0.03/0.95^2)^2)
        #       = 0.02 / 0.95
        expected_sigma = 0.02 / 0.95
        np.testing.assert_allclose(sigma, [[expected_sigma]])

    def test_cal_zero_gives_nan(self):
        """cal_value=0 gives NaN for both outputs."""
        sci = np.array([[0.8]])
        cal = np.array([[0.0]])
        sci_err = np.array([[0.02]])
        cal_err = np.array([[0.03]])
        result, sigma = _calibrate_division(sci, sci_err, cal, cal_err)
        assert np.isnan(result[0, 0])
        assert np.isnan(sigma[0, 0])

    def test_nan_propagation(self):
        """NaN in input produces NaN in output."""
        sci = np.array([[np.nan, 0.8]])
        cal = np.array([[0.95, np.nan]])
        sci_err = np.array([[0.01, 0.01]])
        cal_err = np.array([[0.01, 0.01]])
        result, sigma = _calibrate_division(sci, sci_err, cal, cal_err)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])

    def test_shape_preserved(self):
        """Output shape matches input shape."""
        shape = (3, 5)
        sci = np.ones(shape) * 0.8
        cal = np.ones(shape) * 0.95
        sci_err = np.ones(shape) * 0.01
        cal_err = np.ones(shape) * 0.02
        result, sigma = _calibrate_division(sci, sci_err, cal, cal_err)
        assert result.shape == shape
        assert sigma.shape == shape


class TestGetCalibrationValuesConstant:
    """Tests for _get_calibration_values_constant."""

    def test_single_calibrator_returns_directly(self):
        """Single calibrator returns its values unchanged."""
        val = np.array([[1.0, 2.0, 3.0]])
        err = np.array([[0.1, 0.2, 0.3]])
        result_val, result_err = _get_calibration_values_constant([val], [err])
        np.testing.assert_allclose(result_val, val)
        np.testing.assert_allclose(result_err, err)

    def test_two_cals_equal_errors_unweighted_mean(self):
        """Equal errors give unweighted mean."""
        val1 = np.array([[4.0]])
        val2 = np.array([[6.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        result_val, result_err = _get_calibration_values_constant(
            [val1, val2], [err1, err2]
        )
        # Weighted mean with equal weights = arithmetic mean
        np.testing.assert_allclose(result_val, [[5.0]])
        # sigma = 1/sqrt(1/1 + 1/1) = 1/sqrt(2)
        np.testing.assert_allclose(result_err, [[1.0 / np.sqrt(2.0)]])

    def test_two_cals_unequal_errors_weighted_mean(self):
        """Unequal errors give inverse-variance weighted mean."""
        val1 = np.array([[10.0]])
        val2 = np.array([[20.0]])
        err1 = np.array([[1.0]])  # w1 = 1
        err2 = np.array([[2.0]])  # w2 = 0.25
        result_val, result_err = _get_calibration_values_constant(
            [val1, val2], [err1, err2]
        )
        # mu = (1*10 + 0.25*20) / (1 + 0.25) = 15/1.25 = 12
        expected_val = (10.0 * 1.0 + 20.0 * 0.25) / (1.0 + 0.25)
        np.testing.assert_allclose(result_val, [[expected_val]])
        # sigma = 1/sqrt(1 + 0.25) = 1/sqrt(1.25)
        expected_err = 1.0 / np.sqrt(1.25)
        np.testing.assert_allclose(result_err, [[expected_err]])

    def test_three_cals_weighted_mean(self):
        """Three calibrators weighted mean verified."""
        val1 = np.array([[5.0]])
        val2 = np.array([[10.0]])
        val3 = np.array([[15.0]])
        err1 = np.array([[1.0]])  # w1 = 1.0
        err2 = np.array([[2.0]])  # w2 = 0.25
        err3 = np.array([[0.5]])  # w3 = 4.0
        result_val, result_err = _get_calibration_values_constant(
            [val1, val2, val3], [err1, err2, err3]
        )
        w_sum = 1.0 + 0.25 + 4.0
        expected_val = (5.0 * 1.0 + 10.0 * 0.25 + 15.0 * 4.0) / w_sum
        expected_err = 1.0 / np.sqrt(w_sum)
        np.testing.assert_allclose(result_val, [[expected_val]])
        np.testing.assert_allclose(result_err, [[expected_err]])

    def test_uncertainty_formula(self):
        """Uncertainty is 1/sqrt(sum(1/sigma_i^2))."""
        err1 = np.array([[2.0]])
        err2 = np.array([[3.0]])
        val1 = np.array([[1.0]])
        val2 = np.array([[1.0]])
        _, result_err = _get_calibration_values_constant(
            [val1, val2], [err1, err2]
        )
        sum_inv_var = 1.0 / 4.0 + 1.0 / 9.0
        expected = 1.0 / np.sqrt(sum_inv_var)
        np.testing.assert_allclose(result_err, [[expected]])

    def test_fallback_when_sigma_zero(self):
        """Falls back to unweighted mean when sigma=0."""
        val1 = np.array([[4.0]])
        val2 = np.array([[8.0]])
        err1 = np.array([[0.0]])  # Zero sigma -> inf weight
        err2 = np.array([[2.0]])
        result_val, result_err = _get_calibration_values_constant(
            [val1, val2], [err1, err2]
        )
        # Fallback: unweighted mean
        np.testing.assert_allclose(result_val, [[6.0]])
        # Fallback error: sqrt(sum(sigma_i^2)) / N
        expected_err = np.sqrt(0.0 + 4.0) / 2.0
        np.testing.assert_allclose(result_err, [[expected_err]])

    def test_fallback_when_sigma_nan(self):
        """Falls back to unweighted mean when sigma=NaN."""
        val1 = np.array([[4.0]])
        val2 = np.array([[8.0]])
        err1 = np.array([[np.nan]])
        err2 = np.array([[2.0]])
        result_val, result_err = _get_calibration_values_constant(
            [val1, val2], [err1, err2]
        )
        # Unweighted mean of values
        np.testing.assert_allclose(result_val, [[6.0]])
        # NaN in errors propagates through fallback formula
        assert np.isnan(result_err[0, 0])


class TestGetCalibrationValuesLinear:
    """Tests for _get_calibration_values_linear."""

    def test_two_cals_midpoint_interpolation(self):
        """Science at midpoint gets interpolated value."""
        val1 = np.array([[10.0]])
        val2 = np.array([[20.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        cal_times = np.array([100.0, 200.0])
        sci_time = 150.0  # midpoint
        result_val, result_err = _get_calibration_values_linear(
            [val1, val2], [err1, err2], cal_times, sci_time
        )
        np.testing.assert_allclose(result_val, [[15.0]])
        # f = 0.5; sigma = sqrt(0.25*1 + 0.25*1) = sqrt(0.5)
        expected_err = np.sqrt(0.5)
        np.testing.assert_allclose(result_err, [[expected_err]])

    def test_two_cals_at_t1_returns_val1(self):
        """Science at t_1 returns x_1."""
        val1 = np.array([[10.0]])
        val2 = np.array([[20.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[2.0]])
        cal_times = np.array([100.0, 200.0])
        sci_time = 100.0  # at t_1
        result_val, result_err = _get_calibration_values_linear(
            [val1, val2], [err1, err2], cal_times, sci_time
        )
        # f = 0; value = (1-0)*10 + 0*20 = 10
        np.testing.assert_allclose(result_val, [[10.0]])
        # sigma = sqrt(1^2 * 1^2 + 0) = 1.0
        np.testing.assert_allclose(result_err, [[1.0]])

    def test_two_cals_extrapolation_before(self):
        """Science before calibrators extrapolates correctly."""
        val1 = np.array([[10.0]])
        val2 = np.array([[20.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        cal_times = np.array([200.0, 300.0])
        sci_time = 100.0  # before t_1
        result_val, _ = _get_calibration_values_linear(
            [val1, val2], [err1, err2], cal_times, sci_time
        )
        # f = (100 - 200) / (300 - 200) = -1.0
        # value = (1-(-1))*10 + (-1)*20 = 2*10 - 20 = 0
        np.testing.assert_allclose(result_val, [[0.0]])

    def test_two_cals_extrapolation_after(self):
        """Science after calibrators extrapolates correctly."""
        val1 = np.array([[10.0]])
        val2 = np.array([[20.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        cal_times = np.array([100.0, 200.0])
        sci_time = 300.0  # after t_2
        result_val, _ = _get_calibration_values_linear(
            [val1, val2], [err1, err2], cal_times, sci_time
        )
        # f = (300 - 100) / (200 - 100) = 2.0
        # value = (1-2)*10 + 2*20 = -10 + 40 = 30
        np.testing.assert_allclose(result_val, [[30.0]])

    def test_uncertainty_at_midpoint(self):
        """Uncertainty at midpoint: sqrt(0.25*s1^2 + 0.25*s2^2)."""
        val1 = np.array([[0.0]])
        val2 = np.array([[0.0]])
        err1 = np.array([[2.0]])
        err2 = np.array([[4.0]])
        cal_times = np.array([0.0, 100.0])
        sci_time = 50.0
        _, result_err = _get_calibration_values_linear(
            [val1, val2], [err1, err2], cal_times, sci_time
        )
        # f = 0.5; sigma = sqrt(0.25*4 + 0.25*16) = sqrt(5)
        expected = np.sqrt(0.25 * 4.0 + 0.25 * 16.0)
        np.testing.assert_allclose(result_err, [[expected]])

    def test_three_cals_constant_observable(self):
        """Three calibrators with constant value gives constant."""
        val1 = np.array([[5.0]])
        val2 = np.array([[5.0]])
        val3 = np.array([[5.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        err3 = np.array([[1.0]])
        cal_times = np.array([100.0, 200.0, 300.0])
        sci_time = 250.0
        result_val, _ = _get_calibration_values_linear(
            [val1, val2, val3],
            [err1, err2, err3],
            cal_times,
            sci_time,
        )
        np.testing.assert_allclose(result_val, [[5.0]], atol=1e-10)

    def test_three_cals_known_linear_slope(self):
        """Three calibrators on a line give correct regression."""
        # y = 2 + 0.1*t
        cal_times = np.array([0.0, 100.0, 200.0])
        vals = [
            np.array([[2.0]]),
            np.array([[12.0]]),
            np.array([[22.0]]),
        ]
        errs = [
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
        ]
        sci_time = 150.0
        result_val, _ = _get_calibration_values_linear(
            vals, errs, cal_times, sci_time
        )
        # y(150) = 2 + 0.1*150 = 17
        np.testing.assert_allclose(result_val, [[17.0]], atol=1e-10)

    def test_degenerate_same_time_falls_back(self):
        """All cals at same time fall back to constant."""
        val1 = np.array([[4.0]])
        val2 = np.array([[6.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        cal_times = np.array([100.0, 100.0])
        sci_time = 150.0
        with pytest.warns(UserWarning, match="same time"):
            result_val, _ = _get_calibration_values_linear(
                [val1, val2],
                [err1, err2],
                cal_times,
                sci_time,
            )
        # Falls back to constant: mean = 5
        np.testing.assert_allclose(result_val, [[5.0]])

    def test_fallback_for_zero_weights(self):
        """Zero weights fall back to constant for those elements."""
        # Mixed: one element has zero error (invalid weight),
        # another is fine
        val1 = np.array([[4.0, 10.0]])
        val2 = np.array([[8.0, 20.0]])
        err1 = np.array([[0.0, 1.0]])  # Zero for first element
        err2 = np.array([[2.0, 1.0]])
        cal_times = np.array([100.0, 200.0])
        sci_time = 150.0
        result_val, _ = _get_calibration_values_linear(
            [val1, val2], [err1, err2], cal_times, sci_time
        )
        # First element: fallback (f=0.5 still works for
        # two-calibrator exact interpolation regardless of
        # weights since it's not regression)
        # For two cals, we use exact interpolation, not
        # regression. Weights don't matter.
        np.testing.assert_allclose(result_val[0, 0], 6.0, atol=1e-10)
        # Second element: normal interpolation
        np.testing.assert_allclose(result_val[0, 1], 15.0, atol=1e-10)

    def test_three_cals_degenerate_falls_back_with_warning(self):
        """Three cals at same time fall back to constant."""
        val1 = np.array([[3.0]])
        val2 = np.array([[5.0]])
        val3 = np.array([[7.0]])
        err1 = np.array([[1.0]])
        err2 = np.array([[1.0]])
        err3 = np.array([[1.0]])
        cal_times = np.array([100.0, 100.0, 100.0])
        sci_time = 150.0
        with pytest.warns(UserWarning, match="same time"):
            result_val, _ = _get_calibration_values_linear(
                [val1, val2, val3],
                [err1, err2, err3],
                cal_times,
                sci_time,
            )
        # Falls back to constant weighted mean (equal weights)
        np.testing.assert_allclose(result_val, [[5.0]])
