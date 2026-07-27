"""Tests for ales_nrm.calibrate module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ales_nrm.calibrate import (
    _calibrate_division,
    _calibrate_subtraction,
    _get_calibration_values_constant,
    _get_calibration_values_linear,
    calibrate_block,
)
from ales_nrm.observables import (
    BaselineInfo,
    Observables,
    Station,
    TriangleInfo,
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
        # weights since it is not regression).
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


def _make_observables(
    *,
    target="sci_target",
    n_baselines=3,
    n_triangles=1,
    n_wav=2,
    t3phi=None,
    t3phi_err=None,
    vis2=None,
    vis2_err=None,
    visphi=None,
    visphi_err=None,
    visamp=None,
    visamp_err=None,
):
    """Create an Observables instance with given arrays."""
    wavelengths = np.linspace(2.7, 4.3, n_wav)
    stations = [
        Station(index=i + 1, name=f"H{i + 1}", x=0.0, y=0.0) for i in range(3)
    ]
    baselines = [
        BaselineInfo(sta_index=(1, 2), name="H1H2", u=1.0, v=0.0),
        BaselineInfo(sta_index=(1, 3), name="H1H3", u=0.5, v=0.8),
        BaselineInfo(sta_index=(2, 3), name="H2H3", u=-0.5, v=0.8),
    ][:n_baselines]
    triangles = [
        TriangleInfo(
            sta_index=(1, 2, 3),
            name="H1-H2-H3",
            u1=1.0,
            v1=0.0,
            u2=-0.5,
            v2=0.8,
        ),
    ][:n_triangles]

    return Observables(
        target=target,
        wavelengths=wavelengths,
        stations=stations,
        baselines=baselines,
        triangles=triangles,
        mean_para_angle=30.0,
        mjd=60000.0,
        time_start="08:00:00.000",
        time_end="08:05:00.000",
        block_type="SCI",
        mask_name="test_mask",
        t3phi=t3phi,
        t3phi_err=t3phi_err,
        vis2=vis2,
        vis2_err=vis2_err,
        visphi=visphi,
        visphi_err=visphi_err,
        visamp=visamp,
        visamp_err=visamp_err,
    )


def _make_block(
    target="sci_target",
    timestamps=None,
    obs=None,
    source_label="raw",
):
    """Create a mock ObservingBlock with observables."""
    block = MagicMock()
    block.target = target
    if timestamps is None:
        timestamps = np.array(["08:00:00.000", "08:05:00.000"], dtype=object)
    block.timestamps = timestamps
    block.observables = {}
    if obs is not None:
        block.observables[source_label] = obs
    return block


class TestCalibrateBlockConstant:
    """Tests for calibrate_block with method='constant'."""

    def test_single_cal_all_observables(self):
        """Single calibrator calibrates all four observables."""
        n_bl, n_tri, n_wav = 3, 1, 2

        sci_obs = _make_observables(
            t3phi=np.full((n_tri, n_wav), 5.0),
            t3phi_err=np.full((n_tri, n_wav), 1.0),
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
            visphi=np.full((n_bl, n_wav), 10.0),
            visphi_err=np.full((n_bl, n_wav), 2.0),
            visamp=np.full((n_bl, n_wav), 0.9),
            visamp_err=np.full((n_bl, n_wav), 0.03),
        )
        cal_obs = _make_observables(
            target="cal_target",
            t3phi=np.full((n_tri, n_wav), 2.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
            visphi=np.full((n_bl, n_wav), 3.0),
            visphi_err=np.full((n_bl, n_wav), 1.0),
            visamp=np.full((n_bl, n_wav), 0.98),
            visamp_err=np.full((n_bl, n_wav), 0.01),
        )

        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal_target", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block])

        assert result.t3phi is not None
        assert result.vis2 is not None
        assert result.visphi is not None
        assert result.visamp is not None
        assert result.calibrated is True

    def test_multiple_cals_weighted_mean(self):
        """Multiple calibrators use weighted mean."""
        n_bl, _n_tri, n_wav = 3, 1, 2

        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs1 = _make_observables(
            target="cal1",
            vis2=np.full((n_bl, n_wav), 0.9),
            vis2_err=np.full((n_bl, n_wav), 0.01),
        )
        cal_obs2 = _make_observables(
            target="cal2",
            vis2=np.full((n_bl, n_wav), 1.0),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )

        sci_block = _make_block(obs=sci_obs)
        cal1 = _make_block(target="cal1", obs=cal_obs1)
        cal2 = _make_block(target="cal2", obs=cal_obs2)

        result = calibrate_block(sci_block, [cal1, cal2], calibrate={"vis2"})

        # Weighted mean of 0.9 (w=10000) and 1.0 (w=2500)
        w1 = 1.0 / 0.01**2
        w2 = 1.0 / 0.02**2
        expected_cal = (0.9 * w1 + 1.0 * w2) / (w1 + w2)
        expected_result = 0.8 / expected_cal
        np.testing.assert_allclose(
            result.vis2[0, 0], expected_result, rtol=1e-10
        )

    def test_cp_subtraction_verified(self):
        """CP calibration is subtraction."""
        sci_obs = _make_observables(
            t3phi=np.array([[5.0, 10.0]]),
            t3phi_err=np.array([[1.0, 1.0]]),
        )
        cal_obs = _make_observables(
            target="cal",
            t3phi=np.array([[2.0, 3.0]]),
            t3phi_err=np.array([[0.5, 0.5]]),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block], calibrate={"t3phi"})

        np.testing.assert_allclose(result.t3phi, [[3.0, 7.0]])
        expected_err = np.sqrt(1.0**2 + 0.5**2)
        np.testing.assert_allclose(
            result.t3phi_err, [[expected_err, expected_err]]
        )

    def test_vis2_division_verified(self):
        """VIS2 calibration is division."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs = _make_observables(
            target="cal",
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block], calibrate={"vis2"})

        expected = 0.8 / 0.95
        np.testing.assert_allclose(result.vis2, expected, rtol=1e-10)

    def test_visphi_subtraction_verified(self):
        """Visphi calibration is subtraction."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            visphi=np.full((n_bl, n_wav), 15.0),
            visphi_err=np.full((n_bl, n_wav), 2.0),
        )
        cal_obs = _make_observables(
            target="cal",
            visphi=np.full((n_bl, n_wav), 5.0),
            visphi_err=np.full((n_bl, n_wav), 1.0),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block], calibrate={"visphi"})

        np.testing.assert_allclose(result.visphi, 10.0)

    def test_visamp_division_verified(self):
        """Visamp calibration is division."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            visamp=np.full((n_bl, n_wav), 0.85),
            visamp_err=np.full((n_bl, n_wav), 0.03),
        )
        cal_obs = _make_observables(
            target="cal",
            visamp=np.full((n_bl, n_wav), 0.97),
            visamp_err=np.full((n_bl, n_wav), 0.01),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block], calibrate={"visamp"})

        expected = 0.85 / 0.97
        np.testing.assert_allclose(result.visamp, expected, rtol=1e-10)

    def test_selective_calibrate_t3phi_only(self):
        """calibrate={'t3phi'} leaves others as None."""
        n_bl, n_tri, n_wav = 3, 1, 2
        sci_obs = _make_observables(
            t3phi=np.full((n_tri, n_wav), 5.0),
            t3phi_err=np.full((n_tri, n_wav), 1.0),
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 2.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block], calibrate={"t3phi"})

        assert result.t3phi is not None
        assert result.vis2 is None
        assert result.visphi is None
        assert result.visamp is None

    def test_selective_calibrate_vis2_visamp(self):
        """calibrate={'vis2', 'visamp'} calibrates only those."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
            visamp=np.full((n_bl, n_wav), 0.9),
            visamp_err=np.full((n_bl, n_wav), 0.03),
        )
        cal_obs = _make_observables(
            target="cal",
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
            visamp=np.full((n_bl, n_wav), 0.98),
            visamp_err=np.full((n_bl, n_wav), 0.01),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        result = calibrate_block(
            sci_block,
            [cal_block],
            calibrate={"vis2", "visamp"},
        )

        assert result.vis2 is not None
        assert result.visamp is not None
        assert result.t3phi is None
        assert result.visphi is None

    def test_missing_observable_on_science_warns(self):
        """Missing observable on science issues warning."""
        n_bl, n_wav = 3, 2
        # Science has no t3phi
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs = _make_observables(
            target="cal",
            t3phi=np.full((1, n_wav), 2.0),
            t3phi_err=np.full((1, n_wav), 0.5),
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        with pytest.warns(UserWarning, match="not present"):
            result = calibrate_block(
                sci_block,
                [cal_block],
                calibrate={"t3phi", "vis2"},
            )

        assert result.t3phi is None
        assert result.vis2 is not None

    def test_empty_cal_blocks_raises(self):
        """Empty cal_blocks raises ValueError."""
        sci_obs = _make_observables()
        sci_block = _make_block(obs=sci_obs)

        with pytest.raises(ValueError, match="At least one calibrator"):
            calibrate_block(sci_block, [])

    def test_empty_calibrate_set_raises(self):
        """calibrate=set() raises ValueError."""
        sci_obs = _make_observables()
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(
            target="cal", obs=_make_observables(target="cal")
        )

        with pytest.raises(ValueError, match="must not be empty"):
            calibrate_block(sci_block, [cal_block], calibrate=set())

    def test_invalid_key_in_calibrate_raises(self):
        """Invalid key in calibrate raises ValueError."""
        sci_obs = _make_observables()
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(
            target="cal", obs=_make_observables(target="cal")
        )

        with pytest.raises(ValueError, match="Invalid observable"):
            calibrate_block(
                sci_block,
                [cal_block],
                calibrate={"invalid_key"},
            )

    def test_source_label_missing_on_science_raises(self):
        """source_label missing on science raises ValueError."""
        sci_block = _make_block()  # no observables
        cal_block = _make_block(
            target="cal", obs=_make_observables(target="cal")
        )

        with pytest.raises(ValueError, match="source_label"):
            calibrate_block(sci_block, [cal_block])

    def test_source_label_missing_on_calibrator_raises(self):
        """source_label missing on calibrator raises ValueError."""
        sci_obs = _make_observables()
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal")  # no observables

        with pytest.raises(ValueError, match="source_label"):
            calibrate_block(sci_block, [cal_block])

    def test_output_metadata(self):
        """Output has correct metadata fields."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs = _make_observables(
            target="cal_star",
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal_star", obs=cal_obs)

        result = calibrate_block(sci_block, [cal_block], calibrate={"vis2"})

        assert result.calibrated is True
        assert result.extraction_backend == "native"
        assert result.calibrator_target == "cal_star"
        assert "method='constant'" in result.notes
        assert "n_calibrators=1" in result.notes
        assert "source_label='raw'" in result.notes

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        sci_obs = _make_observables()
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(
            target="cal", obs=_make_observables(target="cal")
        )

        with pytest.raises(ValueError, match="Unknown calibration"):
            calibrate_block(
                sci_block,
                [cal_block],
                method="quadratic",
            )

    def test_multiple_cal_targets_in_metadata(self):
        """Calibrator target string joins unique names."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs_a = _make_observables(
            target="cal_A",
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        cal_obs_b = _make_observables(
            target="cal_B",
            vis2=np.full((n_bl, n_wav), 0.97),
            vis2_err=np.full((n_bl, n_wav), 0.01),
        )

        sci_block = _make_block(obs=sci_obs)
        cal_a = _make_block(target="cal_A", obs=cal_obs_a)
        cal_b = _make_block(target="cal_B", obs=cal_obs_b)

        result = calibrate_block(
            sci_block,
            [cal_a, cal_b],
            calibrate={"vis2"},
        )

        assert "cal_A" in result.calibrator_target
        assert "cal_B" in result.calibrator_target


class TestCalibrateBlockLinear:
    """Tests for calibrate_block with method='linear'."""

    def test_two_cals_interpolation_cp(self):
        """Two calibrators, interpolation for CP."""
        n_tri, n_wav = 1, 2
        sci_obs = _make_observables(
            t3phi=np.full((n_tri, n_wav), 6.0),
            t3phi_err=np.full((n_tri, n_wav), 1.0),
        )
        # Cal at t=80s: CP=2, Cal at t=260s: CP=4
        cal_obs1 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 2.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )
        cal_obs2 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 4.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )

        # Science at midpoint t=160s
        sci_block = _make_block(
            obs=sci_obs,
            timestamps=np.array(
                ["08:02:20.000", "08:02:20.000"], dtype=object
            ),
        )
        cal1 = _make_block(
            target="cal",
            obs=cal_obs1,
            timestamps=np.array(
                ["08:01:00.000", "08:01:40.000"], dtype=object
            ),
        )
        cal2 = _make_block(
            target="cal",
            obs=cal_obs2,
            timestamps=np.array(
                ["08:03:00.000", "08:03:40.000"], dtype=object
            ),
        )

        result = calibrate_block(
            sci_block,
            [cal1, cal2],
            method="linear",
            calibrate={"t3phi"},
        )

        # Interpolated cal at midpoint = 3.0
        # Calibrated = 6.0 - 3.0 = 3.0
        np.testing.assert_allclose(result.t3phi, 3.0, atol=0.01)

    def test_two_cals_interpolation_vis2(self):
        """Two calibrators, interpolation for VIS2."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs1 = _make_observables(
            target="cal",
            vis2=np.full((n_bl, n_wav), 0.90),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        cal_obs2 = _make_observables(
            target="cal",
            vis2=np.full((n_bl, n_wav), 1.00),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )

        sci_block = _make_block(
            obs=sci_obs,
            timestamps=np.array(
                ["08:02:30.000", "08:02:30.000"], dtype=object
            ),
        )
        cal1 = _make_block(
            target="cal",
            obs=cal_obs1,
            timestamps=np.array(
                ["08:01:00.000", "08:01:40.000"], dtype=object
            ),
        )
        cal2 = _make_block(
            target="cal",
            obs=cal_obs2,
            timestamps=np.array(
                ["08:03:00.000", "08:03:40.000"], dtype=object
            ),
        )

        result = calibrate_block(
            sci_block,
            [cal1, cal2],
            method="linear",
            calibrate={"vis2"},
        )

        # Interpolated cal at midpoint = 0.95
        # Calibrated = 0.8 / 0.95
        np.testing.assert_allclose(result.vis2, 0.8 / 0.95, atol=0.01)

    def test_extrapolation_before_cals(self):
        """Science before calibrators extrapolates."""
        n_tri, n_wav = 1, 2
        sci_obs = _make_observables(
            t3phi=np.full((n_tri, n_wav), 10.0),
            t3phi_err=np.full((n_tri, n_wav), 1.0),
        )
        cal_obs1 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 4.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )
        cal_obs2 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 6.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )

        # Science before both calibrators
        sci_block = _make_block(
            obs=sci_obs,
            timestamps=np.array(["08:00:00.000"], dtype=object),
        )
        cal1 = _make_block(
            target="cal",
            obs=cal_obs1,
            timestamps=np.array(["08:02:00.000"], dtype=object),
        )
        cal2 = _make_block(
            target="cal",
            obs=cal_obs2,
            timestamps=np.array(["08:04:00.000"], dtype=object),
        )

        result = calibrate_block(
            sci_block,
            [cal1, cal2],
            method="linear",
            calibrate={"t3phi"},
        )

        # t1=120s, t2=240s, sci=0s
        # f = (0 - 120) / (240 - 120) = -1
        # cal = 2*4 + (-1)*6 = 8-6 = 2
        # calibrated = 10 - 2 = 8
        np.testing.assert_allclose(result.t3phi, 8.0, atol=0.01)

    def test_three_cals_linear_trend(self):
        """Three calibrators with linear trend."""
        n_tri, n_wav = 1, 2
        sci_obs = _make_observables(
            t3phi=np.full((n_tri, n_wav), 20.0),
            t3phi_err=np.full((n_tri, n_wav), 1.0),
        )
        # Linear trend: y = 1 + 0.1*t (t in seconds)
        # t1=0, t2=60, t3=120
        cal_obs1 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 1.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )
        cal_obs2 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 7.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )
        cal_obs3 = _make_observables(
            target="cal",
            t3phi=np.full((n_tri, n_wav), 13.0),
            t3phi_err=np.full((n_tri, n_wav), 0.5),
        )

        sci_block = _make_block(
            obs=sci_obs,
            timestamps=np.array(["08:01:30.000"], dtype=object),
        )
        cal1 = _make_block(
            target="cal",
            obs=cal_obs1,
            timestamps=np.array(["08:00:00.000"], dtype=object),
        )
        cal2 = _make_block(
            target="cal",
            obs=cal_obs2,
            timestamps=np.array(["08:01:00.000"], dtype=object),
        )
        cal3 = _make_block(
            target="cal",
            obs=cal_obs3,
            timestamps=np.array(["08:02:00.000"], dtype=object),
        )

        result = calibrate_block(
            sci_block,
            [cal1, cal2, cal3],
            method="linear",
            calibrate={"t3phi"},
        )

        # sci at t=90s, linear trend: y=1+0.1*90=10
        # calibrated = 20 - 10 = 10
        np.testing.assert_allclose(result.t3phi, 10.0, atol=0.1)

    def test_single_cal_linear_raises(self):
        """Single calibrator with method='linear' raises."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        cal_obs = _make_observables(
            target="cal",
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        with pytest.raises(ValueError, match="at least 2"):
            calibrate_block(sci_block, [cal_block], method="linear")

    def test_uncertainty_propagation_linear(self):
        """Uncertainty propagation for linear interpolation."""
        sci_obs = _make_observables(
            n_wav=1,
            t3phi=np.array([[10.0]]),
            t3phi_err=np.array([[2.0]]),
        )
        cal_obs1 = _make_observables(
            n_wav=1,
            target="cal",
            t3phi=np.array([[3.0]]),
            t3phi_err=np.array([[1.0]]),
        )
        cal_obs2 = _make_observables(
            n_wav=1,
            target="cal",
            t3phi=np.array([[5.0]]),
            t3phi_err=np.array([[1.0]]),
        )

        # Science at midpoint
        sci_block = _make_block(
            obs=sci_obs,
            timestamps=np.array(["08:01:30.000"], dtype=object),
        )
        cal1 = _make_block(
            target="cal",
            obs=cal_obs1,
            timestamps=np.array(["08:00:00.000"], dtype=object),
        )
        cal2 = _make_block(
            target="cal",
            obs=cal_obs2,
            timestamps=np.array(["08:03:00.000"], dtype=object),
        )

        result = calibrate_block(
            sci_block,
            [cal1, cal2],
            method="linear",
            calibrate={"t3phi"},
        )

        # f=0.5, cal_err = sqrt(0.25*1 + 0.25*1) = sqrt(0.5)
        # total_err = sqrt(sci_err^2 + cal_err^2)
        #           = sqrt(4 + 0.5) = sqrt(4.5)
        expected_err = np.sqrt(4.0 + 0.5)
        np.testing.assert_allclose(
            result.t3phi_err[0, 0], expected_err, rtol=1e-5
        )

    def test_cal_calibrating_cal(self):
        """Any block can calibrate any other block."""
        n_bl, n_wav = 3, 2
        # Both are CAL blocks
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.88),
            vis2_err=np.full((n_bl, n_wav), 0.03),
        )
        cal_obs1 = _make_observables(
            target="cal1",
            vis2=np.full((n_bl, n_wav), 0.92),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        cal_obs2 = _make_observables(
            target="cal2",
            vis2=np.full((n_bl, n_wav), 0.96),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )

        sci_block = _make_block(
            target="cal_target",
            obs=sci_obs,
            timestamps=np.array(["08:01:30.000"], dtype=object),
        )
        cal1 = _make_block(
            target="cal1",
            obs=cal_obs1,
            timestamps=np.array(["08:00:00.000"], dtype=object),
        )
        cal2 = _make_block(
            target="cal2",
            obs=cal_obs2,
            timestamps=np.array(["08:03:00.000"], dtype=object),
        )

        # Should not raise
        result = calibrate_block(
            sci_block,
            [cal1, cal2],
            method="linear",
            calibrate={"vis2"},
        )
        assert result.vis2 is not None

    def test_missing_error_on_science_warns_and_skips(self):
        """Missing error on science issues warning, skips."""
        n_bl, n_wav = 3, 2
        # Science has vis2 value but no vis2_err
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
        )
        cal_obs = _make_observables(
            target="cal",
            vis2=np.full((n_bl, n_wav), 0.95),
            vis2_err=np.full((n_bl, n_wav), 0.02),
        )
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        with pytest.warns(UserWarning, match="error not present"):
            result = calibrate_block(
                sci_block,
                [cal_block],
                calibrate={"vis2"},
            )

        assert result.vis2 is None

    def test_calibrator_missing_observable_raises(self):
        """Calibrator missing observable raises ValueError."""
        n_bl, n_wav = 3, 2
        sci_obs = _make_observables(
            vis2=np.full((n_bl, n_wav), 0.8),
            vis2_err=np.full((n_bl, n_wav), 0.05),
        )
        # Calibrator has no vis2
        cal_obs = _make_observables(target="cal")
        sci_block = _make_block(obs=sci_obs)
        cal_block = _make_block(target="cal", obs=cal_obs)

        with pytest.raises(ValueError, match="missing observable"):
            calibrate_block(
                sci_block,
                [cal_block],
                calibrate={"vis2"},
            )
