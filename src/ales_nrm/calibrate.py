"""Native backend for calibration of interferometric observables.

Calibrates science block observables against calibrator blocks. Supports
closure phases, squared visibilities, visibility phases, and visibility
amplitudes using constant (zeroth-order) or linear (first-order)
calibration models.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

_VALID_OBSERVABLES = frozenset({"t3phi", "vis2", "visphi", "visamp"})

_SUBTRACTION_OBSERVABLES = frozenset({"t3phi", "visphi"})
_DIVISION_OBSERVABLES = frozenset({"vis2", "visamp"})

_OBS_ATTR_MAP = {
    "t3phi": {
        "value": "t3phi",
        "error": "t3phi_err",
        "flag": "t3_flag",
    },
    "vis2": {
        "value": "vis2",
        "error": "vis2_err",
        "flag": "vis2_flag",
    },
    "visphi": {
        "value": "visphi",
        "error": "visphi_err",
        "flag": "vis_flag",
    },
    "visamp": {
        "value": "visamp",
        "error": "visamp_err",
        "flag": "vis_flag",
    },
}


def _calibrate_subtraction(
    sci_value: np.ndarray,
    sci_err: np.ndarray,
    cal_value: np.ndarray,
    cal_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate by subtraction: calibrated = sci - cal.

    Used for closure phases and visibility phases.

    Args:
        sci_value: Science observable values, shape
            ``(n_obs, n_wav)``.
        sci_err: Science uncertainties, same shape.
        cal_value: Calibrator values (mean or interpolated), same shape.
        cal_err: Calibrator uncertainties, same shape.

    Returns:
        Tuple of (calibrated_value, calibrated_error), each
        with the same shape as input arrays.
    """
    calibrated = sci_value - cal_value
    sigma = np.sqrt(sci_err**2 + cal_err**2)
    return calibrated, sigma


def _calibrate_division(
    sci_value: np.ndarray,
    sci_err: np.ndarray,
    cal_value: np.ndarray,
    cal_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate by division: calibrated = sci / cal.

    Used for squared visibilities and visibility amplitudes.

    Error propagation for f = a/b:
        sigma_f = sqrt((sigma_a/b)^2 + (a * sigma_b / b^2)^2)

    Where cal_value == 0, both calibrated and error are set
    to NaN.

    Args:
        sci_value: Science observable values, shape
            ``(n_obs, n_wav)``.
        sci_err: Science uncertainties, same shape.
        cal_value: Calibrator values (mean or interpolated), same shape.
        cal_err: Calibrator uncertainties, same shape.

    Returns:
        Tuple of (calibrated_value, calibrated_error), each
        with the same shape as input arrays.
    """
    # Initialize with NaN for safety
    calibrated = np.full_like(sci_value, np.nan, dtype=np.float64)
    sigma = np.full_like(sci_value, np.nan, dtype=np.float64)

    # Mask where calibrator is non-zero
    valid = cal_value != 0.0

    calibrated[valid] = sci_value[valid] / cal_value[valid]
    sigma[valid] = np.sqrt(
        (sci_err[valid] / cal_value[valid]) ** 2
        + (sci_value[valid] * cal_err[valid] / cal_value[valid] ** 2) ** 2
    )

    return calibrated, sigma


def _get_calibration_values_constant(
    cal_values: list[np.ndarray],
    cal_errors: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration values using constant (order-0) model.

    For a single calibrator, returns its values directly. For
    multiple calibrators, computes the inverse-variance weighted
    mean. Falls back to unweighted mean for elements where any
    weight is zero or NaN.

    Args:
        cal_values: List of calibrator value arrays, each of
            shape ``(n_obs, n_wav)``.
        cal_errors: List of calibrator uncertainty arrays,
            each of shape ``(n_obs, n_wav)``.

    Returns:
        Tuple of (mean_value, mean_error), each of shape
        ``(n_obs, n_wav)``.
    """
    n_cal = len(cal_values)

    if n_cal == 1:
        return cal_values[0].copy(), cal_errors[0].copy()

    # Stack to (n_cal, n_obs, n_wav)
    values = np.stack(cal_values, axis=0)
    errors = np.stack(cal_errors, axis=0)

    # Compute weights: w_i = 1 / sigma_i^2
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / errors**2

    # Identify elements where weighted mean is valid
    # (all weights finite and positive)
    valid = np.all(np.isfinite(weights) & (weights > 0), axis=0)

    # Weighted mean
    sum_w = np.sum(weights, axis=0)
    sum_wx = np.sum(weights * values, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_mean = np.where(
            valid,
            sum_wx / sum_w,
            np.nan,
        )
        weighted_err = np.where(
            valid,
            1.0 / np.sqrt(sum_w),
            np.nan,
        )

    # Unweighted fallback for invalid elements
    if not np.all(valid):
        unweighted_mean = np.mean(values, axis=0)
        unweighted_err = np.sqrt(np.sum(errors**2, axis=0)) / n_cal
        weighted_mean = np.where(valid, weighted_mean, unweighted_mean)
        weighted_err = np.where(valid, weighted_err, unweighted_err)

    return weighted_mean, weighted_err


def _get_calibration_values_linear(
    cal_values: list[np.ndarray],
    cal_errors: list[np.ndarray],
    cal_times: np.ndarray,
    sci_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration values using linear (order-1) model.

    For exactly two calibrators, uses exact linear
    interpolation/extrapolation. For more than two, uses
    weighted linear regression (least squares).

    Falls back to constant method (with warning) when
    calibrators are at the same time (degenerate case).

    Args:
        cal_values: List of calibrator value arrays, each of
            shape ``(n_obs, n_wav)``.
        cal_errors: List of calibrator uncertainty arrays,
            each of shape ``(n_obs, n_wav)``.
        cal_times: 1D array of calibrator mean times in
            seconds, shape ``(n_cal,)``.
        sci_time: Science block mean time in seconds.

    Returns:
        Tuple of (interpolated_value, interpolated_error),
        each of shape ``(n_obs, n_wav)``.
    """
    n_cal = len(cal_values)

    if n_cal == 2:
        return _linear_interpolation_two(
            cal_values, cal_errors, cal_times, sci_time
        )

    return _linear_regression_multi(
        cal_values, cal_errors, cal_times, sci_time
    )


def _linear_interpolation_two(
    cal_values: list[np.ndarray],
    cal_errors: list[np.ndarray],
    cal_times: np.ndarray,
    sci_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation/extrapolation for two calibrators.

    Args:
        cal_values: Two calibrator value arrays.
        cal_errors: Two calibrator uncertainty arrays.
        cal_times: Times of the two calibrators.
        sci_time: Science time.

    Returns:
        Tuple of (value, error) at sci_time.
    """
    dt = cal_times[1] - cal_times[0]

    if abs(dt) < 1e-10:
        warnings.warn(
            "Calibrators are at the same time. Falling "
            "back to constant calibration model.",
            UserWarning,
            stacklevel=3,
        )
        return _get_calibration_values_constant(cal_values, cal_errors)

    f = (sci_time - cal_times[0]) / dt

    value = (1.0 - f) * cal_values[0] + f * cal_values[1]
    error = np.sqrt(
        (1.0 - f)**2 * cal_errors[0]**2 + f**2 * cal_errors[1]**2
    )

    return value, error


def _linear_regression_multi(
    cal_values: list[np.ndarray],
    cal_errors: list[np.ndarray],
    cal_times: np.ndarray,
    sci_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted linear regression for >2 calibrators.

    Performs per-element weighted least squares: y = a + b*t.
    Falls back to constant method for elements where the
    regression is degenerate.

    Args:
        cal_values: List of calibrator value arrays.
        cal_errors: List of calibrator uncertainty arrays.
        cal_times: Times of the calibrators.
        sci_time: Science time.

    Returns:
        Tuple of (value, error) at sci_time.
    """
    n_cal = len(cal_values)

    # Stack: (n_cal, n_obs, n_wav)
    values = np.stack(cal_values, axis=0)
    errors = np.stack(cal_errors, axis=0)

    # Weights: (n_cal, n_obs, n_wav)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / errors**2

    # Check validity of weights per element
    all_valid = np.all(np.isfinite(weights) & (weights > 0), axis=0)

    # Times: (n_cal,) -> broadcast-ready (n_cal, 1, 1)
    t = cal_times.reshape(n_cal, 1, 1)

    # Components of normal equations M = X^T W X
    # M = [[sum_w, sum_wt], [sum_wt, sum_wt2]]
    sum_w = np.sum(weights, axis=0)
    sum_wt = np.sum(weights * t, axis=0)
    sum_wt2 = np.sum(weights * t**2, axis=0)

    # Determinant
    det = sum_w * sum_wt2 - sum_wt**2

    # Check for degenerate cases (det ~= 0)
    det_valid = np.abs(det) > 1e-20
    valid = all_valid & det_valid

    # Right-hand side: X^T W y
    sum_wy = np.sum(weights * values, axis=0)
    sum_wty = np.sum(weights * t * values, axis=0)

    # Covariance matrix elements C = M_inv
    # C[0,0] = sum_wt2 / det
    # C[0,1] = C[1,0] = -sum_wt / det
    # C[1,1] = sum_w / det
    # Use safe division
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_det = np.where(valid, 1.0 / det, 0.0)

    c00 = sum_wt2 * inv_det
    c01 = -sum_wt * inv_det
    c11 = sum_w * inv_det

    # Coefficients: beta = C @ [sum_wy, sum_wty]
    beta0 = c00 * sum_wy + c01 * sum_wty
    beta1 = c01 * sum_wy + c11 * sum_wty

    # Evaluate at sci_time
    reg_value = beta0 + beta1 * sci_time
    reg_error = np.sqrt(c00 + 2.0 * sci_time * c01 + sci_time**2 * c11)

    # Fallback to constant for invalid elements
    if not np.all(valid):
        if not np.any(valid):
            warnings.warn(
                "Calibrators are at the same time. Falling "
                "back to constant calibration model.",
                UserWarning,
                stacklevel=3,
            )
        const_value, const_error = _get_calibration_values_constant(
            cal_values, cal_errors
        )
        result_value = np.where(valid, reg_value, const_value)
        result_error = np.where(valid, reg_error, const_error)
        return result_value, result_error

    return reg_value, reg_error
