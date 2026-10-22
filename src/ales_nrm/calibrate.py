"""Native backend for calibration of interferometric observables.

Calibrates science block observables against calibrator blocks. Supports
closure phases, squared visibilities, visibility phases, and visibility
amplitudes using constant (zeroth-order) or linear (first-order)
calibration models.
"""

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from ales_nrm.observables import Observables
from ales_nrm.utilities import (
    _get_block_mean_time_seconds,
    _get_block_times,
)

if TYPE_CHECKING:
    from ales_nrm.observation import ObservingBlock

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
        (1.0 - f) ** 2 * cal_errors[0] ** 2 + f**2 * cal_errors[1] ** 2
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


def calibrate_block(
    sci_block: "ObservingBlock",
    cal_blocks: list["ObservingBlock"],
    *,
    source_label: str = "raw",
    method: str = "constant",
    calibrate: set[str] | None = None,
) -> "Observables":
    """Calibrate a science block against calibrator blocks.

    Removes instrumental systematics from interferometric
    observables using either a constant (zeroth-order) or linear
    (first-order) temporal calibration model.

    Args:
        sci_block: Block to calibrate (any block type).
        cal_blocks: List of calibrator blocks. Each must have
            ``Observables`` under ``source_label``.
        source_label: Label key in ``block.observables``.
        method: Calibration model. ``"constant"`` (zeroth-order)
            or ``"linear"`` (first-order).
        calibrate: Set of observable keys to calibrate.
            ``None`` (default) means all four:
            ``{"t3phi", "vis2", "visphi", "visamp"}``.
            Empty set raises ``ValueError``.

    Returns:
        Calibrated ``Observables`` instance.

    Raises:
        ValueError: If arguments are invalid or required data
            is missing.
    """
    # 1. Resolve calibrate set
    if calibrate is None:
        calibrate_set = set(_VALID_OBSERVABLES)
    else:
        calibrate_set = set(calibrate)
        if not calibrate_set:
            raise ValueError("calibrate set must not be empty.")
        invalid = calibrate_set - _VALID_OBSERVABLES
        if invalid:
            raise ValueError(
                f"Invalid observable keys: {sorted(invalid)}. "
                f"Valid keys: {sorted(_VALID_OBSERVABLES)}."
            )

    # 2. Validate method
    if method not in ("constant", "linear"):
        raise ValueError(
            f"Unknown calibration method '{method}'. "
            f"Supported: 'constant', 'linear'."
        )

    # 3. Validate cal_blocks not empty
    if not cal_blocks:
        raise ValueError("At least one calibrator block is required.")

    # 4. Validate linear requires >= 2 calibrators
    if method == "linear" and len(cal_blocks) < 2:
        raise ValueError(
            f"method='linear' requires at least 2 calibrator "
            f"blocks, but only {len(cal_blocks)} provided. "
            f"Use method='constant' for a single calibrator."
        )

    # 5. Validate source_label on science block
    if source_label not in sci_block.observables:
        raise ValueError(
            f"source_label '{source_label}' not found in "
            f"science block observables. Available: "
            f"{list(sci_block.observables.keys())}."
        )

    # 6. Validate source_label on each calibrator
    for i, cal_block in enumerate(cal_blocks):
        if source_label not in cal_block.observables:
            raise ValueError(
                f"source_label '{source_label}' not found "
                f"in calibrator block {i} "
                f"(target='{cal_block.target}'). Available: "
                f"{list(cal_block.observables.keys())}."
            )

    sci_obs = sci_block.observables[source_label]

    # Compute times for calibrators and science
    sci_time = _get_block_mean_time_seconds(sci_block)
    cal_times = _get_block_times(cal_blocks)

    # Prepare output arrays
    output_kwargs: dict[str, object] = {}

    for key in calibrate_set:
        attr_map = _OBS_ATTR_MAP[key]
        value_attr = attr_map["value"]
        error_attr = attr_map["error"]
        flag_attr = attr_map["flag"]

        sci_value = getattr(sci_obs, value_attr)
        sci_err = getattr(sci_obs, error_attr)

        # Check science block has the observable
        if sci_value is None:
            warnings.warn(
                f"Observable '{key}' requested but not "
                f"present on science block. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if sci_err is None:
            warnings.warn(
                f"Observable '{key}' error not present "
                f"on science block. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Gather calibrator values and errors
        cal_values = []
        cal_errors = []
        for i, cal_block in enumerate(cal_blocks):
            cal_obs = cal_block.observables[source_label]
            cal_val = getattr(cal_obs, value_attr)
            cal_err = getattr(cal_obs, error_attr)
            if cal_val is None or cal_err is None:
                raise ValueError(
                    f"Calibrator block {i} "
                    f"(target='{cal_block.target}') is "
                    f"missing observable '{key}' or its "
                    f"error."
                )
            cal_values.append(cal_val)
            cal_errors.append(cal_err)

        # Compute calibration reference values
        if method == "constant":
            ref_value, ref_err = _get_calibration_values_constant(
                cal_values, cal_errors
            )
        else:
            ref_value, ref_err = _get_calibration_values_linear(
                cal_values, cal_errors, cal_times, sci_time
            )

        # Apply calibration operation
        if key in _SUBTRACTION_OBSERVABLES:
            cal_result, cal_sigma = _calibrate_subtraction(
                sci_value, sci_err, ref_value, ref_err
            )
        else:
            cal_result, cal_sigma = _calibrate_division(
                sci_value, sci_err, ref_value, ref_err
            )

        # Store results
        output_kwargs[value_attr] = cal_result
        output_kwargs[error_attr] = cal_sigma
        output_kwargs[flag_attr] = np.zeros(cal_result.shape, dtype=bool)

    # Build calibrator target string
    seen_targets: list[str] = []
    for cal_block in cal_blocks:
        if cal_block.target not in seen_targets:
            seen_targets.append(cal_block.target)
    calibrator_target = ", ".join(seen_targets)

    # Build notes string
    notes = (
        f"method='{method}', "
        f"n_calibrators={len(cal_blocks)}, "
        f"source_label='{source_label}'"
    )

    # Construct output Observables copying geometry from science
    result = Observables(
        target=sci_obs.target,
        wavelengths=sci_obs.wavelengths.copy(),
        stations=sci_obs.stations,
        baselines=sci_obs.baselines,
        triangles=sci_obs.triangles,
        mean_para_angle=sci_obs.mean_para_angle,
        mjd=sci_obs.mjd,
        time_start=sci_obs.time_start,
        time_end=sci_obs.time_end,
        block_type=sci_obs.block_type,
        mask_name=sci_obs.mask_name,
        calibrated=True,
        calibrator_target=calibrator_target,
        extraction_backend="native",
        notes=notes,
        **output_kwargs,
    )

    return result


def compute_scatter_uncertainties(
    calibrated_obs: "Observables",
    scatter_blocks: list["ObservingBlock"],
    *,
    scatter_label: str = "raw",
    calibrate: set[str] | None = None,
) -> None:
    """Overwrite uncertainties with scatter across blocks.

    For each requested observable, computes the standard deviation
    of the raw values from ``scatter_blocks`` (per baseline/triangle,
    per wavelength) and uses it as the uncertainty on
    ``calibrated_obs``.

    For subtraction-type observables (closure phases, visibility
    phases), the uncertainty is the direct standard deviation.
    For division-type observables (squared visibilities, visibility
    amplitudes), a relative scatter treatment is applied:
    ``uncertainty = std/|mean| * |calibrated_value|``.

    Args:
        calibrated_obs: The calibrated Observables whose error
            fields will be overwritten in-place.
        scatter_blocks: List of ObservingBlocks providing the
            scatter estimate. Must contain at least 2 blocks.
        scatter_label: Label under which to find observables
            on each scatter block. Default ``"raw"``.
        calibrate: Set of observable keys for which to
            overwrite uncertainties. ``None`` means all four.
            Must be a non-empty subset of
            ``{'t3phi', 'vis2', 'visphi', 'visamp'}``.

    Raises:
        ValueError: If fewer than 2 scatter blocks are
            provided, if ``scatter_label`` is missing on a
            block, if ``calibrate`` is invalid, or if a
            required observable is missing on a scatter block.
    """
    # Resolve calibrate set
    if calibrate is None:
        calibrate_set = set(_VALID_OBSERVABLES)
    else:
        calibrate_set = set(calibrate)
        if not calibrate_set:
            raise ValueError("calibrate set must not be empty.")
        invalid = calibrate_set - _VALID_OBSERVABLES
        if invalid:
            raise ValueError(
                f"Invalid observable keys: {sorted(invalid)}. "
                f"Valid keys: {sorted(_VALID_OBSERVABLES)}."
            )

    if len(scatter_blocks) < 2:
        raise ValueError(
            "At least 2 scatter blocks are required for "
            "scatter-based uncertainty estimation, but "
            f"only {len(scatter_blocks)} provided."
        )

    # Validate scatter_label on each block
    for i, block in enumerate(scatter_blocks):
        if scatter_label not in block.observables:
            raise ValueError(
                f"scatter_label '{scatter_label}' not found "
                f"in scatter block {i} "
                f"(target='{block.target}'). Available: "
                f"{list(block.observables.keys())}."
            )

    for key in calibrate_set:
        attr_map = _OBS_ATTR_MAP[key]
        value_attr = attr_map["value"]
        error_attr = attr_map["error"]

        # Check calibrated_obs has the value
        cal_value = getattr(calibrated_obs, value_attr)
        if cal_value is None:
            continue

        # Gather values from scatter blocks
        scatter_values = []
        for i, block in enumerate(scatter_blocks):
            obs = block.observables[scatter_label]
            val = getattr(obs, value_attr)
            if val is None:
                raise ValueError(
                    f"Scatter block {i} "
                    f"(target='{block.target}') is "
                    f"missing observable '{key}'."
                )
            scatter_values.append(val)

        # Stack: (n_blocks, n_obs, n_wav)
        stacked = np.stack(scatter_values, axis=0)

        if key in _SUBTRACTION_OBSERVABLES:
            # Direct std
            uncertainty = np.std(stacked, axis=0, ddof=1)
        else:
            # Relative scatter: std/|mean| * |calibrated|
            std_vals = np.std(stacked, axis=0, ddof=1)
            mean_vals = np.mean(stacked, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_scatter = np.where(
                    np.abs(mean_vals) > 0,
                    std_vals / np.abs(mean_vals),
                    np.nan,
                )
            uncertainty = rel_scatter * np.abs(cal_value)

        setattr(calibrated_obs, error_attr, uncertainty)
