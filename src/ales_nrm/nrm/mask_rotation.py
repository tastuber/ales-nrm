"""Mask rotation angle calibration for ALES NRM data.

This module determines the physical rotation angle of the
non-redundant mask relative to the detector, by comparing
measured power spectrum splodge positions against analytic
predictions from the mask geometry.
Internally, computations are performed in detector pixel
space (y, x). To make the results directly applicable to
rotate the mask in the pupil plane (x, y) the angles are
negated before return.

The approach uses a hybrid two-step workflow:

- **Step A** (coarse): Grid search over trial rotation
  angles, sampling the power spectrum at rotated analytic
  positions to find the angle maximizing total flux.
- **Step B** (refinement): Gaussian centroid fitting of
  individual splodges, followed by least-squares
  minimization of the angle that best maps analytic
  positions onto measured centers.
"""

import logging

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit, minimize_scalar

from ales_nrm.nrm.mask import ALES_PIXEL_SCALE_ARCSEC, NRMMask
from ales_nrm.utilities import rotate_points_2d

logger = logging.getLogger(__name__)


def find_mask_rotation_angle(
    power_spectra: np.ndarray,
    mask: NRMMask,
    wavelengths: np.ndarray,
    pixel_scale_arcsec: float = ALES_PIXEL_SCALE_ARCSEC,
    angle_range: tuple[float, float] = (-20.0, 20.0),
    n_grid: int = 81,
    cutout_size: int = 5,
    refine: bool = True,
) -> dict:
    """Find the mask rotation angle from power spectra.

    Main entry point for mask rotation calibration.
    Implements a hybrid workflow combining a coarse grid
    search (Step A) with optional Gaussian centroid
    refinement (Step B).

    Convention
    ----------
    The returned ``angle_deg`` is negated relative to
    the internal calibration result. Internally, the
    search operates in detector (row, col) = (y, x)
    pixel space, where a positive angle rotates
    counter-clockwise in the (y, x) plane. Because
    ``NRMMask`` rotates hole coordinates in the pupil
    (x, y) plane, and transposing axis order inverts
    the rotation sign, here, the output is negated so
    it can be passed directly to
    ``NRMMask.from_bundled(angle_deg=...)`` or
     ``NRMMask.from_file(angle_deg=...)``.

    Args:
        power_spectra: Power spectrum array. Accepted shapes:
            - ``(ny, nx)`` — single frame, single
              wavelength
            - ``(n_wavelengths, ny, nx)`` — single
              frame, multiple wavelengths
            - ``(n_files, n_wavelengths, ny, nx)`` —
              multiple frames, multiple wavelengths
        mask: NRMMask instance with hole and baseline
            geometry.
        wavelengths: 1D array of wavelengths in microns,
            matching the wavelength dimension.
        pixel_scale_arcsec: Pixel scale in
            arcsec/pixel. Default is ALES (0.0345).
        angle_range: Search range in degrees for the
            coarse grid search. Default is (-20, 20).
        n_grid: Number of grid points for the coarse
            search. Default is 81.
        cutout_size: Side length in pixels for Gaussian
            fitting cutouts in Step B. Default is 5.
        refine: If True (default), perform Step B
            Gaussian refinement after Step A. If False,
            only perform Step A for a quick-look result.

    Returns:
        Dictionary with keys:

        - ``"angle_deg"``: Best-fit rotation angle in degrees.
            (float). This value can be passed directly to
            ``NRMMask.from_bundled(angle_deg=...)`` or
            ``NRMMask.from_file(angle_deg=...)`` to load the
            mask with calibrated orientation.
        - ``"angle_std_deg"``: Standard deviation across
          slices (float or None if only one slice or
          refine=False).
        - ``"angles_all"``: Array of per-slice angles
          from Step B (None if refine=False).
        - ``"measured_centers"``: Dict of fitted
          Gaussian centers per baseline (None if
          refine=False).
        - ``"step_a_angle_deg"``: Angle from Step A
          coarse search (float).
    """
    # Normalize input to 4D.
    power_spectra = np.asarray(power_spectra, dtype=np.float64)
    if power_spectra.ndim == 2:
        power_spectra = power_spectra[np.newaxis, np.newaxis, :, :]
    elif power_spectra.ndim == 3:
        power_spectra = power_spectra[np.newaxis, :, :, :]
    elif power_spectra.ndim != 4:
        raise ValueError(
            f"power_spectra must be 2D, 3D, or 4D, got {power_spectra.ndim}D."
        )

    n_files, n_wav, ny, nx = power_spectra.shape
    wavelengths = np.asarray(wavelengths, dtype=np.float64)

    if len(wavelengths) != n_wav:
        raise ValueError(
            f"wavelengths length ({len(wavelengths)}) "
            f"does not match power_spectra wavelength "
            f"dimension ({n_wav})."
        )

    # Compute analytic splodge positions (unrotated).
    positions = mask.compute_splodge_positions(
        wavelengths,
        pixel_scale_arcsec,
        ny,
        nx,
    )

    center = ((ny - 1) / 2.0, (nx - 1) / 2.0)

    # Build position array: both direct and conjugate.
    # Shape: (n_positions, n_wavelengths, 2)
    all_positions = []
    for bl_name in positions:
        coords = positions[bl_name]  # (n_wav, 2)
        all_positions.append(coords)
        # Conjugate positions.
        conj = np.empty_like(coords)
        conj[:, 0] = 2 * center[0] - coords[:, 0]
        conj[:, 1] = 2 * center[1] - coords[:, 1]
        all_positions.append(conj)

    all_positions = np.array(all_positions)  # (N, n_wav, 2)

    # --- Step A: Coarse grid search ---
    step_a_angle = _step_a_coarse_search(
        power_spectra,
        all_positions,
        center,
        angle_range,
        n_grid,
    )

    logger.info(
        "Step A coarse angle: %.4f deg.",
        step_a_angle,
    )

    if not refine:
        return {
            "angle_deg": -step_a_angle,
            "angle_std_deg": None,
            "angles_all": None,
            "measured_centers": None,
            "step_a_angle_deg": -step_a_angle,
        }

    # --- Step B: Gaussian refinement ---
    angles_all = []
    all_measured_centers = {bl_name: [] for bl_name in positions}

    for f in range(n_files):
        for w in range(n_wav):
            ps_slice = power_spectra[f, w]

            # Get rotated analytic positions for this
            # wavelength using Step A angle.
            slice_positions = {}
            for bl_name in positions:
                pos = positions[bl_name][w : w + 1]  # (1,2)
                slice_positions[bl_name] = pos[0]

            # Fit Gaussian centers for each splodge.
            measured = {}
            for bl_name, pos_yx in slice_positions.items():
                rotated = rotate_points_2d(
                    pos_yx.reshape(1, 2),
                    center,
                    step_a_angle,
                )[0]

                fitted_y, fitted_x, success = _fit_2d_gaussian_at_position(
                    ps_slice,
                    rotated,
                    cutout_size,
                )
                if success:
                    measured[bl_name] = np.array([fitted_y, fitted_x])
                    all_measured_centers[bl_name].append((fitted_y, fitted_x))
                else:
                    measured[bl_name] = rotated
                    all_measured_centers[bl_name].append(
                        (rotated[0], rotated[1])
                    )

            # Compute refined angle from centroids.
            analytic_arr = np.array(
                [slice_positions[bl_name] for bl_name in slice_positions]
            )
            measured_arr = np.array(
                [measured[bl_name] for bl_name in slice_positions]
            )

            refined_angle = _compute_angle_from_centroids(
                analytic_arr,
                measured_arr,
                center,
                step_a_angle,
                bound_deg=5.0,
            )
            angles_all.append(refined_angle)

    angles_all = np.array(angles_all)
    mean_angle = float(np.mean(angles_all))
    std_angle = float(np.std(angles_all)) if len(angles_all) > 1 else None

    # Convert measured centers to arrays.
    measured_centers_out = {}
    for bl_name in all_measured_centers:
        measured_centers_out[bl_name] = np.array(all_measured_centers[bl_name])

    logger.info(
        "Step B refined angle: %.4f ± %.4f deg (%d slices).",
        mean_angle,
        std_angle if std_angle is not None else 0.0,
        len(angles_all),
    )

    return {
        "angle_deg": -mean_angle,
        "angle_std_deg": std_angle,
        "angles_all": -angles_all,
        "measured_centers": measured_centers_out,
        "step_a_angle_deg": -step_a_angle,
    }


def _step_a_coarse_search(
    power_spectra: np.ndarray,
    all_positions: np.ndarray,
    center: tuple[float, float],
    angle_range: tuple[float, float],
    n_grid: int,
) -> float:
    """Coarse grid search followed by bounded refinement.

    Args:
        power_spectra: 4D array (n_files, n_wav, ny, nx).
        all_positions: Array of shape
            (n_splodges, n_wav, 2) with analytic
            positions (direct + conjugate).
        center: (center_y, center_x) of the image.
        angle_range: (min_angle, max_angle) in degrees.
        n_grid: Number of grid points.

    Returns:
        Best-fit angle in degrees from Step A.
    """
    # Average power spectrum across files for speed.
    mean_ps = np.mean(power_spectra, axis=0)  # (n_wav, ny, nx)

    trial_angles = np.linspace(angle_range[0], angle_range[1], n_grid)
    flux_values = np.empty(n_grid)

    n_wav = mean_ps.shape[0]

    for i, angle in enumerate(trial_angles):
        total_flux = 0.0
        for w in range(n_wav):
            # Get positions for this wavelength.
            pos_w = all_positions[:, w, :]  # (N, 2)
            rotated = rotate_points_2d(pos_w, center, angle)
            sampled = _sample_power_at_positions(mean_ps[w], rotated)
            total_flux += np.sum(sampled)
        flux_values[i] = total_flux

    # Find peak.
    best_idx = np.argmax(flux_values)
    coarse_angle = trial_angles[best_idx]

    # Refine with bounded minimization around peak.
    grid_step = (angle_range[1] - angle_range[0]) / (n_grid - 1)
    refine_bound = max(2.0, 2 * grid_step)
    refine_min = max(angle_range[0], coarse_angle - refine_bound)
    refine_max = min(angle_range[1], coarse_angle + refine_bound)

    def neg_flux(angle_deg):
        total = 0.0
        for w in range(n_wav):
            pos_w = all_positions[:, w, :]
            rotated = rotate_points_2d(pos_w, center, angle_deg)
            sampled = _sample_power_at_positions(mean_ps[w], rotated)
            total += np.sum(sampled)
        return -total

    result = minimize_scalar(
        neg_flux,
        bounds=(refine_min, refine_max),
        method="bounded",
        options={"xatol": 0.001},
    )

    return float(result.x)


def _sample_power_at_positions(
    power_spectrum: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Get interpolated power spectrum values.

    Uses bicubic interpolation at fractional pixel
    positions.

    Args:
        power_spectrum: 2D array (ny, nx).
        positions: Array of shape (N, 2) with columns
            [y, x].

    Returns:
        Array of sampled values, shape (N,).
    """
    coords = np.array([positions[:, 0], positions[:, 1]])
    return map_coordinates(
        power_spectrum,
        coords,
        order=3,
        mode="constant",
        cval=0.0,
    )


def _fit_2d_gaussian(
    cutout: np.ndarray,
    initial_center: tuple[float, float],
) -> tuple[float, float, bool]:
    """Fit a 2D Gaussian to a small cutout.

    Model:
        A * exp(-((x-x0)^2/(2*sx^2) + (y-y0)^2/(2*sy^2)))
        + offset

    Args:
        cutout: 2D array (cutout_size, cutout_size).
        initial_center: (y, x) initial guess in cutout
            coordinates.

    Returns:
        Tuple of (y_center, x_center, success) in
        cutout-local coordinates. On failure, returns
        initial_center with success=False.
    """
    ny, nx = cutout.shape
    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    y_flat = y_grid.ravel()
    x_flat = x_grid.ravel()
    data_flat = cutout.ravel()

    def gaussian_2d(coords, amplitude, y0, x0, sy, sx, offset):
        y, x = coords
        return (
            amplitude
            * np.exp(
                -((y - y0) ** 2 / (2 * sy**2) + (x - x0) ** 2 / (2 * sx**2))
            )
            + offset
        )

    # Initial guesses.
    amp_guess = float(cutout.max() - cutout.min())
    offset_guess = float(cutout.min())
    y0_guess = initial_center[0]
    x0_guess = initial_center[1]
    sigma_guess = 1.0

    p0 = [
        amp_guess,
        y0_guess,
        x0_guess,
        sigma_guess,
        sigma_guess,
        offset_guess,
    ]

    # Bounds.
    half = max(ny, nx) / 2.0
    bounds_lower = [
        0.0,
        -1.0,
        -1.0,
        0.5,
        0.5,
        0.0,
    ]
    bounds_upper = [
        np.inf,
        ny,
        nx,
        half,
        half,
        np.inf,
    ]

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (y_flat, x_flat),
            data_flat,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=1000,
        )
        y_center = popt[1]
        x_center = popt[2]
        # Check that center is within cutout.
        if -1.0 <= y_center <= ny and -1.0 <= x_center <= nx:
            return float(y_center), float(x_center), True
        else:
            return (
                initial_center[0],
                initial_center[1],
                False,
            )
    except (RuntimeError, ValueError):
        return initial_center[0], initial_center[1], False


def _fit_2d_gaussian_at_position(
    power_spectrum: np.ndarray,
    position: np.ndarray,
    cutout_size: int,
) -> tuple[float, float, bool]:
    """Extract cutout and fit Gaussian, return global coords.

    Args:
        power_spectrum: 2D array (ny, nx).
        position: (y, x) position in global frame.
        cutout_size: Side length of cutout.

    Returns:
        (y_global, x_global, success).
    """
    ny, nx = power_spectrum.shape
    half = cutout_size // 2

    # Integer corner of cutout.
    iy = int(np.round(position[0]))
    ix = int(np.round(position[1]))

    y_start = iy - half
    x_start = ix - half

    # Check bounds.
    if (
        y_start < 0
        or x_start < 0
        or y_start + cutout_size > ny
        or x_start + cutout_size > nx
    ):
        return float(position[0]), float(position[1]), False

    cutout = power_spectrum[
        y_start : y_start + cutout_size,
        x_start : x_start + cutout_size,
    ]

    # Initial center in cutout coordinates.
    init_y = position[0] - y_start
    init_x = position[1] - x_start

    fit_y, fit_x, success = _fit_2d_gaussian(cutout, (init_y, init_x))

    # Convert back to global.
    global_y = fit_y + y_start
    global_x = fit_x + x_start

    return float(global_y), float(global_x), success


def _compute_angle_from_centroids(
    analytic_positions: np.ndarray,
    measured_centers: np.ndarray,
    center: tuple[float, float],
    initial_angle_deg: float,
    bound_deg: float = 5.0,
) -> float:
    """Find optimal rotation angle from centroid positions.

    Minimizes sum of squared distances between rotated
    analytic positions and measured centers.

    Args:
        analytic_positions: (N, 2) unrotated analytic
            positions.
        measured_centers: (N, 2) measured Gaussian
            centers.
        center: (center_y, center_x) rotation origin.
        initial_angle_deg: Initial angle estimate from
            Step A.
        bound_deg: Search bound around initial angle.

    Returns:
        Optimal rotation angle in degrees.
    """

    def cost(angle_deg):
        rotated = rotate_points_2d(analytic_positions, center, angle_deg)
        residuals = rotated - measured_centers
        return float(np.sum(residuals**2))

    result = minimize_scalar(
        cost,
        bounds=(
            initial_angle_deg - bound_deg,
            initial_angle_deg + bound_deg,
        ),
        method="bounded",
        options={"xatol": 0.0001},
    )

    return float(result.x)
