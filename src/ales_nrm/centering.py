"""Frame centering and flux-conserving shifting.

This module provides functions for finding the center of NRM
interferograms and shifting frames using the Fourier shift
theorem, which exactly conserves total flux.

Because the ALES lenslet IFS introduces a chromatic shift in
the PSF position, the center is determined independently per
wavelength (or wavelength group) across all frames. For each
wavelength where all frames have a successful Gaussian fit,
the mean center across frames is computed and all frames are
shifted to that mean position. For wavelengths where any fit
fails, shifts are propagated from the nearest reliable
wavelength.
"""

import logging

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian2D

logger = logging.getLogger(__name__)


def _find_peak_pixel(image: np.ndarray) -> tuple[int, int]:
    """Find the pixel with maximum flux in a 2D image.

    Args:
        image: 2D array.

    Returns:
        Tuple of ``(y_peak, x_peak)`` pixel indices.
    """
    idx = np.argmax(image)
    return np.unravel_index(idx, image.shape)


def find_center(
    image: np.ndarray,
    cutout_size: int = 5,
) -> tuple[float, float]:
    """Find the sub-pixel center of an NRM interferogram.

    Locates the peak pixel in the image, extracts a small
    cutout, and fits a 2D Gaussian to obtain sub-pixel
    center coordinates.

    Args:
        image: 2D array (single wavelength slice or sum
            of adjacent wavelength slices).
        cutout_size: Side length of the square cutout
            around the peak pixel used for Gaussian
            fitting. Must be odd. Default is 5.

    Returns:
        Tuple of ``(y_center, x_center)`` in pixel
        coordinates of the full image.

    Raises:
        ValueError: If cutout_size is not a positive odd
            integer >= 3, or if the Gaussian fit fails.
    """
    if cutout_size < 3 or cutout_size % 2 == 0:
        raise ValueError(
            f"cutout_size must be an odd integer >= 3, got {cutout_size}."
        )

    ny, nx = image.shape
    y_peak, x_peak = _find_peak_pixel(image)

    half = cutout_size // 2
    y_lo = max(0, y_peak - half)
    y_hi = min(ny, y_peak + half + 1)
    x_lo = max(0, x_peak - half)
    x_hi = min(nx, x_peak + half + 1)

    cutout = image[y_lo:y_hi, x_lo:x_hi].copy()
    cy, cx = np.mgrid[y_lo:y_hi, x_lo:x_hi]

    amplitude = cutout.max()

    model = Gaussian2D(
        amplitude=amplitude,
        x_mean=float(x_peak),
        y_mean=float(y_peak),
        x_stddev=1.5,
        y_stddev=1.5,
    )

    fitter = LevMarLSQFitter()
    fitted = fitter(model, cx, cy, cutout)

    if fitter.fit_info["ierr"] not in [1, 2, 3, 4]:
        raise ValueError(
            "Gaussian fit did not converge. "
            f"Fit info: {fitter.fit_info['message']}"
        )

    y_center = fitted.y_mean.value
    x_center = fitted.x_mean.value

    logger.debug(
        "Center found at (y=%.3f, x=%.3f) from peak pixel (%d, %d).",
        y_center,
        x_center,
        y_peak,
        x_peak,
    )

    return y_center, x_center


def _try_find_center(
    image: np.ndarray,
    cutout_size: int = 5,
) -> tuple[float, float] | None:
    """Attempt to find the center, returning None on failure.

    This is a wrapper around ``find_center`` that catches
    ``ValueError`` from failed Gaussian fits and returns
    ``None`` instead. Used internally by ``center_cubes``
    to handle low-SNR wavelength slices gracefully.

    Args:
        image: 2D array.
        cutout_size: Side length of the Gaussian fitting
            cutout. Must be odd. Default is 5.

    Returns:
        Tuple of ``(y_center, x_center)`` or ``None`` if
        the fit failed.
    """
    try:
        return find_center(image, cutout_size=cutout_size)
    except ValueError as e:
        logger.debug("Center finding failed: %s", e)
        return None


def fourier_shift_2d(
    image: np.ndarray,
    dy: float,
    dx: float,
) -> np.ndarray:
    """Shift a 2D image using the Fourier shift theorem.

    This operation is exactly flux-conserving: the total
    flux (sum of all pixels) is preserved to floating-point
    precision because the DC component of the Fourier
    transform is multiplied by exp(0) = 1.

    The shift assumes periodic boundary conditions. For
    small shifts (a few pixels) on NRM frames where flux
    drops toward the edges, the wrapping artifact is
    negligible.

    Args:
        image: 2D array to shift.
        dy: Shift in the y direction (rows) in pixels.
            Positive values shift the image downward.
        dx: Shift in the x direction (columns) in pixels.
            Positive values shift the image rightward.

    Returns:
        Shifted 2D array with the same shape and dtype.
    """
    ny, nx = image.shape
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    f_grid_x, f_grid_y = np.meshgrid(fx, fy)

    phase = np.exp(-2j * np.pi * (f_grid_x * dx + f_grid_y * dy))

    shifted = np.real(np.fft.ifft2(np.fft.fft2(image) * phase))

    return shifted.astype(image.dtype)


def _fill_shifts(
    shifts: np.ndarray,
) -> np.ndarray:
    """Fill nan values in a shifts array using forward/backward fill.

    For each frame, propagates the last valid shift forward
    through subsequent nan wavelengths (forward fill), then
    propagates the first valid shift backward through
    preceding nan wavelengths (backward fill). If a frame
    has no valid shifts at any wavelength, its shifts remain
    at zero.

    Args:
        shifts: 3D array of shape ``(n_frames, n_wav, 2)``
            with ``np.nan`` for wavelengths that need
            filling.

    Returns:
        Filled shifts array with the same shape.
    """
    filled = shifts.copy()
    n_frames, n_wav, _ = filled.shape

    for f in range(n_frames):
        # Check if any valid shifts exist for this frame.
        valid_mask = np.isfinite(filled[f, :, 0])
        if not np.any(valid_mask):
            # No valid shifts at all — set to zero.
            filled[f] = 0.0
            logger.warning(
                "Frame %d: no valid center fits at any "
                "wavelength. Leaving unshifted.",
                f,
            )
            continue

        # Forward fill.
        for w in range(1, n_wav):
            if np.isnan(filled[f, w, 0]):
                filled[f, w] = filled[f, w - 1]

        # Backward fill.
        for w in range(n_wav - 2, -1, -1):
            if np.isnan(filled[f, w, 0]):
                filled[f, w] = filled[f, w + 1]

    return filled


def center_cubes(
    cubes: np.ndarray,
    cutout_size: int = 5,
    n_wave_sum: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Center multiple frames by aligning to the mean position.

    For each wavelength (or group of adjacent wavelengths),
    the PSF center is found independently in each frame.
    Only wavelengths where all frames have a successful
    Gaussian fit are used for computing the mean center and
    shifts. For wavelengths where any fit fails, shifts are
    propagated from the nearest reliable wavelength using
    forward/backward fill.

    This approach ensures that shifts are only computed from
    fully reliable data, and that noisy spectral edges are
    handled gracefully by borrowing shifts from nearby
    well-behaved wavelengths.

    Args:
        cubes: 4D array with shape
            ``(n_frames, n_wavelengths, ny, nx)``.
        cutout_size: Side length of the cutout used for
            Gaussian fitting. Must be odd. Default is 5.
        n_wave_sum: Number of adjacent wavelength slices
            to sum before centroiding. Default is 1
            (each wavelength independently). Useful for
            low-SNR data where summing adjacent slices
            improves the centroid estimate. The same
            shift is applied to all slices in the group.

    Returns:
        Tuple of ``(centered_cubes, shifts)`` where:
            - ``centered_cubes`` has the same shape and
              dtype as the input.
            - ``shifts`` is a 3D array of shape
              ``(n_frames, n_wavelengths, 2)`` containing
              the ``(dy, dx)`` shift applied to each
              frame and wavelength slice.

    Raises:
        ValueError: If the input is not 4-dimensional or
            if ``n_wave_sum`` is less than 1.
    """
    if cubes.ndim != 4:
        raise ValueError(f"Expected a 4D array, got {cubes.ndim}D.")

    if n_wave_sum < 1:
        raise ValueError(f"n_wave_sum must be >= 1, got {n_wave_sum}.")

    n_frames, n_wav, ny, nx = cubes.shape

    # First pass: find centers for every frame and
    # wavelength group.
    centers = np.full(
        (n_frames, n_wav, 2),
        np.nan,
        dtype=np.float64,
    )
    fit_success = np.zeros(
        (n_frames, n_wav),
        dtype=bool,
    )

    for f in range(n_frames):
        w = 0
        while w < n_wav:
            w_end = min(w + n_wave_sum, n_wav)

            if w_end - w == 1:
                ref_image = cubes[f, w]
            else:
                ref_image = np.sum(
                    cubes[f, w:w_end],
                    axis=0,
                )

            result = _try_find_center(
                ref_image,
                cutout_size=cutout_size,
            )

            if result is not None:
                y_c, x_c = result
                for wi in range(w, w_end):
                    centers[f, wi] = [y_c, x_c]
                    fit_success[f, wi] = True
            else:
                logger.debug(
                    "Center fit failed for frame %d, wavelength slices %d–%d.",
                    f,
                    w,
                    w_end - 1,
                )

            w = w_end

    # Second pass: compute shifts only for wavelengths
    # where ALL frames have successful fits.
    shifts = np.full(
        (n_frames, n_wav, 2),
        np.nan,
        dtype=np.float64,
    )

    n_reliable = 0
    n_unreliable = 0

    for wi in range(n_wav):
        all_success = np.all(fit_success[:, wi])

        if not all_success:
            n_unreliable += 1
            logger.debug(
                "Wavelength %d: not all frames have "
                "successful fits (%d/%d). Will fill "
                "from neighbor.",
                wi,
                np.sum(fit_success[:, wi]),
                n_frames,
            )
            continue

        n_reliable += 1

        # Mean center from all frames at this wavelength.
        y_mean = np.mean(centers[:, wi, 0])
        x_mean = np.mean(centers[:, wi, 1])

        logger.debug(
            "Wavelength %d: mean center (y=%.3f, x=%.3f) from %d frames.",
            wi,
            y_mean,
            x_mean,
            n_frames,
        )

        for f in range(n_frames):
            dy = y_mean - centers[f, wi, 0]
            dx = x_mean - centers[f, wi, 1]
            shifts[f, wi] = [dy, dx]

    logger.info(
        "Center fitting: %d/%d wavelengths reliable, "
        "%d will be filled from neighbors.",
        n_reliable,
        n_wav,
        n_unreliable,
    )

    if n_reliable == 0:
        logger.warning(
            "No wavelength has all frames with "
            "successful center fits. Leaving all "
            "frames unshifted."
        )
        return cubes.copy(), np.zeros(
            (n_frames, n_wav, 2),
            dtype=np.float64,
        )

    # Fill nan shifts from nearest reliable wavelength.
    shifts = _fill_shifts(shifts)

    # Apply shifts.
    centered = np.empty_like(cubes)

    for f in range(n_frames):
        for wi in range(n_wav):
            dy, dx = shifts[f, wi]
            centered[f, wi] = fourier_shift_2d(
                cubes[f, wi],
                dy,
                dx,
            )

    logger.info(
        "Centered %d frames x %d wavelengths, "
        "n_wave_sum=%d, mean abs shift "
        "(dy=%.3f, dx=%.3f).",
        n_frames,
        n_wav,
        n_wave_sum,
        np.mean(np.abs(shifts[:, :, 0])),
        np.mean(np.abs(shifts[:, :, 1])),
    )

    return centered, shifts
