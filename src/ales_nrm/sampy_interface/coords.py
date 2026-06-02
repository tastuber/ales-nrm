"""Mask format conversion, filter generation, and SAMpy setup.

Provides utilities for converting ales_nrm mask geometry and
wavelength information into the file formats expected by
SAMpy's ``make_coords`` function, and a high-level wrapper
that orchestrates the full coordinate setup for all ALES
wavelength channels.
"""

import shutil
from pathlib import Path

import numpy as np

from ales_nrm.nrm.mask import ALES_PIXEL_SCALE_ARCSEC, NRMMask


def _convert_mask_to_sampy(
    mask: NRMMask,
    output_path: Path,
    center: bool = True,
) -> None:
    """Write mask hole coordinates in SAMpy format.

    Extracts (x, y) positions from each hole and writes
    a two-column whitespace-delimited text file with no
    header, one row per hole.

    SAMpy's ``make_coords`` places holes at pixel
    positions ``n_pix_ft//2 + hole_coord/pupil_pixel_scale``.
    The ALES mask coordinates are offset from origin;
    setting ``center=True`` translates them so that their
    centroid is at (0, 0), ensuring symmetric placement
    within SAMpy's internal pupil model grid.

    Args:
        mask: NRMMask instance with loaded hole
            coordinates.
        output_path: Destination file path. Parent
            directories are created if they do not
            exist.
        center: If True (default), subtract the
            centroid from all hole coordinates so they
            are centered at the origin.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords = np.array([[h.x, h.y] for h in mask.holes])

    if center:
        centroid = coords.mean(axis=0)
        coords = coords - centroid

    np.savetxt(output_path, coords, fmt="%.10f")


def _generate_narrow_filter(
    wavelength_um: float,
    output_path: Path,
    bandwidth_um: float = 0.01,
    n_points: int = 3,
) -> None:
    """Write a narrow tophat filter transmission file.

    Generates a synthetic filter file in the format
    expected by SAMpy: one header line followed by data
    rows with wavelength (µm) and throughput columns.
    The filter is a tophat of width ``bandwidth_um``
    centered on ``wavelength_um``, with two boundary
    points at zero throughput just outside the band
    edges.

    Args:
        wavelength_um: Central wavelength in microns.
        output_path: Destination file path. Parent
            directories are created if they do not
            exist.
        bandwidth_um: Full width of the tophat in
            microns. Default is 0.01 µm.
        n_points: Number of interior points within the
            passband (all at throughput 1.0). Default
            is 3.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    half_bw = bandwidth_um / 2.0

    # Interior wavelengths within the passband
    wl_interior = np.linspace(
        wavelength_um - half_bw,
        wavelength_um + half_bw,
        n_points,
    )

    # Small offset for boundary zero-points
    edge_offset = bandwidth_um * 0.01

    # Build full wavelength and throughput arrays
    wavelengths = np.empty(n_points + 2)
    throughputs = np.empty(n_points + 2)

    # Lower boundary (zero throughput)
    wavelengths[0] = wl_interior[0] - edge_offset
    throughputs[0] = 0.0

    # Interior (full throughput)
    wavelengths[1:-1] = wl_interior
    throughputs[1:-1] = 1.0

    # Upper boundary (zero throughput)
    wavelengths[-1] = wl_interior[-1] + edge_offset
    throughputs[-1] = 0.0

    with open(output_path, "w") as f:
        f.write("wavelength throughput\n")
        for wl, tp in zip(wavelengths, throughputs, strict=True):
            f.write(f"{wl:.8f} {tp:.6f}\n")


def _compute_zero_spacing_radius(
    wavelength_um: float,
    subaperture_diameter: float,
    n_pixels: int,
    pixel_scale: float,
    scale_factor: float = 1.0,
) -> int:
    """Compute the DC-peak masking radius for a given wavelength.

    The central (zero-spacing) peak in the power spectrum
    has angular width ~ λ/d. Its extent in pixels on the
    ``n_pixels`` Fourier grid is:

        r ≈ d * n_pixels * pixel_scale / (206265 * λ)

    Compute this radius of the masking region. A scale
    factor allows linear scaling of the diffraction limit
    for fine tuning or testing. Output radius is rounded
    to integer pixels.

    Args:
        wavelength_um: Wavelength in microns.
        subaperture_diameter: Hole diameter in meters.
        n_pixels: Fourier-plane grid size in pixels.
        pixel_scale: Detector pixel scale in arcsec/pixel.
        scale_factor: Scale the diffraction limit to compute
            final radius. Default 1.0.

    Returns:
        Masking radius in pixels (integer, >= 1).
    """
    # Plate scale of the Fourier grid: angular freq
    # per pixel = 1 / (n_pixels * pixel_scale) in rad.
    # DC peak radius in pixels =
    #   d / (lambda / (n_pixels * pixel_scale_rad))
    #   = d * n_pixels * pixel_scale_rad / lambda
    pixel_scale_rad = pixel_scale / 206265.0
    wave_m = wavelength_um * 1e-6
    radius = (
        scale_factor
        * subaperture_diameter
        * n_pixels
        * pixel_scale_rad
        / wave_m
    )
    return max(1, int(np.round(radius)))


def _make_cache_dirname(
    source_name: str,
    n_pixels: int,
    pixel_scale: float,
    pupil_pixel_scale: float,
    fourier_cutoff: float,
) -> str:
    """Construct a parametrized cache directory name.

    Encodes the key SAMpy coordinate-generation parameters
    into a deterministic, human-readable directory name.
    This ensures that different parameter sets produce
    separate cache directories, avoiding collisions.

    The format is:

        {source_name}_npix{n_pixels}_pixscale{pixel_scale}_pupilpixscale{pupil_pixel_scale}_fouriercutoff{fourier_cutoff}

    Float values are formatted to remove trailing zeros
    for compactness while preserving uniqueness.

    Args:
        source_name: Mask source name (e.g., "lbt_nrm").
        n_pixels: Fourier-plane grid size.
        pixel_scale: Detector pixel scale in arcsec/pixel.
        pupil_pixel_scale: Pupil model scale in m/pixel.
        fourier_cutoff: Fourier-plane selection threshold.

    Returns:
        Directory name string.
    """

    def _fmt(value: float) -> str:
        """Format float compactly, removing trailing zeros."""
        return f"{value:g}"

    return (
        f"{source_name}"
        f"_npix{n_pixels}"
        f"_pixscale{_fmt(pixel_scale)}"
        f"_pupilpixscale{_fmt(pupil_pixel_scale)}"
        f"_fouriercutoff{_fmt(fourier_cutoff)}"
    )


def _write_source_mask(mask: NRMMask, output_path: Path) -> None:
    """Write verbatim copy of original mask file.

    Args:
        mask: NRMMask instance with source_content.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mask.source_content)


def _write_rotated_mask(mask: NRMMask, output_path: Path) -> None:
    """Write mask coordinates after rotation.

    Includes a header documenting the source file and
    applied rotation angle.

    Args:
        mask: NRMMask instance with current (possibly
            rotated) hole coordinates.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Rotated mask coordinates\n")
        f.write(f"# Original: {mask.source_name}.txt\n")
        f.write(f"# Rotation: {mask.angle_deg:.4f} deg (CCW)\n")
        f.write("# Columns: name  x  y  radius\n")
        for h in mask.holes:
            f.write(f"{h.name}  {h.x:.10f}  {h.y:.10f}  {h.radius:.10f}\n")


def setup_sampy_coords(
    mask: NRMMask,
    wavelengths: float | np.ndarray,
    cache_dir: Path,
    pixel_scale: float = ALES_PIXEL_SCALE_ARCSEC,
    n_pixels: int = 501,
    pupil_pixel_scale: float = 0.01,
    zero_spacing_radius: int | None = None,
    zero_spacing_scale_factor: float = 1.0,
    force_recompute: bool = False,
    fourier_cutoff: float = 0.4,
    center_mask: bool = True,
    suppress_plots: bool = True,
) -> dict[float, Path]:
    """Set up SAMpy coordinate files for all wavelengths.

    Converts the ales_nrm mask to SAMpy format, generates
    narrow tophat filters for each wavelength channel, and
    calls SAMpy's ``make_coords`` to produce the Fourier-
    plane sampling coordinate files needed by the analysis
    functions.

    Results are cached in a subdirectory of ``cache_dir``
    named after the mask source file and key parameters.
    The directory structure is::

        cache_dir/
        └── {source_name}_npix{n_pixels}_pixscale{pixel_scale}_\
            pupilpixscale{pupil_pixel_scale}_fouriercutoff{fourier_cutoff}/
            ├── {source_name}.txt
            ├── {source_name}_rotated.txt
            ├── {source_name}_sampy.txt
            ├── filters/
            │   └── filter_{wl}um.txt
            └── sampy_coords/
                └── {wl}um/
                    ├── bl_uvs.fits
                    └── ...

    When ``force_recompute=False``, existing wavelength
    coordinate directories are skipped; new wavelengths
    are computed incrementally. When
    ``force_recompute=True``, the entire mask cache
    directory is deleted and rebuilt from scratch.

    Args:
        mask: NRMMask instance (e.g., from
            ``NRMMask.from_bundled()``).
        wavelengths: Wavelength(s) in microns. A single
            float or a 1D array.
        cache_dir: Root directory for cached output.
            Created if needed.
        pixel_scale: Detector pixel scale in
            arcsec/pixel. Default is 0.0345 (ALES).
        n_pixels: Fourier-plane sampling grid size.
            Images are zero-padded to this size before
            FFT in SAMpy's analysis functions.
            Default is 501.
        pupil_pixel_scale: Pupil model scale in
            meters/pixel. Controls resolution of
            subaperture shapes in the internal model.
            Default is 0.01 (~80 pixels across the
            0.784 m ALES hole diameter).
        zero_spacing_radius: Radius in pixels for
            masking the DC peak (circular mask). If
            None, computed per wavelength from the
            subaperture diameter and Fourier geometry.
        zero_spacing_scale_factor: Scale factor for the
            auto-computed zero-spacing radius. Only
            used when ``zero_spacing_radius`` is None.
            Default is 1.0.
        force_recompute: If True, delete the entire
            mask cache directory and regenerate all
            files from scratch.
        fourier_cutoff: Threshold for selecting
            Fourier-plane sampling pixels (fraction of
            peak power in each baseline's splodge).
            Default is 0.4.
        center_mask: If True (default), center the mask
            coordinates before writing the SAMpy mask
            file. See ``_convert_mask_to_sampy``.
        suppress_plots: If True (default), suppress
            interactive matplotlib plots produced by
            SAMpy's ``make_coords``. If False, plots
            are displayed normally.

    Returns:
        Dictionary mapping wavelength (float, µm) to
        the Path of the SAMpy coordinate directory for
        that wavelength.

    Raises:
        ImportError: If SAMpy is not installed.
    """
    try:
        import sampy.mask
    except ImportError as exc:
        raise ImportError(
            "SAMpy is required for setup_sampy_coords. "
            "Install with: pip install ales_nrm[sampy]"
        ) from exc

    import matplotlib.pyplot as plt

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Convert single wavelength to ndarray
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))

    # Mask-specific subdirectory named after source file and parameters
    source_name = mask.source_name or "unknown_mask"
    cache_dirname = _make_cache_dirname(
        source_name, n_pixels, pixel_scale, pupil_pixel_scale, fourier_cutoff
    )
    mask_cache = cache_dir / cache_dirname

    # force_recompute: clean slate
    if force_recompute and mask_cache.exists():
        shutil.rmtree(mask_cache)

    mask_cache.mkdir(parents=True, exist_ok=True)

    # Write mask files if not already present
    mask_file = mask_cache / f"{source_name}_sampy.txt"
    if not mask_file.exists():
        _convert_mask_to_sampy(mask, mask_file, center=center_mask)
        # Verbatim copy of original mask file
        if mask.source_content:
            _write_source_mask(
                mask,
                mask_cache / f"{source_name}.txt",
            )
        # Post-rotation coordinates for provenance
        _write_rotated_mask(
            mask,
            mask_cache / f"{source_name}_rotated.txt",
        )

    # Subaperture diameter from mask (2 * radius)
    subaperture_diameter = 2.0 * mask.holes[0].radius

    coord_dirs: dict[float, Path] = {}

    for wl in wavelengths:
        wl_float = float(wl)
        wl_label = f"{wl_float:.4f}um"

        # Coordinate directory for this wavelength
        coord_dir = mask_cache / "sampy_coords" / wl_label
        coord_dirs[wl_float] = coord_dir

        # Skip if already computed
        if coord_dir.exists():
            continue

        # Generate filter for this wavelength
        filter_dir = mask_cache / "filters"
        filter_file = filter_dir / f"filter_{wl_label}.txt"
        if not filter_file.exists():
            _generate_narrow_filter(wl_float, filter_file)

        coord_dir.mkdir(parents=True, exist_ok=True)

        # Compute zero-spacing radius for this wavelength
        # DC peak width scales as d*n*ps/lambda
        if zero_spacing_radius is not None:
            zsr = zero_spacing_radius
        else:
            zsr = _compute_zero_spacing_radius(
                wl_float,
                subaperture_diameter,
                n_pixels,
                pixel_scale,
                zero_spacing_scale_factor,
            )

        # Optionally suppress interactive plots
        if suppress_plots:
            plt.ioff()
            original_show = plt.show
            plt.show = lambda *args, **kwargs: plt.close("all")

        try:
            sampy.mask.make_coords(
                output_dir=str(coord_dir) + "/",
                mask_file=str(mask_file),
                subaperture_diameter=subaperture_diameter,
                filter_file=str(filter_file),
                hole_shape="circular",
                pixel_scale=pixel_scale,
                n_pixels=n_pixels,
                pupil_pixel_scale=pupil_pixel_scale,
                rotation=0,
                x_offset=0,
                y_offset=0,
                zero_spacing_radius=zsr,
                spectral_sampling=1,
                recompute=True,
                fourier_cutoff=fourier_cutoff,
            )
        finally:
            if suppress_plots:
                plt.show = original_show
                plt.ion()
                plt.close("all")

    return coord_dirs
