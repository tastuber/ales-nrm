"""Functions for reading LMIRCam/ALES FITS data cubes.

The ALES integral field spectrograph produces 3D data cubes where the
third axis corresponds to wavelength. Each FITS file contains a single
sky-subtracted spectral cube with dimensions (n_wavelengths, ny, nx).

Wavelength information is stored in the FITS header as SLICEnnn
keywords, where nnn is a zero-padded three-digit index corresponding to
the slice along NAXIS3.

Typical filenames follow the pattern::

    cube_lm_YYMMDD_NNNNNN.fits
    cube_lm_YYMMDD_NNNNNN.fits.gz

where YYMMDD is the UT date and NNNNNN is a six-digit running file
number.
"""

import logging
import re
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Regex pattern to extract the file number from the filename.
# Matches both .fits and .fits.gz extensions.
_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?P<number>\d{6})\.fits(?:\.gz)?$"
)


def parse_file_number(filepath: str | Path) -> int:
    """Extract the running file number from a LMIRCam/ALES filename.

    Args:
        filepath: Path to the FITS file. Only the filename (stem) is
            used for matching.

    Returns:
        The six-digit file number as an integer.

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    name = Path(filepath).name
    match = _FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(
            f"Filename '{name}' does not match the expected pattern "
            f"'<prefix>_NNNNNN.fits[.gz]'."
        )
    return int(match.group("number"))


def read_wavelengths(header: fits.Header) -> np.ndarray:
    """Extract the wavelength array from SLICEnnn header keywords.

    The ALES data cubes encode wavelengths as SLICE000, SLICE001, etc.
    in the FITS header, one per spectral channel along NAXIS3.

    Args:
        header: FITS header containing SLICEnnn keywords.

    Returns:
        1D array of wavelengths in microns, with length equal to
        NAXIS3.

    Raises:
        KeyError: If NAXIS3 is missing from the header.
        ValueError: If fewer SLICEnnn keywords are found than expected
            from NAXIS3.
    """
    n_slices = header["NAXIS3"]
    wavelengths = np.empty(n_slices, dtype=np.float64)

    for i in range(n_slices):
        key = f"SLICE{i:03d}"
        if key not in header:
            raise ValueError(
                f"Expected {n_slices} wavelength keywords (NAXIS3={n_slices}) "
                f"but keyword '{key}' is missing from the header."
            )
        wavelengths[i] = header[key]

    logger.debug(
        "Read %d wavelengths: %.4f - %.4f µm",
        n_slices,
        wavelengths[0],
        wavelengths[-1],
    )
    return wavelengths


def read_cube(
    filepath: str | Path,
) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    """Read a single ALES spectral cube from a FITS file.

    Reads the primary HDU data and extracts the wavelength array from
    the SLICEnnn header keywords.

    Args:
        filepath: Path to the FITS file (.fits or .fits.gz).

    Returns:
        A tuple of ``(cube, wavelengths, header)`` where:
            - ``cube`` is a 3D numpy array with shape
              ``(n_wavelengths, ny, nx)``.
            - ``wavelengths`` is a 1D numpy array of wavelengths in
              microns.
            - ``header`` is the full FITS header.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the data is not 3-dimensional or wavelength
            keywords are incomplete.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"FITS file not found: {filepath}")

    with fits.open(filepath) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if data is None:
        raise ValueError(f"No data in primary HDU of {filepath}.")

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D data cube, got {data.ndim}D array "
            f"with shape {data.shape} in {filepath}."
        )

    # Convert from FITS big-endian to native byte order.
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="))

    wavelengths = read_wavelengths(header)

    logger.info(
        "Read cube from %s: shape=%s, λ=%.3f–%.3f µm",
        filepath.name,
        data.shape,
        wavelengths[0],
        wavelengths[-1],
    )

    return data, wavelengths, header


def find_cubes(
    directory: str | Path,
    file_range: tuple[int, int] | None = None,
    prefix: str = "cube_lm_",
) -> list[Path]:
    """Find cube FITS files, optionally by file number range.

    Searches for files matching the naming pattern
    ``<prefix>*_NNNNNN.fits[.gz]``. If ``file_range`` is provided, only
    files whose running number falls within the inclusive range are
    returned.

    Args:
        directory: Path to the directory containing FITS files.
        file_range: Optional tuple ``(start, end)`` of file numbers
            (inclusive). If ``None``, all matching files are returned.
        prefix: Filename prefix to filter on. Default is
            ``'cube_lm_'``.

    Returns:
        Sorted list of Path objects for the matching FITS files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Collect both .fits and .fits.gz
    candidates = sorted(
        list(directory.glob(f"{prefix}*.fits"))
        + list(directory.glob(f"{prefix}*.fits.gz"))
    )

    if file_range is None:
        return candidates

    start, end = file_range
    selected = []
    for path in candidates:
        try:
            num = parse_file_number(path)
        except ValueError:
            logger.warning("Skipping file with unexpected name: %s", path.name)
            continue
        if start <= num <= end:
            selected.append(path)

    logger.info(
        "Found %d cubes in %s for file range %d–%d.",
        len(selected),
        directory,
        start,
        end,
    )

    return selected


def read_cubes(
    directory: str | Path,
    file_range: tuple[int, int] | None = None,
    prefix: str = "cube_lm_",
) -> tuple[np.ndarray, np.ndarray, list[fits.Header], np.ndarray]:
    """Read multiple ALES cubes from a directory into a 4D array.

    All cubes are expected to share the same wavelength grid and spatial
    dimensions. The wavelength array and spatial shape are taken from
    the first file and verified against all subsequent files.

    Args:
        directory: Path to the directory containing FITS files.
        file_range: Optional tuple ``(start, end)`` of file numbers
            (inclusive).
        prefix: Filename prefix to filter on. Default is
            ``'cube_lm_'``.

    Returns:
        A tuple of ``(cubes, wavelengths, headers, file_numbers)`` with:
            - ``cubes`` is a 4D numpy array with shape
              ``(n_files, n_wavelengths, ny, nx)``.
            - ``wavelengths`` is a 1D numpy array of wavelengths in
              microns (from the first file).
            - ``headers`` is a list of FITS headers.
            - ``file_numbers`` is a 1D integer numpy array of the
              running file number extracted from each filename, in the
              same order as the first axis of ``cubes``.

    Raises:
        FileNotFoundError: If the directory does not exist or no files
            are found.
        ValueError: If wavelength grids or spatial dimensions are
            inconsistent across files.
    """
    paths = find_cubes(directory, file_range=file_range, prefix=prefix)

    if not paths:
        range_str = f" in range {file_range}" if file_range else ""
        raise FileNotFoundError(
            f"No FITS cubes found in {directory}{range_str}."
        )

    headers = []
    reference_wavelengths = None
    reference_shape = None

    # First pass: read first file to determine array dimensions.
    first_cube, reference_wavelengths, first_header = read_cube(paths[0])
    reference_shape = first_cube.shape
    headers.append(first_header)

    # Pre-allocate the 4D array.
    n_files = len(paths)
    cubes = np.empty((n_files, *reference_shape), dtype=first_cube.dtype)
    cubes[0] = first_cube

    # Collect file numbers.
    file_numbers = np.empty(n_files, dtype=int)
    file_numbers[0] = parse_file_number(paths[0])

    # Read remaining files.
    for i, path in enumerate(paths[1:], start=1):
        cube, wavelengths, header = read_cube(path)

        if cube.shape != reference_shape:
            raise ValueError(
                f"Cube shape {cube.shape} in {path.name} does not match "
                f"the reference shape {reference_shape} from "
                f"{paths[0].name}."
            )

        if not np.allclose(wavelengths, reference_wavelengths, atol=1e-8):
            raise ValueError(
                f"Wavelength grid in {path.name} does not match the "
                f"reference grid from {paths[0].name}."
            )

        cubes[i] = cube
        headers.append(header)
        file_numbers[i] = parse_file_number(path)

    logger.info(
        "Read %d cubes from %s: shape=%s, files %d–%d.",
        n_files,
        directory,
        cubes.shape,
        file_numbers[0],
        file_numbers[-1],
    )

    return cubes, reference_wavelengths, file_numbers, headers
