"""Shared pytest fixtures for ales_nrm tests."""

import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture()
def rng():
    """Provide a seeded random number generator for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture()
def sample_wavelengths():
    """Provide a realistic ALES wavelength array (98 channels, 2.77–4.29µm)."""
    return np.linspace(2.768, 4.290, 98)


@pytest.fixture()
def sample_cube(rng, sample_wavelengths):
    """Create a synthetic ALES data cube with realistic dimensions.

    Returns a 3D array of shape (98, 63, 67) matching the ALES cube
    format (n_wavelengths, ny, nx).
    """
    n_wav = len(sample_wavelengths)
    ny, nx = 63, 67
    return rng.normal(100.0, 10.0, size=(n_wav, ny, nx))


def write_test_cube(
    filepath,
    cube,
    wavelengths,
    extra_header=None,
):
    """Write a synthetic ALES cube to a FITS file for testing.

    Args:
        filepath: Output path (.fits or .fits.gz).
        cube: 3D numpy array (n_wav, ny, nx).
        wavelengths: 1D array of wavelengths in microns.
        extra_header: Optional dict of additional header keywords.
    """
    header = fits.Header()
    for i, wav in enumerate(wavelengths):
        header[f"SLICE{i:03d}"] = (wav, "wavelength microns")
    header["LBT_PARA"] = "0.0"
    header["TIME-OBS"] = "12:00:00.000"
    header["LBT_ALT"] = "60.0"

    if extra_header:
        for key, value in extra_header.items():
            header[key] = value

    hdu = fits.PrimaryHDU(data=cube, header=header)
    hdu.writeto(filepath, overwrite=True)
