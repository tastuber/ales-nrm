"""Tests for FITS I/O routines."""

import numpy as np
import pytest

from ales_nrm.io.read_fits import (
    read_cube,
    read_wavelengths,
)
from tests.conftest import write_test_cube


class TestReadWavelengths:
    """Tests for read_wavelengths."""

    def test_correct_wavelengths(self, sample_wavelengths):
        """Recover the correct wavelength array from SLICEnnn keywords."""
        from astropy.io import fits as fits_mod

        header = fits_mod.Header()
        header["NAXIS3"] = len(sample_wavelengths)
        for i, wav in enumerate(sample_wavelengths):
            header[f"SLICE{i:03d}"] = wav

        result = read_wavelengths(header)
        np.testing.assert_allclose(result, sample_wavelengths)

    def test_missing_naxis3(self):
        """Raise KeyError when NAXIS3 is absent from the header."""
        from astropy.io import fits as fits_mod

        header = fits_mod.Header()
        with pytest.raises(KeyError):
            read_wavelengths(header)

    def test_missing_slice_keyword(self):
        """Raise ValueError when a SLICEnnn keyword is missing."""
        from astropy.io import fits as fits_mod

        header = fits_mod.Header()
        header["NAXIS3"] = 5
        header["SLICE000"] = 3.0
        header["SLICE001"] = 3.1
        # Missing SLICE002 through SLICE004
        with pytest.raises(ValueError, match="missing from the header"):
            read_wavelengths(header)


class TestReadCube:
    """Tests for read_cube."""

    def test_read_fits(self, tmp_path, sample_cube, sample_wavelengths):
        """Read a .fits file and recover data and wavelengths."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cube, wavelengths, header = read_cube(filepath)

        assert cube.shape == sample_cube.shape
        np.testing.assert_allclose(cube, sample_cube)
        np.testing.assert_allclose(wavelengths, sample_wavelengths)

    def test_read_gzipped(self, tmp_path, sample_cube, sample_wavelengths):
        """Read a gzipped .fits.gz file successfully."""
        filepath = tmp_path / "cube_lm_251108_005001.fits.gz"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cube, wavelengths, header = read_cube(filepath)

        assert cube.shape == sample_cube.shape

    def test_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            read_cube(tmp_path / "nonexistent.fits")

    def test_preserves_header_keywords(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Preserve observatory-specific header keywords after reading."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(
            filepath,
            sample_cube,
            sample_wavelengths,
            extra_header={"LBT_PARA": "15.5", "TIME-OBS": "10:30:00.000"},
        )

        _, _, header = read_cube(filepath)
        assert header["LBT_PARA"] == "15.5"
        assert header["TIME-OBS"] == "10:30:00.000"
