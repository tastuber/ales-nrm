"""Tests for FITS I/O routines."""

import numpy as np
import pytest

from ales_nrm.io.read_fits import (
    find_cubes,
    parse_file_number,
    read_cube,
    read_cubes,
    read_wavelengths,
)
from tests.conftest import write_test_cube


class TestParseFileNumber:
    """Tests for parse_file_number."""

    def test_standard_fits(self):
        """Parse file number from a standard .fits filename."""
        assert parse_file_number("cube_lm_251108_005265.fits") == 5265

    def test_gzipped_fits(self):
        """Parse file number from a gzipped .fits.gz filename."""
        assert parse_file_number("cube_lm_251108_005265.fits.gz") == 5265

    def test_full_path(self, tmp_path):
        """Parse file number when given a full filesystem path."""
        path = tmp_path / "cube_lm_251108_000001.fits"
        assert parse_file_number(path) == 1

    def test_leading_zeros(self):
        """Parse file number correctly when it has leading zeros."""
        assert parse_file_number("cube_lm_251108_000042.fits") == 42

    def test_invalid_filename(self):
        """Raise ValueError for a filename that does not match the pattern."""
        with pytest.raises(ValueError, match="does not match"):
            parse_file_number("not_a_valid_file.txt")

    def test_raw_filename(self):
        """Parse file number from a raw (non-cube) LMIRCam filename."""
        assert parse_file_number("lm_251108_005265.fits.gz") == 5265


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


class TestFindCubes:
    """Tests for find_cubes."""

    def test_find_all(self, tmp_path, sample_cube, sample_wavelengths):
        """Find all FITS cubes in a directory."""
        for num in [5001, 5002, 5003]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        paths = find_cubes(tmp_path)
        assert len(paths) == 3

    def test_find_with_range(self, tmp_path, sample_cube, sample_wavelengths):
        """Find only files within the specified file number range."""
        for num in [5001, 5002, 5003, 5004, 5005]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        paths = find_cubes(tmp_path, file_range=(5002, 5004))
        assert len(paths) == 3
        numbers = [parse_file_number(p) for p in paths]
        assert numbers == [5002, 5003, 5004]

    def test_find_gzipped(self, tmp_path, sample_cube, sample_wavelengths):
        """Find gzipped .fits.gz files."""
        filepath = tmp_path / "cube_lm_251108_005001.fits.gz"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        paths = find_cubes(tmp_path)
        assert len(paths) == 1

    def test_sorted_output(self, tmp_path, sample_cube, sample_wavelengths):
        """Return files sorted by file number."""
        for num in [5003, 5001, 5002]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        paths = find_cubes(tmp_path)
        numbers = [parse_file_number(p) for p in paths]
        assert numbers == sorted(numbers)

    def test_directory_not_found(self, tmp_path):
        """Raise FileNotFoundError for a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            find_cubes(tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path):
        """Return an empty list for a directory with no FITS files."""
        paths = find_cubes(tmp_path)
        assert paths == []


class TestReadCubes:
    """Tests for read_cubes."""

    def test_read_multiple(self, tmp_path, sample_cube, sample_wavelengths):
        """Read multiple cubes into a 4D array with correct shape."""
        for num in [5001, 5002, 5003]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, wavelengths, file_numbers, headers = read_cubes(tmp_path)

        assert isinstance(cubes, np.ndarray)
        assert cubes.ndim == 4
        assert cubes.shape == (3, *sample_cube.shape)
        assert len(headers) == 3
        np.testing.assert_allclose(wavelengths, sample_wavelengths)

    def test_read_with_range(self, tmp_path, sample_cube, sample_wavelengths):
        """Read only cubes within the specified file number range."""
        for num in [5001, 5002, 5003, 5004]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, wavelengths, file_numbers, headers = read_cubes(
            tmp_path,
            file_range=(5002, 5003),
        )
        assert cubes.shape[0] == 2

    def test_no_files_found(self, tmp_path):
        """Raise FileNotFoundError when no matching files exist."""
        with pytest.raises(FileNotFoundError, match="No FITS cubes found"):
            read_cubes(tmp_path)

    def test_inconsistent_wavelengths(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Raise ValueError when wavelength grids differ between files."""
        filepath1 = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath1, sample_cube, sample_wavelengths)

        shifted = sample_wavelengths + 0.5
        filepath2 = tmp_path / "cube_lm_251108_005002.fits"
        write_test_cube(filepath2, sample_cube, shifted)

        with pytest.raises(ValueError, match="Wavelength grid"):
            read_cubes(tmp_path)

    def test_inconsistent_shape(
        self,
        tmp_path,
        rng,
        sample_wavelengths,
    ):
        """Raise ValueError when spatial dimensions differ between files."""
        cube1 = rng.normal(size=(98, 63, 67))
        filepath1 = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath1, cube1, sample_wavelengths)

        cube2 = rng.normal(size=(98, 50, 50))
        filepath2 = tmp_path / "cube_lm_251108_005002.fits"
        write_test_cube(filepath2, cube2, sample_wavelengths)

        with pytest.raises(ValueError, match="Cube shape"):
            read_cubes(tmp_path)

    def test_data_values_preserved(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Verify that pixel values are preserved through read/write."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, _, _, _ = read_cubes(tmp_path)

        np.testing.assert_allclose(cubes[0], sample_cube)

    def test_dtype_preserved(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Test dtype output/input match after native byte-order conversion."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, _, _, _ = read_cubes(tmp_path)

        assert cubes.dtype == sample_cube.dtype

    def test_file_numbers_returned(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Return correct file numbers as an integer array."""
        for num in [5010, 5011, 5012]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        _, _, file_numbers, _ = read_cubes(tmp_path)

        assert isinstance(file_numbers, np.ndarray)
        assert file_numbers.dtype == int
        np.testing.assert_array_equal(file_numbers, [5010, 5011, 5012])

    def test_file_numbers_match_range(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Return only file numbers within the requested range."""
        for num in [5001, 5002, 5003, 5004, 5005]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        _, _, file_numbers, _ = read_cubes(
            tmp_path,
            file_range=(5002, 5004),
        )
        np.testing.assert_array_equal(file_numbers, [5002, 5003, 5004])

    def test_file_numbers_length_matches_cubes(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Verify file_numbers array length matches the number of cubes."""
        for num in [5001, 5002, 5003]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, _, file_numbers, _ = read_cubes(tmp_path)

        assert len(file_numbers) == cubes.shape[0]
