"""Tests for FITS I/O routines."""

import numpy as np
import pytest
from astropy.io import fits

from ales_nrm.io.read_fits import (
    _pad_to_square,
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
        """Raise ValueError for a filename not matching the pattern."""
        with pytest.raises(ValueError, match="does not match"):
            parse_file_number("not_a_valid_file.txt")

    def test_raw_filename(self):
        """Parse file number from a raw (non-cube) LMIRCam filename."""
        assert parse_file_number("lm_251108_005265.fits.gz") == 5265


class TestReadWavelengths:
    """Tests for read_wavelengths."""

    def test_correct_wavelengths(self, sample_wavelengths):
        """Recover the wavelength array from SLICEnnn keywords."""
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


class TestPadToSquare:
    """Tests for the _pad_to_square helper."""

    def test_already_odd_square(self):
        """Odd square input is returned unchanged."""
        cube = np.ones((10, 67, 67))
        result = _pad_to_square(cube)
        assert result is cube

    def test_even_square_padded_to_odd(self):
        """Even square input is padded to odd."""
        cube = np.ones((10, 64, 64))
        result = _pad_to_square(cube)
        assert result.shape == (10, 65, 65)

    def test_ny_less_than_nx_odd(self):
        """Pad y when ny < nx, result already odd."""
        cube = np.ones((5, 63, 67))
        result = _pad_to_square(cube)
        assert result.shape == (5, 67, 67)

    def test_nx_less_than_ny_odd(self):
        """Pad x when nx < ny, result already odd."""
        cube = np.ones((5, 67, 63))
        result = _pad_to_square(cube)
        assert result.shape == (5, 67, 67)

    def test_even_max_padded_to_odd(self):
        """Even max dimension is padded to next odd."""
        cube = np.ones((5, 60, 64))
        result = _pad_to_square(cube)
        assert result.shape == (5, 65, 65)

    def test_data_at_origin(self):
        """Data occupies the top-left corner."""
        cube = np.ones((2, 63, 67))
        result = _pad_to_square(cube)
        np.testing.assert_array_equal(
            result[:, :63, :67],
            1.0,
        )

    def test_padding_is_zeros_bottom(self):
        """Bottom rows are zero."""
        cube = np.ones((2, 63, 67))
        result = _pad_to_square(cube)
        np.testing.assert_array_equal(
            result[:, 63:, :],
            0.0,
        )

    def test_padding_right_columns(self):
        """Right columns are zero when nx < n_out."""
        cube = np.ones((2, 67, 63))
        result = _pad_to_square(cube)
        np.testing.assert_array_equal(
            result[:, :, 63:],
            0.0,
        )

    def test_even_square_padding_is_zeros(self):
        """Padded row and column are zero for even input."""
        cube = np.ones((2, 64, 64))
        result = _pad_to_square(cube)
        # Row 64 should be zero.
        np.testing.assert_array_equal(
            result[:, 64, :],
            0.0,
        )
        # Column 64 should be zero.
        np.testing.assert_array_equal(
            result[:, :, 64],
            0.0,
        )

    def test_preserves_dtype(self, rng):
        """Output dtype matches input dtype."""
        cube = rng.normal(size=(3, 63, 67)).astype(np.float32)
        result = _pad_to_square(cube)
        assert result.dtype == np.float32

    def test_total_flux_preserved(self, rng):
        """Total flux is preserved since padding is zeros."""
        cube = rng.normal(100, 10, size=(5, 63, 67))
        result = _pad_to_square(cube)
        np.testing.assert_allclose(
            np.sum(result),
            np.sum(cube),
        )

    def test_difference_of_one_odd_result(self):
        """Dimensions 66×67: max is 67 (odd), no extra pad."""
        cube = np.ones((3, 66, 67))
        result = _pad_to_square(cube)
        assert result.shape == (3, 67, 67)
        np.testing.assert_array_equal(
            result[:, :66, :],
            1.0,
        )
        np.testing.assert_array_equal(
            result[:, 66, :],
            0.0,
        )

    def test_large_difference(self):
        """Handle large difference between dimensions."""
        cube = np.ones((2, 30, 67))
        result = _pad_to_square(cube)
        assert result.shape == (2, 67, 67)
        np.testing.assert_array_equal(
            result[:, :30, :67],
            1.0,
        )
        np.testing.assert_array_equal(
            result[:, 30:, :],
            0.0,
        )

    def test_both_even_dimensions(self):
        """Both even dimensions pad to odd square."""
        cube = np.ones((3, 62, 68))
        result = _pad_to_square(cube)
        # max(62, 68) = 68, ensure_odd(68) = 69
        assert result.shape == (3, 69, 69)
        np.testing.assert_array_equal(
            result[:, :62, :68],
            1.0,
        )
        np.testing.assert_array_equal(
            result[:, 62:, :],
            0.0,
        )
        np.testing.assert_array_equal(
            result[:, :, 68:],
            0.0,
        )

    def test_result_always_odd(self):
        """Result dimension is always odd."""
        for ny, nx in [(63, 67), (64, 64), (100, 50), (51, 52)]:
            cube = np.ones((2, ny, nx))
            result = _pad_to_square(cube)
            assert result.shape[-1] % 2 == 1
            assert result.shape[-2] % 2 == 1
            assert result.shape[-1] == result.shape[-2]


class TestReadCube:
    """Tests for read_cube."""

    def test_read_fits(self, tmp_path, sample_cube, sample_wavelengths):
        """Read a .fits file and recover wavelengths."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cube, wavelengths, header = read_cube(filepath)

        # Padded from (98, 63, 67) to (98, 67, 67).
        assert cube.shape == (98, 67, 67)
        np.testing.assert_allclose(wavelengths, sample_wavelengths)

    def test_read_gzipped(self, tmp_path, sample_cube, sample_wavelengths):
        """Read a gzipped .fits.gz file successfully."""
        filepath = tmp_path / "cube_lm_251108_005001.fits.gz"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cube, wavelengths, header = read_cube(filepath)

        # Padded from (98, 63, 67) to (98, 67, 67).
        assert cube.shape == (98, 67, 67)

    def test_read_pads_to_square(
        self, tmp_path, sample_cube, sample_wavelengths
    ):
        """Non-square cube is padded to odd square on read."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cube, _, _ = read_cube(filepath)
        assert cube.shape[-2] == cube.shape[-1]
        assert cube.shape[-1] % 2 == 1
        assert cube.shape == (98, 67, 67)

    def test_read_preserves_data_in_padded_region(
        self, tmp_path, sample_cube, sample_wavelengths
    ):
        """Data values preserved in top-left corner."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cube, _, _ = read_cube(filepath)

        np.testing.assert_allclose(
            cube[:, :63, :67],
            sample_cube,
        )

    def test_read_even_square_pads_to_odd(
        self, tmp_path, rng, sample_wavelengths
    ):
        """An even square cube is padded to odd square."""
        even_cube = rng.normal(100.0, 10.0, size=(98, 64, 64))
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, even_cube, sample_wavelengths)

        cube, _, _ = read_cube(filepath)
        assert cube.shape == (98, 65, 65)
        np.testing.assert_allclose(
            cube[:, :64, :64],
            even_cube,
        )

    def test_read_odd_square_cube_unchanged(
        self, tmp_path, rng, sample_wavelengths
    ):
        """An odd square cube is not modified."""
        odd_cube = rng.normal(100.0, 10.0, size=(98, 65, 65))
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, odd_cube, sample_wavelengths)

        cube, _, _ = read_cube(filepath)
        assert cube.shape == (98, 65, 65)
        np.testing.assert_allclose(cube, odd_cube)

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
        """Preserve header keywords after reading."""
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

    def test_empty_primary_hdu(self, tmp_path):
        """Raise ValueError when primary HDU contains no data."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        hdu = fits.PrimaryHDU()
        hdu.writeto(filepath)

        with pytest.raises(ValueError, match="No data in primary HDU"):
            read_cube(filepath)

    def test_non_3d_data(self, tmp_path, rng):
        """Raise ValueError when data is not 3-dimensional."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        data_2d = rng.normal(size=(63, 67))
        hdu = fits.PrimaryHDU(data=data_2d)
        hdu.writeto(filepath)

        with pytest.raises(ValueError, match="Expected a 3D data cube"):
            read_cube(filepath)


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

    def test_skips_non_matching_filenames(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Skip files that match the glob but not the number pattern."""
        # Valid file
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        # File that matches glob but has no 6-digit number
        bad_filepath = tmp_path / "cube_lm_badname.fits"
        write_test_cube(bad_filepath, sample_cube, sample_wavelengths)

        paths = find_cubes(tmp_path, file_range=(5001, 5001))
        assert len(paths) == 1


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
        # Padded from (98, 63, 67) to (98, 67, 67).
        assert cubes.shape == (3, 98, 67, 67)
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
        """Raise ValueError when wavelengths differ between files."""
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
        """Raise ValueError when dimensions differ between files."""
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
        """Read/write preserves pixel values in top-left corner."""
        filepath = tmp_path / "cube_lm_251108_005001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, _, _, _ = read_cubes(tmp_path)

        np.testing.assert_allclose(
            cubes[0, :, :63, :67],
            sample_cube,
        )

    def test_dtype_preserved(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Test I/O dtype match after native byte-order conversion."""
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
        """Verify file_numbers array length matches cube number."""
        for num in [5001, 5002, 5003]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(filepath, sample_cube, sample_wavelengths)

        cubes, _, file_numbers, _ = read_cubes(tmp_path)

        assert len(file_numbers) == cubes.shape[0]
