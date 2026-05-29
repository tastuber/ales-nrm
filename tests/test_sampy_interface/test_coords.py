"""Tests for ales_nrm.sampy_interface.coords module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ales_nrm.nrm.mask import ALES_PIXEL_SCALE_ARCSEC, NRMMask
from ales_nrm.sampy_interface.coords import (
    _compute_zero_spacing_radius,
    _convert_mask_to_sampy,
    _generate_narrow_filter,
    setup_sampy_coords,
)


@pytest.fixture()
def ales_mask():
    """Load the bundled ALES 6-hole mask."""
    return NRMMask.from_bundled()


@pytest.fixture()
def sample_wavelengths_short():
    """A small wavelength array for fast tests."""
    return np.array([3.0, 3.5, 4.0])


def _make_sampy_mocks(mock_make_coords):
    """Create properly linked sampy module mocks."""
    mock_module = MagicMock()
    mock_module.make_coords = mock_make_coords
    mock_sampy = MagicMock()
    mock_sampy.mask = mock_module
    return mock_sampy, mock_module


class TestConvertMaskToSampy:
    """Tests for _convert_mask_to_sampy."""

    def test_output_has_correct_shape(self, ales_mask, tmp_path):
        """Output file has 6 lines, 2 columns."""
        out = tmp_path / "mask.txt"
        _convert_mask_to_sampy(ales_mask, out)

        data = np.loadtxt(out)
        assert data.shape == (6, 2)

    def test_centered_coordinates_near_zero(self, ales_mask, tmp_path):
        """With center=True, centroid is at origin."""
        out = tmp_path / "mask.txt"
        _convert_mask_to_sampy(ales_mask, out, center=True)

        data = np.loadtxt(out)
        centroid = data.mean(axis=0)
        assert abs(centroid[0]) < 1e-10
        assert abs(centroid[1]) < 1e-10

    def test_uncentered_preserves_original(self, ales_mask, tmp_path):
        """With center=False, coordinates must match original."""
        out = tmp_path / "mask.txt"
        _convert_mask_to_sampy(ales_mask, out, center=False)

        data = np.loadtxt(out)
        for i, hole in enumerate(ales_mask.holes):
            assert data[i, 0] == pytest.approx(hole.x, abs=1e-8)
            assert data[i, 1] == pytest.approx(hole.y, abs=1e-8)

    def test_centered_values_match_shifted_holes(self, ales_mask, tmp_path):
        """Centered coordinates equal original minus centroid."""
        out = tmp_path / "mask.txt"
        _convert_mask_to_sampy(ales_mask, out, center=True)

        data = np.loadtxt(out)
        coords = np.array([[h.x, h.y] for h in ales_mask.holes])
        centroid = coords.mean(axis=0)
        expected = coords - centroid

        np.testing.assert_allclose(data, expected, atol=1e-8)

    def test_creates_parent_directories(self, ales_mask, tmp_path):
        """Intermediate directories are created."""
        out = tmp_path / "a" / "b" / "c" / "mask.txt"
        _convert_mask_to_sampy(ales_mask, out)

        assert out.exists()
        data = np.loadtxt(out)
        assert data.shape == (6, 2)


class TestGenerateNarrowFilter:
    """Tests for _generate_narrow_filter."""

    def test_default_parameters(self, tmp_path):
        """Default call produces correct structure."""
        out = tmp_path / "filter.txt"
        _generate_narrow_filter(3.5, out)

        with open(out) as f:
            lines = f.readlines()

        # Header + n_points(3) + 2 boundary = 6 lines
        assert len(lines) == 6
        assert "wavelength" in lines[0].lower()

        data = np.loadtxt(out, skiprows=1)
        assert data.shape == (5, 2)
        assert data[0, 1] == 0.0
        assert data[-1, 1] == 0.0
        assert all(data[1:-1, 1] == 1.0)

    def test_wavelength_centered(self, tmp_path):
        """Filter is centered on the given wavelength."""
        wl = 3.5
        out = tmp_path / "filter.txt"
        _generate_narrow_filter(wl, out)

        data = np.loadtxt(out, skiprows=1)
        interior = data[1:-1, 0]
        center = (interior[0] + interior[-1]) / 2.0
        assert center == pytest.approx(wl, abs=1e-8)

    def test_custom_bandwidth(self, tmp_path):
        """Custom bandwidth widens the filter."""
        bw = 0.05
        out = tmp_path / "filter.txt"
        _generate_narrow_filter(3.5, out, bandwidth_um=bw)

        data = np.loadtxt(out, skiprows=1)
        interior = data[1:-1, 0]
        assert interior[-1] - interior[0] == pytest.approx(bw, abs=1e-8)

    def test_custom_npoints(self, tmp_path):
        """Custom n_points changes row count."""
        out = tmp_path / "filter.txt"
        _generate_narrow_filter(3.5, out, n_points=5)

        data = np.loadtxt(out, skiprows=1)
        assert data.shape[0] == 7

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories created if absent."""
        out = tmp_path / "x" / "y" / "filter.txt"
        _generate_narrow_filter(3.5, out)
        assert out.exists()

    def test_sampy_can_read_format(self, tmp_path):
        """File matches SAMpy's np.loadtxt(skiprows=1)."""
        out = tmp_path / "filter.txt"
        _generate_narrow_filter(3.5, out)

        data = np.loadtxt(out, skiprows=1)
        assert data.ndim == 2
        assert data.shape[1] >= 2
        assert all(np.diff(data[:, 0]) > 0)


class TestComputeZeroSpacingRadius:
    """Tests for _compute_zero_spacing_radius."""

    def test_ales_values(self):
        """Produces reasonable value for ALES at edge wavelengths."""
        r_shortest_wave = _compute_zero_spacing_radius(
            wavelength_um=2.76787989,
            subaperture_diameter=0.784,
            n_pixels=501,
            pixel_scale=0.0345,
            scale_factor=1.0,
        )
        r_longest_wave = _compute_zero_spacing_radius(
            wavelength_um=4.2901481,
            subaperture_diameter=0.784,
            n_pixels=501,
            pixel_scale=0.0345,
            scale_factor=1.0,
        )
        assert r_shortest_wave == 24
        assert r_longest_wave == 15
        assert isinstance(r_shortest_wave, int)
        assert isinstance(r_longest_wave, int)

    def test_shorter_wavelength_larger_radius(self):
        """Shorter wavelength gives larger radius."""
        r_short = _compute_zero_spacing_radius(
            wavelength_um=1,
            subaperture_diameter=1.0,
            n_pixels=501,
            pixel_scale=0.01,
            scale_factor=1.0,
        )
        r_long = _compute_zero_spacing_radius(
            wavelength_um=5,
            subaperture_diameter=1.0,
            n_pixels=501,
            pixel_scale=0.01,
            scale_factor=1.0,
        )
        assert r_short > r_long

    def test_larger_npixels_larger_radius(self):
        """More pixels give larger radius."""
        r_small = _compute_zero_spacing_radius(
            wavelength_um=1,
            subaperture_diameter=1.0,
            n_pixels=501,
            pixel_scale=0.01,
            scale_factor=1.0,
        )
        r_large = _compute_zero_spacing_radius(
            wavelength_um=1,
            subaperture_diameter=1.0,
            n_pixels=1001,
            pixel_scale=0.01,
            scale_factor=1.0,
        )
        assert r_large > r_small

    def test_minimum_is_one(self):
        """Never returns less than 1."""
        r = _compute_zero_spacing_radius(
            wavelength_um=100.0,
            subaperture_diameter=1.0,
            n_pixels=11,
            pixel_scale=0.01,
            scale_factor=1.0,
        )
        assert r == 1

    def test_scale_factor(self):
        """Scale factor of 2 doubles radius.

        As radius is computed as the integer of a rounded float, this
        does not work for arbitrary values for the other input
        parameters of `_compute_zero_spacing_radius`.
        """
        r = _compute_zero_spacing_radius(
            wavelength_um=1.0,
            subaperture_diameter=1.0,
            n_pixels=100,
            pixel_scale=0.01,
            scale_factor=1.0,
        )
        r_twice = _compute_zero_spacing_radius(
            wavelength_um=1.0,
            subaperture_diameter=1.0,
            n_pixels=100,
            pixel_scale=0.01,
            scale_factor=2.0,
        )
        assert r_twice == 2.0 * r


class TestSetupSampyCoords:
    """Tests for setup_sampy_coords (mocked SAMpy)."""

    def test_calls_make_coords_per_wavelength(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """make_coords called once per wavelength."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            result = setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                tmp_path / "cache",
                force_recompute=True,
            )

        assert mock_make_coords.call_count == 3
        assert len(result) == 3

    def test_correct_default_arguments_passed(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """make_coords receives correct default params."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                tmp_path / "cache",
                force_recompute=True,
            )

        kw = mock_make_coords.call_args[1]
        assert kw["pixel_scale"] == ALES_PIXEL_SCALE_ARCSEC
        assert kw["n_pixels"] == 501
        assert kw["pupil_pixel_scale"] == 0.01
        assert kw["hole_shape"] == "circular"
        assert kw["rotation"] == 0
        assert kw["x_offset"] == 0
        assert kw["y_offset"] == 0
        assert kw["spectral_sampling"] == 1
        assert kw["recompute"] is True
        assert kw["fourier_cutoff"] == 0.4
        expected_diam = 2.0 * ales_mask.holes[0].radius
        assert kw["subaperture_diameter"] == pytest.approx(expected_diam)

    def test_custom_arguments_passed(self, ales_mask, tmp_path):
        """Custom parameters are forwarded correctly."""
        wls = np.array([3.5])
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                wls,
                tmp_path / "cache",
                pixel_scale=0.05,
                n_pixels=101,
                pupil_pixel_scale=0.005,
                zero_spacing_radius=30,
                force_recompute=True,
                fourier_cutoff=0.3,
            )

        kw = mock_make_coords.call_args[1]
        assert kw["pixel_scale"] == 0.05
        assert kw["n_pixels"] == 101
        assert kw["pupil_pixel_scale"] == 0.005
        assert kw["zero_spacing_radius"] == 30
        assert kw["fourier_cutoff"] == 0.3

    def test_auto_zero_spacing_varies_with_wavelength(
        self, ales_mask, tmp_path
    ):
        """Auto zero_spacing_radius differs per wavelength."""
        wls = np.array([2.8, 4.2])
        calls_kwargs = []
        mock_make_coords = MagicMock(
            side_effect=lambda **kw: calls_kwargs.append(kw)
        )
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                wls,
                tmp_path / "cache",
                zero_spacing_radius=None,
                force_recompute=True,
            )

        # Shorter wavelength -> larger radius
        zsr_short = calls_kwargs[0]["zero_spacing_radius"]
        zsr_long = calls_kwargs[1]["zero_spacing_radius"]
        assert zsr_short > zsr_long

    def test_fixed_zero_spacing_same_for_all(self, ales_mask, tmp_path):
        """Fixed zero_spacing_radius used for all wavelengths."""
        wls = np.array([2.8, 4.2])
        calls_kwargs = []
        mock_make_coords = MagicMock(
            side_effect=lambda **kw: calls_kwargs.append(kw)
        )
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                wls,
                tmp_path / "cache",
                zero_spacing_radius=25,
                force_recompute=True,
            )

        assert calls_kwargs[0]["zero_spacing_radius"] == 25
        assert calls_kwargs[1]["zero_spacing_radius"] == 25

    def test_caching_skips_existing(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Existing coordinate directories are skipped."""
        cache = tmp_path / "cache"
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        # Pre-create mask file and coord dirs
        mask_cache = cache / ales_mask.source_name
        mask_file = mask_cache / f"{ales_mask.source_name}_sampy.txt"
        mask_file.parent.mkdir(parents=True)
        mask_file.write_text("dummy")
        for wl in sample_wavelengths_short:
            d = mask_cache / "sampy_coords" / f"{float(wl):.4f}um"
            d.mkdir(parents=True)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            result = setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                cache,
                force_recompute=False,
            )

        assert mock_make_coords.call_count == 0
        assert len(result) == 3

    def test_incremental_wavelength_addition(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """New wavelengths computed, existing skipped."""
        cache = tmp_path / "cache"
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        # Pre-create mask file and one wavelength
        mask_cache = cache / ales_mask.source_name
        mask_file = mask_cache / f"{ales_mask.source_name}_sampy.txt"
        mask_file.parent.mkdir(parents=True)
        mask_file.write_text("dummy")
        existing = mask_cache / "sampy_coords" / "3.0000um"
        existing.mkdir(parents=True)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            result = setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                cache,
                force_recompute=False,
            )

        # Only 2 new wavelengths computed
        assert mock_make_coords.call_count == 2
        assert len(result) == 3

    def test_force_recompute_deletes_cache(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """force_recompute removes stale cache files."""
        cache = tmp_path / "cache"
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        # Pre-create stale artifact
        mask_cache = cache / ales_mask.source_name
        stale = mask_cache / "stale_artifact.txt"
        stale.parent.mkdir(parents=True)
        stale.write_text("should be deleted")

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                cache,
                force_recompute=True,
            )

        assert not stale.exists()
        assert mask_cache.exists()

    def test_return_dict_keys_are_wavelengths(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Return dict keys match input wavelengths."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            result = setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                tmp_path / "cache",
                force_recompute=True,
            )

        for wl in sample_wavelengths_short:
            assert float(wl) in result

    def test_return_dict_values_are_paths(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Return dict values are Path objects."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            result = setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                tmp_path / "cache",
                force_recompute=True,
            )

        for path in result.values():
            assert isinstance(path, Path)

    def test_directory_structure(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Cache uses mask-named subdirectory structure."""
        cache = tmp_path / "cache"
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                cache,
                force_recompute=True,
            )

        mask_cache = cache / ales_mask.source_name

        sampy_mask = mask_cache / f"{ales_mask.source_name}_sampy.txt"
        assert sampy_mask.exists()
        data = np.loadtxt(sampy_mask)
        assert data.shape == (6, 2)
        centroid = data.mean(axis=0)
        assert abs(centroid[0]) < 1e-10
        assert abs(centroid[1]) < 1e-10

        source_copy = mask_cache / f"{ales_mask.source_name}.txt"
        assert source_copy.exists()

        rotated = mask_cache / f"{ales_mask.source_name}_rotated.txt"
        assert rotated.exists()

        filter_dir = mask_cache / "filters"
        for wl in sample_wavelengths_short:
            f = filter_dir / f"filter_{float(wl):.4f}um.txt"
            assert f.exists()

        for wl in sample_wavelengths_short:
            d = mask_cache / "sampy_coords" / f"{float(wl):.4f}um"
            assert d.exists()

    def test_rotated_mask_has_hole_data(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Rotated mask file contains rotated hole coordinates."""
        cache = tmp_path / "cache"
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {"sampy": mock_sampy, "sampy.mask": mock_module},
        ):
            setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                cache,
                force_recompute=True,
            )

        rotated = (
            cache
            / ales_mask.source_name
            / f"{ales_mask.source_name}_rotated.txt"
        )
        content = rotated.read_text()
        assert "Rotation:" in content
        for hole in ales_mask.holes:
            assert hole.name in content

    def test_source_mask_is_verbatim_copy(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Source mask file contains original content."""
        cache = tmp_path / "cache"
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {"sampy": mock_sampy, "sampy.mask": mock_module},
        ):
            setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                cache,
                force_recompute=True,
            )

        source_copy = (
            cache / ales_mask.source_name / f"{ales_mask.source_name}.txt"
        )
        assert source_copy.read_text() == ales_mask.source_content

    def test_import_error_message(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Clear error when SAMpy is not installed."""
        with patch.dict(
            sys.modules,
            {"sampy": None, "sampy.mask": None},
        ):
            with pytest.raises(ImportError, match="SAMpy is required"):
                setup_sampy_coords(
                    ales_mask,
                    sample_wavelengths_short,
                    tmp_path / "cache",
                )

    def test_output_dir_has_trailing_slash(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """SAMpy expects output_dir to end with '/'."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            setup_sampy_coords(
                ales_mask,
                sample_wavelengths_short,
                tmp_path / "cache",
                force_recompute=True,
            )

        kw = mock_make_coords.call_args[1]
        assert kw["output_dir"].endswith("/")

    def test_suppress_plots_default_true(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Default suppress_plots=True activates ioff."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            with patch("matplotlib.pyplot.ioff") as mock_ioff:
                setup_sampy_coords(
                    ales_mask,
                    sample_wavelengths_short,
                    tmp_path / "cache",
                    force_recompute=True,
                    suppress_plots=True,
                )
                mock_ioff.assert_called()

    def test_suppress_plots_false_no_suppression(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """suppress_plots=False skips plot suppression."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            with patch("matplotlib.pyplot.ioff") as mock_ioff:
                setup_sampy_coords(
                    ales_mask,
                    sample_wavelengths_short,
                    tmp_path / "cache",
                    force_recompute=True,
                    suppress_plots=False,
                )
                mock_ioff.assert_not_called()

    def test_single_wavelength_scalar(self, ales_mask, tmp_path):
        """Single float wavelength accepted."""
        mock_make_coords = MagicMock()
        mock_sampy, mock_module = _make_sampy_mocks(mock_make_coords)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.mask": mock_module,
            },
        ):
            result = setup_sampy_coords(
                ales_mask,
                3.5,
                tmp_path / "cache",
                force_recompute=True,
            )

        assert mock_make_coords.call_count == 1
        assert 3.5 in result


@pytest.mark.sampy
class TestSetupSampyCoordsIntegration:
    """Integration tests that call real SAMpy."""

    def test_single_wavelength_produces_fits(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Real make_coords produces expected FITS files.

        For 6 holes: 15 baselines, 20 closing triangles.
        """
        cache = tmp_path / "cache"

        result = setup_sampy_coords(
            ales_mask,
            sample_wavelengths_short,
            cache,
            force_recompute=True,
        )

        coord_dir = result[sample_wavelengths_short[0]]
        assert (coord_dir / "bl_pix.fits").exists()
        assert (coord_dir / "bl_uvs.fits").exists()
        assert (coord_dir / "cp_pix.fits").exists()
        assert (coord_dir / "cp_uvs.fits").exists()
        assert (coord_dir / "cvis_pix.fits").exists()
        assert (coord_dir / "cvis_uvs.fits").exists()
        assert (coord_dir / "syn_pspec.fits").exists()
        assert (coord_dir / "k_mat.txt").exists()
        for i in range(15):
            assert (coord_dir / f"cvis_ind{i}.fits").exists()
            assert (coord_dir / f"v2_ind{i}.fits").exists()
        for i in range(20):
            for j in range(3):
                assert (coord_dir / f"ind{i}_vert{j}.fits").exists()

    def test_multiple_wavelengths_all_complete(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Multiple wavelengths all produce output."""
        cache = tmp_path / "cache"

        result = setup_sampy_coords(
            ales_mask,
            sample_wavelengths_short,
            cache,
            force_recompute=True,
        )

        for _wl, coord_dir in result.items():
            assert (coord_dir / "bl_pix.fits").exists()
            assert (coord_dir / "bl_uvs.fits").exists()
            assert (coord_dir / "cp_pix.fits").exists()
            assert (coord_dir / "cp_uvs.fits").exists()
            assert (coord_dir / "cvis_pix.fits").exists()
            assert (coord_dir / "cvis_uvs.fits").exists()
            assert (coord_dir / "syn_pspec.fits").exists()
            assert (coord_dir / "k_mat.txt").exists()
            for i in range(15):
                assert (coord_dir / f"cvis_ind{i}.fits").exists()
                assert (coord_dir / f"v2_ind{i}.fits").exists()
            for i in range(20):
                for j in range(3):
                    assert (coord_dir / f"ind{i}_vert{j}.fits").exists()

    def test_no_interactive_plots_shown(
        self, ales_mask, sample_wavelengths_short, tmp_path
    ):
        """Completion confirms plot suppression.

        make_coords completes without blocking on
        plt.show().
        """
        import matplotlib.pyplot as plt

        cache = tmp_path / "cache"

        result = setup_sampy_coords(
            ales_mask,
            sample_wavelengths_short,
            cache,
            force_recompute=True,
        )
        assert len(result) == len(sample_wavelengths_short)
        # No leftover figures
        assert len(plt.get_fignums()) == 0
