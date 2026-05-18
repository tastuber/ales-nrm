"""Tests for mask rotation angle calibration."""

from unittest.mock import patch

import numpy as np
import pytest

from ales_nrm.nrm.mask import (
    ALES_PIXEL_SCALE_ARCSEC,
    NRMMask,
)
from ales_nrm.nrm.mask_rotation import (
    _compute_angle_from_centroids,
    _fit_2d_gaussian,
    _fit_2d_gaussian_at_position,
    _sample_power_at_positions,
    find_mask_rotation_angle,
)
from ales_nrm.utilities import rotate_points_2d


@pytest.fixture()
def bundled_mask():
    """Load the bundled LBTI NRM6 SX mask."""
    return NRMMask.from_bundled("lbti_nrm6_sx")


@pytest.fixture()
def three_hole_mask():
    """Create a simple 3-hole mask for testing."""
    from ales_nrm.nrm.mask import Hole

    holes = [
        Hole("H1", -1.0, 0.0, 0.4),
        Hole("H2", 1.0, 0.0, 0.4),
        Hole("H3", 0.0, 1.5, 0.4),
    ]
    mask = NRMMask(
        primary_diameter=8.4,
        holes=holes,
    )
    mask._compute_baselines()
    return mask


class TestSamplePowerAtPositions:
    """Tests for _sample_power_at_positions."""

    def test_integer_positions(self):
        """Integer positions return exact pixel values."""
        ps = np.zeros((11, 11))
        ps[5, 5] = 100.0
        positions = np.array([[5.0, 5.0]])
        values = _sample_power_at_positions(ps, positions)
        assert values[0] == pytest.approx(100.0, abs=1e-6)

    def test_out_of_bounds_returns_zero(self):
        """Positions outside frame return zero."""
        ps = np.ones((11, 11))
        positions = np.array([[-5.0, -5.0]])
        values = _sample_power_at_positions(ps, positions)
        assert values[0] == pytest.approx(0.0)

    def test_interpolation(self):
        """Sub-pixel positions are interpolated."""
        ps = np.zeros((11, 11))
        ps[5, 5] = 100.0
        positions = np.array([[5.1, 5.0]])
        values = _sample_power_at_positions(ps, positions)
        assert 0.0 < values[0] < 100.0

    def test_multiple_positions(self):
        """Multiple positions return correct array shape and values."""
        ps = np.ones((11, 11)) * 42.0
        positions = np.array([[3.0, 3.0], [5.0, 5.0], [7.0, 7.0]])
        values = _sample_power_at_positions(ps, positions)
        assert values.shape == (3,)
        np.testing.assert_allclose(values, 42.0, atol=1e-6)


class TestFit2DGaussian:
    """Tests for _fit_2d_gaussian."""

    def test_known_center(self):
        """Recovers center of a known Gaussian."""
        size = 7
        y, x = np.mgrid[0:size, 0:size]
        y0, x0 = 3.2, 3.7
        sigma = 1.2
        cutout = (
            50.0
            * np.exp(
                -(
                    (y - y0) ** 2 / (2 * sigma**2)
                    + (x - x0) ** 2 / (2 * sigma**2)
                )
            )
            + 5.0
        )
        fy, fx, success = _fit_2d_gaussian(cutout, (3.0, 4.0))
        assert success is True
        assert fy == pytest.approx(y0, abs=0.1)
        assert fx == pytest.approx(x0, abs=0.1)

    def test_noisy_gaussian(self):
        """Recovers approximate center with noise."""
        rng = np.random.default_rng(42)
        size = 7
        y, x = np.mgrid[0:size, 0:size]
        y0, x0 = 3.5, 3.5
        sigma = 1.0
        cutout = 100.0 * np.exp(
            -((y - y0) ** 2 / (2 * sigma**2) + (x - x0) ** 2 / (2 * sigma**2))
        ) + rng.normal(0, 2, (size, size))
        cutout = np.clip(cutout, 0, None)
        fy, fx, success = _fit_2d_gaussian(cutout, (3.5, 3.5))
        assert success is True
        assert fy == pytest.approx(y0, abs=0.5)
        assert fx == pytest.approx(x0, abs=0.5)

    def test_flat_cutout_fails_gracefully(self):
        """Flat cutout does not crash."""
        cutout = np.ones((5, 5)) * 10.0
        fy, fx, success = _fit_2d_gaussian(cutout, (2.5, 2.5))
        assert isinstance(success, bool)

    def test_runtime_error_returns_initial_center(self):
        """Curve_fit RuntimeError returns initial center and failure."""
        cutout = np.ones((5, 5)) * 10.0
        with patch(
            "ales_nrm.nrm.mask_rotation.curve_fit",
            side_effect=RuntimeError("mock failure"),
        ):
            fy, fx, success = _fit_2d_gaussian(cutout, (2.0, 3.0))
        assert success is False
        assert fy == pytest.approx(2.0)
        assert fx == pytest.approx(3.0)

    def test_value_error_returns_initial_center(self):
        """Curve_fit ValueError returns initial center and failure."""
        cutout = np.ones((5, 5)) * 10.0
        with patch(
            "ales_nrm.nrm.mask_rotation.curve_fit",
            side_effect=ValueError("mock value error"),
        ):
            fy, fx, success = _fit_2d_gaussian(cutout, (1.5, 2.5))
        assert success is False
        assert fy == pytest.approx(1.5)
        assert fx == pytest.approx(2.5)

    def test_center_outside_bounds_returns_initial(self):
        """Non-valid center returns initial center and failure."""
        mock_popt = np.array([100.0, -5.0, -5.0, 1.0, 1.0, 0.0])
        mock_pcov = np.eye(6)
        with patch(
            "ales_nrm.nrm.mask_rotation.curve_fit",
            return_value=(mock_popt, mock_pcov),
        ):
            fy, fx, success = _fit_2d_gaussian(np.ones((5, 5)), (2.5, 2.5))
        assert success is False
        assert fy == pytest.approx(2.5)
        assert fx == pytest.approx(2.5)


class TestFit2DGaussianAtPosition:
    """Tests for _fit_2d_gaussian_at_position edge cases."""

    def test_position_near_top_edge(self):
        """Position too close to top edge returns failure."""
        ps = np.ones((67, 67)) * 10.0
        position = np.array([1.0, 33.0])
        fy, fx, success = _fit_2d_gaussian_at_position(
            ps, position, cutout_size=5
        )
        assert success is False
        assert fy == pytest.approx(1.0)
        assert fx == pytest.approx(33.0)

    def test_position_near_bottom_edge(self):
        """Position too close to bottom edge returns failure."""
        ps = np.ones((67, 67)) * 10.0
        position = np.array([66.0, 33.0])
        fy, fx, success = _fit_2d_gaussian_at_position(
            ps, position, cutout_size=5
        )
        assert success is False
        assert fy == pytest.approx(66.0)
        assert fx == pytest.approx(33.0)

    def test_position_near_left_edge(self):
        """Position too close to left edge returns failure."""
        ps = np.ones((67, 67)) * 10.0
        position = np.array([33.0, 0.0])
        fy, fx, success = _fit_2d_gaussian_at_position(
            ps, position, cutout_size=5
        )
        assert success is False
        assert fy == pytest.approx(33.0)
        assert fx == pytest.approx(0.0)

    def test_position_near_right_edge(self):
        """Position too close to right edge returns failure."""
        ps = np.ones((67, 67)) * 10.0
        position = np.array([33.0, 66.0])
        fy, fx, success = _fit_2d_gaussian_at_position(
            ps, position, cutout_size=5
        )
        assert success is False
        assert fy == pytest.approx(33.0)
        assert fx == pytest.approx(66.0)

    def test_valid_position_with_gaussian(self):
        """Valid interior position with Gaussian succeeds."""
        y, x = np.mgrid[0:67, 0:67]
        ps = 100.0 * np.exp(
            -((y - 33.0) ** 2 + (x - 33.0) ** 2) / (2 * 1.5**2)
        )
        position = np.array([33.0, 33.0])
        fy, fx, success = _fit_2d_gaussian_at_position(
            ps, position, cutout_size=7
        )
        assert success is True
        assert fy == pytest.approx(33.0, abs=0.5)
        assert fx == pytest.approx(33.0, abs=0.5)


class TestComputeAngleFromCentroids:
    """Tests for _compute_angle_from_centroids."""

    def test_zero_rotation(self):
        """Identical positions yield zero angle."""
        center = (33.0, 33.0)
        analytic = np.array([[30.0, 35.0], [36.0, 28.0], [33.0, 40.0]])
        angle = _compute_angle_from_centroids(
            analytic, analytic, center, 0.0, bound_deg=10.0
        )
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_known_rotation(self):
        """Recovers a known rotation angle."""
        center = (33.0, 33.0)
        analytic = np.array([[30.0, 35.0], [36.0, 28.0], [25.0, 40.0]])
        true_angle = 5.0
        measured = rotate_points_2d(analytic, center, true_angle)
        angle = _compute_angle_from_centroids(
            analytic,
            measured,
            center,
            initial_angle_deg=3.0,
            bound_deg=10.0,
        )
        assert angle == pytest.approx(true_angle, abs=0.01)

    def test_single_baseline(self):
        """Works with a single baseline position."""
        center = (33.0, 33.0)
        analytic = np.array([[30.0, 38.0]])
        true_angle = -3.0
        measured = rotate_points_2d(analytic, center, true_angle)
        angle = _compute_angle_from_centroids(
            analytic,
            measured,
            center,
            initial_angle_deg=-2.0,
            bound_deg=5.0,
        )
        assert angle == pytest.approx(true_angle, abs=0.1)


class TestFindMaskRotationAngle:
    """Integration tests for find_mask_rotation_angle."""

    def test_synthetic_zero_rotation(self, bundled_mask):
        """Recover zero rotation from synthetic power spectrum."""
        wavelength = 3.5
        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            pixel_scale_arcsec=ALES_PIXEL_SCALE_ARCSEC,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        wavelengths = np.array([wavelength])
        result = find_mask_rotation_angle(
            ps,
            bundled_mask,
            wavelengths,
            angle_range=(-10.0, 10.0),
            n_grid=41,
            refine=False,
        )
        assert abs(result["angle_deg"]) < 2.0
        assert result["step_a_angle_deg"] is not None
        assert result["angles_all"] is None
        assert result["measured_centers"] is None

    def test_synthetic_known_rotation(self, three_hole_mask):
        """Recover a known rotation from rotated power spectrum."""
        from scipy.ndimage import rotate as ndimage_rotate

        wavelength = 3.5
        true_angle = 7.0
        ps = three_hole_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            pixel_scale_arcsec=ALES_PIXEL_SCALE_ARCSEC,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        ps_rotated = ndimage_rotate(
            ps,
            true_angle,
            reshape=False,
            order=3,
            mode="constant",
            cval=0.0,
        )
        wavelengths = np.array([wavelength])
        result = find_mask_rotation_angle(
            ps_rotated,
            three_hole_mask,
            wavelengths,
            angle_range=(-15.0, 15.0),
            n_grid=61,
            refine=False,
        )
        assert result["angle_deg"] == pytest.approx(-true_angle, abs=2.0)

    def test_2d_input(self, bundled_mask):
        """Accepts 2D input array."""
        wavelength = 3.5
        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        result = find_mask_rotation_angle(
            ps,
            bundled_mask,
            np.array([wavelength]),
            n_grid=21,
            refine=False,
        )
        assert "angle_deg" in result

    def test_3d_input(self, bundled_mask):
        """Accepts 3D input array."""
        wavelengths = np.array([3.0, 3.5, 4.0])
        ps_3d = np.array(
            [
                bundled_mask.compute_synthetic_power_spectrum(
                    wavelength=w,
                    n_pixels_image=67,
                    n_pixels_pupil=501,
                )
                for w in wavelengths
            ]
        )
        result = find_mask_rotation_angle(
            ps_3d,
            bundled_mask,
            wavelengths,
            n_grid=21,
            refine=False,
        )
        assert "angle_deg" in result

    def test_4d_input(self, bundled_mask):
        """Accepts 4D input array."""
        wavelengths = np.array([3.5])
        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=3.5,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        ps_4d = ps[np.newaxis, np.newaxis, :, :]
        result = find_mask_rotation_angle(
            ps_4d,
            bundled_mask,
            wavelengths,
            n_grid=21,
            refine=False,
        )
        assert "angle_deg" in result

    def test_invalid_ndim_raises(self, bundled_mask):
        """5D input raises ValueError."""
        ps_5d = np.zeros((1, 1, 1, 67, 67))
        with pytest.raises(ValueError, match="must be"):
            find_mask_rotation_angle(ps_5d, bundled_mask, np.array([3.5]))

    def test_wavelength_mismatch_raises(self, bundled_mask):
        """Mismatched wavelength count raises ValueError."""
        ps = np.zeros((67, 67))
        with pytest.raises(ValueError, match="does not match"):
            find_mask_rotation_angle(ps, bundled_mask, np.array([3.0, 3.5]))

    def test_refine_true_returns_all_keys(self, bundled_mask):
        """With refine=True all result keys are populated."""
        wavelength = 3.5
        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        result = find_mask_rotation_angle(
            ps,
            bundled_mask,
            np.array([wavelength]),
            angle_range=(-10.0, 10.0),
            n_grid=21,
            cutout_size=5,
            refine=True,
        )
        assert result["angle_deg"] is not None
        assert result["step_a_angle_deg"] is not None
        assert result["angles_all"] is not None
        assert len(result["angles_all"]) == 1
        assert result["measured_centers"] is not None
        assert result["angle_std_deg"] is None

    def test_refine_multiple_slices_gives_std(self, bundled_mask):
        """Multiple slices produce a standard deviation estimate."""
        wavelengths = np.array([3.0, 3.5])
        ps_3d = np.array(
            [
                bundled_mask.compute_synthetic_power_spectrum(
                    wavelength=w,
                    n_pixels_image=67,
                    n_pixels_pupil=501,
                )
                for w in wavelengths
            ]
        )
        result = find_mask_rotation_angle(
            ps_3d,
            bundled_mask,
            wavelengths,
            angle_range=(-10.0, 10.0),
            n_grid=21,
            refine=True,
        )
        assert result["angle_std_deg"] is not None
        assert len(result["angles_all"]) == 2

    def test_refine_with_failed_fits(self, bundled_mask):
        """Step B else branch is exercised when fits fail near edges."""
        wavelength = 3.5
        # Use a small frame so some splodges land near edges,
        # causing _fit_2d_gaussian_at_position to return failure.
        n_pix = 33
        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            pixel_scale_arcsec=ALES_PIXEL_SCALE_ARCSEC,
            n_pixels_image=n_pix,
            n_pixels_pupil=501,
        )
        wavelengths = np.array([wavelength])

        result = find_mask_rotation_angle(
            ps,
            bundled_mask,
            wavelengths,
            angle_range=(-10.0, 10.0),
            n_grid=21,
            cutout_size=7,
            refine=True,
        )

        assert result["angle_deg"] is not None
        assert result["angles_all"] is not None
        assert len(result["angles_all"]) == 1
        assert result["measured_centers"] is not None

    def test_quick_look_no_refine(self, bundled_mask):
        """refine=False gives quick-look result matching step A."""
        wavelength = 3.5
        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        result = find_mask_rotation_angle(
            ps,
            bundled_mask,
            np.array([wavelength]),
            n_grid=21,
            refine=False,
        )
        assert result["angle_deg"] == result["step_a_angle_deg"]
        assert result["angles_all"] is None
        assert result["measured_centers"] is None
        assert result["angle_std_deg"] is None
