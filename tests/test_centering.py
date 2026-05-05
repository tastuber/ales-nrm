"""Tests for frame centering and Fourier shifting."""

import numpy as np
import pytest

from ales_nrm.centering import (
    _fill_shifts,
    _try_find_center,
    center_cubes,
    find_center,
    fourier_shift_2d,
)


@pytest.fixture()
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(seed=42)


def _make_gaussian_image(
    ny,
    nx,
    y_center,
    x_center,
    amplitude=1000.0,
    sigma=2.0,
    noise_level=0.0,
    rng=None,
):
    """Create a 2D Gaussian image with optional noise.

    Args:
        ny: Number of pixels in y dimension.
        nx: Number of pixels in x dimension.
        y_center: Y center of the Gaussian.
        x_center: X center of the Gaussian.
        amplitude: Peak amplitude.
        sigma: Standard deviation of the Gaussian.
        noise_level: Standard deviation of additive noise.
        rng: Random number generator for noise.
    """
    y, x = np.mgrid[0:ny, 0:nx]
    image = amplitude * np.exp(
        -((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma**2)
    )
    if rng is not None and noise_level > 0:
        image += rng.normal(
            0,
            noise_level,
            image.shape,
        )
    return image


def _make_test_4d_cube(
    n_frames,
    n_wav,
    ny,
    nx,
    y_offsets,
    x_offsets,
    chromatic_shift=0.0,
    noise_wavelengths=None,
    noise_level=1000.0,
    rng=None,
):
    """Create a 4D test cube with per-frame offsets.

    Args:
        n_frames: Number of frames.
        n_wav: Number of wavelengths.
        ny: Number of pixels in y dimension.
        nx: Number of pixels in x dimension.
        y_offsets: 1D array of y-offsets per frame.
        x_offsets: 1D array of x-offsets per frame.
        chromatic_shift: Additional x-shift per
            wavelength index, simulating ALES
            dispersion.
        noise_wavelengths: Optional list of wavelength
            indices to fill with pure noise (no signal),
            simulating low-SNR spectral edges.
        noise_level: Amplitude of noise for noisy
            wavelength slices.
        rng: Random number generator for noise.
    """
    y_geo = (ny - 1) / 2.0
    x_geo = (nx - 1) / 2.0
    cubes = np.zeros((n_frames, n_wav, ny, nx))
    for f in range(n_frames):
        for w in range(n_wav):
            x_chrom = x_offsets[f] + chromatic_shift * w
            cubes[f, w] = _make_gaussian_image(
                ny,
                nx,
                y_geo + y_offsets[f],
                x_geo + x_chrom,
                amplitude=1000.0 - w * 5,
            )

    if noise_wavelengths is not None and rng is not None:
        for w in noise_wavelengths:
            cubes[:, w] = rng.normal(
                0,
                noise_level,
                size=(n_frames, ny, nx),
            )

    return cubes


class TestFourierShift2D:
    """Tests for the Fourier shift function."""

    def test_zero_shift_preserves_image(self, rng):
        """A zero shift returns the original image."""
        image = rng.normal(100, 10, size=(63, 67))
        shifted = fourier_shift_2d(image, 0.0, 0.0)
        np.testing.assert_allclose(
            shifted,
            image,
            atol=1e-10,
        )

    def test_integer_shift(self):
        """An integer shift moves pixels exactly."""
        image = np.zeros((64, 64))
        image[30, 32] = 100.0
        shifted = fourier_shift_2d(image, 2.0, 3.0)
        assert shifted[32, 35] == pytest.approx(
            100.0,
            abs=1e-10,
        )

    def test_flux_conservation(self, rng):
        """Total flux is preserved after shifting."""
        image = rng.normal(100, 10, size=(63, 67))
        shifted = fourier_shift_2d(image, 1.7, -2.3)
        np.testing.assert_allclose(
            np.sum(shifted),
            np.sum(image),
            rtol=1e-12,
        )

    def test_fractional_shift_flux_conservation(self):
        """Flux preserved for fractional pixel shifts."""
        image = _make_gaussian_image(
            63,
            67,
            30.0,
            33.0,
        )
        shifted = fourier_shift_2d(image, 0.37, -0.82)
        np.testing.assert_allclose(
            np.sum(shifted),
            np.sum(image),
            rtol=1e-12,
        )

    def test_preserves_dtype(self, rng):
        """Output dtype matches input dtype."""
        image = rng.normal(
            100,
            10,
            size=(63, 67),
        ).astype(np.float64)
        shifted = fourier_shift_2d(image, 1.5, -0.5)
        assert shifted.dtype == image.dtype

    def test_preserves_shape(self, rng):
        """Output shape matches input shape."""
        image = rng.normal(100, 10, size=(63, 67))
        shifted = fourier_shift_2d(image, 1.5, -0.5)
        assert shifted.shape == image.shape

    def test_reverse_shift(self):
        """Shifting and reversing recovers original."""
        image = np.zeros((64, 64))
        image[28:36, 30:34] = 100.0
        shifted = fourier_shift_2d(image, 3.7, -2.1)
        recovered = fourier_shift_2d(shifted, -3.7, 2.1)
        np.testing.assert_allclose(recovered, image, atol=1e-13)


class TestFindCenter:
    """Tests for the center-finding function."""

    def test_centered_gaussian(self):
        """Find center of a centered Gaussian."""
        image = _make_gaussian_image(
            63,
            67,
            31.0,
            33.0,
        )
        y, x = find_center(image)
        assert y == pytest.approx(31.0, abs=0.1)
        assert x == pytest.approx(33.0, abs=0.1)

    def test_offset_gaussian(self):
        """Find center of an off-center Gaussian."""
        image = _make_gaussian_image(
            63,
            67,
            25.3,
            37.8,
        )
        y, x = find_center(image)
        assert y == pytest.approx(25.3, abs=0.2)
        assert x == pytest.approx(37.8, abs=0.2)

    def test_subpixel_precision(self):
        """Achieve sub-pixel precision on noiseless data."""
        y_true, x_true = 30.37, 34.62
        image = _make_gaussian_image(
            63,
            67,
            y_true,
            x_true,
        )
        y, x = find_center(image)
        assert y == pytest.approx(y_true, abs=0.05)
        assert x == pytest.approx(x_true, abs=0.05)

    def test_with_noise(self, rng):
        """Find center in the presence of noise."""
        image = _make_gaussian_image(
            63,
            67,
            31.0,
            33.0,
            amplitude=1000.0,
            noise_level=5.0,
            rng=rng,
        )
        y, x = find_center(image)
        assert y == pytest.approx(31.0, abs=0.5)
        assert x == pytest.approx(33.0, abs=0.5)

    def test_invalid_cutout_size_even(self):
        """Raise ValueError for even cutout_size."""
        image = _make_gaussian_image(
            63,
            67,
            31.0,
            33.0,
        )
        with pytest.raises(ValueError, match="cutout_size must be"):
            find_center(image, cutout_size=6)

    def test_invalid_cutout_size_small(self):
        """Raise ValueError for cutout_size < 3."""
        image = _make_gaussian_image(
            63,
            67,
            31.0,
            33.0,
        )
        with pytest.raises(ValueError, match="cutout_size must be"):
            find_center(image, cutout_size=1)

    def test_different_cutout_sizes(self):
        """Center is consistent across cutout sizes."""
        image = _make_gaussian_image(
            63,
            67,
            31.0,
            33.0,
        )
        y5, x5 = find_center(image, cutout_size=5)
        y7, x7 = find_center(image, cutout_size=7)
        assert y5 == pytest.approx(y7, abs=0.3)
        assert x5 == pytest.approx(x7, abs=0.3)

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_pure_noise_raises(self, rng):
        """Raise ValueError on pure noise image."""
        image = rng.normal(0, 10, size=(63, 67))
        with pytest.raises(ValueError, match="did not converge"):
            find_center(image)


class TestTryFindCenter:
    """Tests for the error-handling wrapper."""

    def test_success_returns_tuple(self):
        """Return center on successful fit."""
        image = _make_gaussian_image(
            63,
            67,
            31.0,
            33.0,
        )
        result = _try_find_center(image)
        assert result is not None
        y, x = result
        assert y == pytest.approx(31.0, abs=0.1)
        assert x == pytest.approx(33.0, abs=0.1)

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_failure_returns_none(self, rng):
        """Return None on failed fit."""
        image = rng.normal(0, 10, size=(63, 67))
        result = _try_find_center(image)
        assert result is None


class TestFillShifts:
    """Tests for the forward/backward fill logic."""

    def test_no_nans(self):
        """Array without nans is unchanged."""
        shifts = np.ones((3, 5, 2))
        filled = _fill_shifts(shifts)
        np.testing.assert_array_equal(filled, shifts)

    def test_forward_fill(self):
        """Nans at the end are filled from the last valid."""
        shifts = np.full((1, 5, 2), np.nan)
        shifts[0, 0] = [1.0, 2.0]
        shifts[0, 1] = [1.1, 2.1]
        shifts[0, 2] = [1.2, 2.2]
        filled = _fill_shifts(shifts)
        np.testing.assert_array_equal(
            filled[0, 3],
            [1.2, 2.2],
        )
        np.testing.assert_array_equal(
            filled[0, 4],
            [1.2, 2.2],
        )

    def test_backward_fill(self):
        """Nans at the start are filled from first valid."""
        shifts = np.full((1, 5, 2), np.nan)
        shifts[0, 3] = [1.0, 2.0]
        shifts[0, 4] = [1.1, 2.1]
        filled = _fill_shifts(shifts)
        np.testing.assert_array_equal(
            filled[0, 0],
            [1.0, 2.0],
        )
        np.testing.assert_array_equal(
            filled[0, 2],
            [1.0, 2.0],
        )

    def test_interior_gap(self):
        """Interior gap is filled from shorter wavelength."""
        shifts = np.full((1, 5, 2), np.nan)
        shifts[0, 0] = [1.0, 2.0]
        shifts[0, 4] = [3.0, 4.0]
        filled = _fill_shifts(shifts)
        # Forward fill fills from index 0.
        np.testing.assert_array_equal(
            filled[0, 1],
            [1.0, 2.0],
        )
        np.testing.assert_array_equal(
            filled[0, 3],
            [1.0, 2.0],
        )

    def test_all_nan_frame_gets_zero(self):
        """Frame with all nans gets zero shifts."""
        shifts = np.full((2, 5, 2), np.nan)
        shifts[0, :] = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        filled = _fill_shifts(shifts)
        # Frame 0 is unchanged.
        np.testing.assert_array_equal(
            filled[0, 0],
            [1.0, 2.0],
        )
        # Frame 1 (all nan) gets zero.
        np.testing.assert_array_equal(
            filled[1, 0],
            [0.0, 0.0],
        )

    def test_per_frame_independence(self):
        """Each frame is filled independently."""
        shifts = np.full((2, 5, 2), np.nan)
        shifts[0, 2] = [1.0, 2.0]
        shifts[1, 4] = [3.0, 4.0]
        filled = _fill_shifts(shifts)
        # Frame 0: backward fill to 0,1; forward to 3,4.
        np.testing.assert_array_equal(
            filled[0, 0],
            [1.0, 2.0],
        )
        np.testing.assert_array_equal(
            filled[0, 4],
            [1.0, 2.0],
        )
        # Frame 1: backward fill from 4.
        np.testing.assert_array_equal(
            filled[1, 0],
            [3.0, 4.0],
        )


class TestCenterCubes:
    """Tests for the 4D cube centering function."""

    def test_non_4d_raises(self):
        """Raise ValueError for non-4D input."""
        cube_3d = np.zeros((10, 63, 67))
        with pytest.raises(ValueError, match="Expected a 4D array"):
            center_cubes(cube_3d)

    def test_invalid_n_wave_sum(self):
        """Raise ValueError for n_wave_sum < 1."""
        cubes = _make_test_4d_cube(
            3,
            5,
            63,
            67,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        with pytest.raises(ValueError, match="n_wave_sum must be"):
            center_cubes(cubes, n_wave_sum=0)

    def test_preserves_shape(self):
        """Output shape matches input shape."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        centered, shifts = center_cubes(cubes)
        assert centered.shape == cubes.shape

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        centered, shifts = center_cubes(cubes)
        assert centered.dtype == cubes.dtype

    def test_shifts_shape(self):
        """Shifts array has correct shape."""
        n_frames, n_wav = 3, 10
        cubes = _make_test_4d_cube(
            n_frames,
            n_wav,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        _, shifts = center_cubes(cubes)
        assert shifts.shape == (n_frames, n_wav, 2)

    def test_flux_conservation(self):
        """Total flux preserved per frame and slice."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        original_flux = np.sum(cubes, axis=(2, 3))
        centered, _ = center_cubes(cubes)
        centered_flux = np.sum(centered, axis=(2, 3))
        np.testing.assert_allclose(
            centered_flux,
            original_flux,
            rtol=1e-12,
        )

    def test_frames_aligned_per_wavelength(self):
        """All frames converge to same position."""
        cubes = _make_test_4d_cube(
            5,
            10,
            63,
            67,
            [3.0, -2.0, 1.5, -0.5, 2.0],
            [-1.0, 2.0, -1.5, 0.5, -2.0],
        )
        centered, _ = center_cubes(cubes)

        for w in range(10):
            centers = []
            for f in range(5):
                y_c, x_c = find_center(centered[f, w])
                centers.append((y_c, x_c))
            centers = np.array(centers)
            assert np.std(centers[:, 0]) < 0.3
            assert np.std(centers[:, 1]) < 0.3

    def test_mean_shift_near_zero(self):
        """Mean shift across frames is near zero."""
        cubes = _make_test_4d_cube(
            5,
            10,
            63,
            67,
            [3.0, -2.0, 1.5, -0.5, 2.0],
            [-1.0, 2.0, -1.5, 0.5, -2.0],
        )
        _, shifts = center_cubes(cubes)

        for w in range(10):
            mean_dy = np.mean(shifts[:, w, 0])
            mean_dx = np.mean(shifts[:, w, 1])
            assert abs(mean_dy) < 0.1
            assert abs(mean_dx) < 0.1

    def test_already_aligned(self):
        """Already-aligned frames get near-zero shifts."""
        cubes = _make_test_4d_cube(
            3,
            5,
            63,
            67,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        _, shifts = center_cubes(cubes)
        assert np.all(np.abs(shifts) < 0.2)

    def test_chromatic_shift_handled(self):
        """Per-wavelength centering handles chromatic shift."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [1.0, -1.0, 0.5],
            [0.5, -0.5, 0.2],
            chromatic_shift=0.3,
        )
        centered, shifts = center_cubes(cubes)

        for w in range(10):
            centers = []
            for f in range(3):
                y_c, x_c = find_center(centered[f, w])
                centers.append((y_c, x_c))
            centers = np.array(centers)
            assert np.std(centers[:, 0]) < 0.3
            assert np.std(centers[:, 1]) < 0.3

    def test_chromatic_different_mean_per_wav(self):
        """Mean center differs per wavelength."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            chromatic_shift=0.5,
        )
        centered, _ = center_cubes(cubes)

        centers_per_wav = []
        for w in range(10):
            y_c, x_c = find_center(centered[0, w])
            centers_per_wav.append(x_c)
        centers_per_wav = np.array(centers_per_wav)
        assert np.std(centers_per_wav) > 0.1

    def test_n_wave_sum_groups_slices(self):
        """Wavelength summing groups slices correctly."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        _, shifts = center_cubes(
            cubes,
            n_wave_sum=5,
        )

        for f in range(3):
            np.testing.assert_array_equal(
                shifts[f, 0],
                shifts[f, 1],
            )
            np.testing.assert_array_equal(
                shifts[f, 0],
                shifts[f, 4],
            )
            np.testing.assert_array_equal(
                shifts[f, 5],
                shifts[f, 6],
            )
            np.testing.assert_array_equal(
                shifts[f, 5],
                shifts[f, 9],
            )

    def test_n_wave_sum_uneven(self):
        """Handle uneven wavelength groups gracefully."""
        cubes = _make_test_4d_cube(
            3,
            7,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        centered, shifts = center_cubes(
            cubes,
            n_wave_sum=3,
        )

        for f in range(3):
            np.testing.assert_array_equal(
                shifts[f, 0],
                shifts[f, 2],
            )
            np.testing.assert_array_equal(
                shifts[f, 3],
                shifts[f, 5],
            )
        assert centered.shape == cubes.shape

    def test_n_wave_sum_larger_than_n_wav(self):
        """n_wave_sum larger than n_wav uses single group."""
        cubes = _make_test_4d_cube(
            3,
            5,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        _, shifts = center_cubes(
            cubes,
            n_wave_sum=100,
        )

        for f in range(3):
            for w in range(5):
                np.testing.assert_array_equal(
                    shifts[f, 0],
                    shifts[f, w],
                )

    def test_n_wave_sum_one_is_default(self):
        """n_wave_sum=1 gives same result as default."""
        cubes = _make_test_4d_cube(
            3,
            5,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )
        centered_default, shifts_default = center_cubes(
            cubes,
        )
        centered_one, shifts_one = center_cubes(
            cubes,
            n_wave_sum=1,
        )
        np.testing.assert_allclose(
            centered_default,
            centered_one,
        )
        np.testing.assert_allclose(
            shifts_default,
            shifts_one,
        )

    def test_single_frame(self):
        """Single frame gets zero shifts."""
        cubes = _make_test_4d_cube(
            1,
            5,
            63,
            67,
            [3.0],
            [-2.0],
        )
        _, shifts = center_cubes(cubes)
        np.testing.assert_allclose(
            shifts,
            0.0,
            atol=1e-10,
        )

    def test_two_frames_opposite_offsets(self):
        """Two frames with opposite offsets center correctly."""
        cubes = _make_test_4d_cube(
            2,
            5,
            63,
            67,
            [2.0, -2.0],
            [1.5, -1.5],
        )
        centered, shifts = center_cubes(cubes)

        np.testing.assert_allclose(
            shifts[0],
            -shifts[1],
            atol=0.3,
        )

        for w in range(5):
            y0, x0 = find_center(centered[0, w])
            y1, x1 = find_center(centered[1, w])
            assert y0 == pytest.approx(y1, abs=0.3)
            assert x0 == pytest.approx(x1, abs=0.3)

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_noisy_wavelengths_filled(self, rng):
        """Noisy wavelengths get shifts from neighbors."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
            noise_wavelengths=[0, 1, 8, 9],
            noise_level=1000.0,
            rng=rng,
        )

        centered, shifts = center_cubes(cubes)

        # Noisy wavelengths should have non-nan shifts
        # (filled from neighbors).
        assert not np.any(np.isnan(shifts))

        # Good wavelengths should still be aligned.
        for w in range(2, 8):
            centers = []
            for f in range(3):
                y_c, x_c = find_center(centered[f, w])
                centers.append((y_c, x_c))
            centers = np.array(centers)
            assert np.std(centers[:, 0]) < 0.3
            assert np.std(centers[:, 1]) < 0.3

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_noisy_edge_shifts_match_neighbor(self, rng):
        """Edge wavelength shifts match nearest good wavelength."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
            noise_wavelengths=[0, 1],
            noise_level=1000.0,
            rng=rng,
        )

        _, shifts = center_cubes(cubes)

        # Wavelengths 0,1 should have same shifts as
        # wavelength 2 (backward filled).
        for f in range(3):
            np.testing.assert_allclose(
                shifts[f, 0],
                shifts[f, 2],
                atol=0.01,
            )
            np.testing.assert_allclose(
                shifts[f, 1],
                shifts[f, 2],
                atol=0.01,
            )

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_all_fits_fail_returns_unshifted(
        self,
        monkeypatch,
    ):
        """All fit failures return unshifted data with zero shifts."""
        from ales_nrm import centering

        monkeypatch.setattr(
            centering,
            "_try_find_center",
            lambda image, cutout_size=5: None,
        )

        cubes = _make_test_4d_cube(
            3,
            5,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )

        centered, shifts = center_cubes(cubes)

        np.testing.assert_allclose(
            shifts,
            0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            centered,
            cubes,
            atol=1e-10,
        )

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_partial_frame_failure_excluded(self, rng):
        """Wavelength with one frame noisy is excluded."""
        cubes = _make_test_4d_cube(
            3,
            5,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
        )

        # Make one frame noisy at wavelength 2.
        cubes[1, 2] = rng.normal(
            0,
            1000,
            size=(63, 67),
        )

        _, shifts = center_cubes(cubes)

        # Wavelength 2 should be filled from neighbor,
        # not computed from partial data.
        # Shifts at wav 2 should match wav 1 (forward fill).
        for f in range(3):
            np.testing.assert_allclose(
                shifts[f, 2],
                shifts[f, 1],
                atol=0.01,
            )

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_output_shape_with_noisy_wavelengths(
        self,
        rng,
    ):
        """Output shape preserved with noisy wavelengths."""
        cubes = _make_test_4d_cube(
            3,
            10,
            63,
            67,
            [2.0, -1.0, 0.5],
            [-1.5, 1.0, -0.3],
            noise_wavelengths=[0, 1, 8, 9],
            noise_level=1000.0,
            rng=rng,
        )

        centered, shifts = center_cubes(cubes)
        assert centered.shape == cubes.shape
        assert shifts.shape == (3, 10, 2)

    def test_one_frame_all_fits_fail(self, monkeypatch):
        """Frame with all fits failed gets zero shifts."""
        from ales_nrm import centering

        cubes = _make_test_4d_cube(
            3, 5, 63, 67, [2.0, -1.0, 0.5], [-1.5, 1.0, -0.3]
        )

        call_count = {"n": 0}
        original = centering._try_find_center

        def fail_frame_1(image, cutout_size=5):
            """Return None for all calls from frame 1."""
            idx = call_count["n"]
            call_count["n"] += 1
            # Frame 1 spans calls 5–9 (5 wavelengths each).
            if 5 <= idx < 10:
                return None
            return original(image, cutout_size=cutout_size)

        monkeypatch.setattr(centering, "_try_find_center", fail_frame_1)

        centered, shifts = center_cubes(cubes)

        # Frame 1 should have zero shifts (all fits failed).
        np.testing.assert_allclose(shifts[1], 0.0, atol=1e-10)
        assert centered.shape == cubes.shape
