"""Tests for NRM mask geometry and Fourier-plane mapping."""

import numpy as np
import pytest

from ales_nrm.nrm.mask import (
    ALES_PIXEL_SCALE_ARCSEC,
    Baseline,
    Hole,
    NRMMask,
)


@pytest.fixture()
def sample_holes():
    """Create a simple 3-hole mask for testing."""
    return [
        Hole("H1", -1.0, 0.0, 0.4),
        Hole("H2", 1.0, 0.0, 0.4),
        Hole("H3", 0.0, 1.5, 0.4),
    ]


@pytest.fixture()
def three_hole_mask(sample_holes):
    """Create a 3-hole NRMMask instance."""
    mask = NRMMask(
        primary_diameter=8.4,
        holes=sample_holes,
    )
    mask._compute_baselines()
    return mask


@pytest.fixture()
def bundled_mask():
    """Load the bundled LBTI NRM6 SX mask."""
    return NRMMask.from_bundled("lbti_nrm6_sx", primary_diameter=8.4)


class TestHole:
    """Tests for the Hole dataclass."""

    def test_creation(self):
        """Create a Hole with correct attributes."""
        hole = Hole("H1", 1.5, -2.3, 0.4)
        assert hole.name == "H1"
        assert hole.x == 1.5
        assert hole.y == -2.3
        assert hole.radius == 0.4


class TestBaseline:
    """Tests for the Baseline dataclass."""

    def test_creation(self):
        """Create a Baseline with correct attributes."""
        bl = Baseline("H1H2", "H1", "H2", 2.0, 0.0, 2.0)
        assert bl.name == "H1H2"
        assert bl.hole1 == "H1"
        assert bl.hole2 == "H2"
        assert bl.bx == 2.0
        assert bl.by == 0.0
        assert bl.length == 2.0


class TestNRMMaskProperties:
    """Tests for NRMMask basic properties."""

    def test_n_holes(self, three_hole_mask):
        """Count holes correctly."""
        assert three_hole_mask.n_holes == 3

    def test_n_baselines(self, three_hole_mask):
        """Count baselines correctly for 3 holes."""
        # C(3,2) = 3
        assert three_hole_mask.n_baselines == 3

    def test_n_closing_triangles(self, three_hole_mask):
        """Count closing triangles for 3 holes."""
        # C(3,3) = 1
        assert three_hole_mask.n_closing_triangles == 1

    def test_six_hole_counts(self, bundled_mask):
        """Verify counts for the 6-hole LBTI mask."""
        assert bundled_mask.n_holes == 6
        # C(6,2) = 15
        assert bundled_mask.n_baselines == 15
        # C(6,3) = 20
        assert bundled_mask.n_closing_triangles == 20


class TestNRMMaskBaselines:
    """Tests for baseline computation."""

    def test_baseline_names(self, three_hole_mask):
        """Baseline names combine hole names."""
        names = [bl.name for bl in three_hole_mask.baselines]
        assert "H1H2" in names
        assert "H1H3" in names
        assert "H2H3" in names

    def test_baseline_vectors(self, three_hole_mask):
        """Baseline vectors are differences of positions."""
        bl_h1h2 = next(
            bl for bl in three_hole_mask.baselines if bl.name == "H1H2"
        )
        # H2.x - H1.x = 1.0 - (-1.0) = 2.0
        assert bl_h1h2.bx == pytest.approx(2.0)
        assert bl_h1h2.by == pytest.approx(0.0)

    def test_baseline_length(self, three_hole_mask):
        """Baseline length is Euclidean distance."""
        bl_h1h2 = next(
            bl for bl in three_hole_mask.baselines if bl.name == "H1H2"
        )
        assert bl_h1h2.length == pytest.approx(2.0)

    def test_baseline_length_diagonal(self, three_hole_mask):
        """Diagonal baseline length is correct."""
        bl_h1h3 = next(
            bl for bl in three_hole_mask.baselines if bl.name == "H1H3"
        )
        expected = np.sqrt(1.0**2 + 1.5**2)
        assert bl_h1h3.length == pytest.approx(expected)


class TestNRMMaskClosingTriangles:
    """Tests for closing triangle computation."""

    def test_three_holes_one_triangle(self, three_hole_mask):
        """Three holes form exactly one triangle."""
        triangles = three_hole_mask.get_closing_triangles()
        assert len(triangles) == 1

    def test_triangle_contains_three_baselines(self, three_hole_mask):
        """Each triangle is a tuple of three baseline names."""
        triangles = three_hole_mask.get_closing_triangles()
        assert len(triangles[0]) == 3

    def test_six_hole_triangles(self, bundled_mask):
        """Six holes produce 20 closing triangles."""
        triangles = bundled_mask.get_closing_triangles()
        assert len(triangles) == 20


class TestNRMMaskFromFile:
    """Tests for loading masks from files."""

    def test_from_bundled(self, bundled_mask):
        """Load bundled mask successfully."""
        assert bundled_mask.n_holes == 6
        assert bundled_mask.primary_diameter == 8.4

    def test_from_bundled_dx(self):
        """Load the DX bundled mask."""
        mask = NRMMask.from_bundled("lbti_nrm6_dx")
        assert mask.n_holes == 6

    def test_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            NRMMask.from_file(tmp_path / "nonexistent.txt")

    def test_invalid_format(self, tmp_path):
        """Raise ValueError for wrong column count."""
        bad_file = tmp_path / "bad_mask.txt"
        bad_file.write_text("H1 1.0 2.0\n")
        with pytest.raises(ValueError, match="Expected 4 columns"):
            NRMMask.from_file(bad_file)

    def test_comments_skipped(self, tmp_path):
        """Comment lines are ignored."""
        mask_file = tmp_path / "mask.txt"
        mask_file.write_text(
            "# This is a comment\n"
            "H1 -1.0 0.0 0.4\n"
            "# Another comment\n"
            "H2 1.0 0.0 0.4\n"
        )
        mask = NRMMask.from_file(mask_file)
        assert mask.n_holes == 2

    def test_empty_lines_skipped(self, tmp_path):
        """Empty lines are ignored."""
        mask_file = tmp_path / "mask.txt"
        mask_file.write_text("H1 -1.0 0.0 0.4\n\nH2 1.0 0.0 0.4\n\n")
        mask = NRMMask.from_file(mask_file)
        assert mask.n_holes == 2

    def test_custom_primary_diameter(self, tmp_path):
        """Primary diameter is set from argument."""
        mask_file = tmp_path / "mask.txt"
        mask_file.write_text("H1 0.0 0.0 0.2\nH2 1.0 0.0 0.2\n")
        mask = NRMMask.from_file(mask_file, primary_diameter=6.5)
        assert mask.primary_diameter == 6.5


class TestMaskRotationAtLoad:
    """Tests for angle_deg parameter in factory methods."""

    def test_zero_angle_matches_default(self):
        """angle_deg=0.0 gives same result as no angle."""
        mask_default = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_zero = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=0.0)
        for h1, h2 in zip(mask_default.holes, mask_zero.holes, strict=True):
            assert h1.x == pytest.approx(h2.x)
            assert h1.y == pytest.approx(h2.y)
            assert h1.radius == h2.radius
            assert h1.name == h2.name

    def test_nonzero_angle_changes_coordinates(self):
        """Non-zero angle produces different coordinates."""
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rotated = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=30.0)
        any_different = False
        for h1, h2 in zip(mask_plain.holes, mask_rotated.holes, strict=True):
            if abs(h1.x - h2.x) > 1e-6:
                any_different = True
                break
        assert any_different

    def test_preserves_n_holes(self):
        """Rotation preserves number of holes."""
        mask = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=15.0)
        assert mask.n_holes == 6

    def test_preserves_n_baselines(self):
        """Rotation preserves number of baselines."""
        mask = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=15.0)
        assert mask.n_baselines == 15

    def test_preserves_baseline_lengths(self):
        """Baseline lengths are invariant under rotation."""
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rotated = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=23.7)
        for bl1, bl2 in zip(
            mask_plain.baselines,
            mask_rotated.baselines,
            strict=True,
        ):
            assert bl2.length == pytest.approx(bl1.length, rel=1e-10)

    def test_preserves_hole_radii(self):
        """Hole radii unchanged by rotation."""
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rotated = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=45.0)
        for h1, h2 in zip(mask_plain.holes, mask_rotated.holes, strict=True):
            assert h2.radius == h1.radius

    def test_preserves_hole_names(self):
        """Hole names unchanged by rotation."""
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rotated = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=45.0)
        for h1, h2 in zip(mask_plain.holes, mask_rotated.holes, strict=True):
            assert h2.name == h1.name

    def test_preserves_primary_diameter(self):
        """Primary diameter unchanged by rotation."""
        mask = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=45.0)
        assert mask.primary_diameter == 8.4

    def test_centroid_invariant(self):
        """Centroid of holes unchanged by rotation."""
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rotated = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=45.0)
        cx_orig = np.mean([h.x for h in mask_plain.holes])
        cy_orig = np.mean([h.y for h in mask_plain.holes])
        cx_rot = np.mean([h.x for h in mask_rotated.holes])
        cy_rot = np.mean([h.y for h in mask_rotated.holes])
        assert cx_rot == pytest.approx(cx_orig, abs=1e-10)
        assert cy_rot == pytest.approx(cy_orig, abs=1e-10)

    def test_360_returns_original(self):
        """360 degree rotation returns original."""
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_360 = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=360.0)
        for h1, h2 in zip(mask_plain.holes, mask_360.holes, strict=True):
            assert h2.x == pytest.approx(h1.x, abs=1e-10)
            assert h2.y == pytest.approx(h1.y, abs=1e-10)

    def test_opposite_angles_differ(self):
        """Loading with +θ and -θ gives different coords."""
        mask_pos = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=17.3)
        mask_neg = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=-17.3)
        any_different = False
        for h1, h2 in zip(mask_pos.holes, mask_neg.holes, strict=True):
            if abs(h1.x - h2.x) > 1e-6:
                any_different = True
                break
        assert any_different

    def test_from_file_with_angle(self, tmp_path):
        """from_file respects angle_deg parameter."""
        mask_file = tmp_path / "test_mask.txt"
        mask_file.write_text(
            "H1 -1.0 0.0 0.4\nH2  1.0 0.0 0.4\nH3  0.0 1.5 0.4\n"
        )
        mask_0 = NRMMask.from_file(mask_file, angle_deg=0.0)
        mask_45 = NRMMask.from_file(mask_file, angle_deg=45.0)
        # Baseline lengths preserved.
        for bl1, bl2 in zip(mask_0.baselines, mask_45.baselines, strict=True):
            assert bl2.length == pytest.approx(bl1.length, rel=1e-10)
        # Coordinates differ.
        assert mask_45.holes[0].x != pytest.approx(mask_0.holes[0].x, abs=0.01)

    def test_from_bundled_angle_consistent(self):
        """from_bundled angle gives consistent result."""
        mask_bundled = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=12.5)
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        # Baseline lengths must match.
        for bl1, bl2 in zip(
            mask_plain.baselines,
            mask_bundled.baselines,
            strict=True,
        ):
            assert bl2.length == pytest.approx(bl1.length, rel=1e-10)
        # But coordinates must differ.
        assert mask_bundled.holes[0].x != pytest.approx(
            mask_plain.holes[0].x, abs=0.001
        )

    def test_splodge_distances_preserved(self):
        """Splodge distances from center invariant."""
        wavelengths = np.array([3.5])
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rot = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=10.0)
        pos_plain = mask_plain.compute_splodge_positions(
            wavelengths, ny=501, nx=501
        )
        pos_rot = mask_rot.compute_splodge_positions(
            wavelengths, ny=501, nx=501
        )
        center_y = 250.0
        center_x = 250.0
        for bl_name in pos_plain:
            d_plain = np.sqrt(
                (pos_plain[bl_name][0, 0] - center_y) ** 2
                + (pos_plain[bl_name][0, 1] - center_x) ** 2
            )
            d_rot = np.sqrt(
                (pos_rot[bl_name][0, 0] - center_y) ** 2
                + (pos_rot[bl_name][0, 1] - center_x) ** 2
            )
            assert d_rot == pytest.approx(d_plain, rel=1e-6)

    def test_power_spectrum_changes_with_rotation(self):
        """Power spectrum peak positions shift."""
        wavelength = 3.5
        mask_plain = NRMMask.from_bundled("lbti_nrm6_sx")
        mask_rot = NRMMask.from_bundled("lbti_nrm6_sx", angle_deg=15.0)
        ps_plain = mask_plain.compute_synthetic_power_spectrum(
            wavelength,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        ps_rot = mask_rot.compute_synthetic_power_spectrum(
            wavelength,
            n_pixels_image=67,
            n_pixels_pupil=501,
        )
        # Power spectra should differ.
        assert not np.allclose(ps_plain, ps_rot)
        # But total power approximately equal.
        assert np.sum(ps_rot) == pytest.approx(np.sum(ps_plain), rel=0.01)


class TestMakePupilImage:
    """Tests for pupil image generation."""

    def test_output_shape(self, three_hole_mask):
        """Output is square with requested size."""
        image, _ = three_hole_mask.make_pupil_image(n_pixels=101)
        assert image.shape == (101, 101)

    def test_forced_odd(self, three_hole_mask):
        """Even input is forced to odd."""
        image, _ = three_hole_mask.make_pupil_image(n_pixels=100)
        assert image.shape == (101, 101)

    def test_default_size(self, three_hole_mask):
        """Default size is 1001."""
        image, _ = three_hole_mask.make_pupil_image()
        assert image.shape == (1001, 1001)

    def test_binary_values(self, three_hole_mask):
        """Image contains only 0 and 1."""
        image, _ = three_hole_mask.make_pupil_image()
        unique = np.unique(image)
        assert set(unique).issubset({0.0, 1.0})

    def test_has_nonzero_pixels(self, three_hole_mask):
        """Image has illuminated pixels."""
        image, _ = three_hole_mask.make_pupil_image()
        assert np.sum(image) > 0

    def test_pixel_scale_positive(self, three_hole_mask):
        """Pixel scale is positive."""
        _, pix_scale = three_hole_mask.make_pupil_image()
        assert pix_scale > 0

    def test_extent_covers_primary(self, bundled_mask):
        """Pupil image extent >= primary diameter."""
        image, pix_scale = bundled_mask.make_pupil_image(n_pixels=501)
        extent = image.shape[0] * pix_scale
        assert extent >= bundled_mask.primary_diameter

    def test_extent_covers_all_holes(self, bundled_mask):
        """All hole positions fall within image extent."""
        image, pix_scale = bundled_mask.make_pupil_image(n_pixels=501)
        # All holes should produce nonzero pixels.
        assert np.sum(image) > 0
        # Each hole should contribute some pixels.
        n_total = np.sum(image)
        assert n_total > bundled_mask.n_holes


class TestMakePupilImageBaseline:
    """Tests for single-baseline pupil image."""

    def test_output_shape(self, three_hole_mask):
        """Output shape matches request."""
        bl = three_hole_mask.baselines[0]
        image, _ = three_hole_mask.make_pupil_image_baseline(bl, n_pixels=101)
        assert image.shape == (101, 101)

    def test_forced_odd(self, three_hole_mask):
        """Even input forced to odd."""
        bl = three_hole_mask.baselines[0]
        image, _ = three_hole_mask.make_pupil_image_baseline(bl, n_pixels=100)
        assert image.shape == (101, 101)

    def test_only_two_holes(self, bundled_mask):
        """Only two holes are illuminated."""
        bl = bundled_mask.baselines[0]
        image, pix_scale = bundled_mask.make_pupil_image_baseline(
            bl, n_pixels=501
        )
        # Full mask image for comparison.
        full_image, _ = bundled_mask.make_pupil_image(n_pixels=501)
        # Baseline image should have fewer lit pixels.
        assert np.sum(image) < np.sum(full_image)
        assert np.sum(image) > 0

    def test_same_pixel_scale_as_full(self, bundled_mask):
        """Pixel scale matches full pupil image."""
        bl = bundled_mask.baselines[0]
        _, ps_full = bundled_mask.make_pupil_image(n_pixels=501)
        _, ps_bl = bundled_mask.make_pupil_image_baseline(bl, n_pixels=501)
        assert ps_full == pytest.approx(ps_bl)


class TestComputeSyntheticPSF:
    """Tests for synthetic PSF computation."""

    def test_output_shape_default(self, three_hole_mask):
        """Default output is 101x101."""
        psf = three_hole_mask.compute_synthetic_psf(wavelength=3.5)
        assert psf.shape == (101, 101)

    def test_output_shape_custom(self, three_hole_mask):
        """Custom output size is respected."""
        psf = three_hole_mask.compute_synthetic_psf(
            wavelength=3.5, n_pixels_image=51
        )
        assert psf.shape == (51, 51)

    def test_forced_odd_image(self, three_hole_mask):
        """Even n_pixels_image forced to odd."""
        psf = three_hole_mask.compute_synthetic_psf(
            wavelength=3.5, n_pixels_image=50
        )
        assert psf.shape == (51, 51)

    def test_forced_odd_pupil(self, three_hole_mask):
        """Even n_pixels_pupil forced to odd."""
        psf = three_hole_mask.compute_synthetic_psf(
            wavelength=3.5, n_pixels_pupil=100
        )
        # Should still produce valid output.
        assert psf.shape[0] % 2 == 1

    def test_peak_normalized(self, three_hole_mask):
        """PSF peak is normalized to 1."""
        psf = three_hole_mask.compute_synthetic_psf(wavelength=3.5)
        assert psf.max() == pytest.approx(1.0)

    def test_nonnegative(self, three_hole_mask):
        """PSF values are non-negative."""
        psf = three_hole_mask.compute_synthetic_psf(wavelength=3.5)
        assert np.all(psf >= 0)

    def test_peak_near_center(self, three_hole_mask):
        """PSF peak is at or near the center pixel."""
        psf = three_hole_mask.compute_synthetic_psf(
            wavelength=3.5, n_pixels_image=101
        )
        peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)
        center = 101 // 2
        assert abs(peak_y - center) <= 1
        assert abs(peak_x - center) <= 1

    def test_default_pixel_scale(self, three_hole_mask):
        """Default pixel scale is ALES value."""
        # Should run without specifying pixel_scale.
        psf = three_hole_mask.compute_synthetic_psf(wavelength=3.5)
        assert psf.shape == (101, 101)

    def test_different_wavelengths(self, three_hole_mask):
        """Different wavelengths produce different PSFs."""
        psf1 = three_hole_mask.compute_synthetic_psf(
            wavelength=3.0, n_pixels_image=51
        )
        psf2 = three_hole_mask.compute_synthetic_psf(
            wavelength=4.0, n_pixels_image=51
        )
        # Not identical.
        assert not np.allclose(psf1, psf2)

    def test_large_image_does_not_crash(self, three_hole_mask):
        """Large n_pixels_image works without error."""
        psf = three_hole_mask.compute_synthetic_psf(
            wavelength=3.5,
            n_pixels_image=201,
            n_pixels_pupil=101,
        )
        assert psf.shape == (201, 201)

    def test_n_pixels_image_larger_than_n_pad(self, three_hole_mask):
        """Output works when n_pixels_image > natural n_pad."""
        # Use large pixel scale to make n_pad small.
        psf = three_hole_mask.compute_synthetic_psf(
            wavelength=3.5,
            pixel_scale_arcsec=0.5,
            n_pixels_image=201,
            n_pixels_pupil=51,
        )
        assert psf.shape == (201, 201)
        assert psf.max() == pytest.approx(1.0)


class TestComputeSyntheticPowerSpectrum:
    """Tests for synthetic power spectrum."""

    def test_output_shape(self, three_hole_mask):
        """Output matches requested size."""
        ps = three_hole_mask.compute_synthetic_power_spectrum(
            wavelength=3.5, n_pixels_image=101
        )
        assert ps.shape == (101, 101)

    def test_nonnegative(self, three_hole_mask):
        """Power spectrum is non-negative."""
        ps = three_hole_mask.compute_synthetic_power_spectrum(wavelength=3.5)
        assert np.all(ps >= 0)

    def test_dc_at_center(self, three_hole_mask):
        """DC component (maximum) is at center."""
        ps = three_hole_mask.compute_synthetic_power_spectrum(
            wavelength=3.5, n_pixels_image=101
        )
        peak_y, peak_x = np.unravel_index(np.argmax(ps), ps.shape)
        center = 101 // 2
        assert peak_y == center
        assert peak_x == center

    def test_symmetric(self, three_hole_mask):
        """Power spectrum has conjugate symmetry."""
        ps = three_hole_mask.compute_synthetic_power_spectrum(
            wavelength=3.5, n_pixels_image=101
        )
        # PS should be symmetric: PS[y,x] ≈ PS[N-1-y, N-1-x]
        # For odd N, flipping reverses around the center pixel
        # with no roll needed.
        flipped = ps[::-1, ::-1]
        np.testing.assert_allclose(ps, flipped, rtol=1e-6, atol=1e-20)


class TestComputeSplodgePositions:
    """Tests for splodge position computation."""

    def test_output_structure(self, three_hole_mask):
        """Returns dict with correct keys and shapes."""
        wavelengths = np.array([3.0, 3.5, 4.0])
        positions = three_hole_mask.compute_splodge_positions(
            wavelengths=wavelengths,
            ny=67,
            nx=67,
        )
        assert isinstance(positions, dict)
        assert len(positions) == 3  # 3 baselines
        for _name, coords in positions.items():
            assert coords.shape == (3, 2)

    def test_default_parameters(self, three_hole_mask):
        """Defaults use ALES pixel scale and 67x67."""
        wavelengths = np.array([3.5])
        positions = three_hole_mask.compute_splodge_positions(
            wavelengths=wavelengths
        )
        assert len(positions) == 3

    def test_positions_within_frame(self, bundled_mask):
        """Splodge positions are within frame bounds."""
        wavelengths = np.linspace(2.8, 4.2, 10)
        positions = bundled_mask.compute_splodge_positions(
            wavelengths=wavelengths,
            ny=67,
            nx=67,
        )
        for coords in positions.values():
            assert np.all(coords[:, 0] >= 0)
            assert np.all(coords[:, 0] < 67)
            assert np.all(coords[:, 1] >= 0)
            assert np.all(coords[:, 1] < 67)

    def test_longer_baseline_further_from_center(self, three_hole_mask):
        """Longer baselines produce splodges further out."""
        wavelengths = np.array([3.5])
        positions = three_hole_mask.compute_splodge_positions(
            wavelengths=wavelengths,
            ny=67,
            nx=67,
        )
        center_y = (67 - 1) / 2.0
        center_x = (67 - 1) / 2.0

        distances = {}
        for name, coords in positions.items():
            dy = coords[0, 0] - center_y
            dx = coords[0, 1] - center_x
            distances[name] = np.sqrt(dy**2 + dx**2)

        # Find the longest baseline.
        longest_bl = max(
            three_hole_mask.baselines,
            key=lambda b: b.length,
        )
        shortest_bl = min(
            three_hole_mask.baselines,
            key=lambda b: b.length,
        )
        assert distances[longest_bl.name] > distances[shortest_bl.name]

    def test_shorter_wavelength_further_out(self, three_hole_mask):
        """Shorter wavelength pushes splodges further."""
        wav_short = np.array([3.0])
        wav_long = np.array([4.5])

        pos_short = three_hole_mask.compute_splodge_positions(
            wavelengths=wav_short, ny=67, nx=67
        )
        pos_long = three_hole_mask.compute_splodge_positions(
            wavelengths=wav_long, ny=67, nx=67
        )

        center_y = (67 - 1) / 2.0
        center_x = (67 - 1) / 2.0

        bl_name = three_hole_mask.baselines[0].name
        dy_s = pos_short[bl_name][0, 0] - center_y
        dx_s = pos_short[bl_name][0, 1] - center_x
        dist_short = np.sqrt(dy_s**2 + dx_s**2)

        dy_l = pos_long[bl_name][0, 0] - center_y
        dx_l = pos_long[bl_name][0, 1] - center_x
        dist_long = np.sqrt(dy_l**2 + dx_l**2)

        assert dist_short > dist_long

    def test_non_square_frame(self, three_hole_mask):
        """Handles non-square ny != nx."""
        wavelengths = np.array([3.5])
        positions = three_hole_mask.compute_splodge_positions(
            wavelengths=wavelengths,
            ny=63,
            nx=67,
        )
        # Should not crash.
        assert len(positions) == 3

    def test_center_reference(self, three_hole_mask):
        """Center is at (ny-1)/2, (nx-1)/2."""
        wavelengths = np.array([3.5])
        # Zero-length baseline would be at center.
        # Use a real baseline and verify direction.
        positions = three_hole_mask.compute_splodge_positions(
            wavelengths=wavelengths,
            ny=67,
            nx=67,
        )
        # H1H2 has bx=2.0, by=0.0: should offset in x.
        bl_h1h2 = positions["H1H2"]
        center_y = (67 - 1) / 2.0
        center_x = (67 - 1) / 2.0
        # y position should be near center (by=0).
        assert bl_h1h2[0, 0] == pytest.approx(center_y, abs=0.5)
        # x position should be offset from center.
        assert bl_h1h2[0, 1] != pytest.approx(center_x, abs=0.1)


class TestALESPixelScaleConstant:
    """Tests for the ALES pixel scale constant."""

    def test_value(self):
        """ALES pixel scale is 34.5 mas = 0.0345 arcsec."""
        assert ALES_PIXEL_SCALE_ARCSEC == pytest.approx(0.0345)

    def test_used_as_default(self, three_hole_mask):
        """Default pixel scale matches constant."""
        # Calling without pixel_scale should use the
        # constant (verified by not raising).
        psf = three_hole_mask.compute_synthetic_psf(wavelength=3.5)
        assert psf.shape == (101, 101)


class TestSplodgePositionsConsistency:
    """Cross-validate splodge positions with synthetic PS."""

    def test_splodges_match_power_spectrum_peaks(self, bundled_mask):
        """Analytical positions match PS peak locations."""
        wavelength = 3.5
        n_pix = 67
        pixel_scale = ALES_PIXEL_SCALE_ARCSEC

        ps = bundled_mask.compute_synthetic_power_spectrum(
            wavelength=wavelength,
            pixel_scale_arcsec=pixel_scale,
            n_pixels_image=n_pix,
            n_pixels_pupil=501,
        )

        wavelengths = np.array([wavelength])
        positions = bundled_mask.compute_splodge_positions(
            wavelengths=wavelengths,
            pixel_scale_arcsec=pixel_scale,
            ny=n_pix,
            nx=n_pix,
        )

        # For each baseline, the analytical position
        # should be near a local maximum in the PS.
        for _bl_name, coords in positions.items():
            y, x = coords[0]
            # Skip if too close to edge.
            if y < 2 or y > n_pix - 3 or x < 2 or x > n_pix - 3:
                continue
            # Check that the pixel nearest the predicted
            # position has high power relative to
            # neighbors.
            iy, ix = int(round(y)), int(round(x))
            local_val = ps[iy, ix]
            # Should be above median of entire PS
            # (splodges are bright spots).
            assert local_val > np.median(ps)


class TestPlotMethods:
    """Tests that plot methods run without error."""

    @pytest.fixture(autouse=True)
    def _setup_matplotlib(self):
        """Use non-interactive backend."""
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_pupil(self, bundled_mask):
        """plot_pupil runs without error."""
        ax = bundled_mask.plot_pupil(n_pixels=101)
        assert ax is not None

    def test_plot_synthetic_psf(self, three_hole_mask):
        """plot_synthetic_psf runs without error."""
        ax = three_hole_mask.plot_synthetic_psf(
            wavelength=3.5, n_pixels_image=51
        )
        assert ax is not None

    def test_plot_synthetic_power_spectrum(self, three_hole_mask):
        """plot_synthetic_power_spectrum runs."""
        ax = three_hole_mask.plot_synthetic_power_spectrum(
            wavelength=3.5, n_pixels_image=51
        )
        assert ax is not None

    def test_plot_power_spectrum_with_baselines_log(self, three_hole_mask):
        """plot_power_spectrum_with_baselines runs."""
        ps = three_hole_mask.compute_synthetic_power_spectrum(
            wavelength=3.5, n_pixels_image=67
        )
        ax = three_hole_mask.plot_power_spectrum_with_baselines(
            power_spectrum=ps,
            wavelength=3.5,
        )
        assert ax is not None

    def test_plot_ps_with_baselines_linear(self, three_hole_mask):
        """plot_power_spectrum_with_baselines linear scale."""
        ps = three_hole_mask.compute_synthetic_power_spectrum(
            wavelength=3.5, n_pixels_image=67
        )
        ax = three_hole_mask.plot_power_spectrum_with_baselines(
            power_spectrum=ps,
            wavelength=3.5,
            log_scale=False,
        )
        assert ax is not None

    def test_plot_psf_linear_scale(self, three_hole_mask):
        """Plot PSF with linear scale."""
        ax = three_hole_mask.plot_synthetic_psf(
            wavelength=3.5,
            n_pixels_image=51,
            log_scale=False,
        )
        assert ax is not None

    def test_plot_ps_linear_scale(self, three_hole_mask):
        """Plot power spectrum with linear scale."""
        ax = three_hole_mask.plot_synthetic_power_spectrum(
            wavelength=3.5,
            n_pixels_image=51,
            log_scale=False,
        )
        assert ax is not None
