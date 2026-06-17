"""Tests for ales_nrm.observables module."""

import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from ales_nrm.nrm.mask import Hole, NRMMask
from ales_nrm.observables import (
    BaselineInfo,
    Observables,
    Station,
    TriangleInfo,
    _compute_mean_mjd,
    _parse_time_string_to_seconds,
    _parse_timestamps_to_seconds,
)


@pytest.fixture()
def four_hole_mask():
    """Create a 4-hole mask for testing of from_block_and_mask."""
    holes = [
        Hole(name="H1", x=0.0, y=1.0, radius=0.25),
        Hole(name="H2", x=1.0, y=0.0, radius=0.25),
        Hole(name="H3", x=-1.0, y=0.0, radius=0.25),
        Hole(name="H4", x=0.0, y=-1.0, radius=0.25),
    ]
    mask = NRMMask(
        primary_diameter=8.4,
        holes=holes,
        source_name="test_mask",
    )
    mask._compute_baselines()

    return mask


@pytest.fixture()
def sample_stations():
    """Provide 4 stations, matching to the four_hole_mask fixture."""
    return [
        Station(index=1, name="H1", x=0.0, y=1.0, diameter=0.5),
        Station(index=2, name="H2", x=1.0, y=0.0, diameter=0.5),
        Station(index=3, name="H3", x=-1.0, y=0.0, diameter=0.5),
        Station(index=4, name="H4", x=0.0, y=-1.0, diameter=0.5),
    ]


@pytest.fixture()
def sample_baselines():
    """Provide 6 baselines, matching to the four_hole_mask fixture."""
    return [
        BaselineInfo(sta_index=(1, 2), name="H1H2", u=1.0, v=-1.0),
        BaselineInfo(sta_index=(1, 3), name="H1H3", u=-1.0, v=-1.0),
        BaselineInfo(sta_index=(1, 4), name="H1H4", u=0.0, v=-2.0),
        BaselineInfo(sta_index=(2, 3), name="H2H3", u=-2.0, v=0.0),
        BaselineInfo(sta_index=(2, 4), name="H2H4", u=-1.0, v=-1.0),
        BaselineInfo(sta_index=(3, 4), name="H3H4", u=1.0, v=-1.0),
    ]


@pytest.fixture()
def sample_triangles():
    """Provide 4 triangles, matching to the four_hole_mask fixture."""
    return [
        TriangleInfo(
            sta_index=(1, 2, 3),
            name="H1-H2-H3",
            u1=1.0,
            v1=-1.0,
            u2=-2.0,
            v2=0.0,
        ),
        TriangleInfo(
            sta_index=(1, 2, 4),
            name="H1-H2-H4",
            u1=1.0,
            v1=-1.0,
            u2=-1.0,
            v2=-1.0,
        ),
        TriangleInfo(
            sta_index=(1, 3, 4),
            name="H1-H3-H4",
            u1=-1.0,
            v1=-1.0,
            u2=1.0,
            v2=-1.0,
        ),
        TriangleInfo(
            sta_index=(2, 3, 4),
            name="H2-H3-H4",
            u1=-2.0,
            v1=0.0,
            u2=1.0,
            v2=-1.0,
        ),
    ]


@pytest.fixture()
def sample_wavelengths_short():
    """A small wavelength array for fast tests."""
    return np.array([2.7, 4.3])


@pytest.fixture()
def sample_observables(
    sample_stations,
    sample_baselines,
    sample_triangles,
    sample_wavelengths_short,
):
    """Create a minimal Observables instance with no data arrays."""
    return Observables(
        target="TestTarget",
        wavelengths=sample_wavelengths_short,
        stations=sample_stations,
        baselines=sample_baselines,
        triangles=sample_triangles,
        mean_para_angle=45.0,
        mjd=59000.5,
        time_start="08:00:00.000",
        time_end="08:30:00.000",
    )


@pytest.fixture()
def populated_observables(sample_observables):
    """Create an Observables with all data arrays filled."""
    obs = sample_observables
    n_bl = obs.n_baselines
    n_tri = obs.n_triangles
    n_wav = obs.n_wav

    rng = np.random.default_rng(42)
    obs.vis2 = rng.uniform(0, 1, (n_bl, n_wav))
    obs.vis2_err = rng.uniform(0, 0.1, (n_bl, n_wav))
    obs.vis2_flag = np.zeros((n_bl, n_wav), dtype=bool)

    obs.t3phi = rng.uniform(-180, 180, (n_tri, n_wav))
    obs.t3phi_err = rng.uniform(0, 10, (n_tri, n_wav))
    obs.t3amp = rng.uniform(0, 1, (n_tri, n_wav))
    obs.t3amp_err = rng.uniform(0, 0.1, (n_tri, n_wav))
    obs.t3_flag = np.zeros((n_tri, n_wav), dtype=bool)

    obs.visamp = rng.uniform(0, 1, (n_bl, n_wav))
    obs.visamp_err = rng.uniform(0, 0.1, (n_bl, n_wav))
    obs.visphi = rng.uniform(-180, 180, (n_bl, n_wav))
    obs.visphi_err = rng.uniform(0, 10, (n_bl, n_wav))
    obs.vis_flag = np.zeros((n_bl, n_wav), dtype=bool)

    return obs


@pytest.fixture()
def mock_block():
    """Create a mock ObservingBlock for from_block_and_mask."""
    block = MagicMock()
    block.parallactic_angles = np.array([10.0, 12.0, 14.0])
    block.observation_date = datetime.date(2023, 6, 15)
    block.timestamps = np.array(
        ["08:00:00.000", "08:15:00.000", "08:30:00.000"],
        dtype=object,
    )
    block.target = "HD12345"
    block.wavelengths = np.array([3.0, 3.5, 4.0])
    block.block_type = "SCI"
    return block


class TestStation:
    """Tests for Station dataclass."""

    def test_creation(self):
        """Station stores attributes correctly."""
        s = Station(index=1, name="H1", x=1.5, y=-2.3, diameter=0.5)
        assert s.index == 1
        assert s.name == "H1"
        assert s.x == 1.5
        assert s.y == -2.3
        assert s.diameter == 0.5
        assert s.z == 0.0

    def test_frozen(self):
        """Station is immutable."""
        s = Station(index=1, name="H1", x=0.0, y=0.0)
        with pytest.raises(AttributeError):
            s.x = 1.0

    def test_defaults(self):
        """Station z and diameter default to 0."""
        s = Station(index=1, name="H1", x=0.0, y=0.0)
        assert s.z == 0.0
        assert s.diameter == 0.0


class TestBaselineInfo:
    """Tests for BaselineInfo dataclass."""

    def test_creation(self):
        """BaselineInfo stores attributes correctly."""
        bl = BaselineInfo(sta_index=(1, 2), name="H1H2", u=0.5, v=-0.3)
        assert bl.sta_index == (1, 2)
        assert bl.name == "H1H2"
        assert bl.u == 0.5
        assert bl.v == -0.3

    def test_frozen(self):
        """BaselineInfo is immutable."""
        bl = BaselineInfo(sta_index=(1, 2), name="H1H2", u=0.5, v=-0.3)
        with pytest.raises(AttributeError):
            bl.u = 2.0


class TestTriangleInfo:
    """Tests for TriangleInfo dataclass."""

    def test_creation(self):
        """TriangleInfo stores attributes correctly."""
        tri = TriangleInfo(
            sta_index=(1, 2, 3),
            name="H1-H2-H3",
            u1=1.0,
            v1=0.5,
            u2=-0.5,
            v2=1.0,
        )
        assert tri.sta_index == (1, 2, 3)
        assert tri.name == "H1-H2-H3"
        assert tri.u1 == 1.0
        assert tri.v1 == 0.5
        assert tri.u2 == -0.5
        assert tri.v2 == 1.0

    def test_frozen(self):
        """TriangleInfo is immutable."""
        tri = TriangleInfo(
            sta_index=(1, 2, 3),
            name="H1-H2-H3",
            u1=1.0,
            v1=0.5,
            u2=-0.5,
            v2=1.0,
        )
        with pytest.raises(AttributeError):
            tri.u1 = 2.0


class TestObservablesProperties:
    """Tests for Observables property accessors."""

    def test_n_wav(self, sample_observables):
        """n_wav returns number of wavelength channels."""
        assert sample_observables.n_wav == 2

    def test_n_baselines(self, sample_observables):
        """n_baselines returns number of baselines."""
        assert sample_observables.n_baselines == 6

    def test_n_triangles(self, sample_observables):
        """n_triangles returns number of triangles."""
        assert sample_observables.n_triangles == 4

    def test_n_stations(self, sample_observables):
        """n_stations returns number of stations."""
        assert sample_observables.n_stations == 4

    def test_has_vis2_false_when_none(self, sample_observables):
        """has_vis2 is False when vis2 is None."""
        assert sample_observables.has_vis2 is False

    def test_has_vis2_true_when_set(self, populated_observables):
        """has_vis2 is True when vis2 is populated."""
        assert populated_observables.has_vis2 is True

    def test_has_vis2_err_false_when_none(self, sample_observables):
        """has_vis2_err is False when vis2_err is None."""
        assert sample_observables.has_vis2_err is False

    def test_has_vis2_err_true_when_set(self, populated_observables):
        """has_vis2_err is True when vis2_err is populated."""
        assert populated_observables.has_vis2_err is True

    def test_has_t3phi_false_when_none(self, sample_observables):
        """has_t3phi is False when t3phi is None."""
        assert sample_observables.has_t3phi is False

    def test_has_t3phi_true_when_set(self, populated_observables):
        """has_t3phi is True when t3phi is populated."""
        assert populated_observables.has_t3phi is True

    def test_has_t3phi_err_false_when_none(self, sample_observables):
        """has_t3phi_err is False when t3phi_err is None."""
        assert sample_observables.has_t3phi_err is False

    def test_has_t3phi_err_true_when_set(self, populated_observables):
        """has_t3phi_err is True when t3phi_err is populated."""
        assert populated_observables.has_t3phi_err is True

    def test_has_t3amp_false_when_none(self, sample_observables):
        """has_t3amp is False when t3amp is None."""
        assert sample_observables.has_t3amp is False

    def test_has_t3amp_true_when_set(self, populated_observables):
        """has_t3amp is True when t3amp is populated."""
        assert populated_observables.has_t3amp is True

    def test_has_t3amp_err_false_when_none(self, sample_observables):
        """has_t3amp_err is False when t3amp_err is None."""
        assert sample_observables.has_t3amp_err is False

    def test_has_t3amp_err_true_when_set(self, populated_observables):
        """has_t3amp_err is True when t3amp_err is populated."""
        assert populated_observables.has_t3amp_err is True

    def test_has_visamp_false_when_none(self, sample_observables):
        """has_visamp is False when visamp is None."""
        assert sample_observables.has_visamp is False

    def test_has_visamp_true_when_set(self, populated_observables):
        """has_visamp is True when visamp is populated."""
        assert populated_observables.has_visamp is True

    def test_has_visamp_err_false_when_none(self, sample_observables):
        """has_visamp_err is False when visamp_err is None."""
        assert sample_observables.has_visamp_err is False

    def test_has_visamp_err_true_when_set(self, populated_observables):
        """has_visamp_err is True when visamp_err is populated."""
        assert populated_observables.has_visamp_err is True

    def test_has_visphi_false_when_none(self, sample_observables):
        """has_visphi is False when visphi is None."""
        assert sample_observables.has_visphi is False

    def test_has_visphi_true_when_set(self, populated_observables):
        """has_visphi is True when visphi is populated."""
        assert populated_observables.has_visphi is True

    def test_has_visphi_err_false_when_none(self, sample_observables):
        """has_visphi_err is False when visphi_err is None."""
        assert sample_observables.has_visphi_err is False

    def test_has_visphi_err_true_when_set(self, populated_observables):
        """has_visphi_err is True when visphi_err is populated."""
        assert populated_observables.has_visphi_err is True


class TestObservablesValidate:
    """Tests for Observables.validate method."""

    def test_valid_empty_observables(self, sample_observables):
        """Validates successfully with no data arrays."""
        sample_observables.validate()

    def test_valid_populated_observables(self, populated_observables):
        """Validates successfully with all arrays populated."""
        populated_observables.validate()

    def test_invalid_wavelengths_shape(self, sample_observables):
        """Raises ValueError if wavelengths is not 1D."""
        sample_observables.wavelengths = np.zeros((3, 2))
        with pytest.raises(ValueError, match="wavelengths must be 1D"):
            sample_observables.validate()

    def test_invalid_vis2_shape(self, sample_observables):
        """Raises ValueError if vis2 shape is wrong."""
        sample_observables.vis2 = np.zeros((3, 3))  # wrong n_bl
        with pytest.raises(ValueError, match="vis2 shape"):
            sample_observables.validate()

    def test_invalid_vis2_err_shape(self, sample_observables):
        """Raises ValueError if vis2_err shape is wrong."""
        n_bl = sample_observables.n_baselines
        n_wav = sample_observables.n_wav
        sample_observables.vis2 = np.zeros((n_bl, n_wav))
        sample_observables.vis2_err = np.zeros((n_bl + 1, n_wav))
        with pytest.raises(ValueError, match="vis2_err shape"):
            sample_observables.validate()

    def test_invalid_vis2_flag_shape(self, sample_observables):
        """Raises ValueError if vis2_flag shape is wrong."""
        n_bl = sample_observables.n_baselines
        n_wav = sample_observables.n_wav
        sample_observables.vis2 = np.zeros((n_bl, n_wav))
        sample_observables.vis2_flag = np.zeros((n_bl, n_wav + 1), dtype=bool)
        with pytest.raises(ValueError, match="vis2_flag shape"):
            sample_observables.validate()

    def test_invalid_t3phi_shape(self, sample_observables):
        """Raises ValueError if t3phi shape is wrong."""
        sample_observables.t3phi = np.zeros((2, 3))  # wrong n_tri
        with pytest.raises(ValueError, match="t3phi shape"):
            sample_observables.validate()

    def test_invalid_t3phi_err_shape(self, sample_observables):
        """Raises ValueError if t3phi_err shape is wrong."""
        n_tri = sample_observables.n_triangles
        n_wav = sample_observables.n_wav
        sample_observables.t3phi_err = np.zeros((n_tri + 1, n_wav))
        with pytest.raises(ValueError, match="t3phi_err shape"):
            sample_observables.validate()

    def test_invalid_t3amp_shape(self, sample_observables):
        """Raises ValueError if t3amp shape is wrong."""
        sample_observables.t3amp = np.zeros((2, 3))
        with pytest.raises(ValueError, match="t3amp shape"):
            sample_observables.validate()

    def test_invalid_t3amp_err_shape(self, sample_observables):
        """Raises ValueError if t3amp_err shape is wrong."""
        n_tri = sample_observables.n_triangles
        n_wav = sample_observables.n_wav
        sample_observables.t3amp_err = np.zeros((n_tri, n_wav + 1))
        with pytest.raises(ValueError, match="t3amp_err shape"):
            sample_observables.validate()

    def test_invalid_t3_flag_shape(self, sample_observables):
        """Raises ValueError if t3_flag shape is wrong."""
        sample_observables.t3_flag = np.zeros((1, 3), dtype=bool)
        with pytest.raises(ValueError, match="t3_flag shape"):
            sample_observables.validate()

    def test_invalid_visamp_shape(self, sample_observables):
        """Raises ValueError if visamp shape is wrong."""
        sample_observables.visamp = np.zeros((3, 3))
        with pytest.raises(ValueError, match="visamp shape"):
            sample_observables.validate()

    def test_invalid_visamp_err_shape(self, sample_observables):
        """Raises ValueError if visamp_err shape is wrong."""
        sample_observables.visamp_err = np.zeros((3, 3))
        with pytest.raises(ValueError, match="visamp_err shape"):
            sample_observables.validate()

    def test_invalid_visphi_shape(self, sample_observables):
        """Raises ValueError if visphi shape is wrong."""
        sample_observables.visphi = np.zeros((3, 3))
        with pytest.raises(ValueError, match="visphi shape"):
            sample_observables.validate()

    def test_invalid_visphi_err_shape(self, sample_observables):
        """Raises ValueError if visphi_err shape is wrong."""
        sample_observables.visphi_err = np.zeros((3, 3))
        with pytest.raises(ValueError, match="visphi_err shape"):
            sample_observables.validate()

    def test_invalid_vis_flag_shape(self, sample_observables):
        """Raises ValueError if vis_flag shape is wrong."""
        sample_observables.vis_flag = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="vis_flag shape"):
            sample_observables.validate()

    def test_invalid_station_indices_not_contiguous(self, sample_observables):
        """Raises ValueError if station indices are not 1..N."""
        sample_observables.stations[0] = Station(
            index=5,
            name="H1",
            x=0.0,
            y=1.0,
        )
        with pytest.raises(ValueError, match="Station indices must be"):
            sample_observables.validate()

    def test_invalid_station_indices_zero_based(self, sample_observables):
        """Raises ValueError if station indices start at 0."""
        sample_observables.stations = [
            Station(index=i, name=f"H{i}", x=0.0, y=0.0) for i in range(4)
        ]
        with pytest.raises(ValueError, match="Station indices must be"):
            sample_observables.validate()

    def test_invalid_baseline_station_reference(self, sample_observables):
        """Raises ValueError if baseline references invalid station."""
        sample_observables.baselines[0] = BaselineInfo(
            sta_index=(1, 99),
            name="bad",
            u=0.0,
            v=0.0,
        )
        with pytest.raises(ValueError, match="Baseline .* references station"):
            sample_observables.validate()

    def test_invalid_triangle_station_reference(self, sample_observables):
        """Raises ValueError if triangle references invalid station."""
        sample_observables.triangles[0] = TriangleInfo(
            sta_index=(1, 2, 99),
            name="bad",
            u1=0.0,
            v1=0.0,
            u2=0.0,
            v2=0.0,
        )
        with pytest.raises(ValueError, match="Triangle .* references station"):
            sample_observables.validate()


class TestArrStatus:
    """Tests for Observables._arr_status helper."""

    def test_none_returns_cross(self, sample_observables):
        """Returns cross mark for None."""
        assert sample_observables._arr_status(None) == "\u2717"

    def test_all_nan_returns_no_data(self, sample_observables):
        """Returns check with 'no data' for all-NaN array."""
        arr = np.full((3, 3), np.nan)
        assert sample_observables._arr_status(arr) == "\u2713 (no data)"

    def test_valid_data_returns_check(self, sample_observables):
        """Returns check mark for array with real values."""
        arr = np.ones((3, 3))
        assert sample_observables._arr_status(arr) == "\u2713"

    def test_partial_nan_returns_check(self, sample_observables):
        """Returns check mark for array with some NaN."""
        arr = np.ones((3, 3))
        arr[0, 0] = np.nan
        assert sample_observables._arr_status(arr) == "\u2713"


class TestObservablesSummary:
    """Tests for Observables.summary method."""

    def test_summary_contains_target(self, sample_observables):
        """Summary includes target name."""
        s = sample_observables.summary()
        assert "TestTarget" in s

    def test_summary_contains_wavelength_info(self, sample_observables):
        """Summary includes wavelength range."""
        s = sample_observables.summary()
        assert "2.7000" in s
        assert "4.3000" in s

    def test_summary_contains_geometry_counts(self, sample_observables):
        """Summary includes station, baseline, triangle counts."""
        s = sample_observables.summary()
        assert "Stations: 4" in s
        assert "Baselines: 6" in s
        assert "Triangles: 4" in s

    def test_summary_contains_para_angle(self, sample_observables):
        """Summary includes mean parallactic angle."""
        s = sample_observables.summary()
        assert "45.00" in s

    def test_summary_shows_cross_for_empty(self, sample_observables):
        """Summary shows cross marks when no data arrays are set."""
        s = sample_observables.summary()
        assert "\u2717" in s

    def test_summary_shows_check_for_populated(self, populated_observables):
        """Summary shows check marks when arrays are populated."""
        s = populated_observables.summary()
        assert "\u2713" in s
        # No "no data" since all arrays have real values
        assert "\u2713 (no data)" not in s

    def test_summary_shows_no_data_for_nan_err(self, sample_observables):
        """Summary shows 'no data' for all-NaN error arrays."""
        n_bl = sample_observables.n_baselines
        n_wav = sample_observables.n_wav
        sample_observables.visamp = np.ones((n_bl, n_wav))
        sample_observables.visamp_err = np.full((n_bl, n_wav), np.nan)
        s = sample_observables.summary()
        assert "\u2713 (no data)" in s

    def test_summary_shows_calibrator(self, sample_observables):
        """Summary includes calibrator when calibrated."""
        sample_observables.calibrated = True
        sample_observables.calibrator_target = "HD_CAL"
        s = sample_observables.summary()
        assert "HD_CAL" in s

    def test_summary_shows_notes(self, sample_observables):
        """Summary includes notes when present."""
        sample_observables.notes = "test note here"
        s = sample_observables.summary()
        assert "test note here" in s

    def test_summary_no_calibrator_when_uncalibrated(self, sample_observables):
        """Summary omits calibrator line when not calibrated."""
        s = sample_observables.summary()
        assert "Calibrator" not in s

    def test_summary_no_notes_when_empty(self, sample_observables):
        """Summary omits notes line when empty."""
        s = sample_observables.summary()
        assert "Notes" not in s


class TestFromBlockAndMask:
    """Tests for Observables.from_block_and_mask factory method."""

    def test_returns_observables_instance(self, mock_block, four_hole_mask):
        """Factory returns an Observables instance."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert isinstance(obs, Observables)

    def test_target_from_block(self, mock_block, four_hole_mask):
        """Target comes from block."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.target == "HD12345"

    def test_wavelengths_from_block(self, mock_block, four_hole_mask):
        """Wavelengths come from block."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        np.testing.assert_array_equal(obs.wavelengths, mock_block.wavelengths)

    def test_wavelengths_are_copy(self, mock_block, four_hole_mask):
        """Wavelengths array is a copy, not a reference."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        mock_block.wavelengths[0] = 999.0
        assert obs.wavelengths[0] != 999.0

    def test_mean_para_angle(self, mock_block, four_hole_mask):
        """Mean parallactic angle is computed from block."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        expected = float(np.mean(mock_block.parallactic_angles))
        assert obs.mean_para_angle == pytest.approx(expected)

    def test_stations_count(self, mock_block, four_hole_mask):
        """Number of stations matches number of holes."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.n_stations == 4

    def test_stations_indices_one_based(self, mock_block, four_hole_mask):
        """Station indices are 1-based."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        indices = [s.index for s in obs.stations]
        assert indices == [1, 2, 3, 4]

    def test_stations_names(self, mock_block, four_hole_mask):
        """Station names come from hole names."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        names = [s.name for s in obs.stations]
        assert names == ["H1", "H2", "H3", "H4"]

    def test_stations_coordinates(self, mock_block, four_hole_mask):
        """Station coordinates come from hole positions."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.stations[0].x == 0.0
        assert obs.stations[0].y == 1.0

    def test_stations_diameter(self, mock_block, four_hole_mask):
        """Station diameter is 2 * hole radius."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.stations[0].diameter == 0.5

    def test_baselines_count(self, mock_block, four_hole_mask):
        """Number of baselines matches mask."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.n_baselines == 6

    def test_baselines_names(self, mock_block, four_hole_mask):
        """Baseline names come from mask baseline names."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        names = [bl.name for bl in obs.baselines]
        assert names == ["H1H2", "H1H3", "H1H4", "H2H3", "H2H4", "H3H4"]

    def test_baselines_sta_index(self, mock_block, four_hole_mask):
        """Baseline station indices are correct."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.baselines[0].sta_index == (1, 2)
        assert obs.baselines[3].sta_index == (2, 3)

    def test_baselines_uv_rotated(self, mock_block, four_hole_mask):
        """Baseline u,v differ from unrotated bx,by."""
        # With non-zero parallactic angle, rotation should change values
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        # First baseline has bx=1.0, by=-1.0 before rotation
        # With mean para angle = 12 deg, values should differ
        bl = obs.baselines[0]
        # Not equal to unrotated values
        assert not (bl.u == pytest.approx(1.0) and bl.v == pytest.approx(-1.0))

    def test_baselines_uv_zero_para_angle(self, mock_block, four_hole_mask):
        """With zero parallactic angle, u,v equal bx,by."""
        mock_block.parallactic_angles = np.array([0.0, 0.0, 0.0])
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        bl = obs.baselines[0]
        assert bl.u == pytest.approx(1.0)
        assert bl.v == pytest.approx(-1.0)

    def test_triangles_count(self, mock_block, four_hole_mask):
        """Number of triangles matches mask."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.n_triangles == 4

    def test_triangles_sta_index(self, mock_block, four_hole_mask):
        """Triangle station indices are correct."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.triangles[0].sta_index == (1, 2, 3)
        assert obs.triangles[1].sta_index == (1, 2, 4)

    def test_mjd_computed(self, mock_block, four_hole_mask):
        """MJD is computed when observation_date is available."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.mjd is not None
        assert obs.mjd > 0

    def test_mjd_none_when_no_date(self, mock_block, four_hole_mask):
        """MJD is None when observation_date is None."""
        mock_block.observation_date = None
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.mjd is None

    def test_time_start(self, mock_block, four_hole_mask):
        """time_start is first timestamp."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.time_start == "08:00:00.000"

    def test_time_end(self, mock_block, four_hole_mask):
        """time_end is last timestamp."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.time_end == "08:30:00.000"

    def test_block_type(self, mock_block, four_hole_mask):
        """block_type comes from block."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.block_type == "SCI"

    def test_mask_name(self, mock_block, four_hole_mask):
        """mask_name comes from mask.source_name."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.mask_name == "test_mask"

    def test_data_arrays_are_none(self, mock_block, four_hole_mask):
        """All observable arrays are None after factory."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        assert obs.vis2 is None
        assert obs.vis2_err is None
        assert obs.t3phi is None
        assert obs.t3phi_err is None
        assert obs.t3amp is None
        assert obs.t3amp_err is None
        assert obs.visamp is None
        assert obs.visamp_err is None
        assert obs.visphi is None
        assert obs.visphi_err is None

    def test_validates_after_creation(self, mock_block, four_hole_mask):
        """Created Observables passes validate."""
        obs = Observables.from_block_and_mask(mock_block, four_hole_mask)
        obs.validate()


class TestParseTimeStringToSeconds:
    """Tests for _parse_time_string_to_seconds helper."""

    def test_midnight(self):
        """Midnight parses to 0 seconds."""
        assert _parse_time_string_to_seconds("00:00:00.000") == 0.0

    def test_noon(self):
        """Noon parses to 43200 seconds."""
        assert _parse_time_string_to_seconds("12:00:00.000") == 43200.0

    def test_fractional_seconds(self):
        """Fractional seconds are preserved."""
        result = _parse_time_string_to_seconds("01:02:03.456")
        expected = 3600 + 120 + 3.456
        assert result == pytest.approx(expected)

    def test_without_fractional(self):
        """Works without fractional seconds."""
        result = _parse_time_string_to_seconds("08:30:00")
        assert result == pytest.approx(8 * 3600 + 30 * 60)

    def test_leading_whitespace(self):
        """Leading/trailing whitespace is handled."""
        result = _parse_time_string_to_seconds("  12:00:00.000  ")
        assert result == pytest.approx(43200.0)


class TestParseTimestampsToSeconds:
    """Tests for _parse_timestamps_to_seconds helper."""

    def test_simple_sequence(self):
        """Normal sequence without midnight crossing."""
        stamps = np.array(["08:00:00.000", "08:15:00.000", "08:30:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        expected = np.array([28800.0, 29700.0, 30600.0])
        np.testing.assert_allclose(result, expected)

    def test_midnight_crossing(self):
        """Timestamps spanning midnight get unwrapped."""
        stamps = np.array(["23:50:00.000", "23:55:00.000", "00:05:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        # Last timestamp should be unwrapped to > 86400
        assert result[2] > 86400.0
        # Should be monotonically increasing
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_midnight_crossing_value(self):
        """Midnight-crossing unwrap gives correct value."""
        stamps = np.array(["23:59:00.000", "00:01:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        expected_second = 86400.0 + 60.0  # 00:01 = 86460 seconds
        assert result[1] == pytest.approx(expected_second)

    def test_no_crossing_near_midnight(self):
        """Sequence near midnight but not crossing is unmodified."""
        stamps = np.array(["23:50:00.000", "23:55:00.000", "23:59:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        # All should be < 86400
        assert np.all(result < 86400.0)

    def test_single_timestamp(self):
        """Single timestamp works."""
        stamps = np.array(["12:00:00.000"])
        result = _parse_timestamps_to_seconds(stamps)
        assert result[0] == pytest.approx(43200.0)


class TestComputeMeanMjd:
    """Tests for _compute_mean_mjd helper."""

    def test_none_date_returns_none(self):
        """Returns None when observation_date is None."""
        stamps = np.array(["08:00:00.000", "09:00:00.000"])
        assert _compute_mean_mjd(None, stamps) is None

    def test_known_date(self):
        """Computes reasonable MJD for known date and time."""
        date = datetime.date(2023, 1, 1)
        stamps = np.array(["12:00:00.000"])
        mjd = _compute_mean_mjd(date, stamps)
        # 2023-01-01 12:00 UT should be MJD ~59945.5
        assert mjd is not None
        assert mjd == pytest.approx(59945.5, abs=0.01)

    def test_mean_of_timestamps(self):
        """MJD reflects mean of multiple timestamps."""
        date = datetime.date(2023, 1, 1)
        stamps_early = np.array(["06:00:00.000"])
        stamps_late = np.array(["18:00:00.000"])
        stamps_both = np.array(["06:00:00.000", "18:00:00.000"])

        mjd_early = _compute_mean_mjd(date, stamps_early)
        mjd_late = _compute_mean_mjd(date, stamps_late)
        mjd_both = _compute_mean_mjd(date, stamps_both)

        # Mean should be between early and late
        assert mjd_early < mjd_both < mjd_late

    def test_midnight_crossing_mjd(self):
        """MJD handles midnight crossing correctly."""
        date = datetime.date(2023, 1, 1)
        stamps = np.array(["23:59:00.000", "00:01:00.000"])
        mjd = _compute_mean_mjd(date, stamps)
        # Mean is midnight = start of Jan 2
        # observation_date is Jan 1 (date before midnight)
        assert mjd is not None
        # Should be very close to MJD of 2023-01-02 00:00
        assert mjd == pytest.approx(59946.0, abs=0.01)

    def test_type_is_float(self):
        """Return type is float."""
        date = datetime.date(2023, 6, 15)
        stamps = np.array(["12:00:00.000"])
        mjd = _compute_mean_mjd(date, stamps)
        assert isinstance(mjd, float)
