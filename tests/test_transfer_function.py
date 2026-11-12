"""Tests for band-averaged series and transfer-function plots."""

import warnings

import numpy as np
import pytest

from ales_nrm.calibrate import _OBS_ATTR_MAP
from ales_nrm.observables import BaselineInfo, Observables, TriangleInfo
from ales_nrm.observation import (
    ObservingBlock,
    ObservingSequence,
)
from ales_nrm.transfer_function import (
    _SPECS,
    BandAveragedSeries,
    _band_average,
    _nan_mean_rows,
    _safe_block_time,
    compute_band_averaged_series,
)

WL = np.array([3.0, 3.5, 4.0, 4.5])
N_TRI = 2
N_BL = 3


def _make_obs(wavelengths, **arrays):
    """Build an Observables with named geometry and given arrays.

    Args:
        wavelengths: 1D wavelength array in µm.
        **arrays: Observable arrays keyed by attribute name.

    Returns:
        Observables with ``N_TRI`` triangles and ``N_BL`` baselines.
    """
    triangles = [
        TriangleInfo(
            sta_index=(1, 2, k + 3),
            name=f"H1-H2-H{k + 3}",
            u1=0.0,
            v1=0.0,
            u2=0.0,
            v2=0.0,
        )
        for k in range(N_TRI)
    ]
    baselines = [
        BaselineInfo(
            sta_index=(1, k + 2),
            name=f"H1H{k + 2}",
            u=0.0,
            v=0.0,
        )
        for k in range(N_BL)
    ]
    return Observables(
        target="t",
        wavelengths=np.asarray(wavelengths, dtype=float),
        stations=[],
        baselines=baselines,
        triangles=triangles,
        mean_para_angle=0.0,
        mjd=None,
        time_start="",
        time_end="",
        **arrays,
    )


def _arr(n_items, channel_values):
    """Repeat per-channel values to shape ``(n_items, n_wav)``."""
    return np.tile(np.asarray(channel_values, dtype=float), (n_items, 1))


@pytest.fixture()
def make_block(tmp_path):
    """Provide a factory for loaded ObservingBlocks."""

    def _make_block(
        block_type="CAL",
        target="cal",
        timestamps=("08:00:00.000",),
        observables=None,
    ):
        """Build a loaded block with timestamps and observables."""
        block = ObservingBlock(
            block_type=block_type,
            target=target,
            directory=tmp_path,
        )
        block.cubes = np.zeros((1, 1, 1, 1))
        if timestamps is None:
            block.timestamps = None
        else:
            block.timestamps = np.array(timestamps, dtype=object)
        block.observables = {} if observables is None else observables
        return block

    return _make_block


def _vis2_obs(wavelengths=WL, **extra):
    """Observables with vis2 = 1..n per channel and unit errors."""
    n_wav = len(wavelengths)
    channel = np.arange(1.0, n_wav + 1.0)
    arrays = {
        "vis2": _arr(N_BL, channel),
        "vis2_err": _arr(N_BL, np.ones(n_wav)),
    }
    arrays.update(extra)
    return _make_obs(wavelengths, **arrays)


class TestRegistry:
    """Tests for the observable registry."""

    def test_all_keys_present(self):
        """All five observables are registered."""
        assert set(_SPECS) == {"t3phi", "t3amp", "vis2", "visamp", "visphi"}

    def test_attributes_exist_on_observables(self):
        """Registered attribute names exist on Observables."""
        obs = _make_obs(WL)
        for spec in _SPECS.values():
            for attr in (
                spec.value_attr,
                spec.error_attr,
                spec.flag_attr,
                spec.geometry,
            ):
                assert hasattr(obs, attr)

    def test_consistent_with_calibrate_map(self):
        """Entries overlapping calibrate._OBS_ATTR_MAP agree."""
        for key, attrs in _OBS_ATTR_MAP.items():
            spec = _SPECS[key]
            assert spec.value_attr == attrs["value"]
            assert spec.error_attr == attrs["error"]
            assert spec.flag_attr == attrs["flag"]

    def test_phase_and_reference(self):
        """Phases reference 0, amplitudes reference 1."""
        phases = {k for k, s in _SPECS.items() if s.is_phase}
        assert phases == {"t3phi", "visphi"}
        for spec in _SPECS.values():
            assert spec.reference == (0.0 if spec.is_phase else 1.0)


class TestBandAverage:
    """Tests for _nan_mean_rows and _band_average."""

    def test_arithmetic_mean(self):
        """Arithmetic mean ignores NaN."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
        np.testing.assert_allclose(_band_average(a, False), [2.0, 5.0])

    def test_nan_mean_rows_ignores_inf(self):
        """Infinite values are excluded from sum and count."""
        a = np.array([[1.0, np.inf, 3.0]])
        np.testing.assert_allclose(_nan_mean_rows(a), [2.0])

    def test_circular_mean_simple(self):
        """Circular mean of nearby phases equals arithmetic mean."""
        a = np.array([[10.0, 20.0]])
        np.testing.assert_allclose(_band_average(a, True), [15.0])

    def test_circular_mean_wraps(self):
        """Circular mean of 179 and -179 has magnitude 180."""
        a = np.array([[179.0, -179.0]])
        result = _band_average(a, True)
        assert abs(result[0]) == pytest.approx(180.0)
        assert -180.0 < result[0] <= 180.0

    @pytest.mark.parametrize("is_phase", [True, False])
    def test_all_nan_row_no_warning(self, is_phase):
        """All-NaN rows yield NaN without any warning."""
        a = np.array([[np.nan, np.nan], [1.0, np.nan]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _band_average(a, is_phase)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(1.0)


class TestSafeBlockTime:
    """Tests for _safe_block_time."""

    def test_valid_timestamps(self, make_block):
        """Valid timestamps give the mean time in seconds."""
        block = make_block(timestamps=["08:00:00.000", "08:10:00.000"])
        assert _safe_block_time(block) == pytest.approx(29100.0)

    @pytest.mark.parametrize(
        "timestamps",
        [[""], ["08"], None],
        ids=["empty", "malformed", "none"],
    )
    def test_invalid_timestamps_give_nan(self, make_block, timestamps):
        """Empty, malformed or missing timestamps give NaN."""
        block = make_block(timestamps=timestamps)
        with pytest.warns(UserWarning, match="Could not determine"):
            result = _safe_block_time(block)
        assert np.isnan(result)


class TestComputeSeries:
    """Tests for compute_band_averaged_series."""

    @pytest.mark.parametrize("key", sorted(_SPECS))
    def test_each_observable(self, make_block, key):
        """Every registry observable is averaged on its geometry."""
        spec = _SPECS[key]
        n_items = N_TRI if spec.geometry == "triangles" else N_BL
        obs = _make_obs(
            WL,
            **{
                spec.value_attr: _arr(n_items, [1.0, 2.0, 3.0, 4.0]),
                spec.error_attr: _arr(n_items, [0.1, 0.2, 0.3, 0.4]),
            },
        )
        block = make_block(observables={"raw": obs})
        series = compute_band_averaged_series([block], observable=key)
        expected_names = tuple(
            item.name for item in getattr(obs, spec.geometry)
        )
        assert isinstance(series, BandAveragedSeries)
        assert series.observable == key
        assert series.label == "raw"
        assert series.names == expected_names
        assert series.values.shape == (1, n_items)
        np.testing.assert_allclose(series.values, 2.5, rtol=1e-9)
        np.testing.assert_allclose(series.errors, 0.25)
        np.testing.assert_array_equal(series.n_wav_used, [4])
        assert series.wl_range == (3.0, 4.5)
        assert series.block_types == ("CAL",)
        assert series.targets == ("cal",)

    def test_names_as_stored(self, make_block):
        """Triangle and baseline names are used exactly as stored."""
        obs = _make_obs(
            WL,
            t3phi=_arr(N_TRI, np.zeros(4)),
            vis2=_arr(N_BL, np.ones(4)),
        )
        block = make_block(observables={"raw": obs})
        tri = compute_band_averaged_series([block], observable="t3phi")
        bl = compute_band_averaged_series([block], observable="vis2")
        assert tri.names == ("H1-H2-H3", "H1-H2-H4")
        assert bl.names == ("H1H2", "H1H3", "H1H4")

    def test_flags_masked_by_default(self, make_block):
        """Flagged samples are excluded from values and errors."""
        flag = np.zeros((N_BL, 4), dtype=bool)
        flag[:, 3] = True
        obs = _make_obs(
            WL,
            vis2=_arr(N_BL, [1.0, 1.0, 1.0, 5.0]),
            vis2_err=_arr(N_BL, [1.0, 1.0, 1.0, 9.0]),
            vis2_flag=flag,
        )
        block = make_block(observables={"raw": obs})
        series = compute_band_averaged_series([block], observable="vis2")
        np.testing.assert_allclose(series.values, 1.0)
        np.testing.assert_allclose(series.errors, 1.0)

    def test_flags_ignored_when_disabled(self, make_block):
        """use_flags=False keeps flagged samples."""
        flag = np.zeros((N_BL, 4), dtype=bool)
        flag[:, 3] = True
        obs = _make_obs(
            WL,
            vis2=_arr(N_BL, [1.0, 1.0, 1.0, 5.0]),
            vis2_err=_arr(N_BL, [1.0, 1.0, 1.0, 9.0]),
            vis2_flag=flag,
        )
        block = make_block(observables={"raw": obs})
        series = compute_band_averaged_series(
            [block], observable="vis2", use_flags=False
        )
        np.testing.assert_allclose(series.values, 2.0)
        np.testing.assert_allclose(series.errors, 3.0)

    def test_flag_none_uses_all_samples(self, make_block):
        """A missing flag array means all samples are used."""
        block = make_block(observables={"raw": _vis2_obs()})
        series = compute_band_averaged_series([block], observable="vis2")
        np.testing.assert_allclose(series.values, 2.5)

    def test_flagging_does_not_modify_input(self, make_block):
        """Flag masking works on copies of the stored arrays."""
        flag = np.ones((N_BL, 4), dtype=bool)
        obs = _vis2_obs(vis2_flag=flag)
        original = obs.vis2.copy()
        original_err = obs.vis2_err.copy()
        block = make_block(observables={"raw": obs})
        compute_band_averaged_series([block], observable="vis2")
        np.testing.assert_array_equal(obs.vis2, original)
        np.testing.assert_array_equal(obs.vis2_err, original_err)

    @pytest.mark.parametrize("key", ["vis2", "t3phi"])
    def test_all_flagged_no_runtime_warning(self, make_block, key):
        """A fully flagged band gives NaN without any warning."""
        spec = _SPECS[key]
        n_items = N_TRI if spec.geometry == "triangles" else N_BL
        obs = _make_obs(
            WL,
            **{
                spec.value_attr: _arr(n_items, np.ones(4)),
                spec.error_attr: _arr(n_items, np.ones(4)),
                spec.flag_attr: np.ones((n_items, 4), dtype=bool),
            },
        )
        block = make_block(observables={"raw": obs})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            series = compute_band_averaged_series([block], observable=key)
        assert np.all(np.isnan(series.values))
        assert np.all(np.isnan(series.errors))

    def test_missing_error_gives_nan(self, make_block):
        """A missing error array gives NaN errors."""
        flag = np.zeros((N_BL, 4), dtype=bool)
        flag[:, 0] = True
        obs = _make_obs(
            WL,
            vis2=_arr(N_BL, [9.0, 2.0, 2.0, 2.0]),
            vis2_flag=flag,
        )
        block = make_block(observables={"raw": obs})
        series = compute_band_averaged_series([block], observable="vis2")
        np.testing.assert_allclose(series.values, 2.0)
        assert np.all(np.isnan(series.errors))

    def test_label_missing_on_some_blocks(self, make_block):
        """Missing label gives a NaN row and a combined warning."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(block_type="SCI", target="sci")
        with pytest.warns(
            UserWarning,
            match=r"\[1\] 'sci' \(SCI\): label/observable missing",
        ):
            series = compute_band_averaged_series([b0, b1], observable="vis2")
        assert series.values.shape == (2, N_BL)
        np.testing.assert_allclose(series.values[0], 2.5)
        assert np.all(np.isnan(series.values[1]))
        assert np.all(np.isnan(series.errors[1]))
        np.testing.assert_array_equal(series.n_wav_used, [4, 0])

    def test_label_missing_on_all_blocks(self, make_block):
        """Missing label on all blocks raises ValueError."""
        b0 = make_block()
        b1 = make_block(observables={"other": _vis2_obs()})
        with pytest.raises(ValueError, match="missing on 2 block"):
            compute_band_averaged_series([b0, b1], observable="vis2")

    def test_observable_none_treated_as_missing(self, make_block):
        """An observable stored as None counts as missing."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(observables={"raw": _make_obs(WL)})
        with pytest.warns(UserWarning, match="label/observable missing"):
            series = compute_band_averaged_series([b0, b1], observable="vis2")
        assert np.all(np.isnan(series.values[1]))

    def test_empty_band_on_one_block(self, make_block):
        """Each block uses its own grid; empty band gives NaN row."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(
            observables={"raw": _vis2_obs(np.array([2.0, 2.5]))},
        )
        with pytest.warns(UserWarning, match="no channels in band"):
            series = compute_band_averaged_series(
                [b0, b1], observable="vis2", wl_min=3.2, wl_max=4.2
            )
        np.testing.assert_allclose(series.values[0], 2.5)
        assert np.all(np.isnan(series.values[1]))
        np.testing.assert_array_equal(series.n_wav_used, [2, 0])
        assert series.wl_range == (3.5, 4.0)

    def test_empty_band_on_all_blocks(self, make_block):
        """Empty band on all blocks raises ValueError."""
        block = make_block(observables={"raw": _vis2_obs()})
        with pytest.raises(ValueError, match="no channels in band"):
            compute_band_averaged_series(
                [block], observable="vis2", wl_min=10.0
            )

    def test_mixed_unusable_reasons_in_message(self, make_block):
        """Error message lists both missing data and empty band."""
        b0 = make_block()
        b1 = make_block(observables={"raw": _vis2_obs()})
        with pytest.raises(ValueError, match="missing.*no channels"):
            compute_band_averaged_series(
                [b0, b1], observable="vis2", wl_max=1.0
            )

    def test_full_band_by_default(self, make_block, sample_wavelengths_short):
        """No band limits means all channels are used."""
        obs = _vis2_obs(sample_wavelengths_short)
        block = make_block(observables={"raw": obs})
        series = compute_band_averaged_series([block], observable="vis2")
        np.testing.assert_array_equal(series.n_wav_used, [2])
        assert series.wl_range == (2.7, 4.3)
        np.testing.assert_allclose(series.values, 1.5)

    def test_wl_range_across_blocks(self, make_block):
        """wl_range spans the channels used across all blocks."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(
            observables={"raw": _vis2_obs(np.array([3.25, 4.75]))},
        )
        series = compute_band_averaged_series([b0, b1], observable="vis2")
        np.testing.assert_array_equal(series.n_wav_used, [4, 2])
        assert series.wl_range == (3.0, 4.75)

    def test_unknown_observable(self, make_block):
        """Unknown observable raises ValueError listing keys."""
        block = make_block(observables={"raw": _vis2_obs()})
        with pytest.raises(ValueError, match="Unknown observable.*vis2"):
            compute_band_averaged_series([block], observable="bogus")

    def test_unloaded_block_skipped(self, make_block):
        """Unloaded blocks are skipped with one warning."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(block_type="SCI", target="sci")
        b1.cubes = None
        with pytest.warns(UserWarning, match=r"1 unloaded.*\[1\] 'sci'"):
            series = compute_band_averaged_series([b0, b1], observable="vis2")
        assert series.values.shape == (1, N_BL)
        assert series.targets == ("cal",)

    def test_no_loaded_blocks(self, make_block):
        """No loaded blocks raises ValueError."""
        block = make_block()
        block.cubes = None
        with (
            pytest.warns(UserWarning, match="unloaded"),
            pytest.raises(ValueError, match="No loaded blocks"),
        ):
            compute_band_averaged_series([block], observable="vis2")

    def test_sequence_input(self, make_block):
        """An ObservingSequence is accepted as input."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(
            block_type="SCI",
            target="sci",
            observables={"raw": _vis2_obs()},
        )
        seq = ObservingSequence(blocks=[b0, b1])
        series = compute_band_averaged_series(seq, observable="vis2")
        assert series.block_types == ("CAL", "SCI")
        assert series.targets == ("cal", "sci")

    def test_times_unwrapped_across_midnight(self, make_block):
        """Block times are unwrapped across midnight."""
        b0 = make_block(
            timestamps=["23:50:00.000"],
            observables={"raw": _vis2_obs()},
        )
        b1 = make_block(
            timestamps=["00:10:00.000"],
            observables={"raw": _vis2_obs()},
        )
        series = compute_band_averaged_series([b0, b1], observable="vis2")
        np.testing.assert_allclose(series.block_times_sec, [85800.0, 87000.0])

    def test_empty_time_obs_gives_nan(self, make_block):
        """Empty TIME-OBS gives NaN time and a warning."""
        b0 = make_block(timestamps=[""], observables={"raw": _vis2_obs()})
        b1 = make_block(observables={"raw": _vis2_obs()})
        with pytest.warns(UserWarning, match="Could not determine"):
            series = compute_band_averaged_series([b0, b1], observable="vis2")
        assert np.isnan(series.block_times_sec[0])
        assert series.block_times_sec[1] == pytest.approx(28800.0)
