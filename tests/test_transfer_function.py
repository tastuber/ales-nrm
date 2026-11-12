"""Tests for band-averaged series and transfer-function plots."""

import warnings
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox

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
    _build_time_axis,
    _default_figsize,
    _format_hhmm,
    _grid_shape,
    _legend_center_x,
    _nan_mean_rows,
    _resolve_select,
    _safe_block_time,
    _sci_runs,
    _shade_sci_runs,
    compute_band_averaged_series,
    plot_transfer_function,
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


NAMES = tuple(f"N{k}" for k in range(6))
DEFAULT_TYPES = ("CAL", "SCI", "SCI", "CAL", "SCI", "CAL")


def _make_series(
    block_types=DEFAULT_TYPES,
    n_items=6,
    observable="t3phi",
    times=None,
    values=None,
):
    """Build a BandAveragedSeries directly for plot tests."""
    n_blocks = len(block_types)
    if values is None:
        values = np.linspace(-5.0, 5.0, n_blocks * n_items)
        values = values.reshape(n_blocks, n_items)
    values = np.asarray(values, dtype=float)
    n_items = values.shape[1]
    if times is None:
        times = 28800.0 + 600.0 * np.arange(n_blocks)
    return BandAveragedSeries(
        observable=observable,
        label="raw",
        values=values,
        errors=np.full(values.shape, 0.5),
        names=tuple(f"N{k}" for k in range(n_items)),
        block_types=tuple(block_types),
        targets=tuple("cal" if bt == "CAL" else "sci" for bt in block_types),
        block_times_sec=np.asarray(times, dtype=float),
        n_wav_used=np.full(n_blocks, 4),
        wl_range=(3.0, 4.5),
    )


def _legend_labels(fig):
    """Return the legend entry texts of a figure."""
    if not fig.legends:
        return []
    return [t.get_text() for t in fig.legends[0].get_texts()]


def _anchored(ax):
    """Return the AnchoredText artists of an axes."""
    return [a for a in ax.get_children() if isinstance(a, AnchoredText)]


def _name_corners(fig, axes):
    """Return the corner of each panel where its name is drawn.

    Corners are ``"upper left"``, ``"upper right"``, ``"lower left"``
    or ``"lower right"``, from the centre of the name box relative to
    the centre of the panel.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    corners = []
    for ax in axes:
        box = _anchored(ax)[0].get_window_extent(renderer)
        frame = ax.get_window_extent(renderer)
        upper = box.y0 + box.y1 > frame.y0 + frame.y1
        left = box.x0 + box.x1 < frame.x0 + frame.x1
        corners.append(
            f"{'upper' if upper else 'lower'} {'left' if left else 'right'}"
        )
    return corners


def _tf_lines(ax):
    """Return the transfer-function lines of an axes."""
    return [
        ln for ln in ax.get_lines() if ln.get_label() == "transfer function"
    ]


def _containers(ax):
    """Return errorbar containers keyed by label."""
    return {c.get_label(): c for c in ax.containers}


def _span_x(patch):
    """Return (xmin, xmax) of an axvspan patch in data units."""
    verts = patch.get_patch_transform().transform(patch.get_path().vertices)
    return float(verts[:, 0].min()), float(verts[:, 0].max())


class _MatplotlibAgg:
    """Mixin selecting the Agg backend and closing all figures."""

    @pytest.fixture(autouse=True)
    def _agg_backend(self):
        """Use the Agg backend and close all figures afterwards."""
        matplotlib.use("Agg")
        yield
        plt.close("all")


class TestResolveSelect:
    """Tests for _resolve_select."""

    def test_none_selects_all(self):
        """None selects all items in order."""
        assert _resolve_select(None, NAMES) == list(range(6))

    @pytest.mark.parametrize(
        ("select", "expected"),
        [
            ([2, 0], [2, 0]),
            (["N2", "N0"], [2, 0]),
            ([1, "N3"], [1, 3]),
            ([np.int64(4)], [4]),
            ("N1", [1]),
        ],
        ids=["indices", "names", "mixed", "numpy_int", "bare_string"],
    )
    def test_valid(self, select, expected):
        """Indices and names resolve in the given order."""
        assert _resolve_select(select, NAMES) == expected

    @pytest.mark.parametrize(
        "select",
        [[6], [-1], [True], ["bogus"], [0, 0], [0, "N0"], [], [1.0]],
        ids=[
            "out_of_range",
            "negative",
            "bool",
            "unknown_name",
            "duplicate",
            "duplicate_mixed",
            "empty",
            "float",
        ],
    )
    def test_invalid_raises(self, select):
        """Invalid selections raise ValueError."""
        with pytest.raises(ValueError):
            _resolve_select(select, NAMES)


class TestGridShape:
    """Tests for _grid_shape and _default_figsize."""

    @pytest.mark.parametrize(
        ("n_panels", "expected"),
        [
            (1, (1, 1)),
            (4, (1, 4)),
            (5, (2, 3)),
            (6, (2, 3)),
            (7, (2, 4)),
            (15, (5, 3)),
            (20, (5, 4)),
        ],
    )
    def test_automatic(self, n_panels, expected):
        """Automatic grid follows the 4-or-3 column rule."""
        assert _grid_shape(n_panels, None) == expected

    @pytest.mark.parametrize(
        ("n_panels", "n_cols", "expected"),
        [(6, 2, (3, 2)), (3, 4, (1, 4)), (5, 1, (5, 1))],
    )
    def test_explicit_n_cols(self, n_panels, n_cols, expected):
        """Explicit n_cols gives rows by ceiling division."""
        assert _grid_shape(n_panels, n_cols) == expected

    def test_n_cols_zero_raises(self):
        """n_cols < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_cols"):
            _grid_shape(4, 0)

    @pytest.mark.parametrize(
        ("rows", "cols", "expected"),
        [(1, 1, (4.0, 3.0)), (5, 4, (8.0, 8.5))],
    )
    def test_default_figsize(self, rows, cols, expected):
        """Default figure size scales with the grid."""
        assert _default_figsize(rows, cols) == pytest.approx(expected)


class TestTimeAxis:
    """Tests for _build_time_axis and _format_hhmm."""

    TIMES = np.array([28800.0, 29400.0, 30600.0])

    def test_elapsed(self):
        """Elapsed mode gives minutes since the first block."""
        ta = _build_time_axis(self.TIMES, "elapsed")
        np.testing.assert_allclose(ta.x, [0.0, 10.0, 30.0])
        assert ta.label == "elapsed time (min)"
        assert ta.formatter is None
        assert ta.locator is None
        assert ta.ticklabels is None

    def test_ut(self):
        """UT mode gives hours and an HH:MM formatter."""
        ta = _build_time_axis(self.TIMES, "ut")
        np.testing.assert_allclose(ta.x, self.TIMES / 3600.0)
        assert ta.label == "UT (hh:mm)"
        assert isinstance(ta.formatter, FuncFormatter)
        assert ta.formatter(8.5, 0) == "08:30"
        assert ta.formatter(24.25, 0) == "00:15"

    def test_index(self):
        """Index mode gives block indices and an integer locator."""
        ta = _build_time_axis(self.TIMES, "index")
        np.testing.assert_allclose(ta.x, [0.0, 1.0, 2.0])
        assert ta.label == "block index"
        assert isinstance(ta.locator, MaxNLocator)

    def test_index_ut(self):
        """index_ut gives one HH:MM label per block, '--' for NaN."""
        times = np.array([28800.0, np.nan, 87000.0])
        ta = _build_time_axis(times, "index_ut")
        np.testing.assert_allclose(ta.x, [0.0, 1.0, 2.0])
        assert ta.label == "UT (hh:mm)"
        assert ta.ticklabels == ("08:00", "--", "00:10")

    @pytest.mark.parametrize("mode", ["elapsed", "ut"])
    def test_nan_raises(self, mode):
        """NaN times raise ValueError suggesting an index mode."""
        times = np.array([28800.0, np.nan])
        with pytest.raises(ValueError, match="index"):
            _build_time_axis(times, mode)

    def test_unknown_mode(self):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown time_mode"):
            _build_time_axis(self.TIMES, "bogus")

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00"),
            (29.9, "00:00"),
            (30.1, "00:01"),
            (45000.0, "12:30"),
            (86385.0, "00:00"),
            (np.nan, "--"),
        ],
    )
    def test_format_hhmm(self, seconds, expected):
        """Seconds are rounded to minutes, modulo 24 h."""
        assert _format_hhmm(seconds) == expected


class TestSciRuns(_MatplotlibAgg):
    """Tests for _sci_runs and _shade_sci_runs."""

    @pytest.mark.parametrize(
        ("types", "expected"),
        [
            (("CAL", "SCI", "SCI", "CAL"), [(1, 2)]),
            (("SCI", "CAL", "SCI"), [(0, 0), (2, 2)]),
            (("SCI", "SCI"), [(0, 1)]),
            (("CAL", "CAL"), []),
        ],
        ids=["cal_sci_sci_cal", "sci_edges", "all_sci", "no_sci"],
    )
    def test_runs(self, types, expected):
        """Contiguous SCI runs are found, including at the edges."""
        assert _sci_runs(types) == expected

    def test_shading_count_and_colors(self):
        """One span per run with alternating colors."""
        _, ax = plt.subplots()
        x = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        _shade_sci_runs(ax, x, [(0, 0), (2, 2), (4, 4)])
        assert len(ax.patches) == 3
        shades = ["#f0f0f0", "#e4e4e4", "#f0f0f0"]
        for patch, shade in zip(ax.patches, shades, strict=True):
            np.testing.assert_allclose(patch.get_facecolor(), to_rgba(shade))
            assert patch.get_zorder() == 0

    def test_padding_uses_median_spacing(self):
        """Padding is 0.4 times the median spacing."""
        _, ax = plt.subplots()
        x = np.array([0.0, 10.0, 20.0, 40.0])
        _shade_sci_runs(ax, x, [(1, 2)])
        assert _span_x(ax.patches[0]) == pytest.approx((6.0, 24.0))

    @pytest.mark.parametrize(
        "x",
        [np.array([5.0]), np.array([5.0, 5.0])],
        ids=["single_point", "zero_spacing"],
    )
    def test_padding_fallback(self, x):
        """Without positive spacing the padding is 0.5."""
        _, ax = plt.subplots()
        _shade_sci_runs(ax, x, [(0, len(x) - 1)])
        assert _span_x(ax.patches[0]) == pytest.approx((4.5, 5.5))


class TestPlot(_MatplotlibAgg):
    """Tests for plot_transfer_function."""

    @pytest.mark.parametrize(
        ("select", "n_visible", "n_hidden"),
        [(None, 6, 0), ([0, 1, 2, 3, 4], 5, 1)],
    )
    def test_visible_and_hidden_axes(self, select, n_visible, n_hidden):
        """Unused grid cells are hidden."""
        fig = plot_transfer_function(_make_series(), select=select)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == n_visible
        assert len(fig.axes) - len(visible) == n_hidden

    def test_panel_order_and_names(self):
        """Panels follow the selection order and show names."""
        fig = plot_transfer_function(_make_series(), select=["N3", 1])
        texts = [_anchored(ax)[0].txt.get_text() for ax in fig.axes[:2]]
        assert texts == ["N3", "N1"]

    @pytest.mark.parametrize(
        ("plot_cal", "plot_sci", "expected"),
        [
            (True, True, ["CAL", "transfer function", "SCI"]),
            (True, False, ["CAL", "transfer function"]),
            (False, True, ["SCI"]),
        ],
    )
    def test_legend_labels(self, plot_cal, plot_sci, expected):
        """Legend order is CAL, transfer function, SCI."""
        fig = plot_transfer_function(
            _make_series(), plot_cal=plot_cal, plot_sci=plot_sci
        )
        assert _legend_labels(fig) == expected

    @pytest.mark.parametrize(
        ("types", "expected"),
        [
            (("CAL", "CAL"), ["CAL", "transfer function"]),
            (("SCI", "SCI"), ["SCI"]),
        ],
    )
    def test_legend_omits_absent_types(self, types, expected):
        """Absent block types are not listed in the legend."""
        fig = plot_transfer_function(_make_series(block_types=types))
        assert _legend_labels(fig) == expected

    def test_no_legend_without_points(self):
        """No legend is drawn if nothing is plotted."""
        fig = plot_transfer_function(
            _make_series(block_types=("SCI", "SCI")), plot_sci=False
        )
        assert fig.legends == []

    def test_both_disabled_raises(self):
        """plot_cal=False and plot_sci=False raises ValueError."""
        with pytest.raises(ValueError, match="plot_cal"):
            plot_transfer_function(
                _make_series(), plot_cal=False, plot_sci=False
            )
        assert plt.get_fignums() == []

    @pytest.mark.parametrize("errorbars", [True, False])
    def test_errorbars(self, errorbars):
        """Error bars are drawn only when requested."""
        fig = plot_transfer_function(_make_series(), errorbars=errorbars)
        for container in _containers(fig.axes[0]).values():
            assert container.has_yerr is errorbars

    def test_transfer_function_through_finite_points(self):
        """The CAL line skips NaN values; no line if none finite."""
        nan = np.nan
        values = np.array([[nan, nan], [1.0, 1.0], [2.0, nan], [3.0, nan]])
        series = _make_series(
            block_types=("CAL", "SCI", "CAL", "CAL"), values=values
        )
        fig = plot_transfer_function(series)
        lines = _tf_lines(fig.axes[0])
        assert len(lines) == 1
        np.testing.assert_allclose(lines[0].get_ydata(), [2.0, 3.0])
        assert _tf_lines(fig.axes[1]) == []

    def test_legend_handles_from_any_panel(self):
        """The legend lists the line even if panel one lacks it."""
        nan = np.nan
        values = np.array([[nan, nan], [1.0, 1.0], [2.0, nan], [3.0, nan]])
        series = _make_series(
            block_types=("CAL", "SCI", "CAL", "CAL"), values=values
        )
        fig = plot_transfer_function(series, select=[1, 0])
        assert "transfer function" in _legend_labels(fig)

    @pytest.mark.parametrize(
        ("observable", "kwargs", "expected"),
        [
            ("t3phi", {}, 0.0),
            ("vis2", {}, 1.0),
            ("t3phi", {"reference_value": 2.5}, 2.5),
            ("t3phi", {"show_reference": False}, None),
        ],
        ids=["phase_default", "amp_default", "override", "hidden"],
    )
    def test_reference_line(self, observable, kwargs, expected):
        """Reference line uses the registry default or override."""
        series = _make_series(observable=observable)
        fig = plot_transfer_function(series, **kwargs)
        ref = [
            ln for ln in fig.axes[0].get_lines() if ln.get_linestyle() == ":"
        ]
        if expected is None:
            assert ref == []
        else:
            assert len(ref) == 1
            np.testing.assert_allclose(ref[0].get_ydata(), expected)

    def test_ylim_shared(self):
        """Ylim is applied to all panels."""
        fig = plot_transfer_function(_make_series(), ylim=(-18.0, 18.0))
        for ax in fig.axes:
            assert ax.get_ylim() == pytest.approx((-18.0, 18.0))

    def test_color_merge(self):
        """A partial colors dict is merged into the defaults."""
        fig = plot_transfer_function(_make_series(), colors={"SCI": "red"})
        ax = fig.axes[0]
        containers = _containers(ax)
        sci_color = containers["SCI"].lines[0].get_color()
        cal_color = containers["CAL"].lines[0].get_color()
        assert to_rgba(sci_color) == to_rgba("red")
        assert to_rgba(cal_color) == to_rgba("tab:blue")
        tf_color = _tf_lines(ax)[0].get_color()
        assert to_rgba(tf_color) == to_rgba("tab:blue")

    def test_name_loc_string(self):
        """A string name_loc applies to all panels."""
        fig = plot_transfer_function(_make_series(), name_loc="lower right")
        assert set(_name_corners(fig, fig.axes)) == {"lower right"}

    def test_name_loc_default(self):
        """Panel names default to the upper left corner."""
        fig = plot_transfer_function(_make_series())
        assert _name_corners(fig, fig.axes[:1]) == ["upper left"]

    def test_name_loc_dict(self):
        """Dict keys by index and name; others use the default."""
        fig = plot_transfer_function(
            _make_series(),
            name_loc={0: "lower right", "N2": "lower left"},
        )
        assert _name_corners(fig, fig.axes[:3]) == [
            "lower right",
            "upper left",
            "lower left",
        ]

    def test_name_loc_unplotted_key_ignored(self):
        """Valid keys for items that are not plotted are ignored."""
        fig = plot_transfer_function(
            _make_series(), select=[0, 1], name_loc={5: "lower right"}
        )
        assert _name_corners(fig, fig.axes[:2]) == ["upper left"] * 2

    def test_name_loc_unknown_key_raises(self):
        """Unknown name_loc keys raise ValueError."""
        with pytest.raises(ValueError, match="Unknown name"):
            plot_transfer_function(
                _make_series(), name_loc={"bogus": "lower right"}
            )

    def test_name_loc_invalid_location_raises(self):
        """Invalid location strings raise matplotlib's error."""
        with pytest.raises(ValueError):
            plot_transfer_function(_make_series(), name_loc="nowhere")

    def test_default_labels_and_no_title(self):
        """Default labels come from the mode and registry."""
        fig = plot_transfer_function(_make_series())
        assert fig.axes[0].get_ylabel() == "closure phase (deg)"
        assert fig.axes[3].get_xlabel() == "elapsed time (min)"
        assert fig.texts == []

    def test_label_and_title_overrides(self):
        """xlabel, ylabel and title can be overridden."""
        fig = plot_transfer_function(
            _make_series(), xlabel="t", ylabel="cp", title="My title"
        )
        assert fig.axes[0].get_ylabel() == "cp"
        assert fig.axes[3].get_xlabel() == "t"
        assert [t.get_text() for t in fig.texts] == ["My title"]

    def test_labels_on_outer_panels(self):
        """x-label on lowest visible panel, y-label first column."""
        fig = plot_transfer_function(_make_series(), select=[0, 1, 2, 3, 4])
        fig.canvas.draw()
        axes = fig.axes
        assert [ax.get_xlabel() != "" for ax in axes[:5]] == [
            False,
            False,
            True,
            True,
            True,
        ]
        assert [ax.get_ylabel() != "" for ax in axes[:5]] == [
            True,
            False,
            False,
            True,
            False,
        ]
        tick = axes[2].xaxis.get_major_ticks()[0]
        assert tick.label1.get_visible()

    def test_empty_column_skipped(self):
        """Columns without any panel get no x-label."""
        fig = plot_transfer_function(
            _make_series(), select=[0, 1, 2], n_cols=4
        )
        assert len(fig.axes) == 4
        assert not fig.axes[3].get_visible()
        assert fig.axes[3].get_xlabel() == ""
        assert fig.axes[2].get_xlabel() == "elapsed time (min)"

    def test_default_figsize(self):
        """Default figure size follows the grid."""
        fig = plot_transfer_function(_make_series())
        expected = _default_figsize(2, 3)
        assert tuple(fig.get_size_inches()) == pytest.approx(expected)

    def test_explicit_figsize(self):
        """An explicit figsize is used."""
        fig = plot_transfer_function(_make_series(), figsize=(6.0, 7.0))
        assert tuple(fig.get_size_inches()) == pytest.approx((6.0, 7.0))

    def test_rc_context_font_size(self):
        """Tick labels follow font.size from rc_context."""
        with matplotlib.rc_context({"font.size": 6}):
            fig = plot_transfer_function(_make_series())
            fig.canvas.draw()
        xlabels = fig.axes[3].get_xticklabels()
        ylabels = fig.axes[0].get_yticklabels()
        assert xlabels and ylabels
        for label in [*xlabels, *ylabels]:
            assert label.get_fontsize() == pytest.approx(6.0)

    def test_outer_edge_ticks_suppressed(self):
        """Right ticks off on last column, top ticks off on top row."""
        params = {"ytick.right": True, "xtick.top": True}
        with matplotlib.rc_context(params):
            fig = plot_transfer_function(_make_series())
            fig.canvas.draw()
        axes = fig.axes

        def right(ax):
            return ax.yaxis.get_major_ticks()[0].tick2line.get_visible()

        def top(ax):
            return ax.xaxis.get_major_ticks()[0].tick2line.get_visible()

        assert not right(axes[2])
        assert right(axes[0])
        assert not top(axes[0])
        assert top(axes[3])

    @pytest.mark.parametrize(
        ("plot_cal", "plot_sci", "n_patches"),
        [(True, True, 2), (False, True, 0), (True, False, 0)],
    )
    def test_shading_only_with_both(self, plot_cal, plot_sci, n_patches):
        """SCI runs are shaded only when both types are plotted."""
        fig = plot_transfer_function(
            _make_series(), plot_cal=plot_cal, plot_sci=plot_sci
        )
        assert len(fig.axes[0].patches) == n_patches

    def test_time_mode_ut(self):
        """UT mode installs an HH:MM tick formatter."""
        fig = plot_transfer_function(_make_series(), time_mode="ut")
        formatter = fig.axes[0].xaxis.get_major_formatter()
        assert isinstance(formatter, FuncFormatter)
        assert fig.axes[3].get_xlabel() == "UT (hh:mm)"

    def test_time_mode_index(self):
        """Index mode uses an integer tick locator."""
        fig = plot_transfer_function(_make_series(), time_mode="index")
        locator = fig.axes[0].xaxis.get_major_locator()
        assert isinstance(locator, MaxNLocator)
        assert fig.axes[3].get_xlabel() == "block index"

    def test_time_mode_index_ut(self):
        """index_ut shows rotated HH:MM labels, '--' for NaN."""
        times = 28800.0 + 600.0 * np.arange(6)
        times[1] = np.nan
        fig = plot_transfer_function(
            _make_series(times=times), time_mode="index_ut"
        )
        fig.canvas.draw()
        labels = fig.axes[3].get_xticklabels()
        texts = [lab.get_text() for lab in labels]
        assert texts == ["08:00", "--", "08:20", "08:30", "08:40", "08:50"]
        for lab in labels:
            assert lab.get_rotation() == pytest.approx(45.0)
            assert lab.get_horizontalalignment() == "right"

    def test_nan_time_elapsed_raises_without_figure(self):
        """NaN times in elapsed mode raise before creating a figure."""
        times = np.array([28800.0, np.nan, 30000.0])
        series = _make_series(block_types=("CAL", "SCI", "CAL"), times=times)
        with pytest.raises(ValueError, match="index"):
            plot_transfer_function(series)
        assert plt.get_fignums() == []

    def test_from_computed_series(self, make_block):
        """A computed series plots one panel per baseline."""
        b0 = make_block(observables={"raw": _vis2_obs()})
        b1 = make_block(
            block_type="SCI",
            target="sci",
            timestamps=("08:10:00.000",),
            observables={"raw": _vis2_obs()},
        )
        series = compute_band_averaged_series([b0, b1], observable="vis2")
        fig = plot_transfer_function(series)
        assert len(fig.axes) == N_BL
        assert fig.axes[0].get_ylabel() == "squared visibility"
        texts = [_anchored(ax)[0].txt.get_text() for ax in fig.axes]
        assert texts == ["H1H2", "H1H3", "H1H4"]
        assert _legend_labels(fig) == ["CAL", "transfer function", "SCI"]

    def test_panels_flush(self):
        """Neighbouring panels share their edges (no spacing)."""
        fig = plot_transfer_function(_make_series())
        pos = [ax.get_position() for ax in fig.axes]
        assert pos[0].x1 == pytest.approx(pos[1].x0)
        assert pos[1].x1 == pytest.approx(pos[2].x0)
        assert pos[0].y0 == pytest.approx(pos[3].y1)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"select": [0]},
            {"title": "My title"},
            {"time_mode": "index_ut"},
        ],
        ids=["full", "single", "title", "index_ut"],
    )
    def test_no_overlap_and_inside_figure(self, kwargs):
        """Legend and title sit above the panels, inside the figure."""
        fig = plot_transfer_function(_make_series(), **kwargs)
        fig.draw_without_rendering()
        frame = fig.bbox
        visible = [ax for ax in fig.axes if ax.get_visible()]
        decos = [ax.get_tightbbox() for ax in visible]
        legend = fig.legends[0].get_window_extent()
        assert legend.y0 >= max(d.y1 for d in decos) - 1.0
        assert legend.y1 <= frame.y1 + 1.0
        assert legend.x0 >= frame.x0 - 1.0
        assert legend.x1 <= frame.x1 + 1.0
        for deco in decos:
            assert deco.x0 >= frame.x0 - 1.0
            assert deco.x1 <= frame.x1 + 1.0
            assert deco.y0 >= frame.y0 - 1.0
        if "title" in kwargs:
            title = fig.texts[0].get_window_extent()
            assert title.y0 >= legend.y1 - 1.0
            assert title.y1 <= frame.y1 + 1.0

    def test_ignores_autolayout_rcparam(self):
        """A user's autolayout setting does not alter the layout."""
        with matplotlib.rc_context({"figure.autolayout": True}):
            fig = plot_transfer_function(_make_series())
        assert fig.get_layout_engine() is None

    def test_too_small_figure_warns(self):
        """A figure too small for its decorations warns."""
        with pytest.warns(UserWarning, match="too small"):
            plot_transfer_function(_make_series(), figsize=(0.5, 0.5))

    @pytest.mark.parametrize(
        "select",
        [None, [0, 1, 2], [0, 1, 2, 3, 4]],
        ids=["full", "one_row", "hidden_cell"],
    )
    def test_legend_centered_on_grid(self, select):
        """The legend is centered on the panel grid."""
        fig = plot_transfer_function(_make_series(), select=select)
        fig.draw_without_rendering()
        grid = Bbox.union(
            [ax.get_position().transformed(fig.transFigure) for ax in fig.axes]
        )
        legend = fig.legends[0].get_window_extent()
        center = (legend.x0 + legend.x1) / 2.0
        assert center == pytest.approx((grid.x0 + grid.x1) / 2.0, abs=1.0)


class TestLegendCenterX:
    """Tests for _legend_center_x."""

    @pytest.mark.parametrize(
        ("grid_center", "legend_width", "expected"),
        [
            (0.6, 100.0, 0.6),
            (0.98, 100.0, 0.95),
            (0.02, 100.0, 0.05),
            (0.6, 1200.0, 0.5),
        ],
        ids=["centered", "clamp_right", "clamp_left", "too_wide"],
    )
    def test_center(self, grid_center, legend_width, expected):
        """Centered on the grid, clamped to the figure edges."""
        legend = MagicMock()
        legend.get_window_extent.return_value = Bbox.from_bounds(
            0.0, 0.0, legend_width, 10.0
        )
        result = _legend_center_x(legend, grid_center, 1000.0, 0.0)
        assert result == pytest.approx(expected)
