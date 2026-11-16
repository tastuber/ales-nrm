"""Band-averaged observable time series and transfer-function plots.

Provides a registry of plottable interferometric observables, a
compute layer that averages an observable over a wavelength band for
each block of an observing sequence, and a plot layer that shows the
band-averaged values per triangle or baseline over time. The dashed
line through the calibrator blocks is the transfer function.
"""

import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ales_nrm.utilities import (
    _get_block_mean_time_seconds,
    _unwrap_block_times,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.text import Text
    from matplotlib.ticker import Formatter, Locator

    from ales_nrm.observation import ObservingBlock


@dataclass(frozen=True)
class _ObservableSpec:
    """Attribute names and plot defaults for one observable.

    Attributes:
        value_attr: Name of the value array on ``Observables``.
        error_attr: Name of the error array on ``Observables``.
        flag_attr: Name of the flag array (``True`` means invalid).
        geometry: ``"triangles"`` or ``"baselines"``; the
            ``Observables`` attribute holding the items with
            ``.name``.
        is_phase: Whether values are phases in degrees.
        reference: Default value of the reference line.
        ylabel: Default y-axis label.
    """

    value_attr: str
    error_attr: str
    flag_attr: str
    geometry: str
    is_phase: bool
    reference: float
    ylabel: str


_SPECS: dict[str, _ObservableSpec] = {
    "t3phi": _ObservableSpec(
        value_attr="t3phi",
        error_attr="t3phi_err",
        flag_attr="t3_flag",
        geometry="triangles",
        is_phase=True,
        reference=0.0,
        ylabel="closure phase (deg)",
    ),
    "t3amp": _ObservableSpec(
        value_attr="t3amp",
        error_attr="t3amp_err",
        flag_attr="t3_flag",
        geometry="triangles",
        is_phase=False,
        reference=1.0,
        ylabel="triple amplitude",
    ),
    "vis2": _ObservableSpec(
        value_attr="vis2",
        error_attr="vis2_err",
        flag_attr="vis2_flag",
        geometry="baselines",
        is_phase=False,
        reference=1.0,
        ylabel="squared visibility",
    ),
    "visamp": _ObservableSpec(
        value_attr="visamp",
        error_attr="visamp_err",
        flag_attr="vis_flag",
        geometry="baselines",
        is_phase=False,
        reference=1.0,
        ylabel="visibility amplitude",
    ),
    "visphi": _ObservableSpec(
        value_attr="visphi",
        error_attr="visphi_err",
        flag_attr="vis_flag",
        geometry="baselines",
        is_phase=True,
        reference=0.0,
        ylabel="visibility phase (deg)",
    ),
}


@dataclass(frozen=True)
class BandAveragedSeries:
    """Band-averaged observable per block, ready for plotting.

    Attributes:
        observable: Registry key, e.g. ``"t3phi"``.
        label: Observables label used, e.g. ``"raw"``.
        values: Band-averaged values, shape ``(n_blocks, n_items)``.
            Phases are circular means in degrees, in (-180, 180].
        errors: Mean of the per-channel errors (a typical, not a
            propagated, uncertainty; channel errors are correlated).
        names: Triangle or baseline names, length ``n_items``.
        block_types: ``"CAL"``/``"SCI"`` per block.
        targets: Target name per block.
        block_times_sec: Mean UT seconds per block, unwrapped across
            midnight; NaN if unavailable.
        n_wav_used: Channels used per block, shape ``(n_blocks,)``.
        wl_range: (min, max) wavelength in µm actually used.
    """

    observable: str
    label: str
    values: np.ndarray
    errors: np.ndarray
    names: tuple[str, ...]
    block_types: tuple[str, ...]
    targets: tuple[str, ...]
    block_times_sec: np.ndarray
    n_wav_used: np.ndarray
    wl_range: tuple[float, float]


def _nan_mean_rows(a: np.ndarray) -> np.ndarray:
    """Average over the last axis, ignoring non-finite values.

    Rows without any finite value yield NaN without emitting a
    ``RuntimeWarning``.

    Args:
        a: 2D array of shape ``(n_items, n_channels)``.

    Returns:
        1D float array of shape ``(n_items,)``.
    """
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    total = np.where(finite, a, 0.0).sum(axis=1)
    count = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


def _band_average(a: np.ndarray, is_phase: bool) -> np.ndarray:
    """Average each row over the wavelength channels.

    Args:
        a: 2D array of shape ``(n_items, n_channels)``. Phases are
            in degrees.
        is_phase: If True, compute the circular mean; otherwise the
            arithmetic mean. Non-finite values are ignored.

    Returns:
        1D float array of shape ``(n_items,)``. Circular means lie
        in (-180, 180]. Rows without finite values are NaN.
    """
    if not is_phase:
        return _nan_mean_rows(a)
    with np.errstate(invalid="ignore"):
        rad = np.deg2rad(np.asarray(a, dtype=float))
        sin_mean = _nan_mean_rows(np.sin(rad))
        cos_mean = _nan_mean_rows(np.cos(rad))
    deg = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    # arctan2 may return exactly -180; map it to +180
    return np.where(deg <= -180.0, deg + 360.0, deg)


def _describe_block(index: int, block: "ObservingBlock") -> str:
    """Return a short block description for warning messages."""
    return f"[{index}] '{block.target}' ({block.block_type.value})"


def _safe_block_time(block: "ObservingBlock") -> float:
    """Return the block mean time in seconds, or NaN on failure.

    Wraps ``_get_block_mean_time_seconds``. Empty, malformed or
    missing timestamps produce a ``UserWarning`` and NaN.

    Args:
        block: A loaded ObservingBlock.

    Returns:
        Mean time in seconds since midnight (may exceed 86400), or
        NaN if it cannot be determined.
    """
    try:
        return _get_block_mean_time_seconds(block)
    except (ValueError, IndexError, TypeError) as exc:
        warnings.warn(
            "Could not determine the mean time of block "
            f"'{block.target}' ({block.block_type.value}): {exc}. "
            "Using NaN.",
            UserWarning,
            stacklevel=3,
        )
        return float("nan")


def compute_band_averaged_series(
    blocks: "Iterable[ObservingBlock]",
    *,
    observable: str = "t3phi",
    label: str = "raw",
    wl_min: float | None = None,
    wl_max: float | None = None,
    use_flags: bool = True,
) -> BandAveragedSeries:
    """Band-average an observable per block for transfer functions.

    For each loaded block, the observable stored under ``label`` is
    averaged over the wavelength channels within the band. Phases
    are averaged circularly, all other observables arithmetically.
    Errors are the mean of the per-channel errors. Non-finite
    samples are ignored.

    The wavelength selection is built from each block's own
    wavelength grid. All blocks are assumed to share the same mask
    geometry (number and order of triangles or baselines); this is
    not checked. Item names are taken from the first usable block.

    Args:
        blocks: Blocks in chronological order, e.g. a list of
            ``ObservingBlock`` or an ``ObservingSequence``.
        observable: Registry key: ``"t3phi"``, ``"t3amp"``,
            ``"vis2"``, ``"visamp"`` or ``"visphi"``.
        label: Key in ``block.observables``. Default ``"raw"``.
        wl_min: Lower wavelength bound in µm (inclusive). ``None``
            means no lower bound.
        wl_max: Upper wavelength bound in µm (inclusive). ``None``
            means no upper bound.
        use_flags: If True (default), flagged samples are excluded
            from values and errors.

    Returns:
        BandAveragedSeries with one row per loaded block.

    Raises:
        ValueError: If ``observable`` is unknown, if no block is
            loaded, or if no loaded block provides the observable
            with at least one channel in the band.

    Warns:
        UserWarning: Once for all skipped unloaded blocks, once for
            all blocks lacking data (rows filled with NaN), and for
            each block whose mean time cannot be determined (time
            set to NaN). Block indices refer to positions in
            ``blocks``.
    """
    if observable not in _SPECS:
        raise ValueError(
            f"Unknown observable '{observable}'. Valid keys: {sorted(_SPECS)}."
        )
    spec = _SPECS[observable]

    loaded: list[tuple[int, ObservingBlock]] = []
    unloaded: list[str] = []
    for idx, block in enumerate(blocks):
        if block.is_loaded:
            loaded.append((idx, block))
        else:
            unloaded.append(_describe_block(idx, block))
    if unloaded:
        warnings.warn(
            f"Skipping {len(unloaded)} unloaded block(s): "
            f"{'; '.join(unloaded)}.",
            UserWarning,
            stacklevel=2,
        )
    if not loaded:
        raise ValueError("No loaded blocks to compute the series from.")

    n_blocks = len(loaded)
    usable: list[tuple[int, np.ndarray, np.ndarray]] = []
    n_wav_used = np.zeros(n_blocks, dtype=int)
    names: tuple[str, ...] | None = None
    gaps: list[str] = []
    n_missing = 0
    n_no_channels = 0
    wl_lo = np.inf
    wl_hi = -np.inf

    for row, (idx, block) in enumerate(loaded):
        desc = _describe_block(idx, block)
        obs = block.observables.get(label)
        value = None if obs is None else getattr(obs, spec.value_attr)
        if value is None:
            n_missing += 1
            gaps.append(f"{desc}: label/observable missing")
            continue

        wl = np.asarray(obs.wavelengths, dtype=float)
        wl_mask = np.ones(wl.shape, dtype=bool)
        if wl_min is not None:
            wl_mask &= wl >= wl_min
        if wl_max is not None:
            wl_mask &= wl <= wl_max
        if not np.any(wl_mask):
            n_no_channels += 1
            gaps.append(f"{desc}: no channels in band")
            continue

        val = np.array(value, dtype=float)
        error = getattr(obs, spec.error_attr)
        err = None if error is None else np.array(error, dtype=float)
        if use_flags:
            flag = getattr(obs, spec.flag_attr)
            if flag is not None:
                flagged = np.asarray(flag, dtype=bool)
                val[flagged] = np.nan
                if err is not None:
                    err[flagged] = np.nan

        val_row = _band_average(val[:, wl_mask], spec.is_phase)
        if err is None:
            err_row = np.full(val.shape[0], np.nan)
        else:
            err_row = _nan_mean_rows(err[:, wl_mask])
        usable.append((row, val_row, err_row))

        n_wav_used[row] = int(np.count_nonzero(wl_mask))
        wl_used = wl[wl_mask]
        wl_lo = min(wl_lo, float(wl_used.min()))
        wl_hi = max(wl_hi, float(wl_used.max()))
        if names is None:
            names = tuple(item.name for item in getattr(obs, spec.geometry))

    if names is None:
        reasons = []
        if n_missing:
            reasons.append(
                f"label '{label}' or observable '{observable}' "
                f"missing on {n_missing} block(s)"
            )
        if n_no_channels:
            reasons.append(
                f"no channels in band (wl_min={wl_min}, "
                f"wl_max={wl_max}) on {n_no_channels} block(s)"
            )
        raise ValueError(
            f"No usable data for observable '{observable}': "
            f"{'; '.join(reasons)}."
        )

    # Explicit loop keeps the warning stacklevel version-independent
    raw_times = np.empty(n_blocks)
    for row, (_, block) in enumerate(loaded):
        raw_times[row] = _safe_block_time(block)
    block_times = _unwrap_block_times(raw_times)

    if gaps:
        warnings.warn(
            f"Observable '{observable}' (label '{label}') unavailable "
            f"for {len(gaps)} of {n_blocks} block(s), filled with NaN: "
            f"{'; '.join(gaps)}.",
            UserWarning,
            stacklevel=2,
        )

    n_items = len(names)
    values = np.full((n_blocks, n_items), np.nan)
    errors = np.full((n_blocks, n_items), np.nan)
    for row, val_row, err_row in usable:
        values[row] = val_row
        errors[row] = err_row

    return BandAveragedSeries(
        observable=observable,
        label=label,
        values=values,
        errors=errors,
        names=names,
        block_types=tuple(b.block_type.value for _, b in loaded),
        targets=tuple(b.target for _, b in loaded),
        block_times_sec=block_times,
        n_wav_used=n_wav_used,
        wl_range=(wl_lo, wl_hi),
    )


_TIME_MODES = ("elapsed", "ut", "index", "index_ut")
_DEFAULT_COLORS = {"CAL": "tab:blue", "SCI": "tab:orange"}
_DEFAULT_NAME_LOC = "upper left"
_SCI_SHADES = ("#f0f0f0", "#e4e4e4")
_LEGEND_ORDER = ("CAL", "transfer function", "SCI")
_LAYOUT_PAD_IN = 0.05
_STATION_BOUNDARY = re.compile(r"(?<=\d)(?=[A-Za-z])")


class _TimeAxis(NamedTuple):
    """X positions and tick configuration for one time mode.

    Attributes:
        x: X position per block.
        label: Default x-axis label.
        formatter: Major tick formatter, or ``None`` for the default.
        locator: Major tick locator, or ``None`` for the default.
        ticklabels: One fixed tick label per block placed at ``x``,
            or ``None``.
    """

    x: np.ndarray
    label: str
    formatter: "Formatter | None" = None
    locator: "Locator | None" = None
    ticklabels: tuple[str, ...] | None = None


def _format_item_name(name: str, hyphenate: bool) -> str:
    """Return the display name of a triangle or baseline.

    Args:
        name: Stored item name, e.g. ``"H1H2H3"``.
        hyphenate: If True, insert a hyphen wherever a digit is
            followed by a letter, e.g. ``"H1-H2-H3"``. Names
            without such a boundary are returned unchanged.

    Returns:
        The name to display in the panel.
    """
    if not hyphenate:
        return name
    return _STATION_BOUNDARY.sub("-", name)


def _resolve_item(entry: object, names: Sequence[str]) -> int:
    """Resolve one index or name to an item index.

    Args:
        entry: Non-negative integer index or item name.
        names: Item names.

    Returns:
        Index into ``names``.

    Raises:
        ValueError: If ``entry`` is a bool, out of range, an unknown
            name, or of another type.
    """
    n_items = len(names)
    if isinstance(entry, bool):
        raise ValueError(f"Boolean entries are not allowed: {entry!r}.")
    if isinstance(entry, int | np.integer):
        index = int(entry)
        if 0 <= index < n_items:
            return index
        raise ValueError(f"Index {index} out of range for {n_items} item(s).")
    if isinstance(entry, str):
        if entry in names:
            return names.index(entry)
        raise ValueError(
            f"Unknown name {entry!r}. Valid names: {list(names)}."
        )
    raise ValueError(f"Invalid entry {entry!r}; expected an index or a name.")


def _resolve_select(
    select: Sequence[int | str] | None,
    names: Sequence[str],
) -> list[int]:
    """Resolve a selection of items to indices, preserving order.

    Args:
        select: ``None`` for all items, or indices and/or names. A
            single string is treated as one name.
        names: Item names.

    Returns:
        List of item indices in selection order.

    Raises:
        ValueError: If the selection is empty, contains invalid
            entries, or selects an item twice.
    """
    if select is None:
        return list(range(len(names)))
    if isinstance(select, str):
        select = [select]
    indices = [_resolve_item(entry, names) for entry in select]
    if not indices:
        raise ValueError("select must not be empty.")
    if len(set(indices)) != len(indices):
        dupes = sorted({names[i] for i in indices if indices.count(i) > 1})
        raise ValueError(f"Items selected more than once: {dupes}.")
    return indices


def _resolve_name_locs(
    name_loc: str | dict[int | str, str],
    names: Sequence[str],
) -> list[str]:
    """Return the panel-name location for every item.

    Args:
        name_loc: One location for all panels, or a mapping from
            index or name to location. Unmapped items use
            ``"upper left"``.
        names: Item names.

    Returns:
        List of location strings, one per item.

    Raises:
        ValueError: If a mapping key is not a valid index or name.
    """
    if isinstance(name_loc, str):
        return [name_loc] * len(names)
    locs = [_DEFAULT_NAME_LOC] * len(names)
    for key, loc in name_loc.items():
        locs[_resolve_item(key, names)] = loc
    return locs


def _grid_shape(n_panels: int, n_cols: int | None) -> tuple[int, int]:
    """Return ``(rows, cols)`` for the panel grid.

    Args:
        n_panels: Number of panels (at least 1).
        n_cols: Explicit number of columns, or ``None`` for automatic
            selection: up to 4 panels in one row, otherwise 4 or 3
            columns, whichever leaves fewer empty cells (4 on a tie).

    Returns:
        Tuple ``(rows, cols)``.

    Raises:
        ValueError: If ``n_cols`` is smaller than 1.
    """
    if n_cols is not None:
        if n_cols < 1:
            raise ValueError(f"n_cols must be >= 1, got {n_cols}.")
        cols = n_cols
    elif n_panels <= 4:
        cols = n_panels
    else:
        cols = min((4, 3), key=lambda c: (-n_panels) % c)
    return -(-n_panels // cols), cols


def _default_figsize(rows: int, cols: int) -> tuple[float, float]:
    """Return the default figure size in inches for a grid."""
    return (max(1.8 * cols + 0.8, 4.0), max(1.5 * rows + 1.0, 3.0))


def _format_hhmm(seconds: float) -> str:
    """Format seconds as ``HH:MM`` modulo 24 h; NaN gives ``"--"``."""
    if not np.isfinite(seconds):
        return "--"
    minutes = int(round(float(seconds) / 60.0)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _format_hours(hours: float, _pos: int | None = None) -> str:
    """Tick formatter callback: hours to ``HH:MM`` modulo 24 h."""
    return _format_hhmm(hours * 3600.0)


def _build_time_axis(times_sec: np.ndarray, mode: str) -> _TimeAxis:
    """Build x positions and tick configuration for a time mode.

    Args:
        times_sec: Block times in seconds, unwrapped across midnight.
        mode: ``"elapsed"``, ``"ut"``, ``"index"`` or ``"index_ut"``.

    Returns:
        The time-axis configuration.

    Raises:
        ValueError: If ``mode`` is unknown, or if ``mode`` is
            ``"elapsed"`` or ``"ut"`` and any time is NaN.
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if mode not in _TIME_MODES:
        raise ValueError(
            f"Unknown time_mode '{mode}'. Valid modes: {list(_TIME_MODES)}."
        )
    times = np.asarray(times_sec, dtype=float)
    if mode in ("elapsed", "ut") and not np.all(np.isfinite(times)):
        raise ValueError(
            f"time_mode='{mode}' requires valid times for all blocks, "
            "but some are NaN. Use time_mode='index' or 'index_ut'."
        )
    if mode == "elapsed":
        return _TimeAxis(
            x=(times - times[0]) / 60.0, label="elapsed time (min)"
        )
    if mode == "ut":
        return _TimeAxis(
            x=times / 3600.0,
            label="UT (hh:mm)",
            formatter=FuncFormatter(_format_hours),
        )
    index = np.arange(times.size, dtype=float)
    if mode == "index":
        return _TimeAxis(
            x=index,
            label="block index",
            locator=MaxNLocator(integer=True),
        )
    return _TimeAxis(
        x=index,
        label="UT (hh:mm)",
        ticklabels=tuple(_format_hhmm(t) for t in times),
    )


def _sci_runs(block_types: Sequence[str]) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` indices of SCI runs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, block_type in enumerate(block_types):
        if block_type == "SCI":
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(block_types) - 1))
    return runs


def _shade_sci_runs(
    ax: "Axes",
    x: np.ndarray,
    runs: list[tuple[int, int]],
) -> None:
    """Shade contiguous SCI runs with alternating gray spans.

    Each span is padded by 0.4 times the median positive finite
    spacing of ``x``, or by 0.5 if no such spacing exists.

    Args:
        ax: Target axes.
        x: X position per block.
        runs: Inclusive ``(start, end)`` block indices per run.
    """
    spacing = np.abs(np.diff(np.asarray(x, dtype=float)))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    pad = 0.4 * float(np.median(spacing)) if spacing.size else 0.5
    for k, (start, end) in enumerate(runs):
        ax.axvspan(
            x[start] - pad,
            x[end] + pad,
            color=_SCI_SHADES[k % 2],
            zorder=0,
        )


def _apply_time_axis(ax: "Axes", time_axis: _TimeAxis) -> None:
    """Apply the tick formatter, locator or labels to an axes."""
    if time_axis.formatter is not None:
        ax.xaxis.set_major_formatter(time_axis.formatter)
    if time_axis.locator is not None:
        ax.xaxis.set_major_locator(time_axis.locator)
    if time_axis.ticklabels is not None:
        ax.set_xticks(
            time_axis.x,
            labels=list(time_axis.ticklabels),
            rotation=45,
            ha="right",
        )


def _draw_panel(
    ax: "Axes",
    x: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    *,
    is_cal: np.ndarray,
    is_sci: np.ndarray,
    plot_cal: bool,
    plot_sci: bool,
    errorbars: bool,
    colors: dict[str, str],
    sci_runs: list[tuple[int, int]],
    reference: float | None,
    name: str,
    name_loc: str,
) -> None:
    """Draw one triangle or baseline panel.

    Args:
        ax: Target axes.
        x: X position per block.
        values: Band-averaged value per block.
        errors: Error per block.
        is_cal: Boolean mask of CAL blocks.
        is_sci: Boolean mask of SCI blocks.
        plot_cal: Whether to draw CAL points and the transfer
            function.
        plot_sci: Whether to draw SCI points.
        errorbars: Whether to draw error bars.
        colors: Colors keyed by ``"CAL"`` and ``"SCI"``.
        sci_runs: SCI runs to shade if both block types are plotted.
        reference: Reference line value, or ``None`` to omit it.
        name: Panel name.
        name_loc: Location of the panel name.
    """
    from matplotlib.offsetbox import AnchoredText

    if plot_cal and plot_sci and sci_runs:
        _shade_sci_runs(ax, x, sci_runs)
    if reference is not None:
        ax.axhline(reference, color="gray", ls=":", lw=0.8, zorder=1)

    markers = (
        ("CAL", plot_cal, is_cal, "o", 3.2),
        ("SCI", plot_sci, is_sci, "s", 2.2),
    )
    for label, enabled, mask, fmt, size in markers:
        if not enabled or not np.any(mask):
            continue
        ax.errorbar(
            x[mask],
            values[mask],
            yerr=errors[mask] if errorbars else None,
            fmt=fmt,
            ms=size,
            color=colors[label],
            elinewidth=0.6,
            zorder=3,
            label=label,
        )

    if plot_cal:
        line = is_cal & np.isfinite(values)
        if np.any(line):
            ax.plot(
                x[line],
                values[line],
                ls="--",
                lw=1.0,
                alpha=0.8,
                color=colors["CAL"],
                zorder=2,
                label="transfer function",
            )

    text = AnchoredText(
        name,
        loc=name_loc,
        prop={"size": "small", "weight": "bold"},
        pad=0.2,
        frameon=True,
    )
    text.patch.set_facecolor("white")
    text.patch.set_edgecolor("none")
    text.patch.set_alpha(0.7)
    ax.add_artist(text)


def _legend_center_x(
    legend: "Legend",
    grid_center: float,
    width: float,
    pad: float,
) -> float:
    """Return the legend's horizontal center in figure fraction.

    The legend is centered on the panel grid and shifted only as
    far as needed to stay inside the figure. A legend wider than
    the figure is centered on the figure.

    Args:
        legend: Figure legend.
        grid_center: Center of the panel grid in figure fraction.
        width: Figure width in pixels.
        pad: Minimum distance to the figure edge in pixels.

    Returns:
        X position of the legend center in figure fraction.
    """
    half = (legend.get_window_extent().width / 2.0 + pad) / width
    if half >= 0.5:
        return 0.5
    return min(max(grid_center, half), 1.0 - half)


def _fit_margins(
    fig: "Figure",
    grid_axes: Sequence["Axes"],
    used_axes: Sequence["Axes"],
    legend: "Legend | None",
    title: "Text | None",
) -> None:
    """Fit the flush panel grid between its decorations.

    The title and then the legend are stacked down from the top
    edge. The grid fills the remaining space so that tick labels,
    axis labels, legend and title neither overlap the panels nor
    are clipped. The legend is centered on the panel grid. Margins
    are sized for the active rcParams and the current figure size.

    Args:
        fig: Figure to adjust.
        grid_axes: All axes of the grid, including hidden ones.
        used_axes: Visible axes whose decorations are measured.
        legend: Figure legend, or ``None``.
        title: Figure title, or ``None``.

    Warns:
        UserWarning: If the figure is too small for its
            decorations; the default margins are kept.
    """
    from matplotlib.transforms import Bbox

    width = fig.bbox.width
    height = fig.bbox.height
    pad = _LAYOUT_PAD_IN * fig.dpi
    legend_x = 0.5
    # Second pass: resized axes may get different tick labels
    for _ in range(2):
        fig.draw_without_rendering()
        grid = Bbox.union(
            [
                ax.get_position().transformed(fig.transFigure)
                for ax in grid_axes
            ]
        )
        deco = Bbox.union([ax.get_tightbbox() for ax in used_axes])

        y_top = height - pad
        if title is not None:
            title.set_y(y_top / height)
            title.set_verticalalignment("top")
            y_top -= title.get_window_extent().height + pad
        legend_y = y_top / height
        if legend is not None:
            legend.set_bbox_to_anchor(
                (legend_x, legend_y), transform=fig.transFigure
            )
            y_top -= legend.get_window_extent().height + pad

        left = (max(grid.x0 - deco.x0, 0.0) + pad) / width
        right = 1.0 - (max(deco.x1 - grid.x1, 0.0) + pad) / width
        bottom = (max(grid.y0 - deco.y0, 0.0) + pad) / height
        top = (y_top - max(deco.y1 - grid.y1, 0.0)) / height
        if left >= right or bottom >= top:
            warnings.warn(
                "Figure is too small for its labels and legend; "
                "decorations may overlap. Increase figsize.",
                UserWarning,
                stacklevel=3,
            )
            return
        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=0.0,
            hspace=0.0,
        )
        if legend is not None:
            legend_x = _legend_center_x(
                legend, (left + right) / 2.0, width, pad
            )
            legend.set_bbox_to_anchor(
                (legend_x, legend_y), transform=fig.transFigure
            )


def plot_transfer_function(
    series: BandAveragedSeries,
    *,
    select: Sequence[int | str] | None = None,
    time_mode: str = "elapsed",
    plot_cal: bool = True,
    plot_sci: bool = True,
    errorbars: bool = False,
    ylim: tuple[float, float] | None = None,
    reference_value: float | None = None,
    show_reference: bool = True,
    colors: dict[str, str] | None = None,
    name_loc: str | dict[int | str, str] = "upper left",
    hyphenate_names: bool = True,
    n_cols: int | None = None,
    figsize: tuple[float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> "Figure":
    """Plot band-averaged CAL/SCI observables per triangle or baseline.

    Each panel shows one triangle or baseline. CAL blocks are drawn
    as circles, SCI blocks as squares, and the dashed line through
    the finite CAL points is the transfer function. Contiguous SCI
    runs are shaded when both block types are plotted.

    Font sizes and tick styles follow the active rcParams. To apply
    a style, create the figure inside ``matplotlib.rc_context`` and
    also call ``savefig`` inside that context, because some settings
    are only read at draw time.

    Panels are flush. Margins for labels, legend and title are
    sized when the figure is created, from the active rcParams and
    ``figsize``; resizing the figure afterwards may clip them. In
    Jupyter, end the call with ``;`` or assign the result to avoid
    the figure being shown twice.

    Args:
        series: Result of ``compute_band_averaged_series``.
        select: ``None`` for all items, or indices and/or names of
            triangles or baselines, plotted in the given order.
        time_mode: ``"elapsed"`` (minutes since the first block),
            ``"ut"`` (UT hours shown as HH:MM), ``"index"`` (block
            index) or ``"index_ut"`` (block index labelled with
            HH:MM).
        plot_cal: Draw CAL points and the transfer function.
        plot_sci: Draw SCI points.
        errorbars: Draw error bars from ``series.errors``.
        ylim: Common y-limits, or ``None`` for auto-scaling.
        reference_value: Reference line value. ``None`` uses the
            registry default (0 for phases, 1 otherwise).
        show_reference: Whether to draw the reference line.
        colors: Colors keyed by ``"CAL"``/``"SCI"``, merged into the
            defaults ``tab:blue``/``tab:orange``.
        name_loc: Location of the panel names, either one matplotlib
            location string or a mapping from index or name to a
            location. Unmapped panels use ``"upper left"``.
        hyphenate_names: Insert a hyphen between the station names
            in the panel names, e.g. ``H1H2H3`` is shown as
            ``H1-H2-H3``. A station boundary is assumed wherever a
            digit is followed by a letter; other names are shown
            unchanged. ``select`` and ``name_loc`` always refer to
            the stored names.
        n_cols: Number of grid columns, or ``None`` for automatic.
        figsize: Figure size in inches, or ``None`` for automatic.
        xlabel: X-axis label; ``None`` uses the time-mode default.
        ylabel: Y-axis label; ``None`` uses the observable default.
        title: Optional figure title.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If both ``plot_cal`` and ``plot_sci`` are False,
            if ``select``, ``name_loc`` keys, ``time_mode`` or
            ``n_cols`` are invalid, or if ``time_mode`` is
            ``"elapsed"`` or ``"ut"`` and some block times are NaN.
    """
    if not (plot_cal or plot_sci):
        raise ValueError("At least one of plot_cal and plot_sci must be True.")
    spec = _SPECS[series.observable]
    indices = _resolve_select(select, series.names)
    locs = _resolve_name_locs(name_loc, series.names)
    time_axis = _build_time_axis(series.block_times_sec, time_mode)
    n_panels = len(indices)
    n_rows, n_grid_cols = _grid_shape(n_panels, n_cols)
    if figsize is None:
        figsize = _default_figsize(n_rows, n_grid_cols)
    merged_colors = {**_DEFAULT_COLORS, **(colors or {})}
    reference = None
    if show_reference:
        reference = (
            spec.reference if reference_value is None else reference_value
        )
    block_types = np.asarray(series.block_types)
    is_cal = block_types == "CAL"
    is_sci = block_types == "SCI"
    sci_runs = _sci_runs(series.block_types)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        n_rows,
        n_grid_cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="none",
        figsize=figsize,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    flat = axes.ravel()
    used = flat[:n_panels]
    for ax in flat[n_panels:]:
        ax.set_visible(False)

    for ax, item in zip(used, indices, strict=True):
        _draw_panel(
            ax,
            time_axis.x,
            series.values[:, item],
            series.errors[:, item],
            is_cal=is_cal,
            is_sci=is_sci,
            plot_cal=plot_cal,
            plot_sci=plot_sci,
            errorbars=errorbars,
            colors=merged_colors,
            sci_runs=sci_runs,
            reference=reference,
            name=_format_item_name(series.names[item], hyphenate_names),
            name_loc=locs[item],
        )
        _apply_time_axis(ax, time_axis)

    x_label = time_axis.label if xlabel is None else xlabel
    y_label = spec.ylabel if ylabel is None else ylabel
    for row in range(n_rows):
        axes[row, 0].set_ylabel(y_label)
        axes[row, -1].tick_params(axis="y", right=False)
    for col in range(n_grid_cols):
        axes[0, col].tick_params(axis="x", top=False)
        if col >= n_panels:
            continue
        bottom = axes[(n_panels - 1 - col) // n_grid_cols, col]
        bottom.set_xlabel(x_label)
        bottom.tick_params(axis="x", labelbottom=True)

    if ylim is not None:
        used[0].set_ylim(ylim)

    handles: dict[str, object] = {}
    for ax in used:
        pairs = zip(*ax.get_legend_handles_labels(), strict=True)
        for handle, text in pairs:
            handles.setdefault(text, handle)
    order = [text for text in _LEGEND_ORDER if text in handles]
    legend = None
    if order:
        legend = fig.legend(
            [handles[text] for text in order],
            order,
            loc="upper center",
            ncols=3,
            framealpha=1,
            borderaxespad=0.0,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    suptitle = None if title is None else fig.suptitle(title)
    _fit_margins(fig, list(flat), list(used), legend, suptitle)

    return fig
