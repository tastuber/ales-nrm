"""Band-averaged observable time series and transfer-function plots.

Provides a registry of plottable interferometric observables and a
compute layer that averages an observable over a wavelength band for
each block of an observing sequence.
"""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ales_nrm.utilities import (
    _get_block_mean_time_seconds,
    _unwrap_block_times,
)

if TYPE_CHECKING:
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
