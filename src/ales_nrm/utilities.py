"""Shared utility functions for ales-nrm."""

import datetime
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

if TYPE_CHECKING:
    from ales_nrm.observation import ObservingBlock


def ensure_odd(n: int) -> int:
    """Round up to the nearest odd integer if even.

    Args:
        n: Input integer.

    Returns:
        The input unchanged if odd, or ``n + 1`` if
        even.
    """
    return n if n % 2 == 1 else n + 1


def rotate_points_2d(
    points: np.ndarray,
    center: tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    """Rotate 2D points about a center.

    Applies a counter-clockwise rotation by the given angle around the
    specified center point.

    Args:
        points: Array of shape (N, 2) with coordinates. The two columns
            represent the first and second spatial axes respectively.
        center: (c0, c1) rotation origin, matching the axis order of
            points.
        angle_deg: Rotation angle in degrees (counter-clockwise
            positive).

    Returns:
        Rotated points, shape (N, 2).
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    d0 = points[:, 0] - center[0]
    d1 = points[:, 1] - center[1]

    rotated = np.empty_like(points)
    rotated[:, 0] = center[0] + d0 * cos_a - d1 * sin_a
    rotated[:, 1] = center[1] + d0 * sin_a + d1 * cos_a

    return rotated


def _parse_time_string_to_seconds(time_str: str) -> float:
    """Parse UT time string (HH:MM:SS.sss) to seconds past midnight."""
    parts = time_str.strip().split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2]) if len(parts) > 2 else 0.0
    return hours * 3600.0 + minutes * 60.0 + seconds


def _parse_timestamps_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Parse UT time strings to seconds, handling midnight crossing.

    If a midnight crossing is detected (a large backward jump of more
    than 12 hours between consecutive timestamps), all subsequent
    timestamps are shifted by +86400 seconds. The observation_date is
    assumed to be the date before midnight.

    Args:
        timestamps: 1D array of UT time strings.

    Returns:
        1D array of seconds (possibly exceeding 86400 for
        post-midnight frames).
    """
    n = len(timestamps)
    seconds = np.empty(n)
    for i in range(n):
        seconds[i] = _parse_time_string_to_seconds(str(timestamps[i]))

    # Detect and handle midnight crossing
    for i in range(1, n):
        if seconds[i] - seconds[i - 1] < -43200.0:  # jump back > 12h
            # All subsequent timestamps are past midnight
            seconds[i:] += 86400.0
            break

    return seconds


def _compute_mean_mjd(
    observation_date: datetime.date | None,
    timestamps: np.ndarray,
) -> float | None:
    """Compute mean MJD from observation date and UT time strings.

    Handles midnight crossing: if timestamps span midnight, the
    observation_date is assumed to be the date before midnight.

    Args:
        observation_date: The date of observation (date before midnight
            if crossing occurs). None if unavailable.
        timestamps: 1D array of UT time strings.

    Returns:
        Mean MJD as float, or None if observation_date is None.
    """
    if observation_date is None:
        return None

    seconds = _parse_timestamps_to_seconds(timestamps)
    mean_seconds = float(np.mean(seconds))

    # Convert to days offset and remaining time
    days_offset = int(mean_seconds // 86400)
    remaining_seconds = mean_seconds - days_offset * 86400.0

    # Build datetime combining observation_date with mean time
    hours = int(remaining_seconds // 3600)
    remaining = remaining_seconds - hours * 3600
    minutes = int(remaining // 60)
    secs = remaining - minutes * 60
    whole_secs = int(secs)
    microsecs = int((secs - whole_secs) * 1e6)

    base_dt = datetime.datetime(
        year=observation_date.year,
        month=observation_date.month,
        day=observation_date.day,
        hour=hours,
        minute=minutes,
        second=whole_secs,
        microsecond=microsecs,
    )
    # Add any full-day offset (from midnight crossing)
    dt = base_dt + datetime.timedelta(days=days_offset)

    return float(Time(dt, scale="utc").mjd)


def _get_block_mean_time_seconds(block: "ObservingBlock") -> float:
    """Compute mean observation time in seconds for a block.

    Parses all timestamps in the block, handles midnight
    crossing, and returns the arithmetic mean in seconds
    since midnight (possibly exceeding 86400 for
    post-midnight observations).

    Args:
        block: A loaded ObservingBlock with timestamps.

    Returns:
        Mean time in seconds.

    Raises:
        ValueError: If no valid timestamps are available.
    """
    seconds = _parse_timestamps_to_seconds(block.timestamps)
    if len(seconds) == 0:
        raise ValueError(f"Block '{block.target}' has no valid timestamps.")
    return float(np.mean(seconds))


def _get_block_times(
    blocks: list["ObservingBlock"],
) -> np.ndarray:
    """Get mean time in seconds for each block.

    Args:
        blocks: List of loaded ObservingBlocks.

    Returns:
        1D array of mean times in seconds, shape
        ``(n_blocks,)``.
    """
    times = np.empty(len(blocks))
    for i, block in enumerate(blocks):
        times[i] = _get_block_mean_time_seconds(block)
    return times
