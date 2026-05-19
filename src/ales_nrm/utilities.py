"""Shared utility functions for ales-nrm."""

import numpy as np


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
