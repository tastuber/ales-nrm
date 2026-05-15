"""Shared utility functions for ales-nrm."""


def ensure_odd(n: int) -> int:
    """Round up to the nearest odd integer if even.

    Args:
        n: Input integer.

    Returns:
        The input unchanged if odd, or ``n + 1`` if
        even.
    """
    return n if n % 2 == 1 else n + 1
