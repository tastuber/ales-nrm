"""Observable extraction wrapper for SAMpy analysis.

Provides a high-level function to extract closure phases,
squared visibilities (VIS2), and complex visibilities from
an ales_nrm ObservingBlock using SAMpy's Fourier-plane
analysis routines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ales_nrm.observation import ObservingBlock


def _ensure_trailing_sep(path: str | Path) -> str:
    """Ensure a directory path string ends with os.sep.

    SAMpy's analysis functions expect mask_dir strings
    to end with a directory separator. Paths returned by
    ``setup_sampy_coords`` are Path objects without a
    trailing separator.

    Args:
        path: Directory path as string or Path object.

    Returns:
        String representation ending with ``os.sep``.
    """
    s = str(path)
    if not s.endswith(os.sep):
        s += os.sep
    return s


def extract_observables(
    block: ObservingBlock,
    mask_dirs: dict[float, Path],
    *,
    extract_cp: bool = True,
    extract_vis2: bool = True,
    extract_compl_vis: bool = False,
    wl_indices: tuple[int, int] | None = None,
    nx: int = 501,
    ny: int = 501,
    display: bool = False,
) -> dict:
    """Extract CP, VIS2, complex visibility per wavelength.

    Calls SAMpy's multi-pixel closure phase, squared
    visibility, and complex visibility extraction for
    each selected wavelength channel in the block, using
    the pre-computed coordinate directories from
    ``setup_sampy_coords``.

    Args:
        block: Loaded ObservingBlock (optionally
            stacked). Must have ``is_loaded == True``.
        mask_dirs: Mapping wavelength (µm) → SAMpy
            mask coordinate directory Path. Typically
            the return value of ``setup_sampy_coords``.
        extract_cp: If True (default), extract closure
            phases via ``calc_cps_multi``.
        extract_vis2: If True (default), extract
            squared visibilities via ``calc_v2s``.
        extract_compl_vis: If True, extract complex
            visibilities via ``calc_cvis``. Default is
            False.
        wl_indices: Optional tuple ``(start, stop)``
            selecting a slice of wavelength channels
            by index (start inclusive, stop exclusive).
            If None (default), all wavelengths in the
            block are processed. Must match the
            wavelengths used in ``setup_sampy_coords``.
        nx: FFT grid size along x-axis passed to SAMpy
            analysis functions. Must match the
            ``n_pixels`` used in ``setup_sampy_coords``.
            Default is 501.
        ny: FFT grid size along y-axis passed to SAMpy
            analysis functions. Must match the
            ``n_pixels`` used in ``setup_sampy_coords``.
            Default is 501.
        display: If True, show SAMpy diagnostic plots
            during extraction. Default is False.

    Returns:
        Dict with keys:
            - ``'wavelengths'``: 1D array of selected
              wavelengths (µm).
            - ``'cp'``: dict mapping each wavelength
              float to the SAMpy CP result dict. Only
              present if ``extract_cp`` is True.
            - ``'vis2'``: dict mapping each wavelength
              float to the SAMpy VIS2 result dict.
              Only present if ``extract_vis2`` is True.
            - ``'compl_vis'``: dict mapping each
              wavelength float to the SAMpy complex
              visibility result dict. Only present if
              ``extract_compl_vis`` is True.

    Raises:
        ImportError: If SAMpy is not installed.
        RuntimeError: If the block has not been loaded.
        KeyError: If a wavelength in the block is not
            found in ``mask_dirs``.
        ValueError: If all extraction flags are False,
            or if wavelength selection is invalid.
    """
    try:
        import sampy.analysis
    except ImportError as exc:
        raise ImportError(
            "SAMpy is required for extract_observables. "
            "Install with: pip install ales_nrm[sampy]"
        ) from exc

    if not block.is_loaded:
        raise RuntimeError(
            "ObservingBlock has not been loaded. Call block.load() first."
        )

    if not (extract_cp or extract_vis2 or extract_compl_vis):
        raise ValueError(
            "At least one extraction flag must be True. "
            "Set extract_cp, extract_vis2, or "
            "extract_compl_vis to True."
        )

    # Resolve wavelength selection
    n_wavelengths = block.cubes.shape[1]
    if wl_indices is not None:
        start, stop = wl_indices
        wl_idx = np.arange(start, stop)
        if len(wl_idx) == 0:
            raise ValueError(
                "wl_indices yielded no channels. Check start < stop."
            )
    else:
        wl_idx = np.arange(n_wavelengths)

    selected_wavelengths = block.wavelengths[wl_idx]

    cp: dict[float, dict] = {}
    vis2: dict[float, dict] = {}
    compl_vis: dict[float, dict] = {}

    for w in wl_idx:
        wl = float(block.wavelengths[w])

        # Extract 3D image stack for this wavelength
        images = block.cubes[:, w, :, :]

        # Get mask directory as string with trailing sep
        mask_dir = _ensure_trailing_sep(mask_dirs[wl])

        # Call SAMpy analysis functions as requested
        if extract_cp:
            _cp_result = sampy.analysis.calc_cps_multi(
                images,
                mask_dir,
                nx=nx,
                ny=ny,
                display=display,
            )
            cp[wl] = _cp_result

        if extract_vis2:
            _vis2_result = sampy.analysis.calc_v2s(
                images,
                mask_dir,
                nx=nx,
                ny=ny,
                display=display,
            )
            vis2[wl] = _vis2_result

        if extract_compl_vis:
            _compl_vis_result = sampy.analysis.calc_cvis(
                images,
                mask_dir,
                nx=nx,
                ny=ny,
                display=display,
            )
            compl_vis[wl] = _compl_vis_result

    result: dict = {
        "wavelengths": selected_wavelengths,
    }
    if extract_cp:
        result["cp"] = cp
    if extract_vis2:
        result["vis2"] = vis2
    if extract_compl_vis:
        result["compl_vis"] = compl_vis

    return result
