"""Polynomial calibration of SAMpy-extracted observables.

Wraps SAMpy's ``polynomial_calibrate`` function to calibrate
science block observables against calibrator blocks, operating
per wavelength channel. Produces calibrated ``Observables``
containing only closure phases and squared visibilities (SAMpy
does not calibrate complex visibilities).
"""

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np

from ales_nrm.observables import (
    Observables,
    _parse_timestamps_to_seconds,
)

if TYPE_CHECKING:
    from ales_nrm.observation import ObservingBlock, ObservingSequence

logger = logging.getLogger(__name__)


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


def _validate_raw_extraction(
    block: "ObservingBlock",
    source_label: str,
) -> dict:
    """Validate and return raw extraction result for a block.

    Args:
        block: ObservingBlock to check.
        source_label: Label key in ``_raw_extraction``.

    Returns:
        The raw extraction result dict.

    Raises:
        ValueError: If extraction is missing, not from SAMpy,
            or missing required observable keys.
    """
    if block._raw_extraction is None:
        raise ValueError(
            f"Block '{block.target}' has no raw extraction. "
            f"Call extract_observables() first."
        )
    if source_label not in block._raw_extraction:
        raise ValueError(
            f"Block '{block.target}' has no extraction "
            f"under label '{source_label}'."
        )
    entry = block._raw_extraction[source_label]
    if entry["backend"] != "sampy":
        raise ValueError(
            f"Block '{block.target}' extraction under "
            f"'{source_label}' uses backend "
            f"'{entry['backend']}', not 'sampy'."
        )
    return entry["result"]


def _assemble_observables_for_wavelength(
    blocks: list["ObservingBlock"],
    source_label: str,
    observable_type: str,
    key: str,
    wl: float,
) -> np.ndarray:
    """Get one observable vector per block for one wavelength.

    Args:
        blocks: List of ObservingBlocks.
        source_label: Label key in ``_raw_extraction``.
        observable_type: ``'cp'`` or ``'vis2'``.
        key: SAMpy result key (``'closure_phases'`` or
            ``'v2'``).
        wl: Wavelength float key.

    Returns:
        Array of shape ``(n_blocks, n_obs)`` where n_obs is
        the number of triangles (for CP) or baselines (for
        V²).

    Raises:
        ValueError: If a block is missing the observable type
            or the specified wavelength.
    """
    rows = []
    for block in blocks:
        result = _validate_raw_extraction(block, source_label)
        if observable_type not in result:
            raise ValueError(
                f"Block '{block.target}' extraction under "
                f"'{source_label}' does not contain "
                f"'{observable_type}'. Was it extracted?"
            )
        wl_dict = result[observable_type]
        if wl not in wl_dict:
            raise ValueError(
                f"Block '{block.target}' extraction "
                f"missing wavelength {wl:.4f} \u00b5m in "
                f"'{observable_type}'."
            )
        rows.append(np.asarray(wl_dict[wl][key]))
    return np.array(rows)


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


def calibrate_block(
    sci_block: "ObservingBlock",
    cal_blocks: list["ObservingBlock"],
    *,
    source_label: str = "raw",
    poly_order: int = 1,
    calibrate_cp: bool = True,
    calibrate_vis2: bool = True,
    display: bool = False,
) -> "Observables":
    """Calibrate one science block against calibrator blocks.

    For each wavelength channel, assembles the mean observables
    from the science block and all calibrator blocks, then calls
    SAMpy's ``polynomial_calibrate`` to remove instrumental
    systematics. Closure phases are calibrated subtractively;
    squared visibilities are calibrated divisively.

    The calibrated ``Observables`` contains only ``vis2``,
    ``vis2_err``, ``t3phi``, ``t3phi_err``, and associated
    flags. Complex visibility fields (``visamp``, ``visphi``,
    etc.) are not populated since SAMpy does not calibrate
    them.

    Args:
        sci_block: The science ObservingBlock to calibrate.
            Must have a SAMpy extraction under ``source_label``.
        cal_blocks: List of calibrator ObservingBlocks. Each
            must have a SAMpy extraction under ``source_label``.
            At least one calibrator is required.
        source_label: Label under which the raw SAMpy
            extraction is stored. Default ``'raw'``.
        poly_order: Polynomial order for calibration. 0 =
            constant (mean), 1 = linear drift, etc. Default 1.
        calibrate_cp: If True, calibrate closure phases.
            Default True.
        calibrate_vis2: If True, calibrate squared
            visibilities. Default True.
        display: If True, show SAMpy calibration diagnostic
            plots. Default False.

    Returns:
        Calibrated ``Observables`` instance with ``vis2``,
        ``vis2_err``, ``t3phi``, ``t3phi_err``, and flags
        populated. Complex visibility fields are None.

    Raises:
        ImportError: If SAMpy is not installed.
        ValueError: If no calibrators provided, blocks lack
            required extractions, or wavelengths don't match.
    """
    try:
        from sampy.calibration import polynomial_calibrate
    except ImportError as exc:
        raise ImportError(
            "SAMpy is required for calibration. "
            "Install with: pip install ales_nrm[sampy]"
        ) from exc

    if not cal_blocks:
        raise ValueError("At least one calibrator block is required.")

    if not (calibrate_cp or calibrate_vis2):
        raise ValueError(
            "At least one of calibrate_cp or calibrate_vis2 must be True."
        )

    # Validate all blocks have extractions
    sci_result = _validate_raw_extraction(sci_block, source_label)
    for cal_block in cal_blocks:
        _validate_raw_extraction(cal_block, source_label)

    # Determine wavelengths from science block extraction
    wavelengths = sci_result["wavelengths"]

    # Require two or more calibrators for polynomial order
    n_cal = len(cal_blocks)
    if poly_order >= 1 and n_cal < 2:
        raise ValueError(
            f"poly_order={poly_order} requires at least 2 calibrator blocks, "
            f"but only {n_cal} provided. Use poly_order=0 for a single "
            "calibrator."
        )

    # Get times
    sci_time = np.array([_get_block_mean_time_seconds(sci_block)])
    cal_times = _get_block_times(cal_blocks)

    # Retrieve existing Observables for geometry/metadata
    if source_label not in sci_block.observables:
        raise ValueError(
            f"Block '{sci_block.target}' has no "
            f"Observables under label '{source_label}'. "
            f"Call extract_observables() first."
        )
    source_obs = sci_block.observables[source_label]

    # Create output Observables copying geometry from source
    obs = Observables(
        target=source_obs.target,
        wavelengths=source_obs.wavelengths.copy(),
        stations=source_obs.stations,
        baselines=source_obs.baselines,
        triangles=source_obs.triangles,
        mean_para_angle=source_obs.mean_para_angle,
        mjd=source_obs.mjd,
        time_start=source_obs.time_start,
        time_end=source_obs.time_end,
        block_type=source_obs.block_type,
        mask_name=source_obs.mask_name,
    )

    # Add calibration and backend information to Observables
    obs.calibrated = True
    obs.extraction_backend = "sampy"

    # Build calibrator name string
    cal_names = list(dict.fromkeys(b.target for b in cal_blocks))
    obs.calibrator_target = ", ".join(cal_names)
    obs.notes = (
        f"poly_order={poly_order}, "
        f"n_calibrators={n_cal}, "
        f"source_label='{source_label}'"
    )

    n_wav = len(wavelengths)
    n_bl = obs.n_baselines
    n_tri = obs.n_triangles

    # Calibrate closure phases
    if calibrate_cp and "cp" in sci_result:
        t3phi = np.full((n_tri, n_wav), np.nan)
        t3phi_err = np.full((n_tri, n_wav), np.nan)
        t3_flag = np.zeros((n_tri, n_wav), dtype=bool)

        for w_idx, wl in enumerate(wavelengths):
            wl_f = float(wl)

            # Assemble science CPs: shape (1, n_tri)
            sci_cps = _assemble_observables_for_wavelength(
                [sci_block],
                source_label,
                "cp",
                "closure_phases",
                wl_f,
            )

            # Assemble calibrator CPs: shape (n_cal, n_tri)
            cal_cps = _assemble_observables_for_wavelength(
                cal_blocks,
                source_label,
                "cp",
                "closure_phases",
                wl_f,
            )

            calibrated, cal_variance, cal_scatter, _ = polynomial_calibrate(
                sci_cps,
                cal_cps,
                sci_time,
                cal_times,
                poly_order,
                "cps",
                display=display,
            )

            # calibrated shape: (n_tri, n_pointings) = (n_tri, 1)
            t3phi[:, w_idx] = calibrated[:, 0]
            # Use scatter as error estimate (more reliable)
            t3phi_err[:, w_idx] = cal_scatter[:, 0]

        obs.t3phi = t3phi
        obs.t3phi_err = t3phi_err
        obs.t3_flag = t3_flag
    elif calibrate_cp and "cp" not in sci_result:
        warnings.warn(
            "calibrate_cp=True but science block has no CP "
            "extraction. Skipping CP calibration.",
            stacklevel=2,
        )

    # Calibrate squared visibilities
    if calibrate_vis2 and "vis2" in sci_result:
        vis2 = np.full((n_bl, n_wav), np.nan)
        vis2_err = np.full((n_bl, n_wav), np.nan)
        vis2_flag = np.zeros((n_bl, n_wav), dtype=bool)

        for w_idx, wl in enumerate(wavelengths):
            wl_f = float(wl)

            # Assemble science V2: shape (1, n_bl)
            sci_v2 = _assemble_observables_for_wavelength(
                [sci_block],
                source_label,
                "vis2",
                "v2",
                wl_f,
            )

            # Assemble calibrator V2: shape (n_cal, n_bl)
            cal_v2 = _assemble_observables_for_wavelength(
                cal_blocks,
                source_label,
                "vis2",
                "v2",
                wl_f,
            )

            calibrated, cal_variance, cal_scatter, _ = polynomial_calibrate(
                sci_v2,
                cal_v2,
                sci_time,
                cal_times,
                poly_order,
                "v2s",
                display=display,
            )

            # calibrated shape: (n_bl, n_pointings) = (n_bl, 1)
            vis2[:, w_idx] = calibrated[:, 0]
            vis2_err[:, w_idx] = cal_scatter[:, 0]

        obs.vis2 = vis2
        obs.vis2_err = vis2_err
        obs.vis2_flag = vis2_flag
    elif calibrate_vis2 and "vis2" not in sci_result:
        warnings.warn(
            "calibrate_vis2=True but science block has no "
            "VIS2 extraction. Skipping VIS2 calibration.",
            stacklevel=2,
        )

    return obs


def calibrate_sequence(
    sequence: "ObservingSequence",
    *,
    source_label: str = "raw",
    output_label: str = "calibrated",
    poly_order: int = 1,
    calibrate_cp: bool = True,
    calibrate_vis2: bool = True,
    display: bool = False,
) -> None:
    """Calibrate all SCI blocks using all CAL blocks in a sequence.

    For each science block in the sequence, calibrates it against
    all calibrator blocks and stores the result under
    ``output_label`` in the science block's ``observables`` dict.

    Args:
        sequence: ObservingSequence containing SCI and CAL blocks.
        source_label: Label under which raw SAMpy extractions
            are stored. Default ``'raw'``.
        output_label: Label under which to store calibrated
            Observables on each science block. Default
            ``'calibrated'``.
        poly_order: Polynomial order for calibration. Default 1.
        calibrate_cp: Calibrate closure phases. Default True.
        calibrate_vis2: Calibrate squared visibilities.
            Default True.
        display: If True, show SAMpy calibration diagnostic
            plots. Default False.

    Raises:
        ValueError: If no calibrator blocks are found in the
            sequence.
    """
    cal_blocks = sequence.calibrator_blocks
    sci_blocks = sequence.science_blocks

    if not cal_blocks:
        raise ValueError(
            f"No calibrator blocks found in sequence '{sequence.name}'."
        )

    if not sci_blocks:
        logger.warning(
            "No science blocks found in sequence '%s'. Nothing to calibrate.",
            sequence.name,
        )
        return

    logger.info(
        "Calibrating %d SCI blocks using %d CAL blocks "
        "in sequence '%s' (poly_order=%d, "
        "output_label='%s').",
        len(sci_blocks),
        len(cal_blocks),
        sequence.name,
        poly_order,
        output_label,
    )

    for sci_block in sci_blocks:
        cal_obs = calibrate_block(
            sci_block,
            cal_blocks,
            source_label=source_label,
            poly_order=poly_order,
            calibrate_cp=calibrate_cp,
            calibrate_vis2=calibrate_vis2,
            display=display,
        )
        sci_block.observables[output_label] = cal_obs

        logger.info(
            "Calibrated block '%s' stored under '%s'.",
            sci_block.target,
            output_label,
        )
