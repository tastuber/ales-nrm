"""OIFITS2 writer and reader for interferometric observables.

This module provides functions to write and read OIFITS2 files
following the standard described in Duvert et al. (2017). It
supports single-block and multi-block files with OI_TARGET,
OI_ARRAY, OI_WAVELENGTH, OI_VIS2, OI_VIS, and OI_T3 tables.
"""

import datetime
import re
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from ales_nrm import __version__
from ales_nrm.observables import (
    BaselineInfo,
    Observables,
    Station,
    TriangleInfo,
)


def write_oifits(
    observables: "Observables | list[Observables]",
    output_dir: str | Path,
    *,
    filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write Observables to an OIFITS2 file.

    Args:
        observables: Single or list of Observables to write.
        output_dir: Directory for output file.
        filename: Explicit filename. If None, auto-generated.
        overwrite: Overwrite existing file.

    Returns:
        Path to the written file.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(observables, Observables):
        observables_list = [observables]
    else:
        observables_list = list(observables)

    if filename is None:
        filename = generate_oifits_filename(observables_list[0])

    filepath = output_dir / filename

    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. Use overwrite=True to replace."
        )

    # Build target ID mapping
    target_id_map = _build_target_id_map(observables_list)

    # Determine array and wavelength configurations
    arrname = _get_arrname(observables_list)
    insname = _get_insname(observables_list)

    # Build HDU list
    hdu_list = fits.HDUList()
    hdu_list.append(_build_primary_hdu(observables_list, target_id_map))
    hdu_list.append(_build_oi_target(observables_list, target_id_map))
    hdu_list.append(_build_oi_array(observables_list[0].stations, arrname))

    # Build wavelength tables (one per unique grid)
    wl_tables = _build_oi_wavelength_tables(observables_list, insname)
    for wl_hdu in wl_tables:
        hdu_list.append(wl_hdu)

    # Build insname map for data tables
    insname_map = _build_insname_map(observables_list, insname)

    # Data tables (one per insname when grids differ)
    for hdu in _build_oi_vis2(
        observables_list, target_id_map, arrname, insname_map
    ):
        hdu_list.append(hdu)

    for hdu in _build_oi_vis(
        observables_list, target_id_map, arrname, insname_map
    ):
        hdu_list.append(hdu)

    for hdu in _build_oi_t3(
        observables_list, target_id_map, arrname, insname_map
    ):
        hdu_list.append(hdu)

    hdu_list.writeto(filepath, overwrite=overwrite)
    return filepath


def read_oifits(filepath: str | Path) -> list[Observables]:
    """Read an OIFITS2 file into Observables objects.

    Returns one Observables per unique (TARGET_ID, MJD) combination
    found in the data tables.

    Args:
        filepath: Path to the OIFITS2 file.

    Returns:
        List of Observables instances.

    Raises:
        ValueError: If file is not a valid OIFITS2 file.
    """
    filepath = Path(filepath)

    with fits.open(filepath) as hdul:
        # Verify OIFITS2
        primary = hdul[0].header
        content = primary.get("CONTENT", "")
        if content != "OIFITS2":
            raise ValueError(
                f"Not an OIFITS2 file: CONTENT='{content}'. "
                f"Expected 'OIFITS2'."
            )

        # Parse supporting tables
        target_map = _parse_oi_target(hdul)
        station_map = _parse_oi_array(hdul)
        wavelength_map = _parse_oi_wavelength(hdul)

        # Parse metadata from primary header
        mask_name = _get_header_value(primary, "NS_MASK", "")
        cal_state = _get_header_value(primary, "NS_CALST", "raw")
        cal_target = _get_header_value(primary, "NS_CALTG", None)
        cal_method = _get_header_value(primary, "NS_CALMT", None)

        # Collect data rows grouped by (TARGET_ID, MJD)
        groups = _collect_data_groups(hdul)

        # Build Observables for each group
        result = []
        for (target_id, mjd), group_data in groups.items():
            target_name = target_map.get(target_id, "UNKNOWN")
            # Determine which insname this group uses
            insname_key = group_data.get("insname", "ALES_NRM")
            wl_info = wavelength_map.get(insname_key)
            if wl_info is None:
                # Fallback to first available
                wl_info = next(iter(wavelength_map.values()))

            wavelengths_m = wl_info["eff_wave"]
            wavelengths_um = wavelengths_m * 1e6
            n_wav = len(wavelengths_um)

            # Get stations for this group
            arr = group_data.get("arrname", "")
            stations = station_map.get(arr, next(iter(station_map.values())))

            # Reconstruct baselines from vis2 or vis data
            baselines = _reconstruct_baselines(group_data, stations)
            triangles = _reconstruct_triangles(group_data, stations)

            # Determine block_type from category
            category = _get_category_for_target(hdul, target_id)
            block_type = "CAL" if category == "CAL" else "SCI"

            # Build observable arrays
            vis2, vis2_err, vis2_flag = _extract_vis2_arrays(
                group_data, len(baselines), n_wav
            )
            t3phi, t3phi_err, t3amp, t3amp_err, t3_flag = _extract_t3_arrays(
                group_data, len(triangles), n_wav
            )
            visamp, visamp_err, visphi, visphi_err, vis_flag = (
                _extract_vis_arrays(group_data, len(baselines), n_wav)
            )

            # Determine time_start and time_end
            time_start = group_data.get("time_start", "")
            time_end = group_data.get("time_end", "")

            # Mean parallactic angle
            mean_para_angle = float(primary.get("NS_PARA", 0.0))

            calibrated = cal_state == "calibrated"
            notes = cal_method if cal_method else ""

            obs = Observables(
                target=target_name,
                wavelengths=wavelengths_um,
                stations=stations,
                baselines=baselines,
                triangles=triangles,
                mean_para_angle=mean_para_angle,
                mjd=mjd,
                time_start=time_start,
                time_end=time_end,
                vis2=vis2,
                vis2_err=vis2_err,
                vis2_flag=vis2_flag,
                t3phi=t3phi,
                t3phi_err=t3phi_err,
                t3amp=t3amp,
                t3amp_err=t3amp_err,
                t3_flag=t3_flag,
                visamp=visamp,
                visamp_err=visamp_err,
                visphi=visphi,
                visphi_err=visphi_err,
                vis_flag=vis_flag,
                calibrated=calibrated,
                calibrator_target=cal_target,
                block_type=block_type,
                mask_name=mask_name,
                notes=notes,
            )
            result.append(obs)

    return result


def generate_oifits_filename(
    obs: Observables,
    label: str = "",
) -> str:
    """Generate a standardized OIFITS filename.

    Format: {date}T{time}_{target}_{state}[_{label}].oifits

    Args:
        obs: Observables providing metadata.
        label: Optional label suffix.

    Returns:
        Filename string (without directory).
    """
    # Date portion
    if obs.mjd is not None:
        t = Time(obs.mjd, format="mjd", scale="utc")
        date_str = t.datetime.strftime("%Y%m%d")
    else:
        date_str = "20000101"

    # Time portion
    time_str = _format_time_for_filename(obs.time_start)

    # Target
    target_str = _sanitize_target_name(obs.target)

    # State
    state = "cal" if obs.calibrated else "raw"

    # Build filename
    parts = [f"{date_str}T{time_str}", target_str, state]

    # Add label if not redundant with state
    if label and label not in ("raw", "calibrated"):
        parts.append(_sanitize_target_name(label))

    return "_".join(parts) + ".fits"


def _build_target_id_map(
    observables_list: list[Observables],
) -> dict[str, int]:
    """Build mapping from target name to TARGET_ID (1-based)."""
    targets = []
    seen = set()
    for obs in observables_list:
        if obs.target not in seen:
            seen.add(obs.target)
            targets.append(obs.target)
    return {name: i + 1 for i, name in enumerate(targets)}


def _get_arrname(observables_list: list[Observables]) -> str:
    """Get ARRNAME from observables."""
    for obs in observables_list:
        if obs.mask_name:
            return obs.mask_name
    return "UNKNOWN"


def _get_insname(observables_list: list[Observables]) -> str:
    """Get base INSNAME."""
    return "ALES_NRM"


def _build_insname_map(
    observables_list: list[Observables],
    base_insname: str,
) -> dict[int, str]:
    """Map obs index to INSNAME based on wavelength grid."""
    grids: dict[str, str] = {}
    result: dict[int, str] = {}
    counter = 0

    for i, obs in enumerate(observables_list):
        key = _wavelength_grid_key(obs.wavelengths)
        if key not in grids:
            if counter == 0:
                grids[key] = base_insname
            else:
                grids[key] = f"{base_insname}_{counter}"
            counter += 1
        result[i] = grids[key]

    return result


def _wavelength_grid_key(wavelengths: np.ndarray) -> str:
    """Create a hashable key for a wavelength grid."""
    return f"{len(wavelengths)}_{wavelengths[0]:.8f}_{wavelengths[-1]:.8f}"


def _build_primary_hdu(
    observables_list: list[Observables],
    target_id_map: dict[str, int],
) -> fits.PrimaryHDU:
    """Construct OIFITS2-compliant primary HDU."""
    header = fits.Header()
    header["CONTENT"] = ("OIFITS2", "OIFITS2 standard file")
    header["ORIGIN"] = (
        "Steward Observatory, The University of Arizona",
        "Institution",
    )

    now = datetime.datetime.now(tz=datetime.timezone.utc)
    header["DATE"] = (
        now.strftime("%Y-%m-%dT%H:%M:%S"),
        "File creation date",
    )

    obs0 = observables_list[0]
    if obs0.mjd is not None:
        t = Time(obs0.mjd, format="mjd", scale="utc")
        header["DATE-OBS"] = (
            t.datetime.strftime("%Y-%m-%d"),
            "Observation date",
        )
    else:
        header["DATE-OBS"] = (
            "2000-01-01",
            "Observation date (fallback)",
        )

    header["TELESCOP"] = ("LBT", "Telescope name")
    header["INSTRUME"] = ("ALES", "Instrument name")
    header["OBSERVER"] = ("", "Observer name")

    if len(target_id_map) > 1:
        header["OBJECT"] = ("MULTI", "Multiple targets")
    else:
        header["OBJECT"] = (obs0.target, "Target name")

    header["INSMODE"] = ("NRM", "Instrument mode")
    header["PROCSOFT"] = (
        f"ales-nrm {__version__}",
        "Processing software",
    )
    header["OBSTECH"] = (
        "APERTURE_MASKING",
        "Observation technique",
    )

    header["NS_ALVER"] = (
        __version__,
        "ales-nrm package version",
    )
    header["NS_CRDAT"] = (
        now.strftime("%Y-%m-%dT%H:%M:%S"),
        "File creation datetime",
    )

    if obs0.mjd is not None:
        t = Time(obs0.mjd, format="mjd", scale="utc")
        header["NS_OBDAT"] = (
            t.datetime.strftime("%Y-%m-%d"),
            "Observation date",
        )
    else:
        header["NS_OBDAT"] = (
            "2000-01-01",
            "Observation date (fallback)",
        )

    if obs0.calibrated:
        header["NS_CALST"] = (
            "calibrated",
            "Calibration state",
        )
    else:
        header["NS_CALST"] = ("raw", "Calibration state")

    if obs0.calibrator_target:
        header["NS_CALTG"] = (
            obs0.calibrator_target,
            "Calibrator target",
        )

    if obs0.notes:
        header["NS_CALMT"] = (
            obs0.notes[:68],
            "Calibration method",
        )

    if obs0.mask_name:
        header["NS_MASK"] = (obs0.mask_name, "Mask name")

    header["NS_PARA"] = (
        obs0.mean_para_angle,
        "Mean parallactic angle [deg]",
    )

    return fits.PrimaryHDU(header=header)


def _build_oi_target(
    observables_list: list[Observables],
    target_id_map: dict[str, int],
) -> fits.BinTableHDU:
    """Build OI_TARGET table from unique targets."""
    n_targets = len(target_id_map)
    target_id = np.array(list(target_id_map.values()), dtype=np.int16)
    target_names = list(target_id_map.keys())
    raep0 = np.zeros(n_targets, dtype=np.float64)
    decep0 = np.zeros(n_targets, dtype=np.float64)
    equinox = np.full(n_targets, 2000.0, dtype=np.float32)
    ra_err = np.zeros(n_targets, dtype=np.float64)
    dec_err = np.zeros(n_targets, dtype=np.float64)
    sysvel = np.zeros(n_targets, dtype=np.float64)
    veltyp = ["UNKNOWN"] * n_targets
    veldef = ["OPTICAL"] * n_targets
    pmra = np.zeros(n_targets, dtype=np.float64)
    pmdec = np.zeros(n_targets, dtype=np.float64)
    pmra_err = np.zeros(n_targets, dtype=np.float64)
    pmdec_err = np.zeros(n_targets, dtype=np.float64)
    parallax = np.zeros(n_targets, dtype=np.float32)
    para_err = np.zeros(n_targets, dtype=np.float32)
    spectyp = [""] * n_targets

    category = []
    for name in target_names:
        cat = "SCI"
        for obs in observables_list:
            if obs.target == name:
                if obs.block_type == "CAL":
                    cat = "CAL"
                break
        category.append(cat)

    cols = [
        fits.Column(name="TARGET_ID", format="I", array=target_id),
        fits.Column(name="TARGET", format="32A", array=target_names),
        fits.Column(name="RAEP0", format="D", unit="deg", array=raep0),
        fits.Column(name="DECEP0", format="D", unit="deg", array=decep0),
        fits.Column(name="EQUINOX", format="E", unit="yr", array=equinox),
        fits.Column(name="RA_ERR", format="D", unit="deg", array=ra_err),
        fits.Column(name="DEC_ERR", format="D", unit="deg", array=dec_err),
        fits.Column(name="SYSVEL", format="D", unit="m/s", array=sysvel),
        fits.Column(name="VELTYP", format="8A", array=veltyp),
        fits.Column(name="VELDEF", format="8A", array=veldef),
        fits.Column(name="PMRA", format="D", unit="deg/yr", array=pmra),
        fits.Column(name="PMDEC", format="D", unit="deg/yr", array=pmdec),
        fits.Column(
            name="PMRA_ERR",
            format="D",
            unit="deg/yr",
            array=pmra_err,
        ),
        fits.Column(
            name="PMDEC_ERR",
            format="D",
            unit="deg/yr",
            array=pmdec_err,
        ),
        fits.Column(
            name="PARALLAX",
            format="E",
            unit="deg",
            array=parallax,
        ),
        fits.Column(
            name="PARA_ERR",
            format="E",
            unit="deg",
            array=para_err,
        ),
        fits.Column(name="SPECTYP", format="32A", array=spectyp),
        fits.Column(name="CATEGORY", format="3A", array=category),
    ]

    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["EXTNAME"] = "OI_TARGET"
    hdu.header["OI_REVN"] = (2, "OIFITS revision number")
    return hdu


def _build_oi_array(
    stations: list[Station],
    arrname: str,
) -> fits.BinTableHDU:
    """Build OI_ARRAY table for a mask."""
    n_sta = len(stations)
    tel_name = np.array(["LBT_SX"] * n_sta)
    sta_name = np.array([s.name for s in stations])
    sta_index = np.array([s.index for s in stations], dtype=np.int16)
    diameter = np.array([s.diameter for s in stations], dtype=np.float32)
    staxyz = np.array([[s.x, s.y, s.z] for s in stations], dtype=np.float64)
    fov = np.zeros(n_sta, dtype=np.float64)
    fovtype = np.array(["FWHM"] * n_sta)

    cols = [
        fits.Column(name="TEL_NAME", format="16A", array=tel_name),
        fits.Column(name="STA_NAME", format="16A", array=sta_name),
        fits.Column(name="STA_INDEX", format="I", array=sta_index),
        fits.Column(
            name="DIAMETER",
            format="E",
            unit="m",
            array=diameter,
        ),
        fits.Column(name="STAXYZ", format="3D", unit="m", array=staxyz),
        fits.Column(name="FOV", format="D", unit="arcsec", array=fov),
        fits.Column(name="FOVTYPE", format="6A", array=fovtype),
    ]

    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["EXTNAME"] = "OI_ARRAY"
    hdu.header["OI_REVN"] = (2, "OIFITS revision number")
    hdu.header["ARRNAME"] = (arrname, "Array name")
    hdu.header["FRAME"] = ("SKY", "Coordinate frame")
    hdu.header["ARRAYX"] = (0.0, "Array center X [m]")
    hdu.header["ARRAYY"] = (0.0, "Array center Y [m]")
    hdu.header["ARRAYZ"] = (0.0, "Array center Z [m]")
    return hdu


def _build_oi_wavelength_tables(
    observables_list: list[Observables],
    base_insname: str,
) -> list[fits.BinTableHDU]:
    """Build OI_WAVELENGTH tables for unique wavelength grids."""
    grids: dict[str, tuple[np.ndarray, str]] = {}
    counter = 0

    for obs in observables_list:
        key = _wavelength_grid_key(obs.wavelengths)
        if key not in grids:
            if counter == 0:
                ins = base_insname
            else:
                ins = f"{base_insname}_{counter}"
            grids[key] = (obs.wavelengths, ins)
            counter += 1

    tables = []
    for _key, (wavelengths, ins) in grids.items():
        eff_wave = wavelengths * 1e-6

        if len(wavelengths) > 1:
            diffs = np.diff(wavelengths)
            eff_band = np.empty_like(wavelengths)
            eff_band[0] = diffs[0]
            eff_band[-1] = diffs[-1]
            eff_band[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
            eff_band = eff_band * 1e-6
        else:
            eff_band = np.zeros_like(eff_wave)

        cols = [
            fits.Column(
                name="EFF_WAVE",
                format="E",
                unit="m",
                array=eff_wave.astype(np.float32),
            ),
            fits.Column(
                name="EFF_BAND",
                format="E",
                unit="m",
                array=eff_band.astype(np.float32),
            ),
        ]

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["EXTNAME"] = "OI_WAVELENGTH"
        hdu.header["OI_REVN"] = (2, "OIFITS revision number")
        hdu.header["INSNAME"] = (ins, "Instrument name")
        tables.append(hdu)

    return tables


def _build_oi_vis2(
    observables_list: list[Observables],
    target_id_map: dict[str, int],
    arrname: str,
    insname_map: dict[int, str],
) -> list[fits.BinTableHDU]:
    """Build OI_VIS2 tables, one per INSNAME."""
    if not any(obs.has_vis2 for obs in observables_list):
        return []

    # Group rows by insname
    grouped: dict[str, list[dict]] = {}
    for obs_idx, obs in enumerate(observables_list):
        if not obs.has_vis2:
            continue
        target_id = target_id_map[obs.target]
        mjd = _get_mjd_or_fallback(obs)
        ins = insname_map[obs_idx]
        n_wav = obs.n_wav

        if ins not in grouped:
            grouped[ins] = []

        for i, bl in enumerate(obs.baselines):
            row = {
                "target_id": target_id,
                "time": 0.0,
                "mjd": mjd,
                "int_time": 0.0,
                "vis2data": obs.vis2[i, :],
                "vis2err": (
                    obs.vis2_err[i, :]
                    if obs.vis2_err is not None
                    else np.full(n_wav, np.nan)
                ),
                "ucoord": bl.u,
                "vcoord": bl.v,
                "sta_index": np.array(bl.sta_index, dtype=np.int16),
                "flag": (
                    obs.vis2_flag[i, :]
                    if obs.vis2_flag is not None
                    else np.zeros(n_wav, dtype=bool)
                ),
            }
            grouped[ins].append(row)

    hdus = []
    for ins, rows in grouped.items():
        n_wav = len(rows[0]["vis2data"])
        target_id_arr = np.array(
            [r["target_id"] for r in rows], dtype=np.int16
        )
        time_arr = np.array([r["time"] for r in rows], dtype=np.float64)
        mjd_arr = np.array([r["mjd"] for r in rows], dtype=np.float64)
        int_time_arr = np.array(
            [r["int_time"] for r in rows], dtype=np.float64
        )
        vis2data_arr = np.array(
            [r["vis2data"] for r in rows], dtype=np.float64
        )
        vis2err_arr = np.array([r["vis2err"] for r in rows], dtype=np.float64)
        ucoord_arr = np.array([r["ucoord"] for r in rows], dtype=np.float64)
        vcoord_arr = np.array([r["vcoord"] for r in rows], dtype=np.float64)
        sta_index_arr = np.array(
            [r["sta_index"] for r in rows], dtype=np.int16
        )
        flag_arr = np.array([r["flag"] for r in rows], dtype=bool)

        cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=target_id_arr,
            ),
            fits.Column(
                name="TIME",
                format="D",
                unit="s",
                array=time_arr,
            ),
            fits.Column(
                name="MJD",
                format="D",
                unit="day",
                array=mjd_arr,
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                unit="s",
                array=int_time_arr,
            ),
            fits.Column(
                name="VIS2DATA",
                format=f"{n_wav}D",
                array=vis2data_arr,
            ),
            fits.Column(
                name="VIS2ERR",
                format=f"{n_wav}D",
                array=vis2err_arr,
            ),
            fits.Column(
                name="UCOORD",
                format="D",
                unit="m",
                array=ucoord_arr,
            ),
            fits.Column(
                name="VCOORD",
                format="D",
                unit="m",
                array=vcoord_arr,
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=sta_index_arr,
            ),
            fits.Column(
                name="FLAG",
                format=f"{n_wav}L",
                array=flag_arr,
            ),
        ]

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["EXTNAME"] = "OI_VIS2"
        hdu.header["OI_REVN"] = (2, "OIFITS revision number")
        hdu.header["DATE-OBS"] = _get_date_obs(observables_list)
        hdu.header["ARRNAME"] = (arrname, "Array name")
        hdu.header["INSNAME"] = (ins, "Instrument name")
        hdus.append(hdu)

    return hdus


def _build_oi_vis(
    observables_list: list[Observables],
    target_id_map: dict[str, int],
    arrname: str,
    insname_map: dict[int, str],
) -> list[fits.BinTableHDU]:
    """Build OI_VIS tables, one per INSNAME."""
    if not any(obs.has_visamp or obs.has_visphi for obs in observables_list):
        return []

    grouped: dict[str, list[dict]] = {}
    for obs_idx, obs in enumerate(observables_list):
        if not (obs.has_visamp or obs.has_visphi):
            continue
        target_id = target_id_map[obs.target]
        mjd = _get_mjd_or_fallback(obs)
        ins = insname_map[obs_idx]
        n_wav = obs.n_wav

        if ins not in grouped:
            grouped[ins] = []

        for i, bl in enumerate(obs.baselines):
            row = {
                "target_id": target_id,
                "time": 0.0,
                "mjd": mjd,
                "int_time": 0.0,
                "visamp": (
                    obs.visamp[i, :]
                    if obs.visamp is not None
                    else np.full(n_wav, np.nan)
                ),
                "visamperr": (
                    obs.visamp_err[i, :]
                    if obs.visamp_err is not None
                    else np.full(n_wav, np.nan)
                ),
                "visphi": (
                    obs.visphi[i, :]
                    if obs.visphi is not None
                    else np.full(n_wav, np.nan)
                ),
                "visphierr": (
                    obs.visphi_err[i, :]
                    if obs.visphi_err is not None
                    else np.full(n_wav, np.nan)
                ),
                "ucoord": bl.u,
                "vcoord": bl.v,
                "sta_index": np.array(bl.sta_index, dtype=np.int16),
                "flag": (
                    obs.vis_flag[i, :]
                    if obs.vis_flag is not None
                    else np.zeros(n_wav, dtype=bool)
                ),
            }
            grouped[ins].append(row)

    hdus = []
    for ins, rows in grouped.items():
        n_wav = len(rows[0]["visamp"])
        target_id_arr = np.array(
            [r["target_id"] for r in rows], dtype=np.int16
        )
        time_arr = np.array([r["time"] for r in rows], dtype=np.float64)
        mjd_arr = np.array([r["mjd"] for r in rows], dtype=np.float64)
        int_time_arr = np.array(
            [r["int_time"] for r in rows], dtype=np.float64
        )
        visamp_arr = np.array([r["visamp"] for r in rows], dtype=np.float64)
        visamperr_arr = np.array(
            [r["visamperr"] for r in rows], dtype=np.float64
        )
        visphi_arr = np.array([r["visphi"] for r in rows], dtype=np.float64)
        visphierr_arr = np.array(
            [r["visphierr"] for r in rows], dtype=np.float64
        )
        ucoord_arr = np.array([r["ucoord"] for r in rows], dtype=np.float64)
        vcoord_arr = np.array([r["vcoord"] for r in rows], dtype=np.float64)
        sta_index_arr = np.array(
            [r["sta_index"] for r in rows], dtype=np.int16
        )
        flag_arr = np.array([r["flag"] for r in rows], dtype=bool)

        cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=target_id_arr,
            ),
            fits.Column(
                name="TIME",
                format="D",
                unit="s",
                array=time_arr,
            ),
            fits.Column(
                name="MJD",
                format="D",
                unit="day",
                array=mjd_arr,
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                unit="s",
                array=int_time_arr,
            ),
            fits.Column(
                name="VISAMP",
                format=f"{n_wav}D",
                array=visamp_arr,
            ),
            fits.Column(
                name="VISAMPERR",
                format=f"{n_wav}D",
                array=visamperr_arr,
            ),
            fits.Column(
                name="VISPHI",
                format=f"{n_wav}D",
                unit="deg",
                array=visphi_arr,
            ),
            fits.Column(
                name="VISPHIERR",
                format=f"{n_wav}D",
                unit="deg",
                array=visphierr_arr,
            ),
            fits.Column(
                name="UCOORD",
                format="D",
                unit="m",
                array=ucoord_arr,
            ),
            fits.Column(
                name="VCOORD",
                format="D",
                unit="m",
                array=vcoord_arr,
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=sta_index_arr,
            ),
            fits.Column(
                name="FLAG",
                format=f"{n_wav}L",
                array=flag_arr,
            ),
        ]

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["EXTNAME"] = "OI_VIS"
        hdu.header["OI_REVN"] = (2, "OIFITS revision number")
        hdu.header["DATE-OBS"] = _get_date_obs(observables_list)
        hdu.header["ARRNAME"] = (arrname, "Array name")
        hdu.header["INSNAME"] = (ins, "Instrument name")
        hdu.header["AMPTYP"] = ("absolute", "Amplitude type")
        hdu.header["PHITYP"] = ("absolute", "Phase type")
        hdus.append(hdu)

    return hdus


def _build_oi_t3(
    observables_list: list[Observables],
    target_id_map: dict[str, int],
    arrname: str,
    insname_map: dict[int, str],
) -> list[fits.BinTableHDU]:
    """Build OI_T3 tables, one per INSNAME."""
    if not any(obs.has_t3phi or obs.has_t3amp for obs in observables_list):
        return []

    grouped: dict[str, list[dict]] = {}
    for obs_idx, obs in enumerate(observables_list):
        if not (obs.has_t3phi or obs.has_t3amp):
            continue
        target_id = target_id_map[obs.target]
        mjd = _get_mjd_or_fallback(obs)
        ins = insname_map[obs_idx]
        n_wav = obs.n_wav

        if ins not in grouped:
            grouped[ins] = []

        for i, tri in enumerate(obs.triangles):
            row = {
                "target_id": target_id,
                "time": 0.0,
                "mjd": mjd,
                "int_time": 0.0,
                "t3amp": (
                    obs.t3amp[i, :]
                    if obs.t3amp is not None
                    else np.full(n_wav, np.nan)
                ),
                "t3amperr": (
                    obs.t3amp_err[i, :]
                    if obs.t3amp_err is not None
                    else np.full(n_wav, np.nan)
                ),
                "t3phi": (
                    obs.t3phi[i, :]
                    if obs.t3phi is not None
                    else np.full(n_wav, np.nan)
                ),
                "t3phierr": (
                    obs.t3phi_err[i, :]
                    if obs.t3phi_err is not None
                    else np.full(n_wav, np.nan)
                ),
                "u1coord": tri.u1,
                "v1coord": tri.v1,
                "u2coord": tri.u2,
                "v2coord": tri.v2,
                "sta_index": np.array(tri.sta_index, dtype=np.int16),
                "flag": (
                    obs.t3_flag[i, :]
                    if obs.t3_flag is not None
                    else np.zeros(n_wav, dtype=bool)
                ),
            }
            grouped[ins].append(row)

    hdus = []
    for ins, rows in grouped.items():
        n_wav = len(rows[0]["t3phi"])
        target_id_arr = np.array(
            [r["target_id"] for r in rows], dtype=np.int16
        )
        time_arr = np.array([r["time"] for r in rows], dtype=np.float64)
        mjd_arr = np.array([r["mjd"] for r in rows], dtype=np.float64)
        int_time_arr = np.array(
            [r["int_time"] for r in rows], dtype=np.float64
        )
        t3amp_arr = np.array([r["t3amp"] for r in rows], dtype=np.float64)
        t3amperr_arr = np.array(
            [r["t3amperr"] for r in rows], dtype=np.float64
        )
        t3phi_arr = np.array([r["t3phi"] for r in rows], dtype=np.float64)
        t3phierr_arr = np.array(
            [r["t3phierr"] for r in rows], dtype=np.float64
        )
        u1coord_arr = np.array([r["u1coord"] for r in rows], dtype=np.float64)
        v1coord_arr = np.array([r["v1coord"] for r in rows], dtype=np.float64)
        u2coord_arr = np.array([r["u2coord"] for r in rows], dtype=np.float64)
        v2coord_arr = np.array([r["v2coord"] for r in rows], dtype=np.float64)
        sta_index_arr = np.array(
            [r["sta_index"] for r in rows], dtype=np.int16
        )
        flag_arr = np.array([r["flag"] for r in rows], dtype=bool)

        cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=target_id_arr,
            ),
            fits.Column(
                name="TIME",
                format="D",
                unit="s",
                array=time_arr,
            ),
            fits.Column(
                name="MJD",
                format="D",
                unit="day",
                array=mjd_arr,
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                unit="s",
                array=int_time_arr,
            ),
            fits.Column(
                name="T3AMP",
                format=f"{n_wav}D",
                array=t3amp_arr,
            ),
            fits.Column(
                name="T3AMPERR",
                format=f"{n_wav}D",
                array=t3amperr_arr,
            ),
            fits.Column(
                name="T3PHI",
                format=f"{n_wav}D",
                unit="deg",
                array=t3phi_arr,
            ),
            fits.Column(
                name="T3PHIERR",
                format=f"{n_wav}D",
                unit="deg",
                array=t3phierr_arr,
            ),
            fits.Column(
                name="U1COORD",
                format="D",
                unit="m",
                array=u1coord_arr,
            ),
            fits.Column(
                name="V1COORD",
                format="D",
                unit="m",
                array=v1coord_arr,
            ),
            fits.Column(
                name="U2COORD",
                format="D",
                unit="m",
                array=u2coord_arr,
            ),
            fits.Column(
                name="V2COORD",
                format="D",
                unit="m",
                array=v2coord_arr,
            ),
            fits.Column(
                name="STA_INDEX",
                format="3I",
                array=sta_index_arr,
            ),
            fits.Column(
                name="FLAG",
                format=f"{n_wav}L",
                array=flag_arr,
            ),
        ]

        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["EXTNAME"] = "OI_T3"
        hdu.header["OI_REVN"] = (2, "OIFITS revision number")
        hdu.header["DATE-OBS"] = _get_date_obs(observables_list)
        hdu.header["ARRNAME"] = (arrname, "Array name")
        hdu.header["INSNAME"] = (ins, "Instrument name")
        hdus.append(hdu)

    return hdus


def _sanitize_target_name(name: str) -> str:
    """Sanitize target name for use in filenames."""
    s = name.replace(" ", "_")
    s = re.sub(r"[^\w\-]", "", s)
    return s


def _format_time_for_filename(time_str: str) -> str:
    """Format time string HH:MM:SS.sss to HHMMSS for filename."""
    if not time_str:
        return "000000"
    parts = time_str.strip().split(":")
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        s = int(float(parts[2])) if len(parts) > 2 else 0
        return f"{h:02d}{m:02d}{s:02d}"
    except (ValueError, IndexError):
        return "000000"


def _compute_fallback_mjd(time_start: str) -> float:
    """Compute MJD using 2000-01-01 and time_start."""
    base_date = datetime.date(2000, 1, 1)
    if time_start:
        parts = time_start.strip().split(":")
        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            s = float(parts[2]) if len(parts) > 2 else 0.0
        except (ValueError, IndexError):
            h, m, s = 0, 0, 0.0
    else:
        h, m, s = 0, 0, 0.0

    whole_s = int(s)
    micro = int((s - whole_s) * 1e6)

    dt = datetime.datetime(
        base_date.year,
        base_date.month,
        base_date.day,
        h,
        m,
        whole_s,
        micro,
    )
    return float(Time(dt, scale="utc").mjd)


def _get_mjd_or_fallback(obs: Observables) -> float:
    """Get MJD from Observables, warning if fallback used."""
    if obs.mjd is not None:
        return obs.mjd
    warnings.warn(
        f"Observables for target '{obs.target}' has mjd=None. "
        f"Using fallback date 2000-01-01 with time_start="
        f"'{obs.time_start}'.",
        UserWarning,
        stacklevel=3,
    )
    return _compute_fallback_mjd(obs.time_start)


def _get_date_obs(
    observables_list: list[Observables],
) -> str:
    """Get DATE-OBS string from first obs."""
    obs0 = observables_list[0]
    if obs0.mjd is not None:
        t = Time(obs0.mjd, format="mjd", scale="utc")
        return t.datetime.strftime("%Y-%m-%d")
    return "2000-01-01"


def _get_header_value(header: fits.Header, key: str, default):
    """Get header value with default."""
    return header.get(key, default)


def _parse_oi_target(
    hdul: fits.HDUList,
) -> dict[int, str]:
    """Parse OI_TARGET table into target_id -> name mapping."""
    result = {}
    for hdu in hdul:
        if hasattr(hdu, "header") and hdu.header.get("EXTNAME") == "OI_TARGET":
            for row in hdu.data:
                tid = int(row["TARGET_ID"])
                name = str(row["TARGET"]).strip()
                result[tid] = name
            break
    return result


def _parse_oi_array(
    hdul: fits.HDUList,
) -> dict[str, list[Station]]:
    """Parse OI_ARRAY tables into arrname -> stations mapping."""
    result = {}
    for hdu in hdul:
        if hasattr(hdu, "header") and hdu.header.get("EXTNAME") == "OI_ARRAY":
            arrname = hdu.header.get("ARRNAME", "UNKNOWN")
            stations = []
            for row in hdu.data:
                xyz = row["STAXYZ"]
                stations.append(
                    Station(
                        index=int(row["STA_INDEX"]),
                        name=str(row["STA_NAME"]).strip(),
                        x=float(xyz[0]),
                        y=float(xyz[1]),
                        z=float(xyz[2]),
                        diameter=float(row["DIAMETER"]),
                    )
                )
            result[arrname] = stations
    return result


def _parse_oi_wavelength(
    hdul: fits.HDUList,
) -> dict[str, dict]:
    """Parse OI_WAVELENGTH tables into insname -> info."""
    result = {}
    for hdu in hdul:
        if (
            hasattr(hdu, "header")
            and hdu.header.get("EXTNAME") == "OI_WAVELENGTH"
        ):
            ins = hdu.header.get("INSNAME", "ALES_NRM")
            result[ins] = {
                "eff_wave": np.array(hdu.data["EFF_WAVE"], dtype=np.float64),
                "eff_band": np.array(hdu.data["EFF_BAND"], dtype=np.float64),
            }
    return result


def _collect_data_groups(
    hdul: fits.HDUList,
) -> dict[tuple[int, float], dict]:
    """Collect data rows grouped by (TARGET_ID, MJD)."""
    groups: dict[tuple[int, float], dict] = {}

    for hdu in hdul:
        extname = hdu.header.get("EXTNAME", "")
        arrname = hdu.header.get("ARRNAME", "")
        insname = hdu.header.get("INSNAME", "")

        if extname == "OI_VIS2":
            for row in hdu.data:
                key = (
                    int(row["TARGET_ID"]),
                    float(row["MJD"]),
                )
                if key not in groups:
                    groups[key] = _empty_group(arrname, insname)
                groups[key]["vis2_rows"].append(row)

        elif extname == "OI_VIS":
            for row in hdu.data:
                key = (
                    int(row["TARGET_ID"]),
                    float(row["MJD"]),
                )
                if key not in groups:
                    groups[key] = _empty_group(arrname, insname)
                groups[key]["vis_rows"].append(row)

        elif extname == "OI_T3":
            for row in hdu.data:
                key = (
                    int(row["TARGET_ID"]),
                    float(row["MJD"]),
                )
                if key not in groups:
                    groups[key] = _empty_group(arrname, insname)
                groups[key]["t3_rows"].append(row)

    return groups


def _empty_group(arrname: str, insname: str) -> dict:
    """Create an empty group data structure."""
    return {
        "vis2_rows": [],
        "vis_rows": [],
        "t3_rows": [],
        "arrname": arrname,
        "insname": insname,
        "time_start": "",
        "time_end": "",
    }


def _reconstruct_baselines(
    group_data: dict,
    stations: list[Station],
) -> list[BaselineInfo]:
    """Reconstruct BaselineInfo from vis2 or vis rows."""
    baselines = []
    seen = set()
    rows = group_data["vis2_rows"] or group_data["vis_rows"]
    sta_name_map = {s.index: s.name for s in stations}

    for row in rows:
        sta_idx = tuple(int(x) for x in row["STA_INDEX"])
        if sta_idx in seen:
            continue
        seen.add(sta_idx)
        name1 = sta_name_map.get(sta_idx[0], f"S{sta_idx[0]}")
        name2 = sta_name_map.get(sta_idx[1], f"S{sta_idx[1]}")
        baselines.append(
            BaselineInfo(
                sta_index=(sta_idx[0], sta_idx[1]),
                name=f"{name1}{name2}",
                u=float(row["UCOORD"]),
                v=float(row["VCOORD"]),
            )
        )
    return baselines


def _reconstruct_triangles(
    group_data: dict,
    stations: list[Station],
) -> list[TriangleInfo]:
    """Reconstruct TriangleInfo from t3 rows."""
    triangles = []
    seen = set()
    sta_name_map = {s.index: s.name for s in stations}

    for row in group_data["t3_rows"]:
        sta_idx = tuple(int(x) for x in row["STA_INDEX"])
        if sta_idx in seen:
            continue
        seen.add(sta_idx)
        names = [sta_name_map.get(idx, f"S{idx}") for idx in sta_idx]
        triangles.append(
            TriangleInfo(
                sta_index=(sta_idx[0], sta_idx[1], sta_idx[2]),
                name=f"{names[0]}-{names[1]}-{names[2]}",
                u1=float(row["U1COORD"]),
                v1=float(row["V1COORD"]),
                u2=float(row["U2COORD"]),
                v2=float(row["V2COORD"]),
            )
        )
    return triangles


def _extract_vis2_arrays(
    group_data: dict,
    n_baselines: int,
    n_wav: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract vis2 arrays from group data."""
    rows = group_data["vis2_rows"]
    if not rows:
        return None, None, None

    vis2 = np.empty((n_baselines, n_wav), dtype=np.float64)
    vis2_err = np.empty((n_baselines, n_wav), dtype=np.float64)
    vis2_flag = np.empty((n_baselines, n_wav), dtype=bool)

    for i, row in enumerate(rows):
        if i >= n_baselines:
            break
        vis2[i, :] = np.array(row["VIS2DATA"], dtype=np.float64)
        vis2_err[i, :] = np.array(row["VIS2ERR"], dtype=np.float64)
        vis2_flag[i, :] = np.array(row["FLAG"], dtype=bool)

    return vis2, vis2_err, vis2_flag


def _extract_t3_arrays(
    group_data: dict,
    n_triangles: int,
    n_wav: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract t3 arrays from group data."""
    rows = group_data["t3_rows"]
    if not rows:
        return None, None, None, None, None

    t3phi = np.empty((n_triangles, n_wav), dtype=np.float64)
    t3phi_err = np.empty((n_triangles, n_wav), dtype=np.float64)
    t3amp = np.empty((n_triangles, n_wav), dtype=np.float64)
    t3amp_err = np.empty((n_triangles, n_wav), dtype=np.float64)
    t3_flag = np.empty((n_triangles, n_wav), dtype=bool)

    for i, row in enumerate(rows):
        if i >= n_triangles:
            break
        t3phi[i, :] = np.array(row["T3PHI"], dtype=np.float64)
        t3phi_err[i, :] = np.array(row["T3PHIERR"], dtype=np.float64)
        t3amp[i, :] = np.array(row["T3AMP"], dtype=np.float64)
        t3amp_err[i, :] = np.array(row["T3AMPERR"], dtype=np.float64)
        t3_flag[i, :] = np.array(row["FLAG"], dtype=bool)

    return t3phi, t3phi_err, t3amp, t3amp_err, t3_flag


def _extract_vis_arrays(
    group_data: dict,
    n_baselines: int,
    n_wav: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Extract complex visibility arrays from group data."""
    rows = group_data["vis_rows"]
    if not rows:
        return None, None, None, None, None

    visamp = np.empty((n_baselines, n_wav), dtype=np.float64)
    visamp_err = np.empty((n_baselines, n_wav), dtype=np.float64)
    visphi = np.empty((n_baselines, n_wav), dtype=np.float64)
    visphi_err = np.empty((n_baselines, n_wav), dtype=np.float64)
    vis_flag = np.empty((n_baselines, n_wav), dtype=bool)

    for i, row in enumerate(rows):
        if i >= n_baselines:
            break
        visamp[i, :] = np.array(row["VISAMP"], dtype=np.float64)
        visamp_err[i, :] = np.array(row["VISAMPERR"], dtype=np.float64)
        visphi[i, :] = np.array(row["VISPHI"], dtype=np.float64)
        visphi_err[i, :] = np.array(row["VISPHIERR"], dtype=np.float64)
        vis_flag[i, :] = np.array(row["FLAG"], dtype=bool)

    return visamp, visamp_err, visphi, visphi_err, vis_flag


def _get_category_for_target(
    hdul: fits.HDUList,
    target_id: int,
) -> str:
    """Get CATEGORY for a target from OI_TARGET table."""
    for hdu in hdul:
        if hasattr(hdu, "header") and hdu.header.get("EXTNAME") == "OI_TARGET":
            for row in hdu.data:
                if int(row["TARGET_ID"]) == target_id:
                    cat = str(row["CATEGORY"]).strip()
                    return cat if cat else "SCI"
            break
    return "SCI"
