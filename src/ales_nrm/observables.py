"""Container for extracted interferometric observables and helpers."""

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.time import Time

from ales_nrm.utilities import rotate_points_2d

if TYPE_CHECKING:
    from ales_nrm.observation import ObservingBlock
from ales_nrm.nrm.mask import NRMMask


@dataclass(frozen=True)
class Station:
    """A telescope element / mask hole. Maps to one row in OI_ARRAY.

    Attributes:
        index: 1-based station index.
        name: Station label, e.g. "H1".
        x: Pupil-plane x-coordinate in meters (unrotated py parallactic
            angle).
        y: Pupil-plane y-coordinate in meters (unrotated py parallactic
            angle).
        z: Height coordinate in meters (0 for aperture masks).
        diameter: Element diameter in meters.
    """

    index: int
    name: str
    x: float
    y: float
    z: float = 0.0
    diameter: float = 0.0


@dataclass(frozen=True)
class BaselineInfo:
    """Baseline between two stations. Maps to one row in OI_VIS2/OI_VIS.

    u, v are spatial coordinates in meters, computed from hole
    separations rotated by the mean parallactic angle (astronomical
    convention: positive east of north). To convert to spatial
    frequencies (cycles/rad), divide by wavelength in meters.

    Attributes:
        sta_index: (sta1, sta2) 1-based station indices.
        name: Baseline label, e.g. "H1-H2".
        u: u-coordinate in meters, parallactic-angle-rotated.
        v: v-coordinate in meters, parallactic-angle-rotated.
    """

    sta_index: tuple[int, int]
    name: str
    u: float
    v: float


@dataclass(frozen=True)
class TriangleInfo:
    """A closing triangle of three stations. Maps to one row in OI_T3.

    u, v coordinates follow the same convention as BaselineInfo.

    Attributes:
        sta_index: (A, B, C) 1-based station indices.
        name: Triangle label, e.g. "H1-H2-H3".
        u1: u-coordinate of baseline AB in meters.
        v1: v-coordinate of baseline AB in meters.
        u2: u-coordinate of baseline BC in meters.
        v2: v-coordinate of baseline BC in meters.
    """

    sta_index: tuple[int, int, int]
    name: str
    u1: float
    v1: float
    u2: float
    v2: float


@dataclass
class Observables:
    """Container for extracted interferometric observables.

    Observable arrays are shaped ``(n_baselines, n_wav)`` or
    ``(n_triangles, n_wav)``.

    Attributes:
        target: Target name.
        wavelengths: 1D array of wavelengths in µm, shape ``(n_wav,)``.
        stations: Ordered list of Station objects (mask holes).
        baselines: Ordered list of BaselineInfo objects.
        triangles: Ordered list of TriangleInfo objects.
        mean_para_angle: Mean parallactic angle in degrees used for u,v.
        mjd: Mean MJD of the observing sequence (None if date
            unavailable).
        time_start: First UT timestamp string of the sequence.
        time_end: Last UT timestamp string of the sequence.
        vis2: Squared visibilities, shape ``(n_baselines, n_wav)``.
        vis2_err: Uncertainties on vis2.
        vis2_flag: Boolean flags for vis2.
        t3phi: Closure phases in degrees, shape
            ``(n_triangles, n_wav)``.
        t3phi_err: Uncertainties on closure phases.
        t3amp: Triple amplitudes, shape ``(n_triangles, n_wav)``.
        t3amp_err: Uncertainties on triple amplitudes.
        t3_flag: Boolean flags for t3 quantities.
        visamp: Visibility amplitudes (amplitude of complex visibility),
            shape ``(n_baselines, n_wav)``.
        visamp_err: Uncertainties on visibility amplitudes.
        visphi: Visibility phases (phase of complex visibility) in
            degrees, shape ``(n_baselines, n_wav)``.
        visphi_err: Uncertainties on visibility phases.
        vis_flag: Boolean flags for complex visibility quantities.
        calibrated: Whether observables are calibrated.
        calibrator_target: Name of calibrator used (if calibrated).
        extraction_backend: Name of extraction backend, e.g. "sampy".
        block_type: "SCI" or "CAL".
        mask_name: Name of the mask used.
        notes: Free-form notes string.
    """

    # Metadata
    target: str
    wavelengths: np.ndarray
    stations: list[Station]
    baselines: list[BaselineInfo]
    triangles: list[TriangleInfo]
    mean_para_angle: float

    # Timing
    mjd: float | None
    time_start: str
    time_end: str

    # Observables (None if not extracted)
    vis2: np.ndarray | None = None
    vis2_err: np.ndarray | None = None
    vis2_flag: np.ndarray | None = None

    t3phi: np.ndarray | None = None
    t3phi_err: np.ndarray | None = None
    t3amp: np.ndarray | None = None
    t3amp_err: np.ndarray | None = None
    t3_flag: np.ndarray | None = None

    visamp: np.ndarray | None = None
    visamp_err: np.ndarray | None = None
    visphi: np.ndarray | None = None
    visphi_err: np.ndarray | None = None
    vis_flag: np.ndarray | None = None

    # Calibration state
    calibrated: bool = False
    calibrator_target: str | None = None

    # Provenance
    extraction_backend: str = ""
    block_type: str = ""
    mask_name: str = ""
    notes: str = ""

    @property
    def n_wav(self) -> int:
        """Number of wavelength channels."""
        return len(self.wavelengths)

    @property
    def n_baselines(self) -> int:
        """Number of baselines."""
        return len(self.baselines)

    @property
    def n_triangles(self) -> int:
        """Number of closing triangles."""
        return len(self.triangles)

    @property
    def n_stations(self) -> int:
        """Number of stations (mask holes)."""
        return len(self.stations)

    @property
    def has_vis2(self) -> bool:
        """Whether squared visibilities are present."""
        return self.vis2 is not None

    @property
    def has_vis2_err(self) -> bool:
        """Whether squared visibility uncertainties are present."""
        return self.vis2_err is not None

    @property
    def has_t3phi(self) -> bool:
        """Whether closure phases are present."""
        return self.t3phi is not None

    @property
    def has_t3phi_err(self) -> bool:
        """Whether closure phase uncertainties are present."""
        return self.t3phi_err is not None

    @property
    def has_t3amp(self) -> bool:
        """Whether triple amplitudes are present."""
        return self.t3amp is not None

    @property
    def has_t3amp_err(self) -> bool:
        """Whether triple amplitude uncertainties are present."""
        return self.t3amp_err is not None

    @property
    def has_visamp(self) -> bool:
        """Whether visibility amplitudes are present."""
        return self.visamp is not None

    @property
    def has_visamp_err(self) -> bool:
        """Whether visibility amplitude uncertainties are present."""
        return self.visamp_err is not None

    @property
    def has_visphi(self) -> bool:
        """Whether visibility phases are present."""
        return self.visphi is not None

    @property
    def has_visphi_err(self) -> bool:
        """Whether visibility phase uncertainties are present."""
        return self.visphi_err is not None

    def validate(self) -> None:
        """Check internal consistency of shapes and index references.

        Raises:
            ValueError: If any inconsistency is detected.
        """
        n_wav = self.n_wav
        n_bl = self.n_baselines
        n_tri = self.n_triangles
        n_sta = self.n_stations

        # Check wavelengths
        if self.wavelengths.ndim != 1:
            raise ValueError(
                f"wavelengths must be 1D, got shape {self.wavelengths.shape}"
            )

        # Check vis2 shapes
        if self.vis2 is not None:
            if self.vis2.shape != (n_bl, n_wav):
                raise ValueError(
                    f"vis2 shape {self.vis2.shape} "
                    f"!= expected ({n_bl}, {n_wav})"
                )
        if self.vis2_err is not None:
            if self.vis2_err.shape != (n_bl, n_wav):
                raise ValueError(
                    f"vis2_err shape {self.vis2_err.shape} != "
                    f"expected ({n_bl}, {n_wav})"
                )
        if self.vis2_flag is not None:
            if self.vis2_flag.shape != (n_bl, n_wav):
                raise ValueError(
                    f"vis2_flag shape {self.vis2_flag.shape} != "
                    f"expected ({n_bl}, {n_wav})"
                )

        # Check T3 shapes
        for attr_name in (
            "t3phi",
            "t3phi_err",
            "t3amp",
            "t3amp_err",
            "t3_flag",
        ):
            arr = getattr(self, attr_name)
            if arr is not None and arr.shape != (n_tri, n_wav):
                raise ValueError(
                    f"{attr_name} shape {arr.shape} != "
                    f"expected ({n_tri}, {n_wav})"
                )

        # Check vis shapes
        for attr_name in (
            "visamp",
            "visamp_err",
            "visphi",
            "visphi_err",
            "vis_flag",
        ):
            arr = getattr(self, attr_name)
            if arr is not None and arr.shape != (n_bl, n_wav):
                raise ValueError(
                    f"{attr_name} shape {arr.shape} != "
                    f"expected ({n_bl}, {n_wav})"
                )

        # Check station indices are 1-based and contiguous
        sta_indices = [s.index for s in self.stations]
        if sorted(sta_indices) != list(range(1, n_sta + 1)):
            raise ValueError(
                f"Station indices must be 1..{n_sta}, got {sta_indices}"
            )

        # Check baseline station references
        valid_indices = set(sta_indices)
        for bl in self.baselines:
            for idx in bl.sta_index:
                if idx not in valid_indices:
                    raise ValueError(
                        f"Baseline {bl.name} references station {idx} "
                        f"not in stations list"
                    )

        # Check triangle station references
        for tri in self.triangles:
            for idx in tri.sta_index:
                if idx not in valid_indices:
                    raise ValueError(
                        f"Triangle {tri.name} references station {idx} "
                        f"not in stations list"
                    )

    def _arr_status(self, arr: np.ndarray | None) -> str:
        """Return a status string for an observable array."""
        if arr is None:
            return "✗"
        if np.all(np.isnan(arr)):
            return "✓ (no data)"
        return "✓"

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Observables: {self.target}",
            f"  Backend: {self.extraction_backend or 'unknown'}",
            f"  Block type: {self.block_type or 'unknown'}",
            f"  Mask: {self.mask_name or 'unknown'}",
            f"  Calibrated: {self.calibrated}",
            f"  Stations: {self.n_stations}",
            f"  Baselines: {self.n_baselines}",
            f"  Triangles: {self.n_triangles}",
            f"  Wavelengths: {self.n_wav} channels, "
            f"{self.wavelengths[0]:.4f}\u2013"
            f"{self.wavelengths[-1]:.4f} \u00b5m",
            f"  Mean para. angle: {self.mean_para_angle:.2f} deg",
            f"  MJD: {self.mjd}",
            f"  Time range: {self.time_start} \u2013 {self.time_end}",
            f"  vis2: {self._arr_status(self.vis2)}  |  "
            f"vis2_err: {self._arr_status(self.vis2_err)}",
            f"  t3phi: {self._arr_status(self.t3phi)}  |  "
            f"t3phi_err: {self._arr_status(self.t3phi_err)}",
            f"  t3amp: {self._arr_status(self.t3amp)}  |  "
            f"t3amp_err: {self._arr_status(self.t3amp_err)}",
            f"  visamp: {self._arr_status(self.visamp)}  |  "
            f"visamp_err: {self._arr_status(self.visamp_err)}",
            f"  visphi: {self._arr_status(self.visphi)}  |  "
            f"visphi_err: {self._arr_status(self.visphi_err)}",
        ]
        if self.calibrated and self.calibrator_target:
            lines.append(f"  Calibrator: {self.calibrator_target}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)

    @classmethod
    def from_block_and_mask(
        cls,
        block: "ObservingBlock",
        mask: "NRMMask",
    ) -> "Observables":
        """Create Observables with geometry and timing populated.

        Computes stations, baselines (with u,v rotated by mean
        parallactic angle), and triangles from the mask. Computes timing
        from the block. Observable arrays are left as None.

        Args:
            block: The ObservingBlock providing timing, target, and
                metadata.
            mask: The NRMMask providing hole geometry and closing
                triangles.

        Returns:
            An Observables instance with geometry and timing filled in,
            ready to receive observable arrays.
        """
        mean_para_angle = float(np.mean(block.parallactic_angles))

        # Build stations from Hole objects
        stations = [
            Station(
                index=i + 1,
                name=hole.name,
                x=hole.x,
                y=hole.y,
                diameter=hole.radius * 2,
            )
            for i, hole in enumerate(mask.holes)
        ]

        # Hole name -> 1-based index lookup
        hole_name_to_index = {
            hole.name: i + 1 for i, hole in enumerate(mask.holes)
        }

        # Compute rotated baseline u,v from Baseline objects
        baseline_vectors = np.array([[bl.bx, bl.by] for bl in mask.baselines])

        # Rotate by mean parallactic angle
        rotated_bl = rotate_points_2d(
            points=baseline_vectors,
            center=(0.0, 0.0),
            angle_deg=mean_para_angle,
        )

        # Apply parity flip when converting from pupil-plane x to
        # sky u.
        # If this flip is not applied, the source plane on-sky would be
        # mirrored along the vertical axis (North-South), thus East and
        # West would be swapped.
        rotated_bl[:, 0] *= -1.0

        baselines = [
            BaselineInfo(
                sta_index=(
                    hole_name_to_index[bl.hole1],
                    hole_name_to_index[bl.hole2],
                ),
                name=bl.name,
                u=float(rotated_bl[k, 0]),
                v=float(rotated_bl[k, 1]),
            )
            for k, bl in enumerate(mask.baselines)
        ]

        # Baseline name -> (u, v, sta_index) lookup
        bl_name_to_info = {
            bl.name: (float(rotated_bl[k, 0]), float(rotated_bl[k, 1]), bl)
            for k, bl in enumerate(mask.baselines)
        }

        # Build triangles
        # Returns list of (bl_name_ij, bl_name_jk, bl_name_ik)
        triangle_tuples = mask.get_closing_triangles()

        triangles = []
        for bl_ij_name, bl_jk_name, _bl_ik_name in triangle_tuples:
            u_ij, v_ij, bl_ij = bl_name_to_info[bl_ij_name]
            u_jk, v_jk, bl_jk = bl_name_to_info[bl_jk_name]

            # Stations: i, j, k
            sta_i = hole_name_to_index[bl_ij.hole1]
            sta_j = hole_name_to_index[bl_ij.hole2]
            sta_k = hole_name_to_index[bl_jk.hole2]

            triangles.append(
                TriangleInfo(
                    sta_index=(sta_i, sta_j, sta_k),
                    name=f"{bl_ij.hole1}-{bl_ij.hole2}-{bl_jk.hole2}",
                    u1=u_ij,
                    v1=v_ij,
                    u2=u_jk,
                    v2=v_jk,
                )
            )

        mjd = _compute_mean_mjd(block.observation_date, block.timestamps)
        time_start = str(block.timestamps[0])
        time_end = str(block.timestamps[-1])

        block_type = block.block_type if hasattr(block, "block_type") else ""

        return cls(
            target=block.target,
            wavelengths=block.wavelengths.copy(),
            stations=stations,
            baselines=baselines,
            triangles=triangles,
            mean_para_angle=mean_para_angle,
            mjd=mjd,
            time_start=time_start,
            time_end=time_end,
            block_type=block_type,
            mask_name=mask.source_name,
        )


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
