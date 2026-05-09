"""Observing block and observation sequence management.

This module provides data structures for organizing ALES NRM
observations into observing blocks and calibration sequences. An
ObservingBlock represents a contiguous set of files from a single target
and configuration. An ObservingSequence groups multiple blocks into a
calibration sequence such as CAL-SCI-CAL.
"""

import enum
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from ales_nrm.io.read_fits import read_cubes

logger = logging.getLogger(__name__)


class BlockType(enum.Enum):
    """Classification of an observing block.

    Attributes:
        SCI: Science target observation.
        CAL: Calibrator star observation.
    """

    SCI = "SCI"
    CAL = "CAL"


@dataclass
class ObservingBlock:
    """A contiguous set of ALES NRM observations (frames) of one target.

    It holds metadata describing the observations and, after loading,
    the data themselves.

    Attributes:
        block_type: Whether this is a science or calibrator block.
        target: Target name (e.g., ``'eps Hya'``, ``'FX CnC'``).
        directory: Path to the directory containing the FITS files.
        file_range: Optional tuple ``(start, end)`` of file numbers
            (inclusive). If ``None``, all matching files in the
            directory are loaded.
        prefix: Filename prefix for FITS file matching. Default is
            ``'cube_lm_'``.
        cubes: 4D numpy array of shape
            ``(n_files, n_wavelengths, ny, nx)`` after loading. ``None``
            before ``load()`` is called. After stacking, contains the
            stacked cubes.
        wavelengths: 1D numpy array of wavelengths in microns after
            loading. ``None`` before ``load()`` is called.
        file_numbers: 1D integer numpy array of file numbers after
            loading. ``None`` before ``load()`` is called. After
            stacking, contains the first file number of each stacking
            group as a representative identifier.
        headers: List of FITS headers after loading. ``None`` before
            ``load()`` is called. After stacking, contains the header of
            the first file in each stacking group.
        is_stacked: Whether the data has been stacked via
            ``stack_frames()``. Default is ``False``.
        stacking_groups: List of integer arrays after stacking, one per
            stacked cube, containing all file numbers that were combined
            into that cube. ``None`` before ``stack_frames()`` is
            called. Provides full traceability of which frames
            contributed to each stacked cube.
        stacking_method: The combination method used for stacking
            (``'median'`` or ``'mean'``). ``None`` before
            ``stack_frames()`` is called.
        complex_visibilities: 4D complex numpy array of shape
            ``(n_files, n_wavelengths, ny, nx)`` containing the complex
            Fourier transform for each frame and wavelength, with zero
            frequency at center. ``None`` before
            ``compute_complex_visibilities()`` is called.
        power_spectra: 4D numpy array of shape
            ``(n_files, n_wavelengths, ny, nx)`` containing the power
            spectrum (|FFT|^2) for each frame and wavelength, with zero
            frequency at center. ``None`` before computed.
    """

    block_type: BlockType
    target: str
    directory: str | Path
    file_range: tuple[int, int] | None = None
    prefix: str = "cube_lm_"
    cubes: np.ndarray | None = field(default=None, repr=False)
    wavelengths: np.ndarray | None = field(
        default=None,
        repr=False,
    )
    file_numbers: np.ndarray | None = field(
        default=None,
        repr=False,
    )
    headers: list[fits.Header] | None = field(
        default=None,
        repr=False,
    )
    is_stacked: bool = field(default=False, repr=True)
    stacking_groups: list[np.ndarray] | None = field(
        default=None,
        repr=False,
    )
    stacking_method: str | None = field(
        default=None,
        repr=False,
    )
    complex_visibilities: np.ndarray | None = field(
        default=None,
        repr=False,
    )
    power_spectra: np.ndarray | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self):
        """Validate inputs after initialization."""
        self.directory = Path(self.directory)

        if not isinstance(self.block_type, BlockType):
            self.block_type = BlockType(self.block_type)

        if self.file_range is not None:
            start, end = self.file_range
            if start > end:
                raise ValueError(
                    f"file_range start ({start}) must be <= end ({end})."
                )

    @property
    def n_files(self) -> int | None:
        """Number of loaded files, or None if not loaded."""
        if self.cubes is None:
            return None
        return self.cubes.shape[0]

    @property
    def is_loaded(self) -> bool:
        """Whether data has been loaded into this block."""
        return self.cubes is not None

    def _validate_loaded_files(self) -> None:
        """Check loaded files against the expected file range.

        Issues warnings if the number of loaded files does not match
        the expected count from ``file_range``, or if any expected
        file numbers are missing.

        This method is called automatically by ``load()`` and does
        nothing if ``file_range`` is ``None``.
        """
        if self.file_range is None:
            return

        start, end = self.file_range
        expected_count = end - start + 1
        expected_numbers = set(range(start, end + 1))
        loaded_numbers = set(self.file_numbers.tolist())

        if self.n_files != expected_count:
            warnings.warn(
                f"Block '{self.target}' "
                f"({self.block_type.value}): "
                f"expected {expected_count} files from "
                f"range {start}–{end}, but loaded "
                f"{self.n_files}.",
                stacklevel=3,
            )

        missing = expected_numbers - loaded_numbers
        if missing:
            missing_sorted = sorted(missing)
            warnings.warn(
                f"Block '{self.target}' "
                f"({self.block_type.value}): "
                f"missing file numbers: "
                f"{missing_sorted}.",
                stacklevel=3,
            )

    def load(self) -> None:
        """Load FITS cubes from disk into this block.

        Reads all files in ``directory`` matching the ``file_range`` and
        ``prefix``, and stores the results in ``cubes``,
        ``wavelengths``, ``file_numbers``, and ``headers``.

        If ``file_range`` is ``None``, all matching files in the
        directory are loaded.

        After loading, the file numbers and count are validated against
        ``file_range`` if it was specified. Warnings are issued for
        missing files or count mismatches.

        Raises:
            FileNotFoundError: If the directory does not exist or no
                matching files are found.
            ValueError: If cube dimensions or wavelength grids are
                inconsistent across files.
        """
        if self.file_range is not None:
            range_str = f"files {self.file_range[0]}–{self.file_range[1]}"
        else:
            range_str = "all files"

        logger.info(
            "Loading %s block '%s': %s from %s",
            self.block_type.value,
            self.target,
            range_str,
            self.directory,
        )

        (
            self.cubes,
            self.wavelengths,
            self.file_numbers,
            self.headers,
        ) = read_cubes(
            self.directory,
            file_range=self.file_range,
            prefix=self.prefix,
        )

        self._validate_loaded_files()

        logger.info(
            "Loaded %s block '%s': %d cubes, shape=%s.",
            self.block_type.value,
            self.target,
            self.n_files,
            self.cubes.shape,
        )

    def _resolve_stacking_groups(
        self,
        group_size: int | None = None,
        groups: list[list[int]] | None = None,
        remainder: str = "discard",
    ) -> list[np.ndarray]:
        """Build stacking groups from group_size or explicit groups.

        Args:
            group_size: Number of consecutive frames per group.
            groups: Explicit list of file number groups.
            remainder: How to handle leftover frames when using
                ``group_size``. ``'discard'`` (default) drops them,
                ``'keep'`` includes them as a smaller group, ``'add'``
                appends them to the last complete group.

        Returns:
            List of 1D integer arrays, each containing the file numbers
            for one stacking group.

        Raises:
            ValueError: If arguments are invalid or file numbers in
                ``groups`` are not found in the loaded data.
        """
        if (group_size is None) == (groups is None):
            raise ValueError(
                "Specify exactly one of 'group_size' or 'groups'."
            )

        if group_size is not None:
            if group_size < 1:
                raise ValueError(f"group_size must be >= 1, got {group_size}.")
            if remainder not in ("discard", "keep", "add"):
                raise ValueError(
                    f"remainder must be 'discard', 'keep', or "
                    f"'add', got '{remainder}'."
                )

            n = self.n_files
            n_complete = n // group_size
            leftover = n % group_size

            resolved = []
            for i in range(n_complete):
                start = i * group_size
                end = start + group_size
                resolved.append(self.file_numbers[start:end].copy())

            if leftover > 0:
                leftover_fnums = self.file_numbers[n_complete * group_size :]
                if remainder == "keep":
                    resolved.append(leftover_fnums.copy())
                    logger.info(
                        "Keeping remainder: last group has "
                        "%d frames (less than group_size=%d).",
                        leftover,
                        group_size,
                    )
                elif remainder == "add":
                    if not resolved:
                        resolved.append(leftover_fnums.copy())
                        logger.info(
                            "No complete groups; single group of %d frames.",
                            leftover,
                        )
                    else:
                        resolved[-1] = np.concatenate(
                            [resolved[-1], leftover_fnums]
                        )
                        logger.info(
                            "Added %d remainder frames to last "
                            "group (now %d frames).",
                            leftover,
                            len(resolved[-1]),
                        )
                else:
                    logger.info(
                        "Discarding %d remainder frames: %s.",
                        leftover,
                        leftover_fnums.tolist(),
                    )

            return resolved

        # Explicit groups: validate file numbers.
        loaded_set = set(self.file_numbers.tolist())
        resolved = []
        for group in groups:
            arr = np.array(group, dtype=int)
            missing = set(arr.tolist()) - loaded_set
            if missing:
                raise ValueError(
                    f"File numbers {sorted(missing)} in "
                    f"stacking group are not loaded in "
                    f"block '{self.target}'."
                )
            resolved.append(arr)
        return resolved

    def stack_frames(
        self,
        group_size: int | None = None,
        groups: list[list[int]] | None = None,
        method: str = "median",
        remainder: str = "discard",
        center: bool = False,
        center_kwargs: dict | None = None,
    ) -> None:
        """Stack frames within this block, replacing loaded data.

        Combines multiple frames into fewer stacked cubes. The stacked
        data replaces the original data in ``cubes``, ``file_numbers``,
        and ``headers``. The attribute ``stacking_groups`` records which
        file numbers were combined into each resulting cube.

        Frames can be grouped either by specifying a fixed
        ``group_size`` (stack every N consecutive frames) or by
        providing explicit ``groups`` of file numbers.

        Args:
            group_size: Number of consecutive frames per stacking group.
                Cannot be used together with ``groups``.
            groups: List of lists, where each inner list contains file
                numbers to stack together. Cannot be used together with
                ``group_size``.
            method: Combination method. ``'median'`` (default) or
                ``'mean'``.
            remainder: How to handle leftover frames when the total
                number of frames is not evenly divisible by
                ``group_size``. ``'discard'`` (default) drops leftover
                frames. ``'keep'`` stacks them into a smaller final
                group. ``'add'`` appends them to the last complete
                group. Ignored when using explicit ``groups``.
            center: If ``True``, center frames before stacking. Each
                wavelength slice is centered independently to account
                for ALES chromatic PSF position shifts.
            center_kwargs: Optional dictionary of keyword arguments
                passed to the centering function. Supported keys include
                ``n_wave_sum`` (number of adjacent wavelength slices to
                sum before centroiding, default 1) and ``cutout_size``
                (side length of the Gaussian fitting cutout, default 5).
                Ignored if ``center=False``.

        Raises:
            RuntimeError: If data has not been loaded, or if data has
                already been stacked.
            ValueError: If arguments are invalid, if ``method`` is not
                recognized, or if file numbers in ``groups`` are not
                found in the loaded data.
            NotImplementedError: If ``center=True``.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"Block '{self.target}' "
                f"({self.block_type.value}) has not been "
                f"loaded. Call load() first."
            )

        if self.is_stacked:
            raise RuntimeError(
                f"Block '{self.target}' "
                f"({self.block_type.value}) has already "
                f"been stacked. Reload data to re-stack."
            )

        if center:
            from ales_nrm.centering import center_cubes

            if center_kwargs is None:
                center_kwargs = {}

            logger.info(
                "Centering %d frames before stacking (center_kwargs=%s).",
                self.cubes.shape[0],
                center_kwargs,
            )
            self.cubes, centering_shifts = center_cubes(
                self.cubes,
                **center_kwargs,
            )
            logger.info(
                "Centering complete. Mean abs shift: dy=%.3f, dx=%.3f.",
                np.mean(np.abs(centering_shifts[:, :, 0])),
                np.mean(np.abs(centering_shifts[:, :, 1])),
            )

        if method not in ("median", "mean"):
            raise ValueError(
                f"Unknown stacking method '{method}'. Use 'median' or 'mean'."
            )

        combine_func = np.median if method == "median" else np.mean

        resolved_groups = self._resolve_stacking_groups(
            group_size=group_size,
            groups=groups,
            remainder=remainder,
        )

        if not resolved_groups:
            raise ValueError(
                "No stacking groups could be formed. "
                "Check group_size relative to the number "
                "of loaded frames."
            )

        # Build index mapping from file number to cube index.
        fnum_to_idx = {int(fn): i for i, fn in enumerate(self.file_numbers)}

        n_groups = len(resolved_groups)
        cube_shape = self.cubes.shape[1:]
        stacked = np.empty(
            (n_groups, *cube_shape),
            dtype=self.cubes.dtype,
        )
        new_file_numbers = np.empty(n_groups, dtype=int)
        new_headers = []
        stacking_groups = []

        for g, group_fnums in enumerate(resolved_groups):
            indices = np.array([fnum_to_idx[int(fn)] for fn in group_fnums])
            stacked[g] = combine_func(
                self.cubes[indices],
                axis=0,
            )
            new_file_numbers[g] = group_fnums[0]
            new_headers.append(self.headers[indices[0]])
            stacking_groups.append(group_fnums.copy())

        n_original = self.cubes.shape[0]
        self.cubes = stacked
        self.file_numbers = new_file_numbers
        self.headers = new_headers
        self.is_stacked = True
        self.stacking_groups = stacking_groups
        self.stacking_method = method

        logger.info(
            "Stacked %s block '%s': %d frames -> "
            "%d stacked cubes (method='%s', remainder='%s').",
            self.block_type.value,
            self.target,
            n_original,
            n_groups,
            method,
            remainder,
        )

    def summary(self) -> str:
        """Return a one-line summary of this block's state.

        Returns:
            String describing block type, target, and current
            data state including loading and stacking info.
        """
        parts = [f"{self.block_type.value} '{self.target}'"]

        if not self.is_loaded:
            if self.file_range is not None:
                expected = self.file_range[1] - self.file_range[0] + 1
                parts.append(
                    f"files {self.file_range[0]}"
                    f"–{self.file_range[1]}"
                    f" ({expected} files expected,"
                    f" not loaded)"
                )
            else:
                parts.append("all files (file count unknown, not loaded)")
            return " ".join(parts)

        if not self.is_stacked:
            if self.file_range is not None:
                expected = self.file_range[1] - self.file_range[0] + 1
                parts.append(
                    f"{self.n_files}/{expected} files loaded/expected"
                )
            else:
                parts.append(f"{self.n_files} files loaded")
            return " ".join(parts)

        # Stacked state.
        group_sizes = [len(g) for g in self.stacking_groups]
        if len(set(group_sizes)) == 1:
            group_desc = (
                f"{len(group_sizes)} cubes from groups of {group_sizes[0]}"
            )
        else:
            group_desc = (
                f"{len(group_sizes)} cubes from groups of {group_sizes}"
            )
        parts.append(
            f"stacked ({group_desc}, method='{self.stacking_method}')"
        )
        return " ".join(parts)

    def compute_complex_visibilities(
        self,
        compute_power: bool = True,
    ) -> None:
        """Compute complex visibilities for all frames.

        Computes the Fourier transform of each 2D wavelength
        slice with pre- and post-fftshift for correct phase
        reference at image center.

        Args:
            compute_power: If True (default), also compute
                and store the power spectrum as |FFT|^2.

        Raises:
            RuntimeError: If data has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"Block '{self.target}' "
                f"({self.block_type.value}) has not been "
                f"loaded. Call load() first."
            )

        n_files, n_wav, ny, nx = self.cubes.shape
        self.complex_visibilities = np.empty_like(
            self.cubes, dtype=np.complex128
        )

        # Avoiding the loops and using the axes argument of fft2 and
        # fftshift brings no speed gain.
        for f in range(n_files):
            for w in range(n_wav):
                self.complex_visibilities[f, w] = np.fft.fftshift(
                    np.fft.fft2(np.fft.fftshift(self.cubes[f, w]))
                )

        logger.info(
            "Computed complex visibilities for %s block '%s': shape=%s.",
            self.block_type.value,
            self.target,
            self.complex_visibilities.shape,
        )

        if compute_power:
            self.power_spectra = np.abs(self.complex_visibilities) ** 2
            logger.info(
                "Computed power spectra for %s block '%s': shape=%s.",
                self.block_type.value,
                self.target,
                self.power_spectra.shape,
            )

    def compute_power_spectra(self) -> None:
        """Compute power spectra from complex visibilities.

        If complex visibilities have not been computed, calls
        ``compute_complex_visibilities()`` first.

        Raises:
            RuntimeError: If data has not been loaded.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"Block '{self.target}' "
                f"({self.block_type.value}) has not been "
                f"loaded. Call load() first."
            )

        if self.complex_visibilities is None:
            self.compute_complex_visibilities(compute_power=True)
        else:
            self.power_spectra = np.abs(self.complex_visibilities) ** 2
            logger.info(
                "Computed power spectra for %s block '%s': shape=%s.",
                self.block_type.value,
                self.target,
                self.power_spectra.shape,
            )


@dataclass
class ObservingSequence:
    """An ordered sequence of blocks forming a calibration sequence.

    An ObservingSequence groups science and calibrator blocks into a
    sequence such as CAL1-SCI-CAL2 or SCI-CAL-SCI. The ordering
    reflects the time sequence of the observations, which is important
    for calibration interpolation.

    Attributes:
        blocks: Ordered list of ObservingBlock objects.
        name: Optional descriptive name for this sequence
            (e.g., ``'eps Hya Nov 8 sequence 1'``).
    """

    blocks: list[ObservingBlock] = field(default_factory=list)
    name: str = ""

    @property
    def science_blocks(self) -> list[ObservingBlock]:
        """Return all science blocks in sequence order."""
        return [b for b in self.blocks if b.block_type == BlockType.SCI]

    @property
    def calibrator_blocks(self) -> list[ObservingBlock]:
        """Return all calibrator blocks in sequence order."""
        return [b for b in self.blocks if b.block_type == BlockType.CAL]

    @property
    def targets(self) -> list[str]:
        """Return unique target names in order of first appearance."""
        seen = set()
        result = []
        for block in self.blocks:
            if block.target not in seen:
                seen.add(block.target)
                result.append(block.target)
        return result

    @property
    def is_loaded(self) -> bool:
        """Whether all blocks have been loaded."""
        if not self.blocks:
            return False
        return all(b.is_loaded for b in self.blocks)

    def add_block(self, block: ObservingBlock) -> None:
        """Append a block to the observing sequence.

        Args:
            block: ObservingBlock to add.
        """
        self.blocks.append(block)
        logger.info(
            "Added %s block '%s' to sequence '%s'.",
            block.block_type.value,
            block.target,
            self.name,
        )

    def load_all(self) -> None:
        """Load data for all blocks that are not yet loaded."""
        logger.info(
            "Loading all %d blocks in sequence '%s'.",
            len(self.blocks),
            self.name,
        )
        for block in self.blocks:
            if not block.is_loaded:
                block.load()

    def get_blocks_by_target(
        self,
        target: str,
    ) -> list[ObservingBlock]:
        """Return all blocks matching a given target name.

        Args:
            target: Target name to filter by.

        Returns:
            List of matching blocks in sequence order.
        """
        return [b for b in self.blocks if b.target == target]

    def get_blocks_by_type(
        self,
        block_type: BlockType,
    ) -> list[ObservingBlock]:
        """Return all blocks matching a given type.

        Args:
            block_type: BlockType to filter by.

        Returns:
            List of matching blocks in sequence order.
        """
        if not isinstance(block_type, BlockType):
            block_type = BlockType(block_type)
        return [b for b in self.blocks if b.block_type == block_type]

    def summary(self) -> str:
        """Return a human-readable summary of the sequence.

        Returns:
            Multi-line string describing each block.
        """
        lines = [f"Sequence: {self.name or '(unnamed)'}"]
        lines.append(f"  Blocks: {len(self.blocks)}")
        lines.append(f"  Science blocks: {len(self.science_blocks)}")
        lines.append(f"  Calibrator blocks: {len(self.calibrator_blocks)}")
        lines.append(f"  Targets: {', '.join(self.targets)}")
        lines.append("")

        for i, block in enumerate(self.blocks):
            lines.append(f"  [{i}] {block.summary()}")

        return "\n".join(lines)

    def compute_all_complex_visibilities(
        self,
        compute_power: bool = True,
    ) -> None:
        """Compute complex visibilities for all loaded blocks.

        Args:
            compute_power: If True, also compute power spectra.
        """
        logger.info(
            "Computing complex visibilities for all %d "
            "blocks in sequence '%s'.",
            len(self.blocks),
            self.name,
        )
        for block in self.blocks:
            if not block.is_loaded:
                logger.warning(
                    "Skipping unloaded block '%s'.",
                    block.target,
                )
                continue
            if block.complex_visibilities is None:
                block.compute_complex_visibilities(compute_power=compute_power)

    def compute_all_power_spectra(self) -> None:
        """Compute power spectra for all loaded blocks.

        Skips blocks that already have power spectra or are not loaded.
        """
        logger.info(
            "Computing power spectra for all %d blocks in sequence '%s'.",
            len(self.blocks),
            self.name,
        )
        for block in self.blocks:
            if not block.is_loaded:
                logger.warning(
                    "Skipping unloaded block '%s'.",
                    block.target,
                )
                continue
            if block.power_spectra is None:
                block.compute_power_spectra()

    def __len__(self) -> int:
        """Return the number of blocks."""
        return len(self.blocks)

    def __getitem__(self, index: int) -> ObservingBlock:
        """Access a block by index."""
        return self.blocks[index]

    def __iter__(self):
        """Iterate over blocks in sequence order."""
        return iter(self.blocks)
