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
            before ``load()`` is called.
        wavelengths: 1D numpy array of wavelengths in microns after
            loading. ``None`` before ``load()`` is called.
        file_numbers: 1D integer numpy array of file numbers after
            loading. ``None`` before ``load()`` is called.
        headers: List of FITS headers after loading. ``None`` before
            ``load()`` is called.
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

    def summary(self) -> str:
        """Return a one-line summary of this block's state.

        Returns:
            String describing block type, target, file range,
            and loading state.
        """
        if self.file_range is not None:
            files = f"files {self.file_range[0]}–{self.file_range[1]}"
            expected = self.file_range[1] - self.file_range[0] + 1
            if self.is_loaded:
                load_info = f"{self.n_files}/{expected} files loaded/expected"
            else:
                load_info = f"{expected} files expected, not loaded"
        else:
            files = "all files"
            if self.is_loaded:
                load_info = f"{self.n_files} files loaded"
            else:
                load_info = "file count unknown, not loaded"

        return f"{self.block_type.value} '{self.target}' {files} ({load_info})"


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

    def __len__(self) -> int:
        """Return the number of blocks."""
        return len(self.blocks)

    def __getitem__(self, index: int) -> ObservingBlock:
        """Access a block by index."""
        return self.blocks[index]

    def __iter__(self):
        """Iterate over blocks in sequence order."""
        return iter(self.blocks)
