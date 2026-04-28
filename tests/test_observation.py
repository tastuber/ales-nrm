"""Tests for observing block and observation sequence management."""

import warnings

import numpy as np
import pytest

from ales_nrm.observation import (
    BlockType,
    ObservingBlock,
    ObservingSequence,
)
from tests.conftest import write_test_cube


@pytest.fixture()
def sample_directory(tmp_path, sample_cube, sample_wavelengths):
    """Create a directory with synthetic FITS cubes."""
    for num in range(5001, 5006):
        filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
        write_test_cube(
            filepath,
            sample_cube,
            sample_wavelengths,
        )
    return tmp_path


@pytest.fixture()
def sci_block(sample_directory):
    """Create a science block for testing."""
    return ObservingBlock(
        block_type=BlockType.SCI,
        target="test target",
        directory=sample_directory,
        file_range=(5001, 5003),
    )


@pytest.fixture()
def cal_block(sample_directory):
    """Create a calibrator block for testing."""
    return ObservingBlock(
        block_type=BlockType.CAL,
        target="test calibrator",
        directory=sample_directory,
        file_range=(5004, 5005),
    )


class TestBlockType:
    """Tests for the BlockType enum."""

    def test_sci_value(self):
        """Verify SCI enum value."""
        assert BlockType.SCI.value == "SCI"

    def test_cal_value(self):
        """Verify CAL enum value."""
        assert BlockType.CAL.value == "CAL"

    def test_construct_from_string(self):
        """Construct BlockType from its string value."""
        assert BlockType("SCI") == BlockType.SCI
        assert BlockType("CAL") == BlockType.CAL

    def test_invalid_string(self):
        """Raise ValueError for invalid block type string."""
        with pytest.raises(ValueError):
            BlockType("INVALID")


class TestObservingBlock:
    """Tests for the ObservingBlock dataclass."""

    def test_creation(self, sci_block):
        """Create a block with correct attributes."""
        assert sci_block.block_type == BlockType.SCI
        assert sci_block.target == "test target"
        assert sci_block.file_range == (5001, 5003)

    def test_creation_without_file_range(self, sample_directory):
        """Create a block without specifying file_range."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
        )
        assert block.file_range is None

    def test_block_type_from_string(self, sample_directory):
        """Accept a string and convert to BlockType enum."""
        block = ObservingBlock(
            block_type="SCI",
            target="test target",
            directory=sample_directory,
            file_range=(5001, 5003),
        )
        assert block.block_type == BlockType.SCI

    def test_invalid_file_range(self, sample_directory):
        """Raise ValueError if file_range start > end."""
        with pytest.raises(
            ValueError,
            match="file_range start",
        ):
            ObservingBlock(
                block_type=BlockType.SCI,
                target="test target",
                directory=sample_directory,
                file_range=(5005, 5001),
            )

    def test_not_loaded_initially(self, sci_block):
        """Block is not loaded before calling load()."""
        assert not sci_block.is_loaded
        assert sci_block.n_files is None
        assert sci_block.cubes is None

    def test_load_with_file_range(self, sci_block, sample_wavelengths):
        """Load data from FITS files into the block."""
        sci_block.load()

        assert sci_block.is_loaded
        assert sci_block.n_files == 3
        assert sci_block.cubes.ndim == 4
        assert sci_block.cubes.shape[0] == 3
        np.testing.assert_allclose(
            sci_block.wavelengths,
            sample_wavelengths,
        )
        assert len(sci_block.headers) == 3
        np.testing.assert_array_equal(
            sci_block.file_numbers,
            [5001, 5002, 5003],
        )

    def test_load_without_file_range(
        self, sample_directory, sample_wavelengths
    ):
        """Load all files when file_range is None."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
        )
        block.load()

        assert block.is_loaded
        assert block.n_files == 5
        np.testing.assert_array_equal(
            block.file_numbers,
            [5001, 5002, 5003, 5004, 5005],
        )

    def test_directory_converted_to_path(
        self,
        sample_directory,
    ):
        """Convert string directory to Path object."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=str(sample_directory),
            file_range=(5001, 5003),
        )
        assert isinstance(
            block.directory,
            type(sample_directory),
        )


class TestObservingBlockValidation:
    """Tests for file validation after loading."""

    def test_no_warning_when_all_files_present(self, sci_block):
        """No warning when loaded files match file_range."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sci_block.load()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_warning_on_missing_files(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn when fewer files found than expected."""
        for num in [5001, 5003]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(
                filepath,
                sample_cube,
                sample_wavelengths,
            )

        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=tmp_path,
            file_range=(5001, 5003),
        )

        with pytest.warns(UserWarning, match="expected 3"):
            block.load()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_warning_lists_missing_numbers(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn with specific missing file numbers."""
        for num in [5001, 5003]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(
                filepath,
                sample_cube,
                sample_wavelengths,
            )

        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=tmp_path,
            file_range=(5001, 5003),
        )

        with pytest.warns(UserWarning, match="missing file numbers"):
            block.load()

    def test_no_validation_without_file_range(self, sample_directory):
        """No validation warnings when file_range is None."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            block.load()


class TestObservingSequence:
    """Tests for the ObservingSequence container."""

    def test_empty_sequence(self):
        """Create an empty sequence."""
        seq = ObservingSequence(name="test sequence")
        assert len(seq) == 0
        assert seq.science_blocks == []
        assert seq.calibrator_blocks == []
        assert seq.targets == []
        assert not seq.is_loaded

    def test_construct_with_blocks(self, sci_block, cal_block):
        """Construct a sequence with a list of blocks."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block],
            name="test sequence",
        )
        assert len(seq) == 2

    def test_add_block(self, sci_block, cal_block):
        """Add blocks to a sequence incrementally."""
        seq = ObservingSequence(name="test sequence")
        seq.add_block(cal_block)
        seq.add_block(sci_block)
        assert len(seq) == 2

    def test_science_blocks(self, sci_block, cal_block):
        """Filter science blocks from the sequence."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block, cal_block],
        )
        sci_blocks = seq.science_blocks
        assert len(sci_blocks) == 1
        assert sci_blocks[0].target == "test target"

    def test_calibrator_blocks(self, sci_block, cal_block):
        """Filter calibrator blocks from the sequence."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block, cal_block],
        )
        cal_blocks = seq.calibrator_blocks
        assert len(cal_blocks) == 2
        assert all(b.target == "test calibrator" for b in cal_blocks)

    def test_targets(self, sci_block, cal_block):
        """Return unique targets in order of first appearance."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block, cal_block],
        )
        assert seq.targets == [
            "test calibrator",
            "test target",
        ]

    def test_get_blocks_by_target(
        self,
        sci_block,
        cal_block,
    ):
        """Filter blocks by target name."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block, cal_block],
        )
        result = seq.get_blocks_by_target("test calibrator")
        assert len(result) == 2

    def test_get_blocks_by_type(
        self,
        sci_block,
        cal_block,
    ):
        """Filter blocks by block type enum."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block, cal_block],
        )
        result = seq.get_blocks_by_type(BlockType.CAL)
        assert len(result) == 2

    def test_get_blocks_by_type_string(
        self,
        sci_block,
        cal_block,
    ):
        """Filter blocks by type using a string."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block, cal_block],
        )
        result = seq.get_blocks_by_type("SCI")
        assert len(result) == 1

    def test_is_loaded_false_when_not_loaded(
        self,
        sci_block,
        cal_block,
    ):
        """Report not loaded when blocks are unloaded."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block],
        )
        assert not seq.is_loaded

    def test_is_loaded_true_after_load_all(
        self,
        sci_block,
        cal_block,
    ):
        """Report loaded after load_all()."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block],
        )
        seq.load_all()
        assert seq.is_loaded

    def test_load_all(self, sci_block, cal_block):
        """Load all blocks via the sequence."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        assert sci_block.is_loaded
        assert cal_block.is_loaded

    def test_load_all_skips_loaded(
        self,
        sci_block,
        cal_block,
    ):
        """Skip already-loaded blocks during load_all()."""
        sci_block.load()
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        assert sci_block.is_loaded
        assert cal_block.is_loaded

    def test_iteration(self, sci_block, cal_block):
        """Iterate over blocks in sequence order."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block],
        )
        block_list = list(seq)
        assert block_list[0] is cal_block
        assert block_list[1] is sci_block

    def test_indexing(self, sci_block, cal_block):
        """Access blocks by index."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block],
        )
        assert seq[0] is cal_block
        assert seq[1] is sci_block

    def test_summary_unloaded(self, sci_block, cal_block):
        """Generate summary showing unloaded state."""
        seq = ObservingSequence(
            blocks=[cal_block, sci_block],
            name="test sequence",
        )
        summary = seq.summary()
        assert "test sequence" in summary
        assert "test target" in summary
        assert "test calibrator" in summary
        assert "SCI" in summary
        assert "CAL" in summary
        assert "not loaded" in summary

    def test_summary_loaded(self, sci_block, cal_block):
        """Summary reflects loaded state after loading."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        summary = seq.summary()
        assert "loaded" in summary
        assert "not loaded" not in summary


class TestObservingSequenceIntegration:
    """Integration tests for multi-block sequences."""

    def test_cal_sci_cal_sequence(self, sample_directory):
        """Build a CAL-SCI-CAL sequence."""
        cal1 = ObservingBlock(
            block_type=BlockType.CAL,
            target="test calibrator",
            directory=sample_directory,
            file_range=(5001, 5002),
        )
        sci = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
            file_range=(5003, 5004),
        )
        cal2 = ObservingBlock(
            block_type=BlockType.CAL,
            target="test calibrator",
            directory=sample_directory,
            file_range=(5005, 5005),
        )

        seq = ObservingSequence(
            blocks=[cal1, sci, cal2],
            name="test sequence 1",
        )

        assert len(seq) == 3
        assert len(seq.science_blocks) == 1
        assert len(seq.calibrator_blocks) == 2
        assert seq.targets == [
            "test calibrator",
            "test target",
        ]
        assert seq[0].block_type == BlockType.CAL
        assert seq[1].block_type == BlockType.SCI
        assert seq[2].block_type == BlockType.CAL

    def test_load_cal_sci_cal(self, sample_directory):
        """Load data for a full CAL-SCI-CAL sequence."""
        cal1 = ObservingBlock(
            "CAL",
            "test calibrator",
            sample_directory,
            file_range=(5001, 5002),
        )
        sci = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5003, 5004),
        )
        cal2 = ObservingBlock(
            "CAL",
            "test calibrator",
            sample_directory,
            file_range=(5005, 5005),
        )

        seq = ObservingSequence(blocks=[cal1, sci, cal2])
        seq.load_all()

        assert seq.is_loaded
        assert cal1.n_files == 2
        assert sci.n_files == 2
        assert cal2.n_files == 1
