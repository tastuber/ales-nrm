"""Tests for observing block management."""

import warnings

import numpy as np
import pytest

from ales_nrm.observation import BlockType, ObservingBlock
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
        target="test Target",
        directory=sample_directory,
        file_range=(5001, 5003),
    )


@pytest.fixture()
def cal_block(sample_directory):
    """Create a calibrator block for testing."""
    return ObservingBlock(
        block_type=BlockType.CAL,
        target="test Target",
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
        assert sci_block.target == "test Target"
        assert sci_block.file_range == (5001, 5003)

    def test_creation_without_file_range(self, sample_directory):
        """Create a block without specifying file_range."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test Target",
            directory=sample_directory,
        )
        assert block.file_range is None

    def test_block_type_from_string(self, sample_directory):
        """Accept a string and convert to BlockType enum."""
        block = ObservingBlock(
            block_type="SCI",
            target="test Target",
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
                target="test Target",
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
            target="test Target",
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
            target="test Target",
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
            target="test Target",
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
            target="test Target",
            directory=tmp_path,
            file_range=(5001, 5003),
        )

        with pytest.warns(UserWarning, match="missing file numbers"):
            block.load()

    def test_no_validation_without_file_range(self, sample_directory):
        """No validation warnings when file_range is None."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test Target",
            directory=sample_directory,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            block.load()
