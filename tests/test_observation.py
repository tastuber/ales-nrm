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
    para_angles = [10.0, 20.0, 30.0, 40.0, 50.0]
    altitudes = [55.0, 60.0, 65.0, 70.0, 75.0]
    times = [
        "08:00:00.000",
        "08:01:00.000",
        "08:02:00.000",
        "08:03:00.000",
        "08:04:00.000",
    ]
    for i, num in enumerate(range(5001, 5006)):
        filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
        write_test_cube(
            filepath,
            sample_cube,
            sample_wavelengths,
            extra_header={
                "LBT_PARA": para_angles[i],
                "LBT_ALT": altitudes[i],
                "TIME-OBS": times[i],
            },
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


class TestObservingBlockSummary:
    """Tests for ObservingBlock.summary()."""

    def test_summary_unloaded_with_file_range(self, sci_block):
        """Summary shows expected count when not loaded."""
        summary = sci_block.summary()
        assert "SCI" in summary
        assert "test target" in summary
        assert "3 files expected" in summary
        assert "not loaded" in summary

    def test_summary_loaded_with_file_range(self, sci_block):
        """Summary shows loaded/expected count after loading."""
        sci_block.load()
        summary = sci_block.summary()
        assert "SCI" in summary
        assert "test target" in summary
        assert "3/3 files loaded/expected" in summary

    def test_summary_unloaded_without_file_range(
        self,
        sample_directory,
    ):
        """Summary shows unknown count when no range set."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
        )
        summary = block.summary()
        assert "file count unknown" in summary
        assert "not loaded" in summary

    def test_summary_loaded_without_file_range(
        self,
        sample_directory,
    ):
        """Summary shows actual count when loaded without range."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
        )
        block.load()
        summary = block.summary()
        assert "5 files loaded" in summary

    def test_summary_cal_block(self, cal_block):
        """Summary shows CAL type for calibrator blocks."""
        summary = cal_block.summary()
        assert "CAL" in summary
        assert "test calibrator" in summary

    def test_summary_stacked(self, sample_directory):
        """Summary shows stacking state and group sizes."""
        block = ObservingBlock(
            block_type=BlockType.SCI,
            target="test target",
            directory=sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(group_size=5)
        summary = block.summary()
        assert "stacked" in summary
        assert "1 cubes from groups of 5" in summary
        assert "method='median'" in summary


class TestStackFrames:
    """Tests for ObservingBlock.stack_frames()."""

    def test_stack_not_loaded(self, sci_block):
        """Raise RuntimeError if stacking before loading."""
        with pytest.raises(RuntimeError, match="not been loaded"):
            sci_block.stack_frames(group_size=3)

    def test_stack_already_stacked(self, sci_block):
        """Raise RuntimeError if stacking already-stacked data."""
        sci_block.load()
        sci_block.stack_frames(group_size=3)
        with pytest.raises(RuntimeError, match="already been stacked"):
            sci_block.stack_frames(group_size=1)

    def test_stack_no_args(self, sci_block):
        """Raise ValueError if neither group_size nor groups."""
        sci_block.load()
        with pytest.raises(ValueError, match="Specify exactly one"):
            sci_block.stack_frames()

    def test_stack_both_args(self, sci_block):
        """Raise ValueError if both group_size and groups."""
        sci_block.load()
        with pytest.raises(ValueError, match="Specify exactly one"):
            sci_block.stack_frames(
                group_size=2,
                groups=[[5001, 5002]],
            )

    def test_stack_invalid_method(self, sci_block):
        """Raise ValueError for invalid stacking method."""
        sci_block.load()
        with pytest.raises(ValueError, match="Unknown stacking method"):
            sci_block.stack_frames(group_size=3, method="sum")

    def test_stack_invalid_remainder(self, sci_block):
        """Raise ValueError for invalid remainder option."""
        sci_block.load()
        with pytest.raises(ValueError, match="remainder must be"):
            sci_block.stack_frames(group_size=2, remainder="invalid")

    @pytest.mark.filterwarnings(
        "ignore::astropy.utils.exceptions.AstropyUserWarning"
    )
    def test_stack_with_centering(self, sci_block):
        """Stack with centering enabled completes."""
        sci_block.load()
        sci_block.stack_frames(
            group_size=3,
            center=True,
        )
        assert sci_block.is_stacked
        assert sci_block.n_files == 1

    def test_stack_invalid_group_size(self, sci_block):
        """Raise ValueError for group_size < 1."""
        sci_block.load()
        with pytest.raises(ValueError, match="group_size must be"):
            sci_block.stack_frames(group_size=0)

    def test_stack_group_size_all(self, sci_block):
        """Stack all frames into a single cube."""
        sci_block.load()
        original_shape = sci_block.cubes.shape[1:]
        sci_block.stack_frames(group_size=3)

        assert sci_block.is_stacked
        assert sci_block.n_files == 1
        assert sci_block.cubes.shape == (1, *original_shape)
        assert sci_block.file_numbers[0] == 5001
        assert len(sci_block.headers) == 1
        assert len(sci_block.stacking_groups) == 1
        np.testing.assert_array_equal(
            sci_block.stacking_groups[0],
            [5001, 5002, 5003],
        )

    def test_stack_group_size_one(self, sample_directory):
        """Stack with group_size=1 preserves all frames."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        original_cubes = block.cubes.copy()
        block.stack_frames(group_size=1)

        assert block.is_stacked
        assert block.n_files == 5
        np.testing.assert_allclose(
            block.cubes,
            original_cubes,
        )

    def test_stack_group_size_even_split(
        self,
        sample_directory,
    ):
        """Stack 4 frames into 2 groups of 2."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5004),
        )
        block.load()
        block.stack_frames(group_size=2)

        assert block.n_files == 2
        np.testing.assert_array_equal(
            block.file_numbers,
            [5001, 5003],
        )
        assert len(block.stacking_groups) == 2
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[1],
            [5003, 5004],
        )

    def test_stack_remainder_discard(
        self,
        sample_directory,
    ):
        """Discard remainder frames by default."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(group_size=2)

        assert block.n_files == 2
        np.testing.assert_array_equal(
            block.file_numbers,
            [5001, 5003],
        )
        assert len(block.stacking_groups) == 2
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[1],
            [5003, 5004],
        )

    def test_stack_remainder_keep(self, sample_directory):
        """Keep remainder frames as a smaller final group."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(
            group_size=2,
            remainder="keep",
        )

        assert block.n_files == 3
        np.testing.assert_array_equal(
            block.file_numbers,
            [5001, 5003, 5005],
        )
        assert len(block.stacking_groups) == 3
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[1],
            [5003, 5004],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[2],
            [5005],
        )

    def test_stack_explicit_groups(self, sample_directory):
        """Stack using explicit file number groups."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(
            groups=[[5001, 5002], [5003, 5004, 5005]],
        )

        assert block.n_files == 2
        np.testing.assert_array_equal(
            block.file_numbers,
            [5001, 5003],
        )
        assert len(block.stacking_groups) == 2
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[1],
            [5003, 5004, 5005],
        )

    def test_stack_explicit_groups_invalid_file(
        self,
        sci_block,
    ):
        """Raise ValueError for file numbers not in data."""
        sci_block.load()
        with pytest.raises(ValueError, match="not loaded"):
            sci_block.stack_frames(
                groups=[[5001, 9999]],
            )

    def test_stack_median_values(self, sample_directory):
        """Verify median stacking produces correct values."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5003),
        )
        block.load()
        expected = np.median(block.cubes, axis=0)
        block.stack_frames(group_size=3, method="median")

        np.testing.assert_allclose(
            block.cubes[0],
            expected,
        )

    def test_stack_mean_values(self, sample_directory):
        """Verify mean stacking produces correct values."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5003),
        )
        block.load()
        expected = np.mean(block.cubes, axis=0)
        block.stack_frames(group_size=3, method="mean")

        np.testing.assert_allclose(
            block.cubes[0],
            expected,
        )

    def test_stack_preserves_wavelengths(
        self,
        sci_block,
        sample_wavelengths,
    ):
        """Stacking does not modify the wavelength array."""
        sci_block.load()
        sci_block.stack_frames(group_size=3)
        np.testing.assert_allclose(
            sci_block.wavelengths,
            sample_wavelengths,
        )

    def test_stack_preserves_dtype(self, sci_block):
        """Stacked cubes preserve the original dtype."""
        sci_block.load()
        original_dtype = sci_block.cubes.dtype
        sci_block.stack_frames(group_size=3)
        assert sci_block.cubes.dtype == original_dtype

    def test_stack_empty_after_discard(
        self,
        sample_directory,
    ):
        """Raise ValueError when all frames are discarded."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5002),
        )
        block.load()
        with pytest.raises(ValueError, match="No stacking groups"):
            block.stack_frames(
                group_size=5,
                remainder="discard",
            )

    def test_stacking_groups_traceability(
        self,
        sample_directory,
    ):
        """Verify stacking_groups allows full traceability."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(
            group_size=2,
            remainder="keep",
        )

        # Each stacked cube can be traced back.
        all_stacked_files = np.concatenate(block.stacking_groups)
        np.testing.assert_array_equal(
            np.sort(all_stacked_files),
            [5001, 5002, 5003, 5004, 5005],
        )

    def test_summary_shows_stacking_info(
        self,
        sample_directory,
    ):
        """Summary shows stacking details."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(
            group_size=2,
            remainder="keep",
        )
        summary = block.summary()
        assert "stacked" in summary
        assert "[2, 2, 1]" in summary
        assert "method='median'" in summary

    def test_stack_remainder_add(self, sample_directory):
        """Add remainder frames to the last complete group."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(
            group_size=2,
            remainder="add",
        )

        assert block.n_files == 2
        np.testing.assert_array_equal(
            block.file_numbers,
            [5001, 5003],
        )
        assert len(block.stacking_groups) == 2
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[1],
            [5003, 5004, 5005],
        )

    def test_stack_remainder_add_no_complete_groups(
        self,
        sample_directory,
    ):
        """Handle add when no complete groups exist."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5002),
        )
        block.load()
        block.stack_frames(
            group_size=5,
            remainder="add",
        )

        assert block.n_files == 1
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )

    def test_stack_remainder_add_even_split(
        self,
        sample_directory,
    ):
        """Add with even split behaves like normal stacking."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5004),
        )
        block.load()
        block.stack_frames(
            group_size=2,
            remainder="add",
        )

        assert block.n_files == 2
        np.testing.assert_array_equal(
            block.stacking_groups[0],
            [5001, 5002],
        )
        np.testing.assert_array_equal(
            block.stacking_groups[1],
            [5003, 5004],
        )


class TestHeaderMetadata:
    """Tests for header metadata extraction and stacking."""

    def test_metadata_extracted_on_load(self, sci_block):
        """Parallactic angles, altitudes, timestamps extracted."""
        sci_block.load()
        np.testing.assert_allclose(
            sci_block.parallactic_angles,
            [10.0, 20.0, 30.0],
        )
        np.testing.assert_allclose(
            sci_block.altitudes,
            [55.0, 60.0, 65.0],
        )
        assert list(sci_block.timestamps) == [
            "08:00:00.000",
            "08:01:00.000",
            "08:02:00.000",
        ]

    def test_metadata_mean_after_stacking(self, sample_directory):
        """Mean angles/altitudes and mean timestamp after stacking."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5004),
        )
        block.load()
        block.stack_frames(group_size=2)
        np.testing.assert_allclose(block.parallactic_angles, [15.0, 35.0])
        np.testing.assert_allclose(block.altitudes, [57.5, 67.5])
        assert block.timestamps[0] == "08:00:30.000"
        assert block.timestamps[1] == "08:02:30.000"

    def test_metadata_explicit_groups(self, sample_directory):
        """Mean metadata for explicit stacking groups."""
        block = ObservingBlock(
            "SCI",
            "test target",
            sample_directory,
            file_range=(5001, 5005),
        )
        block.load()
        block.stack_frames(
            groups=[[5001, 5002, 5003], [5004, 5005]],
        )
        np.testing.assert_allclose(block.parallactic_angles, [20.0, 45.0])
        np.testing.assert_allclose(block.altitudes, [60.0, 72.5])
        assert block.timestamps[0] == "08:01:00.000"
        assert block.timestamps[1] == "08:03:30.000"

    def test_metadata_group_size_one(self, sci_block):
        """Stacking with group_size=1 preserves metadata."""
        sci_block.load()
        expected_para = sci_block.parallactic_angles.copy()
        expected_alt = sci_block.altitudes.copy()
        expected_ts = list(sci_block.timestamps)
        sci_block.stack_frames(group_size=1)
        np.testing.assert_allclose(sci_block.parallactic_angles, expected_para)
        np.testing.assert_allclose(sci_block.altitudes, expected_alt)
        assert list(sci_block.timestamps) == expected_ts

    def test_missing_para_warns_and_stores_nan(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn and use NaN when LBT_PARA is missing."""
        filepath = tmp_path / "cube_lm_251108_006001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)
        from astropy.io import fits

        with fits.open(filepath, mode="update") as hdul:
            del hdul[0].header["LBT_PARA"]
            hdul.flush()

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(6001, 6001)
        )
        with pytest.warns(UserWarning, match="missing.*LBT_PARA"):
            block.load()
        assert np.isnan(block.parallactic_angles[0])

    def test_missing_alt_warns_and_stores_nan(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn and use NaN when LBT_ALT is missing."""
        filepath = tmp_path / "cube_lm_251108_006001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)
        from astropy.io import fits

        with fits.open(filepath, mode="update") as hdul:
            del hdul[0].header["LBT_ALT"]
            hdul.flush()

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(6001, 6001)
        )
        with pytest.warns(UserWarning, match="missing.*LBT_ALT"):
            block.load()
        assert np.isnan(block.altitudes[0])

    def test_missing_time_warns_and_stores_empty(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn and use empty string when TIME-OBS missing."""
        filepath = tmp_path / "cube_lm_251108_006001.fits"
        write_test_cube(filepath, sample_cube, sample_wavelengths)
        from astropy.io import fits

        with fits.open(filepath, mode="update") as hdul:
            del hdul[0].header["TIME-OBS"]
            hdul.flush()

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(6001, 6001)
        )
        with pytest.warns(UserWarning, match="missing.*TIME-OBS"):
            block.load()
        assert block.timestamps[0] == ""

    def test_invalid_para_warns_and_stores_nan(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn and use NaN when LBT_PARA is non-numeric."""
        filepath = tmp_path / "cube_lm_251108_006001.fits"
        write_test_cube(
            filepath,
            sample_cube,
            sample_wavelengths,
            extra_header={"LBT_PARA": "NOT_A_NUMBER"},
        )
        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(6001, 6001)
        )
        with pytest.warns(UserWarning, match="invalid.*LBT_PARA"):
            block.load()
        assert np.isnan(block.parallactic_angles[0])

    def test_invalid_alt_warns_and_stores_nan(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Warn and use NaN when LBT_ALT is non-numeric."""
        filepath = tmp_path / "cube_lm_251108_006001.fits"
        write_test_cube(
            filepath,
            sample_cube,
            sample_wavelengths,
            extra_header={"LBT_ALT": "BAD"},
        )
        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(6001, 6001)
        )
        with pytest.warns(UserWarning, match="invalid.*LBT_ALT"):
            block.load()
        assert np.isnan(block.altitudes[0])

    def test_nan_propagates_through_stacking(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """NaN in angles/altitudes propagates through mean."""
        for num in [7001, 7002]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(
                filepath,
                sample_cube,
                sample_wavelengths,
                extra_header={
                    "LBT_PARA": 25.0,
                    "LBT_ALT": 70.0,
                },
            )
        from astropy.io import fits

        fp2 = tmp_path / "cube_lm_251108_007002.fits"
        with fits.open(fp2, mode="update") as hdul:
            del hdul[0].header["LBT_PARA"]
            del hdul[0].header["LBT_ALT"]
            hdul.flush()

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(7001, 7002)
        )
        with pytest.warns(UserWarning):
            block.load()
        block.stack_frames(group_size=2)
        assert np.isnan(block.parallactic_angles[0])
        assert np.isnan(block.altitudes[0])

    def test_mean_timestamp_excludes_empty(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Empty timestamps excluded from mean computation."""
        for num in [7001, 7002]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(
                filepath,
                sample_cube,
                sample_wavelengths,
                extra_header={"TIME-OBS": "10:00:00.000"},
            )
        from astropy.io import fits

        fp2 = tmp_path / "cube_lm_251108_007002.fits"
        with fits.open(fp2, mode="update") as hdul:
            del hdul[0].header["TIME-OBS"]
            hdul.flush()

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(7001, 7002)
        )
        with pytest.warns(UserWarning, match="missing.*TIME-OBS"):
            block.load()
        block.stack_frames(group_size=2)
        assert block.timestamps[0] == "10:00:00.000"

    def test_mean_timestamp_all_empty_yields_empty(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """All-empty timestamps yield empty string."""
        for num in [7001, 7002]:
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            write_test_cube(
                filepath,
                sample_cube,
                sample_wavelengths,
            )
        from astropy.io import fits

        for num in [7001, 7002]:
            fp = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            with fits.open(fp, mode="update") as hdul:
                del hdul[0].header["TIME-OBS"]
                hdul.flush()

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(7001, 7002)
        )
        with pytest.warns(UserWarning, match="missing.*TIME-OBS"):
            block.load()
        block.stack_frames(group_size=2)
        assert block.timestamps[0] == ""

    def test_mean_timestamp_malformed_excluded(
        self,
        tmp_path,
        sample_cube,
        sample_wavelengths,
    ):
        """Malformed timestamps excluded from mean."""
        for i, num in enumerate([7001, 7002]):
            filepath = tmp_path / f"cube_lm_251108_{num:06d}.fits"
            ts = "10:00:00.000" if i == 0 else "BADTIME"
            write_test_cube(
                filepath,
                sample_cube,
                sample_wavelengths,
                extra_header={"TIME-OBS": ts},
            )

        block = ObservingBlock(
            "SCI", "test", tmp_path, file_range=(7001, 7002)
        )
        block.load()
        block.stack_frames(group_size=2)
        # "BADTIME" fails parsing, only "10:00:00.000" contributes.
        assert block.timestamps[0] == "10:00:00.000"


class TestComplexVisibilities:
    """Tests for complex visibility computation."""

    def test_compute_complex_visibilities(self, sci_block):
        """Compute complex visibilities after loading."""
        sci_block.load()
        sci_block.compute_complex_visibilities()
        assert sci_block.complex_visibilities is not None
        n_files, n_wav, ny, nx = sci_block.cubes.shape
        assert sci_block.complex_visibilities.shape == (
            n_files,
            n_wav,
            501,
            501,
        )
        assert np.iscomplexobj(sci_block.complex_visibilities)

    def test_default_n_fft(self, sci_block):
        """Default n_fft produces 501x501 output."""
        sci_block.load()
        sci_block.compute_complex_visibilities()
        assert sci_block.complex_visibilities.shape[-1] == 501
        assert sci_block.complex_visibilities.shape[-2] == 501

    def test_custom_n_fft(self, sci_block):
        """Custom n_fft changes output dimensions."""
        sci_block.load()
        sci_block.compute_complex_visibilities(n_fft=101)
        assert sci_block.complex_visibilities.shape[-1] == 101
        assert sci_block.complex_visibilities.shape[-2] == 101

    def test_even_n_fft_forced_odd(self, sci_block):
        """Even n_fft is forced to next odd value."""
        sci_block.load()
        sci_block.compute_complex_visibilities(n_fft=100)
        assert sci_block.complex_visibilities.shape[-1] == 101
        assert sci_block.complex_visibilities.shape[-2] == 101

    def test_n_fft_too_small(self, sci_block):
        """Raise ValueError if n_fft < spatial dimensions."""
        sci_block.load()
        with pytest.raises(ValueError, match="n_fft"):
            sci_block.compute_complex_visibilities(n_fft=50)

    def test_n_fft_equal_to_cube_size(self, sci_block):
        """n_fft equal to cube size works (no padding)."""
        sci_block.load()
        sci_block.compute_complex_visibilities(n_fft=67)
        n_files, n_wav, ny, nx = sci_block.cubes.shape
        assert sci_block.complex_visibilities.shape == (
            n_files,
            n_wav,
            67,
            67,
        )

    def test_power_computed_by_default(self, sci_block):
        """Power spectra computed by default."""
        sci_block.load()
        sci_block.compute_complex_visibilities()
        assert sci_block.power_spectra is not None
        assert (
            sci_block.power_spectra.shape
            == sci_block.complex_visibilities.shape
        )

    def test_power_not_computed_when_disabled(self, sci_block):
        """Power spectra not computed when disabled."""
        sci_block.load()
        sci_block.compute_complex_visibilities(compute_power=False)
        assert sci_block.complex_visibilities is not None
        assert sci_block.power_spectra is None

    def test_complex_vis_not_loaded(self, sci_block):
        """Raise RuntimeError if not loaded."""
        with pytest.raises(RuntimeError, match="not been loaded"):
            sci_block.compute_complex_visibilities()

    def test_power_equals_abs_squared(self, sci_block):
        """Power spectrum equals |complex_vis|^2."""
        sci_block.load()
        sci_block.compute_complex_visibilities()
        expected = np.abs(sci_block.complex_visibilities) ** 2
        np.testing.assert_allclose(sci_block.power_spectra, expected)

    def test_zero_padding_preserves_total_power(
        self,
        sci_block,
    ):
        """Zero-padding scales total power by (n_fft/n)^2.

        Parseval's theorem: sum |FFT|^2 = N * sum |x|^2.
        The ratio of total power between padded and unpadded
        FFTs equals (n_fft / n_original)^2.
        """
        sci_block.load()

        # Compute without padding.
        sci_block.compute_complex_visibilities(n_fft=67)
        power_no_pad = sci_block.power_spectra[0, 0].sum()

        # Reset and compute with padding.
        sci_block.complex_visibilities = None
        sci_block.power_spectra = None
        sci_block.compute_complex_visibilities(n_fft=501)
        power_padded = sci_block.power_spectra[0, 0].sum()

        # Ratio should be (501/67)^2.
        expected_ratio = (501 / 67) ** 2
        np.testing.assert_allclose(
            power_padded / power_no_pad,
            expected_ratio,
            rtol=1e-10,
        )

    def test_zero_padding_center_pixel_is_total_flux(
        self,
        sci_block,
    ):
        """DC component equals sum of image pixels."""
        sci_block.load()
        sci_block.compute_complex_visibilities(n_fft=501)
        center = 501 // 2
        for w in range(3):
            expected_dc = np.sum(sci_block.cubes[0, w])
            actual_dc = sci_block.complex_visibilities[0, w, center, center]
            np.testing.assert_allclose(
                actual_dc.real,
                expected_dc,
                rtol=1e-10,
            )
            np.testing.assert_allclose(
                actual_dc.imag,
                0.0,
                atol=1e-8,
            )

    def test_n_fft_output_is_always_odd(self, sci_block):
        """Output dimensions are always odd."""
        sci_block.load()
        for n in [67, 100, 128, 256, 500, 501]:
            sci_block.complex_visibilities = None
            sci_block.power_spectra = None
            sci_block.compute_complex_visibilities(n_fft=n)
            assert sci_block.complex_visibilities.shape[-1] % 2 == 1
            assert sci_block.complex_visibilities.shape[-2] % 2 == 1


class TestPowerSpectra:
    """Tests for power spectrum computation."""

    def test_compute_power_spectra(self, sci_block):
        """Compute power spectra after loading."""
        sci_block.load()
        sci_block.compute_power_spectra()
        assert sci_block.power_spectra is not None
        n_files, n_wav, ny, nx = sci_block.cubes.shape
        assert sci_block.power_spectra.shape == (
            n_files,
            n_wav,
            501,
            501,
        )

    def test_power_spectra_custom_n_fft(self, sci_block):
        """Power spectra with custom n_fft."""
        sci_block.load()
        sci_block.compute_power_spectra(n_fft=101)
        assert sci_block.power_spectra.shape[-1] == 101
        assert sci_block.power_spectra.shape[-2] == 101

    def test_power_spectra_computes_complex_vis(self, sci_block):
        """Power spectra triggers complex visibilities computation."""
        sci_block.load()
        sci_block.compute_power_spectra()
        assert sci_block.complex_visibilities is not None

    def test_power_from_existing_complex_vis(self, sci_block):
        """Compute power from pre-existing complex visibilities."""
        sci_block.load()
        sci_block.compute_complex_visibilities(compute_power=False)
        assert sci_block.power_spectra is None
        sci_block.compute_power_spectra()
        assert sci_block.power_spectra is not None

    def test_power_spectra_not_loaded(self, sci_block):
        """Raise RuntimeError if not loaded."""
        with pytest.raises(RuntimeError, match="not been loaded"):
            sci_block.compute_power_spectra()

    def test_power_spectra_nonnegative(self, sci_block):
        """Power spectra are non-negative."""
        sci_block.load()
        sci_block.compute_power_spectra()
        assert np.all(sci_block.power_spectra >= 0)

    def test_power_spectra_dtype(self, sci_block):
        """Power spectra have float dtype."""
        sci_block.load()
        sci_block.compute_power_spectra()
        assert sci_block.power_spectra.dtype == np.float64

    def test_sequence_compute_all_power_spectra(self, sci_block, cal_block):
        """Compute power spectra for all blocks."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
            name="test sequence",
        )
        seq.load_all()
        seq.compute_all_power_spectra()
        for block in seq:
            assert block.power_spectra is not None
            n_files = block.cubes.shape[0]
            n_wav = block.cubes.shape[1]
            assert block.power_spectra.shape == (
                n_files,
                n_wav,
                501,
                501,
            )

    def test_sequence_compute_all_power_spectra_custom_n_fft(
        self,
        sci_block,
        cal_block,
    ):
        """Sequence passes n_fft to all blocks."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        seq.compute_all_power_spectra(n_fft=201)
        for block in seq:
            assert block.power_spectra.shape[-1] == 201
            assert block.power_spectra.shape[-2] == 201

    def test_sequence_compute_all_complex_vis(self, sci_block, cal_block):
        """Compute complex visibilities for all blocks."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
            name="test sequence",
        )
        seq.load_all()
        seq.compute_all_complex_visibilities()
        for block in seq:
            assert block.complex_visibilities is not None
            assert block.power_spectra is not None

    def test_sequence_compute_all_complex_vis_custom_n_fft(
        self,
        sci_block,
        cal_block,
    ):
        """Sequence passes n_fft to complex visibilities computation."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        seq.compute_all_complex_visibilities(n_fft=101, compute_power=False)
        for block in seq:
            assert block.complex_visibilities.shape[-1] == 101
            assert block.complex_visibilities.shape[-2] == 101

    def test_sequence_complex_vis_no_power(
        self,
        sci_block,
        cal_block,
    ):
        """Compute complex visibilities without power."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        seq.compute_all_complex_visibilities(
            compute_power=False,
        )
        for block in seq:
            assert block.complex_visibilities is not None
            assert block.power_spectra is None

    def test_sequence_skips_unloaded(self, sci_block, cal_block):
        """Skip unloaded blocks without error."""
        sci_block.load()
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.compute_all_power_spectra()
        assert sci_block.power_spectra is not None
        assert cal_block.power_spectra is None

    def test_sequence_skips_already_computed(self, sci_block):
        """Do not recompute existing power spectra."""
        sci_block.load()
        sci_block.compute_power_spectra()
        original = sci_block.power_spectra.copy()
        seq = ObservingSequence(blocks=[sci_block])
        seq.compute_all_power_spectra()
        np.testing.assert_array_equal(sci_block.power_spectra, original)

    def test_sequence_complex_vis_skips_unloaded(self, sci_block, cal_block):
        """Skip unloaded blocks when computing complex visibility."""
        sci_block.load()
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.compute_all_complex_visibilities()
        assert sci_block.complex_visibilities is not None
        assert cal_block.complex_visibilities is None


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
        assert "expected" in summary

    def test_summary_loaded(self, sci_block, cal_block):
        """Summary reflects loaded state after loading."""
        seq = ObservingSequence(
            blocks=[sci_block, cal_block],
        )
        seq.load_all()
        summary = seq.summary()
        assert "loaded/expected" in summary
        assert "not loaded" not in summary
        assert "3/3 files" in summary
        assert "2/2 files" in summary


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
