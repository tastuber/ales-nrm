"""Tests for ales_nrm.sampy_interface.extract module."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ales_nrm.observables import Observables
from ales_nrm.sampy_interface.extract import (
    _ensure_trailing_sep,
    build_observables_from_sampy,
    extract_observables,
)


@pytest.fixture()
def mock_block():
    """Create a mock ObservingBlock with known shape."""
    block = MagicMock()
    block.is_loaded = True
    # 3 files, 4 wavelengths, 67x67 spatial
    block.cubes = np.random.default_rng(42).normal(size=(3, 4, 67, 67))
    block.wavelengths = np.array([3.0, 3.5, 4.0, 4.2])
    return block


@pytest.fixture()
def mock_mask_dirs(mock_block, tmp_path):
    """Create mask_dirs dict matching mock_block."""
    dirs = {}
    for wl in mock_block.wavelengths:
        d = tmp_path / f"{float(wl):.4f}um"
        d.mkdir()
        dirs[float(wl)] = d
    return dirs


def _make_analysis_mocks():
    """Create SAMpy analysis module mocks."""
    mock_analysis = MagicMock()
    mock_analysis.calc_cps_multi = MagicMock(
        return_value={"closure_phases": np.zeros(20)}
    )
    mock_analysis.calc_v2s = MagicMock(return_value={"v2": np.zeros(15)})
    mock_analysis.calc_cvis = MagicMock(
        return_value={"amplitudes": np.zeros(15)}
    )
    mock_sampy = MagicMock()
    mock_sampy.analysis = mock_analysis
    return mock_sampy, mock_analysis


class TestEnsureTrailingSep:
    """Tests for _ensure_trailing_sep."""

    def test_path_without_trailing_sep(self, tmp_path):
        """Path object gets trailing os.sep added."""
        result = _ensure_trailing_sep(tmp_path / "subdir")
        assert result.endswith(os.sep)

    def test_string_without_trailing_sep(self):
        """String without separator gets one appended."""
        result = _ensure_trailing_sep("/some/path")
        assert result.endswith(os.sep)

    def test_string_with_trailing_sep(self):
        """String already ending with sep is unchanged."""
        path = f"/some/path{os.sep}"
        result = _ensure_trailing_sep(path)
        assert result == path
        assert result.count(os.sep) == path.count(os.sep)

    def test_path_object_returns_string(self, tmp_path):
        """Return type is always string."""
        result = _ensure_trailing_sep(tmp_path)
        assert isinstance(result, str)


class TestExtractObservables:
    """Tests for extract_observables."""

    def test_calls_sampy_once_per_wavelength(self, mock_block, mock_mask_dirs):
        """Each SAMpy function called once per wl."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            extract_observables(
                mock_block,
                mock_mask_dirs,
                extract_cp=True,
                extract_vis2=True,
                extract_compl_vis=True,
            )

        n_wl = len(mock_block.wavelengths)
        assert mock_analysis.calc_cps_multi.call_count == n_wl
        assert mock_analysis.calc_v2s.call_count == n_wl
        assert mock_analysis.calc_cvis.call_count == n_wl

    def test_correct_image_shape_passed(self, mock_block, mock_mask_dirs):
        """Images have shape (n_files, ny, nx)."""
        mock_sampy, mock_analysis = _make_analysis_mocks()
        captured_shapes = []

        def capture_cps(images, mask_dir, **_kwargs):
            captured_shapes.append(images.shape)
            return {"closure_phases": np.zeros(20)}

        mock_analysis.calc_cps_multi = MagicMock(side_effect=capture_cps)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            extract_observables(mock_block, mock_mask_dirs)

        for shape in captured_shapes:
            assert shape == (3, 67, 67)

    def test_mask_dir_has_trailing_sep(self, mock_block, mock_mask_dirs):
        """mask_dir passed to SAMpy ends with os.sep."""
        mock_sampy, mock_analysis = _make_analysis_mocks()
        captured_dirs = []

        def capture_vis2(images, mask_dir, **_kwargs):
            captured_dirs.append(mask_dir)
            return {"v2": np.zeros(15)}

        mock_analysis.calc_v2s = MagicMock(side_effect=capture_vis2)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            extract_observables(mock_block, mock_mask_dirs)

        for d in captured_dirs:
            assert d.endswith(os.sep)

    def test_correct_mask_dir_content_passed(self, mock_block, mock_mask_dirs):
        """mask_dir matches wavelength's directory."""
        mock_sampy, mock_analysis = _make_analysis_mocks()
        captured_dirs = []

        def capture_vis2(images, mask_dir, **_kwargs):
            captured_dirs.append(mask_dir)
            return {"v2": np.zeros(15)}

        mock_analysis.calc_v2s = MagicMock(side_effect=capture_vis2)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            extract_observables(mock_block, mock_mask_dirs)

        for i, wl in enumerate(mock_block.wavelengths):
            expected = str(mock_mask_dirs[float(wl)]) + os.sep
            assert captured_dirs[i] == expected

    def test_nx_ny_display_passed(self, mock_block, mock_mask_dirs):
        """nx, ny, display forwarded to SAMpy."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            extract_observables(
                mock_block,
                mock_mask_dirs,
                nx=256,
                ny=128,
                display=True,
            )

        call_kw = mock_analysis.calc_cps_multi.call_args[1]
        assert call_kw["nx"] == 256
        assert call_kw["ny"] == 128
        assert call_kw["display"] is True

    def test_default_nx_ny_display(self, mock_block, mock_mask_dirs):
        """Default nx=501, ny=501, display=False."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            extract_observables(mock_block, mock_mask_dirs)

        call_kw = mock_analysis.calc_v2s.call_args[1]
        assert call_kw["nx"] == 501
        assert call_kw["ny"] == 501
        assert call_kw["display"] is False

    def test_wavelength_keys_in_result(self, mock_block, mock_mask_dirs):
        """Result dict has correct wavelength keys."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                extract_cp=True,
                extract_vis2=True,
                extract_compl_vis=True,
            )

        for wl in mock_block.wavelengths:
            wl_f = float(wl)
            assert wl_f in result["cp"]
            assert wl_f in result["vis2"]
            assert wl_f in result["compl_vis"]

    def test_wavelengths_array_in_result(self, mock_block, mock_mask_dirs):
        """Result contains the wavelengths array."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(mock_block, mock_mask_dirs)

        np.testing.assert_array_equal(
            result["wavelengths"], mock_block.wavelengths
        )

    def test_result_structure_all_flags_true(self, mock_block, mock_mask_dirs):
        """Result has all keys when all flags True."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                extract_cp=True,
                extract_vis2=True,
                extract_compl_vis=True,
            )

        assert "wavelengths" in result
        assert "cp" in result
        assert "vis2" in result
        assert "compl_vis" in result

    def test_extract_cp_only(self, mock_block, mock_mask_dirs):
        """Only CP extracted when other flags False."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                extract_cp=True,
                extract_vis2=False,
                extract_compl_vis=False,
            )

        assert "cp" in result
        assert "vis2" not in result
        assert "compl_vis" not in result
        assert mock_analysis.calc_v2s.call_count == 0
        assert mock_analysis.calc_cvis.call_count == 0

    def test_extract_vis2_only(self, mock_block, mock_mask_dirs):
        """Only VIS2 extracted when other flags False."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                extract_cp=False,
                extract_vis2=True,
                extract_compl_vis=False,
            )

        assert "cp" not in result
        assert "vis2" in result
        assert "compl_vis" not in result
        assert mock_analysis.calc_cps_multi.call_count == 0
        assert mock_analysis.calc_cvis.call_count == 0

    def test_extract_compl_vis_only(self, mock_block, mock_mask_dirs):
        """Only complex visibility extracted when others False."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                extract_cp=False,
                extract_vis2=False,
                extract_compl_vis=True,
            )

        assert "cp" not in result
        assert "vis2" not in result
        assert "compl_vis" in result
        assert mock_analysis.calc_cps_multi.call_count == 0
        assert mock_analysis.calc_v2s.call_count == 0

    def test_default_flags_all(self, mock_block, mock_mask_dirs):
        """Default flags extract CP, VIS2, and complex visibility."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(mock_block, mock_mask_dirs)

        assert "cp" in result
        assert "vis2" in result
        assert "compl_vis" in result
        n_wl = len(mock_block.wavelengths)
        assert mock_analysis.calc_cps_multi.call_count == n_wl
        assert mock_analysis.calc_v2s.call_count == n_wl
        assert mock_analysis.calc_cvis.call_count == n_wl

    def test_all_flags_false_raises_value_error(
        self, mock_block, mock_mask_dirs
    ):
        """ValueError raised if all flags are False."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            with pytest.raises(ValueError, match="At least one"):
                extract_observables(
                    mock_block,
                    mock_mask_dirs,
                    extract_cp=False,
                    extract_vis2=False,
                    extract_compl_vis=False,
                )

    def test_not_loaded_raises_runtime_error(self, mock_mask_dirs):
        """RuntimeError raised if block not loaded."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        block = MagicMock()
        block.is_loaded = False

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            with pytest.raises(RuntimeError, match="not been loaded"):
                extract_observables(block, mock_mask_dirs)

    def test_import_error_when_sampy_unavailable(
        self, mock_block, mock_mask_dirs
    ):
        """Clear ImportError when SAMpy not installed."""
        with patch.dict(
            sys.modules,
            {
                "sampy": None,
                "sampy.analysis": None,
            },
        ):
            with pytest.raises(ImportError, match="SAMpy is required"):
                extract_observables(mock_block, mock_mask_dirs)

    def test_missing_wavelength_raises_key_error(self, mock_block, tmp_path):
        """KeyError if mask_dirs missing a wavelength."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        # Only provide dirs for first 2 wavelengths
        partial_dirs = {}
        for wl in mock_block.wavelengths[:2]:
            d = tmp_path / f"{float(wl):.4f}um"
            d.mkdir()
            partial_dirs[float(wl)] = d

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            with pytest.raises(KeyError):
                extract_observables(mock_block, partial_dirs)

    def test_single_file_block(self, tmp_path):
        """Works with a single-file block."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        block = MagicMock()
        block.is_loaded = True
        block.cubes = np.zeros((1, 2, 67, 67))
        block.wavelengths = np.array([3.0, 3.5])

        mask_dirs = {}
        for wl in block.wavelengths:
            d = tmp_path / f"{float(wl):.4f}um"
            d.mkdir()
            mask_dirs[float(wl)] = d

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(block, mask_dirs)

        assert len(result["cp"]) == 2
        assert mock_analysis.calc_cps_multi.call_count == 2

    def test_string_mask_dir_handled(self, mock_block, tmp_path):
        """String paths in mask_dirs work correctly."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        mask_dirs = {}
        for wl in mock_block.wavelengths:
            d = tmp_path / f"{float(wl):.4f}um"
            d.mkdir()
            mask_dirs[float(wl)] = str(d)

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(mock_block, mask_dirs)

        assert "cp" in result
        assert "vis2" in result

    def test_wl_indices_limits_extraction(self, mock_block, mock_mask_dirs):
        """wl_indices restricts to subset of channels."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                wl_indices=(1, 3),
            )

        # Only wavelengths at index 1 and 2
        assert mock_analysis.calc_cps_multi.call_count == 2
        np.testing.assert_array_equal(
            result["wavelengths"],
            mock_block.wavelengths[1:3],
        )
        assert 3.5 in result["cp"]
        assert 4.0 in result["cp"]
        assert 3.0 not in result["cp"]

    def test_wl_indices_none_uses_all(self, mock_block, mock_mask_dirs):
        """wl_indices=None processes all wavelengths."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                wl_indices=None,
            )

        n_wl = len(mock_block.wavelengths)
        assert mock_analysis.calc_cps_multi.call_count == n_wl
        np.testing.assert_array_equal(
            result["wavelengths"],
            mock_block.wavelengths,
        )

    def test_wl_indices_empty_raises(self, mock_block, mock_mask_dirs):
        """wl_indices with start >= stop raises."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            with pytest.raises(ValueError, match="no channels"):
                extract_observables(
                    mock_block,
                    mock_mask_dirs,
                    wl_indices=(3, 3),
                )

    def test_wl_indices_single_channel(self, mock_block, mock_mask_dirs):
        """wl_indices selecting one channel works."""
        mock_sampy, mock_analysis = _make_analysis_mocks()

        with patch.dict(
            sys.modules,
            {
                "sampy": mock_sampy,
                "sampy.analysis": mock_analysis,
            },
        ):
            result = extract_observables(
                mock_block,
                mock_mask_dirs,
                wl_indices=(2, 3),
            )

        assert mock_analysis.calc_cps_multi.call_count == 1
        np.testing.assert_array_equal(
            result["wavelengths"],
            np.array([4.0]),
        )
        assert 4.0 in result["cp"]


@pytest.fixture()
def sampy_mock_block():
    """Create a mock ObservingBlock for build_observables_from_sampy."""
    block = MagicMock()
    block.parallactic_angles = np.array([0.0, 0.0, 0.0])
    block.observation_date = None
    block.timestamps = np.array(
        ["10:00:00.000", "10:10:00.000", "10:20:00.000"],
        dtype=object,
    )
    block.target = "TestStar"
    block.wavelengths = np.array([3.0, 3.5, 4.0])
    block.block_type = "SCI"
    block.observables_raw = None
    return block


@pytest.fixture()
def sampy_extracted_full(sampy_mock_block):
    """Create a full SAMpy extracted dict with all observable types."""
    n_bl = 3
    n_tri = 1
    wavelengths = sampy_mock_block.wavelengths

    cp_dict = {}
    vis2_dict = {}
    cvis_dict = {}
    for wl in wavelengths:
        cp_dict[float(wl)] = {
            "raw": np.zeros((5, n_tri), dtype=complex),
            "closure_phases": np.ones(n_tri) * 10.0,
            "triple_amps": np.ones(n_tri) * 0.8,
            "covariance": np.eye(n_tri),
            "variance": np.ones(n_tri),
            "std_error": np.ones(n_tri) * 2.0,
        }
        vis2_dict[float(wl)] = {
            "v2": np.ones(n_bl) * 0.9,
            "covariance": np.eye(n_bl),
            "variance": np.ones(n_bl) * 0.01,
            "std_error": np.ones(n_bl) * 0.05,
            "v2_scatter": np.ones((5, n_bl)) * 0.9,
            "amplitudes": np.ones(5) * 100.0,
            "unnormalized": np.ones((5, n_bl)) * 90.0,
            "bias": np.ones((5, n_bl)) * 0.01,
        }
        cvis_dict[float(wl)] = {
            "amplitudes": np.ones(n_bl) * 0.95,
            "phases": np.ones(n_bl) * 5.0,
            "covariance": None,
            "variance": None,
            "std_error": None,
            "phases_per_image": np.ones((5, n_bl)) * 5.0,
        }

    return {
        "wavelengths": wavelengths,
        "cp": cp_dict,
        "vis2": vis2_dict,
        "compl_vis": cvis_dict,
    }


@pytest.fixture()
def sampy_extracted_cp_only(sampy_mock_block):
    """Create a SAMpy extracted dict with only CP."""
    n_tri = 1
    wavelengths = sampy_mock_block.wavelengths
    cp_dict = {}
    for wl in wavelengths:
        cp_dict[float(wl)] = {
            "raw": np.zeros((5, n_tri), dtype=complex),
            "closure_phases": np.ones(n_tri) * 15.0,
            "triple_amps": np.ones(n_tri) * 0.7,
            "covariance": np.eye(n_tri),
            "variance": np.ones(n_tri),
            "std_error": np.ones(n_tri) * 3.0,
        }
    return {
        "wavelengths": wavelengths,
        "cp": cp_dict,
    }


@pytest.fixture()
def sampy_extracted_vis2_only(sampy_mock_block):
    """Create a SAMpy extracted dict with only VIS2."""
    n_bl = 3
    wavelengths = sampy_mock_block.wavelengths
    vis2_dict = {}
    for wl in wavelengths:
        vis2_dict[float(wl)] = {
            "v2": np.ones(n_bl) * 0.85,
            "covariance": np.eye(n_bl),
            "variance": np.ones(n_bl),
            "std_error": np.ones(n_bl) * 0.04,
            "v2_scatter": np.ones((5, n_bl)),
            "amplitudes": np.ones(5) * 100.0,
            "unnormalized": np.ones((5, n_bl)),
            "bias": np.ones((5, n_bl)),
        }
    return {
        "wavelengths": wavelengths,
        "vis2": vis2_dict,
    }


@pytest.fixture()
def sampy_extracted_cvis_only(sampy_mock_block):
    """Create a SAMpy extracted dict with only complex visibility."""
    n_bl = 3
    wavelengths = sampy_mock_block.wavelengths
    cvis_dict = {}
    for wl in wavelengths:
        cvis_dict[float(wl)] = {
            "amplitudes": np.ones(n_bl) * 0.92,
            "phases": np.ones(n_bl) * -3.0,
            "covariance": None,
            "variance": None,
            "std_error": None,
            "phases_per_image": np.ones((5, n_bl)) * -3.0,
        }
    return {
        "wavelengths": wavelengths,
        "compl_vis": cvis_dict,
    }


class TestBuildObservablesFromSampy:
    """Tests for build_observables_from_sampy."""

    def test_returns_observables(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """Returns an Observables instance."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert isinstance(obs, Observables)

    def test_extraction_backend_set(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """extraction_backend is set to 'sampy'."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert obs.extraction_backend == "sampy"

    def test_vis2_shape(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """vis2 has shape (n_baselines, n_wav)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert obs.vis2.shape == (3, 3)

    def test_vis2_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """vis2 contains correct values from SAMpy result."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.vis2, 0.9)

    def test_vis2_err_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """vis2_err contains std_error from SAMpy."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.vis2_err, 0.05)

    def test_vis2_flag_all_false(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """vis2_flag is all False (valid data)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert not np.any(obs.vis2_flag)

    def test_t3phi_shape(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """t3phi has shape (n_triangles, n_wav)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert obs.t3phi.shape == (1, 3)

    def test_t3phi_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """t3phi contains closure_phases from SAMpy."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.t3phi, 10.0)

    def test_t3phi_err_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """t3phi_err contains std_error from SAMpy CP result."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.t3phi_err, 2.0)

    def test_t3amp_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """t3amp contains triple_amps from SAMpy."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.t3amp, 0.8)

    def test_t3amp_err_is_nan(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """t3amp_err is all NaN (not provided by SAMpy)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert np.all(np.isnan(obs.t3amp_err))

    def test_t3_flag_all_false(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """t3_flag is all False (valid data)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert not np.any(obs.t3_flag)

    def test_visamp_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """Visamp contains amplitudes from SAMpy calc_cvis."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.visamp, 0.95)

    def test_visphi_values(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """Visphi contains phases from SAMpy calc_cvis."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        np.testing.assert_allclose(obs.visphi, 5.0)

    def test_visamp_err_is_nan(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """visamp_err is all NaN (not provided by SAMpy)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert np.all(np.isnan(obs.visamp_err))

    def test_visphi_err_is_nan(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """visphi_err is all NaN (not provided by SAMpy)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert np.all(np.isnan(obs.visphi_err))

    def test_vis_flag_all_false(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """vis_flag is all False (valid data)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert not np.any(obs.vis_flag)

    def test_cp_only_extraction(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_cp_only
    ):
        """Only CP arrays set when only CP is extracted."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_cp_only,
        )
        assert obs.has_t3phi
        assert obs.has_t3amp
        assert not obs.has_vis2
        assert not obs.has_visamp
        assert not obs.has_visphi

    def test_vis2_only_extraction(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_vis2_only
    ):
        """Only VIS2 arrays set when only VIS2 is extracted."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_vis2_only,
        )
        assert obs.has_vis2
        assert not obs.has_t3phi
        assert not obs.has_visamp

    def test_cvis_only_extraction(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_cvis_only
    ):
        """Only visamp, visphi set when only compl_vis in extracted."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_cvis_only,
        )
        assert obs.has_visamp
        assert obs.has_visphi
        assert not obs.has_vis2
        assert not obs.has_t3phi

    def test_std_error_none_leaves_nan(
        self, sampy_mock_block, three_hole_mask
    ):
        """When SAMpy std_error is None, error arrays remain NaN."""
        n_bl = 3
        n_tri = 1
        wavelengths = sampy_mock_block.wavelengths

        cp_dict = {}
        vis2_dict = {}
        for wl in wavelengths:
            cp_dict[float(wl)] = {
                "raw": np.zeros((1, n_tri), dtype=complex),
                "closure_phases": np.ones(n_tri) * 5.0,
                "triple_amps": np.ones(n_tri) * 0.5,
                "covariance": None,
                "variance": None,
                "std_error": None,
            }
            vis2_dict[float(wl)] = {
                "v2": np.ones(n_bl) * 0.7,
                "covariance": None,
                "variance": None,
                "std_error": None,
                "v2_scatter": np.ones((1, n_bl)),
                "amplitudes": np.ones(1),
                "unnormalized": np.ones((1, n_bl)),
                "bias": np.ones((1, n_bl)),
            }

        extracted = {
            "wavelengths": wavelengths,
            "cp": cp_dict,
            "vis2": vis2_dict,
        }

        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            extracted,
        )
        assert np.all(np.isnan(obs.t3phi_err))
        assert np.all(np.isnan(obs.vis2_err))

    def test_validates_after_build(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """Built Observables passes validate."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        obs.validate()

    def test_wavelength_column_mapping_vis2(
        self, sampy_mock_block, three_hole_mask
    ):
        """Each wavelength maps to the correct column for VIS2."""
        n_bl = 3
        wavelengths = sampy_mock_block.wavelengths

        vis2_dict = {}
        for i, wl in enumerate(wavelengths):
            # Each wavelength gets a distinct value
            vis2_dict[float(wl)] = {
                "v2": np.ones(n_bl) * (i + 1) * 0.1,
                "covariance": None,
                "variance": None,
                "std_error": None,
                "v2_scatter": np.ones((1, n_bl)),
                "amplitudes": np.ones(1),
                "unnormalized": np.ones((1, n_bl)),
                "bias": np.ones((1, n_bl)),
            }

        extracted = {"wavelengths": wavelengths, "vis2": vis2_dict}
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            extracted,
        )

        np.testing.assert_allclose(obs.vis2[:, 0], 0.1)
        np.testing.assert_allclose(obs.vis2[:, 1], 0.2)
        np.testing.assert_allclose(obs.vis2[:, 2], 0.3)

    def test_wavelength_column_mapping_cp(
        self, sampy_mock_block, three_hole_mask
    ):
        """Each wavel. maps to the correct column for closure phases."""
        n_tri = 1
        wavelengths = sampy_mock_block.wavelengths

        cp_dict = {}
        for i, wl in enumerate(wavelengths):
            cp_dict[float(wl)] = {
                "raw": np.zeros((1, n_tri), dtype=complex),
                "closure_phases": np.ones(n_tri) * (i + 1) * 10.0,
                "triple_amps": np.ones(n_tri) * (i + 1) * 0.1,
                "covariance": None,
                "variance": None,
                "std_error": None,
            }

        extracted = {"wavelengths": wavelengths, "cp": cp_dict}
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            extracted,
        )

        np.testing.assert_allclose(obs.t3phi[:, 0], 10.0)
        np.testing.assert_allclose(obs.t3phi[:, 1], 20.0)
        np.testing.assert_allclose(obs.t3phi[:, 2], 30.0)
        np.testing.assert_allclose(obs.t3amp[:, 0], 0.1)
        np.testing.assert_allclose(obs.t3amp[:, 1], 0.2)
        np.testing.assert_allclose(obs.t3amp[:, 2], 0.3)

    def test_wavelength_column_mapping_cvis(
        self, sampy_mock_block, three_hole_mask
    ):
        """Each wavel. maps to the correct column for complex vis."""
        n_bl = 3
        wavelengths = sampy_mock_block.wavelengths

        cvis_dict = {}
        for i, wl in enumerate(wavelengths):
            cvis_dict[float(wl)] = {
                "amplitudes": np.ones(n_bl) * (i + 1) * 0.1,
                "phases": np.ones(n_bl) * (i + 1) * 5.0,
                "covariance": None,
                "variance": None,
                "std_error": None,
                "phases_per_image": np.ones((1, n_bl)),
            }

        extracted = {"wavelengths": wavelengths, "compl_vis": cvis_dict}
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            extracted,
        )

        np.testing.assert_allclose(obs.visamp[:, 0], 0.1)
        np.testing.assert_allclose(obs.visamp[:, 1], 0.2)
        np.testing.assert_allclose(obs.visamp[:, 2], 0.3)
        np.testing.assert_allclose(obs.visphi[:, 0], 5.0)
        np.testing.assert_allclose(obs.visphi[:, 1], 10.0)
        np.testing.assert_allclose(obs.visphi[:, 2], 15.0)

    def test_visamp_shape(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """Visamp has shape (n_baselines, n_wav)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert obs.visamp.shape == (3, 3)

    def test_visphi_shape(
        self, sampy_mock_block, three_hole_mask, sampy_extracted_full
    ):
        """Visphi has shape (n_baselines, n_wav)."""
        obs = build_observables_from_sampy(
            sampy_mock_block,
            three_hole_mask,
            sampy_extracted_full,
        )
        assert obs.visphi.shape == (3, 3)
