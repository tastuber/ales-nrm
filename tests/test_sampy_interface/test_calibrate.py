"""Tests for ales_nrm.sampy_interface.calibrate module.

Calibration test fixtures model a CAL-SCI-CAL sequence:
  cal_block_1 (07:52:30 mean) -> sci_block (08:10:00 mean)
    -> cal_block_2 (08:32:30 mean)

CP and VIS2 values differ between blocks to verify that the
correct data is assembled and passed to SAMpy's
polynomial_calibrate. Science CP=15°, cal1 CP=2°, cal2
CP=3°; science VIS2=0.85, cal1 VIS2=1.0, cal2 VIS2=0.98.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ales_nrm.observables import (
    BaselineInfo,
    Observables,
    Station,
    TriangleInfo,
)
from ales_nrm.sampy_interface.calibrate import (
    _assemble_observables_for_wavelength,
    _get_block_mean_time_seconds,
    _get_block_times,
    _validate_raw_extraction,
    calibrate_block,
    calibrate_sequence,
)


@pytest.fixture()
def make_mock_block():
    """Factory for mock blocks with SAMpy extractions.

    Blocks differ in timestamps (for temporal calibration
    polynomial fitting) and observable values (to verify
    calibration arithmetic produces correct results).
    """

    def _make(
        target: str,
        timestamps: list[str],
        cp_val: float,
        vis2_val: float,
        wavelengths: np.ndarray,
        block_type: str = "SCI",
    ) -> MagicMock:
        block = MagicMock()
        block.target = target
        block.block_type = MagicMock()
        block.block_type.value = block_type
        block.is_loaded = True
        block.timestamps = np.array(timestamps, dtype=object)
        block.parallactic_angles = np.zeros(len(timestamps))
        block.observation_date = None
        block.wavelengths = wavelengths

        n_tri = 1
        n_bl = 3
        cp_dict = {}
        vis2_dict = {}
        for wl in wavelengths:
            cp_dict[float(wl)] = {
                "closure_phases": (np.ones(n_tri) * cp_val),
                "triple_amps": np.ones(n_tri) * 0.8,
                "std_error": np.ones(n_tri) * 1.0,
            }
            vis2_dict[float(wl)] = {
                "v2": np.ones(n_bl) * vis2_val,
                "std_error": np.ones(n_bl) * 0.02,
            }

        block._raw_extraction = {
            "raw": {
                "backend": "sampy",
                "result": {
                    "wavelengths": wavelengths,
                    "cp": cp_dict,
                    "vis2": vis2_dict,
                },
            }
        }
        block.observables = {}

        stations = [
            Station(index=1, name="H1", x=-1.0, y=0.0, diameter=0.8),
            Station(index=2, name="H2", x=1.0, y=0.0, diameter=0.8),
            Station(index=3, name="H3", x=0.0, y=1.5, diameter=0.8),
        ]
        baselines_info = [
            BaselineInfo(sta_index=(1, 2), name="H1H2", u=2.0, v=0.0),
            BaselineInfo(sta_index=(1, 3), name="H1H3", u=1.0, v=1.5),
            BaselineInfo(sta_index=(2, 3), name="H2H3", u=-1.0, v=1.5),
        ]
        triangles_info = [
            TriangleInfo(
                sta_index=(1, 2, 3),
                name="H1-H2-H3",
                u1=2.0,
                v1=0.0,
                u2=-1.0,
                v2=1.5,
            ),
        ]
        block.observables["raw"] = Observables(
            target=target,
            wavelengths=wavelengths.copy(),
            stations=stations,
            baselines=baselines_info,
            triangles=triangles_info,
            mean_para_angle=0.0,
            mjd=None,
            time_start=timestamps[0],
            time_end=timestamps[-1],
            block_type=block_type,
            mask_name="test_mask",
        )

        return block

    return _make


@pytest.fixture()
def mock_sci_block(make_mock_block, sample_wavelengths_short):
    """Science block observed at ~08:10 UT."""
    return make_mock_block(
        target="SciTarget",
        timestamps=[
            "08:00:00.000",
            "08:10:00.000",
            "08:20:00.000",
        ],
        cp_val=15.0,
        vis2_val=0.85,
        wavelengths=sample_wavelengths_short,
        block_type="SCI",
    )


@pytest.fixture()
def mock_cal_block_1(make_mock_block, sample_wavelengths_short):
    """Calibrator block 1 observed at ~07:52 UT (before SCI)."""
    return make_mock_block(
        target="CalStar",
        timestamps=["07:50:00.000", "07:55:00.000"],
        cp_val=2.0,
        vis2_val=1.0,
        wavelengths=sample_wavelengths_short,
        block_type="CAL",
    )


@pytest.fixture()
def mock_cal_block_2(make_mock_block, sample_wavelengths_short):
    """Calibrator block 2 observed at ~08:32 UT (after SCI)."""
    return make_mock_block(
        target="CalStar",
        timestamps=["08:30:00.000", "08:35:00.000"],
        cp_val=3.0,
        vis2_val=0.98,
        wavelengths=sample_wavelengths_short,
        block_type="CAL",
    )


@pytest.fixture()
def patched_sampy():
    """Context-manager fixture that patches SAMpy modules.

    Yields the mock calibration module for assertion access.
    """

    class _Ctx:
        def __init__(self):
            self.mock_sampy = MagicMock()
            self.mock_calibration = MagicMock()
            self._patcher = None

        def __enter__(self):
            self._patcher = patch.dict(
                sys.modules,
                {
                    "sampy": self.mock_sampy,
                    "sampy.calibration": self.mock_calibration,
                },
            )
            self._patcher.start()
            return self

        def __exit__(self, *args):
            self._patcher.stop()

        def set_side_effect(self, fn):
            """Set the polynomial_calibrate mock."""
            self.mock_calibration.polynomial_calibrate = MagicMock(
                side_effect=fn
            )

    return _Ctx


def _zero_poly_cal(target, cal, t_times, c_times, order, dtype, **kw):
    """Mock returning zeros with correct shape."""
    n_obs = target.shape[1]
    n_pts = target.shape[0]
    return (
        np.zeros((n_obs, n_pts)),
        np.zeros((n_obs, n_pts)),
        np.zeros((n_obs, n_pts)),
        np.array([None]),
    )


class TestGetBlockMeanTimeSeconds:
    """Tests for _get_block_mean_time_seconds."""

    def test_simple_timestamps(self):
        """Computes correct mean for simple timestamps."""
        block = MagicMock()
        block.timestamps = np.array(
            ["08:00:00.000", "08:10:00.000"],
            dtype=object,
        )
        result = _get_block_mean_time_seconds(block)
        expected = 8 * 3600 + 5 * 60
        assert result == pytest.approx(expected)

    def test_midnight_crossing(self):
        """Handles midnight crossing correctly."""
        block = MagicMock()
        block.timestamps = np.array(
            ["23:55:00.000", "00:05:00.000"],
            dtype=object,
        )
        result = _get_block_mean_time_seconds(block)
        assert result == pytest.approx(86400.0)

    def test_empty_timestamps_raises(self):
        """Raises ValueError if no valid timestamps."""
        block = MagicMock()
        block.target = "test"
        block.timestamps = np.array([], dtype=object)
        with pytest.raises(ValueError, match="no valid timestamps"):
            _get_block_mean_time_seconds(block)


class TestValidateRawExtraction:
    """Tests for _validate_raw_extraction."""

    def test_valid_extraction(self, mock_sci_block):
        """Returns result for valid extraction."""
        result = _validate_raw_extraction(mock_sci_block, "raw")
        assert "cp" in result
        assert "vis2" in result

    def test_no_raw_extraction(self):
        """Raises if _raw_extraction is None."""
        block = MagicMock()
        block.target = "test"
        block._raw_extraction = None
        with pytest.raises(ValueError, match="no raw extraction"):
            _validate_raw_extraction(block, "raw")

    def test_missing_label(self, mock_sci_block):
        """Raises if label not found."""
        with pytest.raises(ValueError, match="no extraction"):
            _validate_raw_extraction(mock_sci_block, "nonexistent")

    def test_wrong_backend(self):
        """Raises if backend is not sampy."""
        block = MagicMock()
        block.target = "test"
        block._raw_extraction = {"raw": {"backend": "other", "result": {}}}
        with pytest.raises(ValueError, match="not 'sampy'"):
            _validate_raw_extraction(block, "raw")


class TestAssembleObservablesForWavelength:
    """Tests for _assemble_observables_for_wavelength."""

    def test_single_block_cp(self, mock_sci_block, sample_wavelengths_short):
        """Assembles CP for a single block."""
        wl = float(sample_wavelengths_short[0])
        result = _assemble_observables_for_wavelength(
            [mock_sci_block],
            "raw",
            "cp",
            "closure_phases",
            wl,
        )
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0], 15.0)

    def test_multiple_blocks_vis2(
        self,
        mock_cal_block_1,
        mock_cal_block_2,
        sample_wavelengths_short,
    ):
        """Assembles V2 for multiple blocks."""
        wl = float(sample_wavelengths_short[0])
        result = _assemble_observables_for_wavelength(
            [mock_cal_block_1, mock_cal_block_2],
            "raw",
            "vis2",
            "v2",
            wl,
        )
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[0], 1.0)
        np.testing.assert_allclose(result[1], 0.98)

    def test_missing_wavelength_raises(self, mock_sci_block):
        """Raises if wavelength not in extraction."""
        with pytest.raises(ValueError, match="missing wavelength"):
            _assemble_observables_for_wavelength(
                [mock_sci_block],
                "raw",
                "cp",
                "closure_phases",
                99.0,
            )

    def test_missing_observable_type_raises(self):
        """Raises if observable type not in extraction."""
        block = MagicMock()
        block.target = "test"
        block._raw_extraction = {
            "raw": {
                "backend": "sampy",
                "result": {
                    "wavelengths": np.array([3.0]),
                    "vis2": {3.0: {"v2": np.ones(3)}},
                },
            }
        }
        with pytest.raises(ValueError, match="does not contain"):
            _assemble_observables_for_wavelength(
                [block], "raw", "cp", "closure_phases", 3.0
            )


class TestGetBlockTimes:
    """Tests for _get_block_times."""

    def test_multiple_blocks(self, mock_sci_block, mock_cal_block_1):
        """Returns correct times for multiple blocks."""
        times = _get_block_times([mock_cal_block_1, mock_sci_block])
        assert times.shape == (2,)
        expected_cal = 7 * 3600 + 52 * 60 + 30
        assert times[0] == pytest.approx(expected_cal)


class TestCalibrateBlock:
    """Tests for calibrate_block."""

    def test_returns_observables(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Returns an Observables instance."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
            )
        assert isinstance(obs, Observables)

    def test_calibrated_flag_and_metadata(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Observables has calibrated=True and metadata."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                poly_order=0,
            )
        assert obs.calibrated is True
        assert "CalStar" in obs.calibrator_target
        assert "poly_order=0" in obs.notes

    def test_no_complex_vis_in_result(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Calibrated Observables has no visamp/visphi."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
            )
        assert obs.visamp is None
        assert obs.visphi is None
        assert obs.visamp_err is None
        assert obs.visphi_err is None
        assert obs.vis_flag is None

    def test_calls_per_wavelength_and_type(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """polynomial_calibrate called per wl per type."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
            )
            pc = ctx.mock_calibration.polynomial_calibrate
            # 2 wavelengths * 2 types = 4 calls
            assert pc.call_count == 4

    def test_display_forwarded(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Display parameter forwarded."""
        captured = []

        def _capture(*args, **kwargs):
            captured.append(kwargs.get("display"))
            return _zero_poly_cal(*args, **kwargs)

        with patched_sampy() as ctx:
            ctx.set_side_effect(_capture)
            calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_vis2=False,
                display=True,
            )
        assert all(d is True for d in captured)

    def test_display_default_false(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Display defaults to False."""
        captured = []

        def _capture(*args, **kwargs):
            captured.append(kwargs.get("display"))
            return _zero_poly_cal(*args, **kwargs)

        with patched_sampy() as ctx:
            ctx.set_side_effect(_capture)
            calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_vis2=False,
            )
        assert all(d is False for d in captured)

    def test_no_calibrators_raises(
        self,
        mock_sci_block,
        patched_sampy,
    ):
        """Raises if no calibrator blocks provided."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.raises(ValueError, match="At least one"):
                calibrate_block(mock_sci_block, [])

    def test_both_flags_false_raises(
        self,
        mock_sci_block,
        mock_cal_block_1,
        patched_sampy,
    ):
        """Raises if both calibrate flags False."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.raises(ValueError, match="At least one"):
                calibrate_block(
                    mock_sci_block,
                    [mock_cal_block_1],
                    calibrate_cp=False,
                    calibrate_vis2=False,
                )

    def test_poly_order_ge1_single_cal_raises(
        self,
        mock_sci_block,
        mock_cal_block_1,
        patched_sampy,
    ):
        """poly_order>=1 with single calibrator raises."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.raises(ValueError, match="requires at least 2"):
                calibrate_block(
                    mock_sci_block,
                    [mock_cal_block_1],
                    poly_order=1,
                    calibrate_vis2=False,
                )

    def test_poly_order_ge2_single_cal_raises(
        self,
        mock_sci_block,
        mock_cal_block_1,
        patched_sampy,
    ):
        """poly_order>=2 with single calibrator raises."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.raises(ValueError, match="requires at least 2"):
                calibrate_block(
                    mock_sci_block,
                    [mock_cal_block_1],
                    poly_order=2,
                )

    def test_import_error_when_sampy_unavailable(
        self,
        mock_sci_block,
        mock_cal_block_1,
    ):
        """Clear ImportError when SAMpy not installed."""
        with patch.dict(
            sys.modules,
            {
                "sampy": None,
                "sampy.calibration": None,
            },
        ):
            with pytest.raises(ImportError, match="SAMpy is required"):
                calibrate_block(
                    mock_sci_block,
                    [mock_cal_block_1],
                )

    def test_cp_only_leaves_vis2_none(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """CP-only calibration leaves VIS2 as None."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_cp=True,
                calibrate_vis2=False,
            )
        assert obs.t3phi is not None
        assert obs.vis2 is None

    def test_vis2_only_leaves_t3phi_none(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """V2-only calibration leaves t3phi as None."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_cp=False,
                calibrate_vis2=True,
            )
        assert obs.vis2 is not None
        assert obs.t3phi is None

    def test_arbitrary_block_can_calibrate(
        self,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Any block can calibrate any other block."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_cal_block_2,
                [mock_cal_block_1, mock_cal_block_2],
                poly_order=0,
            )
        assert isinstance(obs, Observables)
        assert obs.calibrated is True

    def test_scatter_used_as_error(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """cal_scatter is used as the error estimate."""
        scatter_val = 2.5

        def _scaled(target, cal, t_times, c_times, order, dtype, **kw):
            n_obs = target.shape[1]
            n_pts = target.shape[0]
            return (
                np.ones((n_obs, n_pts)) * 10.0,
                np.ones((n_obs, n_pts)) * 0.1,
                np.ones((n_obs, n_pts)) * scatter_val,
                np.array([None]),
            )

        with patched_sampy() as ctx:
            ctx.set_side_effect(_scaled)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_vis2=False,
            )
        np.testing.assert_allclose(obs.t3phi_err, scatter_val)

    def test_t3phi_shape(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Calibrated t3phi has correct shape."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_vis2=False,
            )
        # n_tri=1, n_wav=2
        assert obs.t3phi.shape == (1, 2)

    def test_vis2_shape(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Calibrated VIS2 has correct shape."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                calibrate_cp=False,
            )
        # n_bl=3, n_wav=2
        assert obs.vis2.shape == (3, 2)

    def test_missing_source_observables_raises(
        self,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Raises if source_label not in sci_block.observables."""
        block = MagicMock()
        block.target = "test"
        block.block_type = MagicMock()
        block.block_type.value = "SCI"
        block.timestamps = np.array(["08:00:00.000"], dtype=object)
        block.parallactic_angles = np.array([0.0])
        block.observation_date = None
        block.wavelengths = np.array([3.0])
        block._raw_extraction = {
            "raw": {
                "backend": "sampy",
                "result": {
                    "wavelengths": np.array([3.0]),
                    "cp": {3.0: {"closure_phases": np.zeros(1)}},
                    "vis2": {3.0: {"v2": np.ones(3)}},
                },
            }
        }
        block.observables = {}  # No "raw" key

        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.raises(ValueError, match="no Observables under label"):
                calibrate_block(
                    block,
                    [mock_cal_block_1, mock_cal_block_2],
                    poly_order=0,
                )

    def test_missing_cp_in_extraction_warns(
        self,
        make_mock_block,
        mock_cal_block_1,
        mock_cal_block_2,
        sample_wavelengths_short,
        patched_sampy,
    ):
        """Warns if calibrate_cp=True but no CP in extraction."""
        block = make_mock_block(
            target="test",
            timestamps=["08:00:00.000"],
            cp_val=0.0,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )
        # Remove CP from raw extraction to trigger the warning
        del block._raw_extraction["raw"]["result"]["cp"]

        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.warns(UserWarning, match="no CP"):
                calibrate_block(
                    block,
                    [mock_cal_block_1, mock_cal_block_2],
                    poly_order=0,
                    calibrate_cp=True,
                    calibrate_vis2=True,
                )

    def test_missing_vis2_in_extraction_warns(
        self,
        make_mock_block,
        mock_cal_block_1,
        mock_cal_block_2,
        sample_wavelengths_short,
        patched_sampy,
    ):
        """Warns if calibrate_vis2=True but no VIS2 in extraction."""
        block = make_mock_block(
            target="test",
            timestamps=["08:00:00.000"],
            cp_val=5.0,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )
        # Remove VIS2 from raw extraction to trigger the warning
        del block._raw_extraction["raw"]["result"]["vis2"]

        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            with pytest.warns(UserWarning, match="no VIS2"):
                calibrate_block(
                    block,
                    [mock_cal_block_1, mock_cal_block_2],
                    poly_order=0,
                    calibrate_cp=True,
                    calibrate_vis2=True,
                )


class TestCalibrateNumerical:
    """Numerical tests using real polynomial_calibrate.

    Verifies that the calibration module correctly
    assembles data and produces numerically correct results
    for different polynomial orders. These tests use
    SAMpy's actual calibration routine via import.
    """

    @pytest.mark.sampy
    def test_order0_single_cal_cp_naive(
        self,
        make_mock_block,
        sample_wavelengths_short,
    ):
        """Order 0, single cal: calibrated = sci - cal."""
        sci_cp = 20.0
        cal_cp = 7.0
        cal = make_mock_block(
            "Cal",
            ["07:00:00.000"],
            cp_val=cal_cp,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000"],
            cp_val=sci_cp,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal],
            poly_order=0,
            calibrate_vis2=False,
        )

        expected = sci_cp - cal_cp
        np.testing.assert_allclose(
            obs.t3phi,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    def test_order0_single_cal_vis2_naive(
        self,
        make_mock_block,
        sample_wavelengths_short,
    ):
        """Order 0, single cal: calibrated = sci / cal."""
        sci_v2 = 0.80
        cal_v2 = 0.95
        cal = make_mock_block(
            "Cal",
            ["07:00:00.000"],
            cp_val=0.0,
            vis2_val=cal_v2,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000"],
            cp_val=0.0,
            vis2_val=sci_v2,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal],
            poly_order=0,
            calibrate_cp=False,
        )

        expected = sci_v2 / cal_v2
        np.testing.assert_allclose(
            obs.vis2,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    def test_order0_multi_cal_cp_equals_mean(
        self,
        make_mock_block,
        sample_wavelengths_short,
    ):
        """Order 0, multi cal: calibrated = sci - mean(cals)."""
        sci_cp = 25.0
        cal_cp_1 = 4.0
        cal_cp_2 = 8.0
        cal1 = make_mock_block(
            "Cal",
            ["07:00:00.000"],
            cp_val=cal_cp_1,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        cal2 = make_mock_block(
            "Cal",
            ["09:00:00.000"],
            cp_val=cal_cp_2,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000"],
            cp_val=sci_cp,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal1, cal2],
            poly_order=0,
            calibrate_vis2=False,
        )

        mean_cal = (cal_cp_1 + cal_cp_2) / 2.0
        expected = sci_cp - mean_cal
        np.testing.assert_allclose(
            obs.t3phi,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    def test_order0_multi_cal_vis2_equals_mean(
        self,
        make_mock_block,
        sample_wavelengths_short,
    ):
        """Order 0, multi cal: calibrated = sci / mean(cals)."""
        sci_v2 = 0.75
        cal_v2_1 = 0.90
        cal_v2_2 = 1.10
        cal1 = make_mock_block(
            "Cal",
            ["07:00:00.000"],
            cp_val=0.0,
            vis2_val=cal_v2_1,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        cal2 = make_mock_block(
            "Cal",
            ["09:00:00.000"],
            cp_val=0.0,
            vis2_val=cal_v2_2,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000"],
            cp_val=0.0,
            vis2_val=sci_v2,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal1, cal2],
            poly_order=0,
            calibrate_cp=False,
        )

        mean_cal = (cal_v2_1 + cal_v2_2) / 2.0
        expected = sci_v2 / mean_cal
        np.testing.assert_allclose(
            obs.vis2,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    @pytest.mark.parametrize("poly_order", [0, 1])
    def test_cp_subtraction(
        self,
        make_mock_block,
        poly_order,
        sample_wavelengths_short,
    ):
        """Calibrated CP equals sci_cp - interpolated_cal_cp.

        With two identical calibrators, the instrumental
        CP is constant regardless of polynomial order.
        """
        cal_cp = 5.0
        sci_cp = 20.0
        cal1 = make_mock_block(
            "Cal",
            ["07:00:00.000", "07:10:00.000"],
            cp_val=cal_cp,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        cal2 = make_mock_block(
            "Cal",
            ["09:00:00.000", "09:10:00.000"],
            cp_val=cal_cp,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000", "08:10:00.000"],
            cp_val=sci_cp,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal1, cal2],
            poly_order=poly_order,
            calibrate_vis2=False,
        )

        expected = sci_cp - cal_cp
        np.testing.assert_allclose(
            obs.t3phi,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    @pytest.mark.parametrize("poly_order", [0, 1])
    def test_vis2_division(
        self,
        make_mock_block,
        poly_order,
        sample_wavelengths_short,
    ):
        """Calibrated VIS2 equals sci_v2 / interpolated_cal_v2.

        With identical calibrator VIS2 values, calibrated =
        sci / cal regardless of polynomial order.
        """
        cal_v2 = 0.95
        sci_v2 = 0.80
        cal1 = make_mock_block(
            "Cal",
            ["07:00:00.000", "07:10:00.000"],
            cp_val=0.0,
            vis2_val=cal_v2,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        cal2 = make_mock_block(
            "Cal",
            ["09:00:00.000", "09:10:00.000"],
            cp_val=0.0,
            vis2_val=cal_v2,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000", "08:10:00.000"],
            cp_val=0.0,
            vis2_val=sci_v2,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal1, cal2],
            poly_order=poly_order,
            calibrate_cp=False,
        )

        expected = sci_v2 / cal_v2
        np.testing.assert_allclose(
            obs.vis2,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    def test_linear_interpolation_cp(
        self,
        make_mock_block,
        sample_wavelengths_short,
    ):
        """Linear calibration interpolates between cal CPs.

        cal1 at t=7h has CP=2°, cal2 at t=9h has CP=6°.
        Science at t=8h (midpoint) sees instrumental
        CP = 4° (linear interpolation).
        Calibrated = 20 - 4 = 16.
        """
        cal1 = make_mock_block(
            "Cal",
            ["07:00:00.000"],
            cp_val=2.0,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        cal2 = make_mock_block(
            "Cal",
            ["09:00:00.000"],
            cp_val=6.0,
            vis2_val=1.0,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000"],
            cp_val=20.0,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal1, cal2],
            poly_order=1,
            calibrate_vis2=False,
        )

        expected = 20.0 - 4.0
        np.testing.assert_allclose(
            obs.t3phi,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    def test_linear_interpolation_vis2(
        self,
        make_mock_block,
        sample_wavelengths_short,
    ):
        """Linear calibration interpolates between cal V2s.

        cal1 at t=7h has VIS2=0.9, cal2 at t=9h has VIS2=1.1.
        Science at t=8h: instrumental VIS2 = 1.0.
        Calibrated = 0.8 / 1.0 = 0.8.
        """
        cal1 = make_mock_block(
            "Cal",
            ["07:00:00.000"],
            cp_val=0.0,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        cal2 = make_mock_block(
            "Cal",
            ["09:00:00.000"],
            cp_val=0.0,
            vis2_val=1.1,
            wavelengths=sample_wavelengths_short,
            block_type="CAL",
        )
        sci = make_mock_block(
            "Sci",
            ["08:00:00.000"],
            cp_val=0.0,
            vis2_val=0.8,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            [cal1, cal2],
            poly_order=1,
            calibrate_cp=False,
        )

        expected = 0.8 / 1.0
        np.testing.assert_allclose(
            obs.vis2,
            expected,
            atol=1e-10,
        )

    @pytest.mark.sampy
    @pytest.mark.parametrize("poly_order", [0, 1, 2, 3])
    def test_poly_orders_with_enough_calibrators(
        self,
        make_mock_block,
        poly_order,
        sample_wavelengths_short,
    ):
        """Various poly orders run without error.

        Uses 4 calibrators so orders 0-3 are all
        well-determined. All calibrators have the same CP
        so the result should be sci_cp - cal_cp = 10.
        """
        cal_cp = 5.0
        sci_cp = 15.0
        cal_blocks = []
        for hour in [7, 8, 9, 10]:
            cal_blocks.append(
                make_mock_block(
                    "Cal",
                    [f"{hour:02d}:00:00.000"],
                    cp_val=cal_cp,
                    vis2_val=1.0,
                    wavelengths=sample_wavelengths_short,
                    block_type="CAL",
                )
            )
        sci = make_mock_block(
            "Sci",
            ["08:30:00.000"],
            cp_val=sci_cp,
            vis2_val=0.9,
            wavelengths=sample_wavelengths_short,
            block_type="SCI",
        )

        obs = calibrate_block(
            sci,
            cal_blocks,
            poly_order=poly_order,
            calibrate_vis2=False,
        )

        # Constant calibrator => all orders give the same result
        expected = sci_cp - cal_cp
        np.testing.assert_allclose(
            obs.t3phi,
            expected,
            atol=1e-8,
        )


class TestCalibrateSequence:
    """Tests for calibrate_sequence."""

    def test_calibrates_all_sci_blocks(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """All SCI blocks get calibrated observables."""
        seq = MagicMock()
        seq.science_blocks = [mock_sci_block]
        seq.calibrator_blocks = [
            mock_cal_block_1,
            mock_cal_block_2,
        ]
        seq.name = "test_seq"

        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            calibrate_sequence(seq)

        assert "calibrated" in mock_sci_block.observables

    def test_custom_output_label(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Custom output_label used."""
        seq = MagicMock()
        seq.science_blocks = [mock_sci_block]
        seq.calibrator_blocks = [
            mock_cal_block_1,
            mock_cal_block_2,
        ]
        seq.name = "test_seq"

        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            calibrate_sequence(
                seq,
                output_label="my_cal",
                poly_order=0,
            )

        assert "my_cal" in mock_sci_block.observables

    def test_no_calibrators_raises(self, mock_sci_block):
        """Raises if no calibrator blocks in sequence."""
        seq = MagicMock()
        seq.science_blocks = [mock_sci_block]
        seq.calibrator_blocks = []
        seq.name = "test_seq"

        with pytest.raises(ValueError, match="No calibrator"):
            calibrate_sequence(seq)

    def test_no_science_blocks_no_error(self, mock_cal_block_1):
        """No error if no science blocks."""
        seq = MagicMock()
        seq.science_blocks = []
        seq.calibrator_blocks = [mock_cal_block_1]
        seq.name = "test_seq"

        # Should not raise
        calibrate_sequence(seq)

    def test_display_forwarded(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Display parameter forwarded through sequence."""
        captured = []

        def _capture(*args, **kwargs):
            captured.append(kwargs.get("display"))
            return _zero_poly_cal(*args, **kwargs)

        seq = MagicMock()
        seq.science_blocks = [mock_sci_block]
        seq.calibrator_blocks = [
            mock_cal_block_1,
            mock_cal_block_2,
        ]
        seq.name = "test_seq"

        with patched_sampy() as ctx:
            ctx.set_side_effect(_capture)
            calibrate_sequence(seq, display=True)

        assert all(d is True for d in captured)


class TestCalibrateObservablesMethod:
    """Tests for ObservingBlock.calibrate_observables."""

    def test_stores_under_output_label(
        self,
        mock_sci_block,
        mock_cal_block_1,
        mock_cal_block_2,
        patched_sampy,
    ):
        """Calibration stored under output_label."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1, mock_cal_block_2],
                poly_order=0,
            )
            mock_sci_block.observables["my_cal"] = obs

        assert "my_cal" in mock_sci_block.observables
        assert mock_sci_block.observables["my_cal"].calibrated is True

    def test_single_calibrator_order0(
        self,
        mock_sci_block,
        mock_cal_block_1,
        patched_sampy,
    ):
        """Single calibrator with poly_order=0 works."""
        with patched_sampy() as ctx:
            ctx.set_side_effect(_zero_poly_cal)
            obs = calibrate_block(
                mock_sci_block,
                [mock_cal_block_1],
                poly_order=0,
            )
        assert isinstance(obs, Observables)
