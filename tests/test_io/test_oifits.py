"""Tests for OIFITS2 writer and reader."""

import copy

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from ales_nrm.io.oifits import (
    _build_insname_map,
    _build_primary_hdu,
    _build_target_id_map,
    _compute_fallback_mjd,
    _extract_t3_arrays,
    _extract_vis2_arrays,
    _extract_vis_arrays,
    _format_time_for_filename,
    _reconstruct_baselines,
    _reconstruct_triangles,
    _sanitize_target_name,
    generate_oifits_filename,
    read_oifits,
    write_oifits,
)
from ales_nrm.observables import (
    BaselineInfo,
    Observables,
    Station,
    TriangleInfo,
)

# ------------------------------------------------------------------ #
# Shared geometry fixture
# ------------------------------------------------------------------ #


@pytest.fixture()
def geometry():
    """Three stations, three baselines, one triangle."""
    stations = [
        Station(index=1, name="H1", x=-1.0, y=0.0, diameter=0.8),
        Station(index=2, name="H2", x=1.0, y=0.0, diameter=0.8),
        Station(index=3, name="H3", x=0.0, y=1.5, diameter=0.8),
    ]
    baselines = [
        BaselineInfo(sta_index=(1, 2), name="H1H2", u=2.0, v=0.0),
        BaselineInfo(sta_index=(1, 3), name="H1H3", u=1.0, v=1.5),
        BaselineInfo(sta_index=(2, 3), name="H2H3", u=-1.0, v=1.5),
    ]
    triangles = [
        TriangleInfo(
            sta_index=(1, 2, 3),
            name="H1-H2-H3",
            u1=2.0,
            v1=0.0,
            u2=-1.0,
            v2=1.5,
        ),
    ]
    return stations, baselines, triangles


# ------------------------------------------------------------------ #
# Core Observables fixtures
# ----------------------------------------------------------------------


@pytest.fixture()
def full_observables(rng, sample_wavelengths_short, geometry):
    """Full Observables with all arrays populated."""
    stations, baselines, triangles = geometry
    n_bl = 3
    n_tri = 1
    n_wav = 2

    return Observables(
        target="alpha_Cen",
        wavelengths=sample_wavelengths_short.copy(),
        stations=stations,
        baselines=baselines,
        triangles=triangles,
        mean_para_angle=15.3,
        mjd=60000.0,
        time_start="12:00:00.000",
        time_end="12:05:00.000",
        vis2=rng.uniform(0.5, 1.0, size=(n_bl, n_wav)),
        vis2_err=rng.uniform(0.01, 0.05, size=(n_bl, n_wav)),
        vis2_flag=np.zeros((n_bl, n_wav), dtype=bool),
        t3phi=rng.uniform(-180, 180, size=(n_tri, n_wav)),
        t3phi_err=rng.uniform(1.0, 5.0, size=(n_tri, n_wav)),
        t3amp=rng.uniform(0.8, 1.0, size=(n_tri, n_wav)),
        t3amp_err=rng.uniform(0.01, 0.05, size=(n_tri, n_wav)),
        t3_flag=np.zeros((n_tri, n_wav), dtype=bool),
        visamp=rng.uniform(0.7, 1.0, size=(n_bl, n_wav)),
        visamp_err=rng.uniform(0.01, 0.05, size=(n_bl, n_wav)),
        visphi=rng.uniform(-180, 180, size=(n_bl, n_wav)),
        visphi_err=rng.uniform(1.0, 5.0, size=(n_bl, n_wav)),
        vis_flag=np.zeros((n_bl, n_wav), dtype=bool),
        calibrated=False,
        block_type="SCI",
        mask_name="test_mask",
    )


@pytest.fixture()
def minimal_observables(sample_wavelengths_short, geometry):
    """Minimal Observables with only geometry/metadata."""
    stations, baselines, triangles = geometry

    return Observables(
        target="alpha_Cen",
        wavelengths=sample_wavelengths_short.copy(),
        stations=stations,
        baselines=baselines,
        triangles=triangles,
        mean_para_angle=0.0,
        mjd=60000.0,
        time_start="12:00:00.000",
        time_end="12:05:00.000",
        calibrated=False,
        block_type="SCI",
        mask_name="test_mask",
    )


@pytest.fixture()
def output_dir(tmp_path):
    """Temporary output directory for written OIFITS files."""
    return tmp_path / "oifits_output"


# ------------------------------------------------------------------ #
# Helper for manually-built OIFITS HDULists
# ------------------------------------------------------------------ #


def _minimal_oifits_hdulist(
    *,
    n_stations=1,
    station_names=None,
    station_coords=None,
    arrname="test_arr",
    insname="TEST",
    eff_wave=None,
    target_name="test_target",
    target_id=1,
    category="SCI",
):
    """Build a minimal OIFITS2 HDUList with support tables.

    Returns the HDUList to which data tables can be appended.
    """
    hdu_list = fits.HDUList()
    primary = fits.PrimaryHDU()
    primary.header["CONTENT"] = "OIFITS2"
    hdu_list.append(primary)

    # OI_TARGET
    target_cols = [
        fits.Column(
            name="TARGET_ID",
            format="I",
            array=np.array([target_id], dtype=np.int16),
        ),
        fits.Column(name="TARGET", format="32A", array=[target_name]),
        fits.Column(name="RAEP0", format="D", array=np.zeros(1)),
        fits.Column(name="DECEP0", format="D", array=np.zeros(1)),
        fits.Column(
            name="EQUINOX",
            format="E",
            array=np.full(1, 2000.0),
        ),
        fits.Column(name="RA_ERR", format="D", array=np.zeros(1)),
        fits.Column(name="DEC_ERR", format="D", array=np.zeros(1)),
        fits.Column(name="SYSVEL", format="D", array=np.zeros(1)),
        fits.Column(name="VELTYP", format="8A", array=["UNKNOWN"]),
        fits.Column(name="VELDEF", format="8A", array=["OPTICAL"]),
        fits.Column(name="PMRA", format="D", array=np.zeros(1)),
        fits.Column(name="PMDEC", format="D", array=np.zeros(1)),
        fits.Column(name="PMRA_ERR", format="D", array=np.zeros(1)),
        fits.Column(name="PMDEC_ERR", format="D", array=np.zeros(1)),
        fits.Column(name="PARALLAX", format="E", array=np.zeros(1)),
        fits.Column(name="PARA_ERR", format="E", array=np.zeros(1)),
        fits.Column(name="SPECTYP", format="32A", array=[""]),
        fits.Column(name="CATEGORY", format="3A", array=[category]),
    ]
    target_hdu = fits.BinTableHDU.from_columns(target_cols)
    target_hdu.header["EXTNAME"] = "OI_TARGET"
    target_hdu.header["OI_REVN"] = 2
    hdu_list.append(target_hdu)

    # OI_ARRAY
    if station_names is None:
        station_names = [f"H{i + 1}" for i in range(n_stations)]
    if station_coords is None:
        station_coords = np.zeros((n_stations, 3))

    arr_cols = [
        fits.Column(
            name="TEL_NAME",
            format="16A",
            array=["LBT_SX"] * n_stations,
        ),
        fits.Column(
            name="STA_NAME",
            format="16A",
            array=station_names,
        ),
        fits.Column(
            name="STA_INDEX",
            format="I",
            array=np.arange(1, n_stations + 1, dtype=np.int16),
        ),
        fits.Column(
            name="DIAMETER",
            format="E",
            array=np.full(n_stations, 0.8, dtype=np.float32),
        ),
        fits.Column(
            name="STAXYZ",
            format="3D",
            array=station_coords,
        ),
        fits.Column(name="FOV", format="D", array=np.zeros(n_stations)),
        fits.Column(
            name="FOVTYPE",
            format="6A",
            array=["FWHM"] * n_stations,
        ),
    ]
    arr_hdu = fits.BinTableHDU.from_columns(arr_cols)
    arr_hdu.header["EXTNAME"] = "OI_ARRAY"
    arr_hdu.header["OI_REVN"] = 2
    arr_hdu.header["ARRNAME"] = arrname
    arr_hdu.header["FRAME"] = "SKY"
    arr_hdu.header["ARRAYX"] = 0.0
    arr_hdu.header["ARRAYY"] = 0.0
    arr_hdu.header["ARRAYZ"] = 0.0
    hdu_list.append(arr_hdu)

    # OI_WAVELENGTH
    if eff_wave is None:
        eff_wave = np.array([3.5e-6], dtype=np.float32)
    wl_cols = [
        fits.Column(
            name="EFF_WAVE",
            format="E",
            array=eff_wave,
        ),
        fits.Column(
            name="EFF_BAND",
            format="E",
            array=np.zeros_like(eff_wave),
        ),
    ]
    wl_hdu = fits.BinTableHDU.from_columns(wl_cols)
    wl_hdu.header["EXTNAME"] = "OI_WAVELENGTH"
    wl_hdu.header["OI_REVN"] = 2
    wl_hdu.header["INSNAME"] = insname
    hdu_list.append(wl_hdu)

    return hdu_list


# ------------------------------------------------------------------ #
# Test Classes
# ------------------------------------------------------------------ #


class TestSanitizeTargetName:
    """Tests for _sanitize_target_name."""

    @pytest.mark.parametrize(
        ("input_name", "expected"),
        [
            ("eps Hya", "eps_Hya"),
            ("HD+123/4", "HD1234"),
            ("AB-Dor", "AB-Dor"),
            ("eps_Hya", "eps_Hya"),
        ],
        ids=[
            "spaces_to_underscores",
            "special_chars_removed",
            "hyphens_preserved",
            "already_clean",
        ],
    )
    def test_sanitize(self, input_name, expected):
        """Target name sanitized correctly."""
        assert _sanitize_target_name(input_name) == expected


class TestFormatTimeForFilename:
    """Tests for _format_time_for_filename."""

    @pytest.mark.parametrize(
        ("time_str", "expected"),
        [
            ("12:05:30.123", "120530"),
            ("", "000000"),
            ("invalid", "000000"),
            ("14", "140000"),
            ("14:30", "143000"),
        ],
        ids=[
            "normal_time",
            "empty_time",
            "malformed_time",
            "hours_only",
            "hours_minutes_only",
        ],
    )
    def test_format(self, time_str, expected):
        """Time string formatted correctly for filename."""
        assert _format_time_for_filename(time_str) == expected


class TestComputeFallbackMjd:
    """Tests for _compute_fallback_mjd."""

    def test_with_time(self):
        """Computes MJD for 2000-01-01 with given time."""
        mjd = _compute_fallback_mjd("12:00:00.000")
        expected = float(Time("2000-01-01T12:00:00", scale="utc").mjd)
        assert abs(mjd - expected) < 1e-6

    @pytest.mark.parametrize(
        "time_str",
        ["", "xx:yy:zz"],
        ids=["empty", "malformed"],
    )
    def test_invalid_time_gives_midnight(self, time_str):
        """Invalid time gives midnight on 2000-01-01."""
        mjd = _compute_fallback_mjd(time_str)
        expected = float(Time("2000-01-01T00:00:00", scale="utc").mjd)
        assert abs(mjd - expected) < 1e-6


class TestGenerateOifitsFilename:
    """Tests for generate_oifits_filename."""

    def test_correct_format_all_metadata(self, full_observables):
        """Correct format with all metadata present."""
        fname = generate_oifits_filename(full_observables)
        assert fname.endswith(".fits")
        assert "alpha_Cen" in fname
        assert "raw" in fname
        assert "T" in fname

    def test_missing_date_uses_fallback(self, full_observables):
        """Missing date uses 20000101."""
        obs = copy.deepcopy(full_observables)
        obs.mjd = None
        fname = generate_oifits_filename(obs)
        assert "20000101" in fname

    def test_sanitizes_special_chars(self, full_observables):
        """Special characters sanitized in target name."""
        obs = copy.deepcopy(full_observables)
        obs.target = "HD+123/A B"
        fname = generate_oifits_filename(obs)
        assert "HD123A_B" in fname

    def test_custom_label_appended(self, full_observables):
        """Custom label appended when not raw/calibrated."""
        fname = generate_oifits_filename(full_observables, label="method2")
        assert "method2" in fname

    def test_raw_label_not_duplicated(self, full_observables):
        """Raw label does not add extra suffix."""
        fname = generate_oifits_filename(full_observables, label="raw")
        assert fname.count("raw") == 1

    def test_calibrated_label_maps_to_cal_state(self, full_observables):
        """Calibrated obs uses 'cal' state in filename."""
        obs = copy.deepcopy(full_observables)
        obs.calibrated = True
        fname = generate_oifits_filename(obs, label="calibrated")
        assert "_cal" in fname
        assert "calibrated" not in fname


class TestWriteOifitsSingleBlock:
    """Tests for writing a single-block OIFITS file."""

    def test_file_created(self, full_observables, output_dir):
        """File created at output_dir."""
        path = write_oifits(full_observables, output_dir)
        assert path.exists()
        assert path.suffix == ".fits"

    def test_explicit_filename(self, full_observables, output_dir):
        """Explicit filename is used when provided."""
        path = write_oifits(
            full_observables,
            output_dir,
            filename="custom.fits",
        )
        assert path.name == "custom.fits"

    def test_primary_header_keywords(self, full_observables, output_dir):
        """Primary header contains required OIFITS2 keywords."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            assert hdr["CONTENT"] == "OIFITS2"
            assert hdr["TELESCOP"] == "LBT"
            assert hdr["INSTRUME"] == "ALES"
            assert hdr["OBJECT"] == "alpha_Cen"
            assert hdr["INSMODE"] == "NRM"
            assert "ales-nrm" in hdr["PROCSOFT"]
            assert hdr["OBSTECH"] == "APERTURE_MASKING"

    def test_nonstandard_keywords(self, full_observables, output_dir):
        """Non-standard keywords present."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            assert "NS_ALVER" in hdr
            assert "NS_CRDAT" in hdr
            assert "NS_OBDAT" in hdr
            assert hdr["NS_CALST"] == "raw"

    def test_oi_target_present(self, full_observables, output_dir):
        """OI_TARGET table present with correct values."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            oi_target = hdul["OI_TARGET"]
            assert oi_target.header["OI_REVN"] == 2
            data = oi_target.data
            assert len(data) == 1
            assert int(data[0]["TARGET_ID"]) >= 1
            assert data[0]["TARGET"].strip() == "alpha_Cen"
            assert data[0]["CATEGORY"].strip() == "SCI"

    def test_oi_array_frame_sky(self, full_observables, output_dir):
        """OI_ARRAY has FRAME=SKY and correct stations."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            oi_array = hdul["OI_ARRAY"]
            assert oi_array.header["FRAME"] == "SKY"
            assert oi_array.header["OI_REVN"] == 2
            assert float(oi_array.header["ARRAYX"]) == 0.0
            assert float(oi_array.header["ARRAYY"]) == 0.0
            assert float(oi_array.header["ARRAYZ"]) == 0.0
            assert len(oi_array.data) == 3

    def test_oi_wavelength_in_meters(self, full_observables, output_dir):
        """OI_WAVELENGTH with correct wavelengths in meters."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            oi_wl = hdul["OI_WAVELENGTH"]
            assert oi_wl.header["OI_REVN"] == 2
            eff_wave = oi_wl.data["EFF_WAVE"]
            expected = full_observables.wavelengths * 1e-6
            np.testing.assert_allclose(eff_wave, expected, rtol=1e-5)

    def test_oi_vis2_shape(self, full_observables, output_dir):
        """OI_VIS2 present with correct row count and shapes."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            oi_vis2 = hdul["OI_VIS2"]
            assert oi_vis2.header["OI_REVN"] == 2
            assert len(oi_vis2.data) == 3
            assert oi_vis2.data[0]["VIS2DATA"].shape == (2,)

    def test_oi_vis_present(self, full_observables, output_dir):
        """OI_VIS present when visamp/visphi available."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            oi_vis = hdul["OI_VIS"]
            assert oi_vis.header["OI_REVN"] == 2
            assert oi_vis.header["AMPTYP"] == "absolute"
            assert oi_vis.header["PHITYP"] == "absolute"
            assert len(oi_vis.data) == 3

    def test_oi_t3_present(self, full_observables, output_dir):
        """OI_T3 present with correct u1, v1, u2, v2."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            oi_t3 = hdul["OI_T3"]
            assert oi_t3.header["OI_REVN"] == 2
            data = oi_t3.data
            assert len(data) == 1
            assert float(data[0]["U1COORD"]) == pytest.approx(2.0)
            assert float(data[0]["V1COORD"]) == pytest.approx(0.0)
            assert float(data[0]["U2COORD"]) == pytest.approx(-1.0)
            assert float(data[0]["V2COORD"]) == pytest.approx(1.5)
            assert len(data[0]["STA_INDEX"]) == 3

    def test_no_vis_table_when_absent(
        self, minimal_observables, rng, output_dir
    ):
        """No OI_VIS/OI_VIS2 when only t3 present."""
        obs = copy.deepcopy(minimal_observables)
        n_tri = 1
        n_wav = 2
        obs.t3phi = rng.uniform(-180, 180, size=(n_tri, n_wav))
        obs.t3phi_err = rng.uniform(1.0, 5.0, size=(n_tri, n_wav))
        obs.t3amp = rng.uniform(0.8, 1.0, size=(n_tri, n_wav))
        obs.t3amp_err = rng.uniform(0.01, 0.05, size=(n_tri, n_wav))
        obs.t3_flag = np.zeros((n_tri, n_wav), dtype=bool)

        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            extnames = [
                h.header.get("EXTNAME", "")
                for h in hdul
                if hasattr(h, "header")
            ]
            assert "OI_VIS" not in extnames
            assert "OI_VIS2" not in extnames

    def test_flag_defaults_false(self, full_observables, output_dir):
        """FLAG column defaults to False."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            for row in hdul["OI_VIS2"].data:
                assert not np.any(row["FLAG"])

    def test_mjd_and_time_columns(self, full_observables, output_dir):
        """MJD matches obs.mjd and TIME is zero."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            for row in hdul["OI_VIS2"].data:
                assert float(row["MJD"]) == pytest.approx(60000.0)
                assert float(row["TIME"]) == 0.0

    def test_sta_index_cross_reference(self, full_observables, output_dir):
        """STA_INDEX values reference OI_ARRAY correctly."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            valid = set(int(x) for x in hdul["OI_ARRAY"].data["STA_INDEX"])
            for row in hdul["OI_VIS2"].data:
                for idx in row["STA_INDEX"]:
                    assert int(idx) in valid

    def test_overwrite_false_raises(self, full_observables, output_dir):
        """overwrite=False raises if file exists."""
        write_oifits(full_observables, output_dir)
        with pytest.raises(FileExistsError):
            write_oifits(full_observables, output_dir)

    def test_overwrite_true(self, full_observables, output_dir):
        """overwrite=True overwrites existing file."""
        path = write_oifits(full_observables, output_dir)
        path2 = write_oifits(full_observables, output_dir, overwrite=True)
        assert path == path2
        assert path.exists()


class TestWriteOifitsMultiBlock:
    """Tests for writing multi-block OIFITS files."""

    @pytest.fixture()
    def multi_block_observables(self, full_observables, rng):
        """Two Observables with different targets and MJDs."""
        obs1 = copy.deepcopy(full_observables)
        obs2 = copy.deepcopy(full_observables)
        obs2.target = "HD_12345"
        obs2.mjd = 60001.0
        obs2.time_start = "12:30:00.000"
        obs2.time_end = "12:35:00.000"
        obs2.mean_para_angle = 20.1
        obs2.block_type = "CAL"
        # Regenerate arrays for different data
        n_bl = 3
        n_wav = 2
        n_tri = 1
        obs2.vis2 = rng.uniform(0.5, 1.0, size=(n_bl, n_wav))
        obs2.vis2_err = rng.uniform(0.01, 0.05, size=(n_bl, n_wav))
        obs2.t3phi = rng.uniform(-180, 180, size=(n_tri, n_wav))
        obs2.t3phi_err = rng.uniform(1.0, 5.0, size=(n_tri, n_wav))
        return [obs1, obs2]

    def test_multi_targets_object_multi(
        self, multi_block_observables, output_dir
    ):
        """Multiple targets sets OBJECT=MULTI."""
        path = write_oifits(multi_block_observables, output_dir)
        with fits.open(path) as hdul:
            assert hdul[0].header["OBJECT"] == "MULTI"

    def test_multi_target_ids(self, multi_block_observables, output_dir):
        """OI_TARGET has multiple rows with unique IDs >= 1."""
        path = write_oifits(multi_block_observables, output_dir)
        with fits.open(path) as hdul:
            data = hdul["OI_TARGET"].data
            assert len(data) == 2
            ids = [int(row["TARGET_ID"]) for row in data]
            assert all(i >= 1 for i in ids)
            assert len(set(ids)) == 2

    def test_data_tables_have_all_rows(
        self, multi_block_observables, output_dir
    ):
        """Data tables have rows from all blocks."""
        path = write_oifits(multi_block_observables, output_dir)
        with fits.open(path) as hdul:
            assert len(hdul["OI_VIS2"].data) == 6

    def test_single_oi_array(self, multi_block_observables, output_dir):
        """Single mask produces one OI_ARRAY table."""
        path = write_oifits(multi_block_observables, output_dir)
        with fits.open(path) as hdul:
            count = sum(
                1
                for h in hdul
                if hasattr(h, "header")
                and h.header.get("EXTNAME") == "OI_ARRAY"
            )
            assert count == 1

    def test_same_wavelength_grid_one_table(
        self, multi_block_observables, output_dir
    ):
        """Same wavelength grid produces one OI_WAVELENGTH."""
        path = write_oifits(multi_block_observables, output_dir)
        with fits.open(path) as hdul:
            count = sum(
                1
                for h in hdul
                if hasattr(h, "header")
                and h.header.get("EXTNAME") == "OI_WAVELENGTH"
            )
            assert count == 1

    def test_different_wavelength_grids(
        self, full_observables, rng, output_dir
    ):
        """Different wavelength grids produce multiple tables."""
        obs_a = copy.deepcopy(full_observables)
        obs_a.target = "A"

        # obs_b has a 3-channel wavelength grid
        n_bl = 3
        n_tri = 1
        n_wav_b = 3
        obs_b = copy.deepcopy(full_observables)
        obs_b.target = "B"
        obs_b.mjd = 60001.0
        obs_b.wavelengths = np.array([3.0, 3.5, 4.0])
        obs_b.vis2 = rng.uniform(0.5, 1.0, size=(n_bl, n_wav_b))
        obs_b.vis2_err = rng.uniform(0.01, 0.05, size=(n_bl, n_wav_b))
        obs_b.vis2_flag = np.zeros((n_bl, n_wav_b), dtype=bool)
        obs_b.visamp = None
        obs_b.visamp_err = None
        obs_b.visphi = None
        obs_b.visphi_err = None
        obs_b.vis_flag = None
        obs_b.t3phi = rng.uniform(-180, 180, size=(n_tri, n_wav_b))
        obs_b.t3phi_err = rng.uniform(1.0, 5.0, size=(n_tri, n_wav_b))
        obs_b.t3amp = rng.uniform(0.8, 1.0, size=(n_tri, n_wav_b))
        obs_b.t3amp_err = rng.uniform(0.01, 0.05, size=(n_tri, n_wav_b))
        obs_b.t3_flag = np.zeros((n_tri, n_wav_b), dtype=bool)

        path = write_oifits([obs_a, obs_b], output_dir)
        with fits.open(path) as hdul:
            wl_count = sum(
                1
                for h in hdul
                if hasattr(h, "header")
                and h.header.get("EXTNAME") == "OI_WAVELENGTH"
            )
            assert wl_count == 2
            vis2_count = sum(
                1
                for h in hdul
                if hasattr(h, "header")
                and h.header.get("EXTNAME") == "OI_VIS2"
            )
            assert vis2_count == 2

    def test_mixed_obs_some_without_vis2(
        self, full_observables, minimal_observables, output_dir
    ):
        """Observables without data skipped while writing."""
        obs_with = copy.deepcopy(full_observables)
        obs_with.target = "A"
        obs_without = copy.deepcopy(minimal_observables)
        obs_without.target = "B"
        obs_without.mjd = 60001.0

        path = write_oifits([obs_with, obs_without], output_dir)
        with fits.open(path) as hdul:
            assert len(hdul["OI_VIS2"].data) == 3
            assert len(hdul["OI_VIS"].data) == 3
            assert len(hdul["OI_T3"].data) == 1


class TestWriteOifitsEdgeCases:
    """Tests for edge cases in OIFITS writing."""

    def test_mjd_none_warning_and_fallback(self, full_observables, output_dir):
        """mjd=None triggers UserWarning and uses fallback date."""
        obs = copy.deepcopy(full_observables)
        obs.mjd = None
        with pytest.warns(UserWarning, match="mjd=None"):
            path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            assert hdul[0].header["DATE-OBS"] == "2000-01-01"
            assert hdul[0].header["NS_OBDAT"] == "2000-01-01"

    def test_calibrated_ns_keywords(self, full_observables, output_dir):
        """Calibrated observables include NS_CALTG and NS_CALMT."""
        obs = copy.deepcopy(full_observables)
        obs.calibrated = True
        obs.calibrator_target = "HD_12345"
        obs.notes = "poly_order=1, calibrate_cp=True"
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            assert hdr["NS_CALST"] == "calibrated"
            assert hdr["NS_CALTG"] == "HD_12345"
            assert "poly_order" in hdr["NS_CALMT"]

    def test_category_cal(self, full_observables, output_dir):
        """CATEGORY column set to CAL for calibrator block."""
        obs = copy.deepcopy(full_observables)
        obs.block_type = "CAL"
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            assert hdul["OI_TARGET"].data[0]["CATEGORY"].strip() == "CAL"

    def test_nan_errors_preserved(self, full_observables, output_dir):
        """NaN errors written correctly."""
        obs = copy.deepcopy(full_observables)
        obs.vis2_err[0, 0] = np.nan
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            assert np.isnan(hdul["OI_VIS2"].data[0]["VIS2ERR"][0])

    def test_vis2_err_none_fills_nan(self, full_observables, output_dir):
        """vis2_err=None fills NaN, vis2_flag=None fills False."""
        obs = copy.deepcopy(full_observables)
        obs.vis2_err = None
        obs.vis2_flag = None
        obs.visamp = None
        obs.visamp_err = None
        obs.visphi = None
        obs.visphi_err = None
        obs.vis_flag = None
        obs.t3phi = None
        obs.t3phi_err = None
        obs.t3amp = None
        obs.t3amp_err = None
        obs.t3_flag = None
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            row = hdul["OI_VIS2"].data[0]
            assert np.all(np.isnan(row["VIS2ERR"]))
            assert not np.any(row["FLAG"])

    @pytest.mark.parametrize(
        ("present_field", "absent_col", "table"),
        [
            ("visphi", "VISAMP", "OI_VIS"),
            ("visamp", "VISPHI", "OI_VIS"),
            ("t3phi", "T3AMP", "OI_T3"),
        ],
        ids=[
            "visamp_absent",
            "visphi_absent",
            "t3amp_absent",
        ],
    )
    def test_absent_complement_fills_nan(
        self,
        full_observables,
        rng,
        output_dir,
        present_field,
        absent_col,
        table,
    ):
        """Absent complement field filled with NaN."""
        obs = copy.deepcopy(full_observables)
        # Clear all vis/t3 fields, then selectively populate
        obs.vis2 = None
        obs.vis2_err = None
        obs.vis2_flag = None
        obs.visamp = None
        obs.visamp_err = None
        obs.visphi = None
        obs.visphi_err = None
        obs.vis_flag = None
        obs.t3phi = None
        obs.t3phi_err = None
        obs.t3amp = None
        obs.t3amp_err = None
        obs.t3_flag = None

        n_bl = 3
        n_tri = 1
        n_wav = 2

        if present_field == "visphi":
            obs.visphi = rng.uniform(-180, 180, size=(n_bl, n_wav))
            obs.visphi_err = rng.uniform(1.0, 5.0, size=(n_bl, n_wav))
        elif present_field == "visamp":
            obs.visamp = rng.uniform(0.7, 1.0, size=(n_bl, n_wav))
            obs.visamp_err = rng.uniform(0.01, 0.05, size=(n_bl, n_wav))
        elif present_field == "t3phi":
            obs.t3phi = rng.uniform(-180, 180, size=(n_tri, n_wav))
            obs.t3phi_err = rng.uniform(1.0, 5.0, size=(n_tri, n_wav))

        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            row = hdul[table].data[0]
            assert np.all(np.isnan(row[absent_col]))

    def test_single_wavelength_zero_bandwidth(
        self, minimal_observables, rng, output_dir
    ):
        """Single wavelength produces zero EFF_BAND."""
        obs = copy.deepcopy(minimal_observables)
        obs.wavelengths = np.array([3.5])
        n_bl = 3
        obs.vis2 = rng.uniform(0.5, 1.0, size=(n_bl, 1))
        obs.vis2_err = rng.uniform(0.01, 0.05, size=(n_bl, 1))
        obs.vis2_flag = np.zeros((n_bl, 1), dtype=bool)
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            assert float(hdul["OI_WAVELENGTH"].data["EFF_BAND"][0]) == 0.0

    def test_no_mask_name_arrname_fallback(self, full_observables, output_dir):
        """Empty mask_name produces ARRNAME='UNKNOWN'."""
        obs = copy.deepcopy(full_observables)
        obs.mask_name = ""
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            assert hdul["OI_ARRAY"].header["ARRNAME"] == "UNKNOWN"

    def test_empty_notes_no_ns_calmt(self, full_observables, output_dir):
        """Empty notes does not write NS_CALMT."""
        obs = copy.deepcopy(full_observables)
        obs.notes = ""
        path = write_oifits(obs, output_dir)
        with fits.open(path) as hdul:
            assert "NS_CALMT" not in hdul[0].header


class TestReadOifits:
    """Tests for reading OIFITS2 files."""

    @pytest.fixture()
    def round_trip(self, full_observables, output_dir):
        """Write and read back full_observables."""
        path = write_oifits(full_observables, output_dir)
        result = read_oifits(path)
        return result[0], full_observables

    def test_round_trip_target(self, round_trip):
        """Write then read recovers target name."""
        obs, _orig = round_trip
        assert obs.target == "alpha_Cen"

    def test_wavelengths_converted_back(self, round_trip):
        """Wavelengths converted back from meters to um."""
        obs, orig = round_trip
        np.testing.assert_allclose(
            obs.wavelengths, orig.wavelengths, rtol=1e-4
        )

    def test_station_geometry_reconstructed(self, round_trip):
        """Station geometry reconstructed correctly."""
        obs, _orig = round_trip
        assert len(obs.stations) == 3
        assert obs.stations[0].name == "H1"
        assert obs.stations[0].x == pytest.approx(-1.0)

    def test_baseline_uv_recovered(self, round_trip):
        """Baseline u, v recovered correctly."""
        obs, _orig = round_trip
        bl = next(b for b in obs.baselines if b.sta_index == (1, 2))
        assert bl.u == pytest.approx(2.0)
        assert bl.v == pytest.approx(0.0)

    def test_triangle_uv_recovered(self, round_trip):
        """Triangle u1, v1, u2, v2 recovered correctly."""
        obs, _orig = round_trip
        tri = obs.triangles[0]
        assert tri.u1 == pytest.approx(2.0)
        assert tri.v1 == pytest.approx(0.0)
        assert tri.u2 == pytest.approx(-1.0)
        assert tri.v2 == pytest.approx(1.5)

    def test_observable_arrays_match(self, round_trip):
        """Observable arrays match original within precision."""
        obs, orig = round_trip
        np.testing.assert_allclose(obs.vis2, orig.vis2, rtol=1e-10)
        np.testing.assert_allclose(obs.t3phi, orig.t3phi, rtol=1e-10)

    def test_flags_preserved(self, full_observables, output_dir):
        """Flags preserved through round-trip."""
        obs = copy.deepcopy(full_observables)
        obs.vis2_flag[1, 0] = True
        path = write_oifits(obs, output_dir)
        result = read_oifits(path)[0]
        assert result.vis2_flag[1, 0] is np.bool_(True)

    def test_metadata_recovered(self, round_trip):
        """Metadata: target, calibrated, mask_name, mjd."""
        obs, _orig = round_trip
        assert obs.target == "alpha_Cen"
        assert obs.calibrated is False
        assert obs.mask_name == "test_mask"
        assert obs.mjd == pytest.approx(60000.0)

    def test_multi_block_produces_multiple(
        self, full_observables, rng, output_dir
    ):
        """Multi-block file produces correct count."""
        obs1 = copy.deepcopy(full_observables)
        obs2 = copy.deepcopy(full_observables)
        obs2.target = "HD_12345"
        obs2.mjd = 60001.0
        n_bl = 3
        n_wav = 2
        obs2.vis2 = rng.uniform(0.5, 1.0, size=(n_bl, n_wav))
        obs2.vis2_err = rng.uniform(0.01, 0.05, size=(n_bl, n_wav))
        path = write_oifits([obs1, obs2], output_dir)
        assert len(read_oifits(path)) == 2

    def test_raises_on_non_oifits2(self, tmp_path):
        """Raises on non-OIFITS2 file."""
        hdu = fits.PrimaryHDU()
        hdu.header["CONTENT"] = "NOT_OIFITS"
        filepath = tmp_path / "bad.fits"
        hdu.writeto(filepath)
        with pytest.raises(ValueError, match="Not an OIFITS2"):
            read_oifits(filepath)

    def test_calibrated_state_round_trip(self, full_observables, output_dir):
        """Calibrated state persists through round-trip."""
        obs = copy.deepcopy(full_observables)
        obs.calibrated = True
        obs.calibrator_target = "HD_12345"
        obs.notes = "poly_order=1"
        path = write_oifits(obs, output_dir)
        result = read_oifits(path)[0]
        assert result.calibrated is True
        assert result.calibrator_target == "HD_12345"

    def test_block_type_from_category(self, full_observables, output_dir):
        """Block type derived from CATEGORY column."""
        obs = copy.deepcopy(full_observables)
        obs.block_type = "CAL"
        path = write_oifits(obs, output_dir)
        result = read_oifits(path)[0]
        assert result.block_type == "CAL"

    def test_missing_content_keyword(self, tmp_path):
        """File without CONTENT keyword raises ValueError."""
        hdu = fits.PrimaryHDU()
        filepath = tmp_path / "no_content.fits"
        hdu.writeto(filepath)
        with pytest.raises(ValueError, match="Not an OIFITS2"):
            read_oifits(filepath)


class TestReadOifitsEdgeCases:
    """Tests for reader edge cases."""

    def test_only_vis2(self, full_observables, output_dir):
        """File with only OI_VIS2 gives vis2 only."""
        obs = copy.deepcopy(full_observables)
        obs.visamp = None
        obs.visamp_err = None
        obs.visphi = None
        obs.visphi_err = None
        obs.vis_flag = None
        obs.t3phi = None
        obs.t3phi_err = None
        obs.t3amp = None
        obs.t3amp_err = None
        obs.t3_flag = None

        path = write_oifits(obs, output_dir)
        result = read_oifits(path)[0]
        assert result.vis2 is not None
        assert result.visamp is None
        assert result.t3phi is None

    def test_only_t3(self, full_observables, rng, output_dir):
        """File with only OI_T3 gives t3 only."""
        obs = copy.deepcopy(full_observables)
        obs.vis2 = None
        obs.vis2_err = None
        obs.vis2_flag = None
        obs.visamp = None
        obs.visamp_err = None
        obs.visphi = None
        obs.visphi_err = None
        obs.vis_flag = None

        path = write_oifits(obs, output_dir)
        result = read_oifits(path)[0]
        assert result.vis2 is None
        assert result.visamp is None
        assert result.t3phi is not None

    def test_only_vis_no_vis2(self, full_observables, output_dir):
        """File with only OI_VIS reconstructs baselines."""
        obs = copy.deepcopy(full_observables)
        obs.vis2 = None
        obs.vis2_err = None
        obs.vis2_flag = None
        obs.t3phi = None
        obs.t3phi_err = None
        obs.t3amp = None
        obs.t3amp_err = None
        obs.t3_flag = None

        path = write_oifits(obs, output_dir)
        result = read_oifits(path)[0]
        assert result.visamp is not None
        assert len(result.baselines) == 3
        assert result.vis2 is None

    def test_insname_mismatch_fallback(self, tmp_path):
        """Reader falls back to first wavelength on mismatch."""
        hdu_list = _minimal_oifits_hdulist(
            n_stations=1,
            insname="REAL_INS",
        )

        # OI_VIS2 with mismatched insname
        vis2_cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=np.array([1], dtype=np.int16),
            ),
            fits.Column(name="TIME", format="D", array=np.array([0.0])),
            fits.Column(
                name="MJD",
                format="D",
                array=np.array([60000.0]),
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                array=np.array([0.0]),
            ),
            fits.Column(
                name="VIS2DATA",
                format="1D",
                array=np.array([[0.9]]),
            ),
            fits.Column(
                name="VIS2ERR",
                format="1D",
                array=np.array([[0.01]]),
            ),
            fits.Column(
                name="UCOORD",
                format="D",
                array=np.array([1.0]),
            ),
            fits.Column(
                name="VCOORD",
                format="D",
                array=np.array([0.0]),
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=np.array([[1, 1]], dtype=np.int16),
            ),
            fits.Column(
                name="FLAG",
                format="1L",
                array=np.array([[False]]),
            ),
        ]
        vis2_hdu = fits.BinTableHDU.from_columns(vis2_cols)
        vis2_hdu.header["EXTNAME"] = "OI_VIS2"
        vis2_hdu.header["OI_REVN"] = 2
        vis2_hdu.header["DATE-OBS"] = "2023-11-08"
        vis2_hdu.header["ARRNAME"] = "test_arr"
        vis2_hdu.header["INSNAME"] = "OTHER_INS"
        hdu_list.append(vis2_hdu)

        filepath = tmp_path / "mismatch.fits"
        hdu_list.writeto(filepath)

        result = read_oifits(filepath)
        assert len(result) == 1
        assert result[0].wavelengths[0] == pytest.approx(3.5, rel=1e-3)

    def test_duplicate_baselines_deduplicated(self, tmp_path):
        """Duplicate STA_INDEX rows are deduplicated."""
        hdu_list = _minimal_oifits_hdulist(
            n_stations=2,
            station_names=["H1", "H2"],
            station_coords=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            arrname="test",
            insname="TEST",
        )

        # OI_VIS2 with duplicate STA_INDEX rows
        vis2_cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=np.array([1, 1], dtype=np.int16),
            ),
            fits.Column(
                name="TIME",
                format="D",
                array=np.array([0.0, 0.0]),
            ),
            fits.Column(
                name="MJD",
                format="D",
                array=np.array([60000.0, 60000.0]),
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                array=np.array([0.0, 0.0]),
            ),
            fits.Column(
                name="VIS2DATA",
                format="1D",
                array=np.array([[0.9], [0.85]]),
            ),
            fits.Column(
                name="VIS2ERR",
                format="1D",
                array=np.array([[0.01], [0.02]]),
            ),
            fits.Column(
                name="UCOORD",
                format="D",
                array=np.array([2.0, 2.0]),
            ),
            fits.Column(
                name="VCOORD",
                format="D",
                array=np.array([0.0, 0.0]),
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=np.array([[1, 2], [1, 2]], dtype=np.int16),
            ),
            fits.Column(
                name="FLAG",
                format="1L",
                array=np.array([[False], [False]]),
            ),
        ]
        vis2_hdu = fits.BinTableHDU.from_columns(vis2_cols)
        vis2_hdu.header["EXTNAME"] = "OI_VIS2"
        vis2_hdu.header["OI_REVN"] = 2
        vis2_hdu.header["DATE-OBS"] = "2023-11-08"
        vis2_hdu.header["ARRNAME"] = "test"
        vis2_hdu.header["INSNAME"] = "TEST"
        hdu_list.append(vis2_hdu)

        # OI_T3 with duplicate STA_INDEX rows
        t3_cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=np.array([1, 1], dtype=np.int16),
            ),
            fits.Column(
                name="TIME",
                format="D",
                array=np.array([0.0, 0.0]),
            ),
            fits.Column(
                name="MJD",
                format="D",
                array=np.array([60000.0, 60000.0]),
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                array=np.array([0.0, 0.0]),
            ),
            fits.Column(
                name="T3AMP",
                format="1D",
                array=np.array([[0.9], [0.8]]),
            ),
            fits.Column(
                name="T3AMPERR",
                format="1D",
                array=np.array([[0.01], [0.02]]),
            ),
            fits.Column(
                name="T3PHI",
                format="1D",
                array=np.array([[10.0], [20.0]]),
            ),
            fits.Column(
                name="T3PHIERR",
                format="1D",
                array=np.array([[1.0], [2.0]]),
            ),
            fits.Column(
                name="U1COORD",
                format="D",
                array=np.array([2.0, 2.0]),
            ),
            fits.Column(
                name="V1COORD",
                format="D",
                array=np.array([0.0, 0.0]),
            ),
            fits.Column(
                name="U2COORD",
                format="D",
                array=np.array([-1.0, -1.0]),
            ),
            fits.Column(
                name="V2COORD",
                format="D",
                array=np.array([1.5, 1.5]),
            ),
            fits.Column(
                name="STA_INDEX",
                format="3I",
                array=np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int16),
            ),
            fits.Column(
                name="FLAG",
                format="1L",
                array=np.array([[False], [False]]),
            ),
        ]
        t3_hdu = fits.BinTableHDU.from_columns(t3_cols)
        t3_hdu.header["EXTNAME"] = "OI_T3"
        t3_hdu.header["OI_REVN"] = 2
        t3_hdu.header["DATE-OBS"] = "2023-11-08"
        t3_hdu.header["ARRNAME"] = "test"
        t3_hdu.header["INSNAME"] = "TEST"
        hdu_list.append(t3_hdu)

        filepath = tmp_path / "duplicates.fits"
        hdu_list.writeto(filepath)

        result = read_oifits(filepath)
        assert len(result) == 1
        assert len(result[0].baselines) == 1
        assert len(result[0].triangles) == 1

    def test_target_id_not_in_oi_target(self, tmp_path):
        """Unknown TARGET_ID returns 'SCI' and 'UNKNOWN'."""
        hdu_list = _minimal_oifits_hdulist(
            n_stations=1,
            arrname="test",
            insname="TEST",
            target_name="known",
            target_id=1,
        )

        # OI_VIS2 with TARGET_ID=99
        vis2_cols = [
            fits.Column(
                name="TARGET_ID",
                format="I",
                array=np.array([99], dtype=np.int16),
            ),
            fits.Column(name="TIME", format="D", array=np.array([0.0])),
            fits.Column(
                name="MJD",
                format="D",
                array=np.array([60000.0]),
            ),
            fits.Column(
                name="INT_TIME",
                format="D",
                array=np.array([0.0]),
            ),
            fits.Column(
                name="VIS2DATA",
                format="1D",
                array=np.array([[0.9]]),
            ),
            fits.Column(
                name="VIS2ERR",
                format="1D",
                array=np.array([[0.01]]),
            ),
            fits.Column(
                name="UCOORD",
                format="D",
                array=np.array([1.0]),
            ),
            fits.Column(
                name="VCOORD",
                format="D",
                array=np.array([0.0]),
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=np.array([[1, 1]], dtype=np.int16),
            ),
            fits.Column(
                name="FLAG",
                format="1L",
                array=np.array([[False]]),
            ),
        ]
        vis2_hdu = fits.BinTableHDU.from_columns(vis2_cols)
        vis2_hdu.header["EXTNAME"] = "OI_VIS2"
        vis2_hdu.header["OI_REVN"] = 2
        vis2_hdu.header["DATE-OBS"] = "2023-11-08"
        vis2_hdu.header["ARRNAME"] = "test"
        vis2_hdu.header["INSNAME"] = "TEST"
        hdu_list.append(vis2_hdu)

        filepath = tmp_path / "unknown_target.fits"
        hdu_list.writeto(filepath)

        result = read_oifits(filepath)
        assert len(result) == 1
        assert result[0].block_type == "SCI"
        assert result[0].target == "UNKNOWN"


class TestOifitsStandardCompliance:
    """Tests for OIFITS2 standard compliance."""

    def test_oi_revn_in_all_tables(self, full_observables, output_dir):
        """OI_REVN = 2 in all OI extension tables."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            for hdu in hdul[1:]:
                extname = hdu.header.get("EXTNAME", "")
                if extname.startswith("OI_"):
                    assert hdu.header["OI_REVN"] == 2, (
                        f"OI_REVN != 2 in {extname}"
                    )

    def test_required_columns_oi_vis2(self, full_observables, output_dir):
        """OI_VIS2 contains all required columns."""
        path = write_oifits(full_observables, output_dir)
        required = {
            "TARGET_ID",
            "TIME",
            "MJD",
            "INT_TIME",
            "VIS2DATA",
            "VIS2ERR",
            "UCOORD",
            "VCOORD",
            "STA_INDEX",
            "FLAG",
        }
        with fits.open(path) as hdul:
            cols = set(hdul["OI_VIS2"].columns.names)
            assert required.issubset(cols)

    def test_required_columns_oi_t3(self, full_observables, output_dir):
        """OI_T3 contains all required columns."""
        path = write_oifits(full_observables, output_dir)
        required = {
            "TARGET_ID",
            "TIME",
            "MJD",
            "INT_TIME",
            "T3AMP",
            "T3AMPERR",
            "T3PHI",
            "T3PHIERR",
            "U1COORD",
            "V1COORD",
            "U2COORD",
            "V2COORD",
            "STA_INDEX",
            "FLAG",
        }
        with fits.open(path) as hdul:
            cols = set(hdul["OI_T3"].columns.names)
            assert required.issubset(cols)

    def test_eff_wave_positive(self, full_observables, output_dir):
        """EFF_WAVE values are positive."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            eff_wave = hdul["OI_WAVELENGTH"].data["EFF_WAVE"]
            assert np.all(eff_wave > 0)

    def test_sta_index_ge_one(self, full_observables, output_dir):
        """STA_INDEX values >= 1 in all data tables."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            for hdu in hdul[1:]:
                extname = hdu.header.get("EXTNAME", "")
                if extname in ("OI_VIS2", "OI_VIS", "OI_T3"):
                    for row in hdu.data:
                        assert np.all(np.array(row["STA_INDEX"]) >= 1)

    def test_target_id_ge_one(self, full_observables, output_dir):
        """TARGET_ID values >= 1."""
        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            for row in hdul["OI_TARGET"].data:
                assert int(row["TARGET_ID"]) >= 1

    def test_date_obs_format(self, full_observables, output_dir):
        """DATE-OBS format is YYYY-MM-DD."""
        import re

        path = write_oifits(full_observables, output_dir)
        with fits.open(path) as hdul:
            date_obs = hdul[0].header["DATE-OBS"]
            assert re.match(r"\d{4}-\d{2}-\d{2}$", date_obs)


class TestInternalHelpers:
    """Tests for internal helper functions."""

    def test_build_target_id_map_deduplicates(self, minimal_observables):
        """Duplicate target names get single ID."""
        result = _build_target_id_map(
            [minimal_observables, minimal_observables]
        )
        assert len(result) == 1
        assert result["alpha_Cen"] == 1

    def test_build_insname_map_multiple_grids(self, minimal_observables):
        """Different wavelength grids produce different insnames."""
        obs_a = copy.deepcopy(minimal_observables)
        obs_a.wavelengths = np.array([2.7, 4.3])
        obs_b = copy.deepcopy(minimal_observables)
        obs_b.wavelengths = np.array([3.0, 3.5, 4.0])
        insname_map = _build_insname_map([obs_a, obs_b], "ALES_NRM")
        assert insname_map[0] == "ALES_NRM"
        assert insname_map[1] == "ALES_NRM_1"

    def test_reconstruct_baselines_from_vis_rows(self, geometry):
        """Baselines reconstructed from vis_rows."""
        stations, _, _ = geometry
        group_data = {
            "vis2_rows": [],
            "vis_rows": [
                {
                    "STA_INDEX": np.array([1, 2], dtype=np.int16),
                    "UCOORD": 2.0,
                    "VCOORD": 0.0,
                },
                {
                    "STA_INDEX": np.array([1, 3], dtype=np.int16),
                    "UCOORD": 1.0,
                    "VCOORD": 1.5,
                },
            ],
            "t3_rows": [],
        }
        baselines = _reconstruct_baselines(group_data, stations)
        assert len(baselines) == 2
        assert baselines[0].sta_index == (1, 2)
        assert baselines[0].u == pytest.approx(2.0)

    def test_reconstruct_triangles_unknown_station(self):
        """Triangle with unknown station uses fallback name."""
        stations = [
            Station(index=1, name="H1", x=0.0, y=0.0),
            Station(index=2, name="H2", x=1.0, y=0.0),
        ]
        group_data = {
            "vis2_rows": [],
            "vis_rows": [],
            "t3_rows": [
                {
                    "STA_INDEX": np.array([1, 2, 99], dtype=np.int16),
                    "U1COORD": 1.0,
                    "V1COORD": 0.0,
                    "U2COORD": -0.5,
                    "V2COORD": 0.8,
                },
            ],
        }
        triangles = _reconstruct_triangles(group_data, stations)
        assert len(triangles) == 1
        assert "S99" in triangles[0].name

    def test_build_primary_hdu_mjd_none(self, minimal_observables):
        """Primary HDU uses fallback date when mjd is None."""
        obs = copy.deepcopy(minimal_observables)
        obs.mjd = None
        obs.time_start = ""
        obs.time_end = ""
        hdu = _build_primary_hdu([obs], {"alpha_Cen": 1})
        assert hdu.header["DATE-OBS"] == "2000-01-01"
        assert hdu.header["NS_OBDAT"] == "2000-01-01"

    def test_build_primary_hdu_no_calibrator(self, minimal_observables):
        """No NS_CALTG when calibrator_target is None."""
        obs = copy.deepcopy(minimal_observables)
        obs.calibrator_target = None
        obs.notes = ""
        hdu = _build_primary_hdu([obs], {"alpha_Cen": 1})
        assert "NS_CALTG" not in hdu.header
        assert "NS_CALMT" not in hdu.header

    @pytest.mark.parametrize(
        ("func", "row_key", "n_limit", "n_wav", "expected_len"),
        [
            (_extract_vis2_arrays, "vis2_rows", 2, 2, 3),
            (_extract_t3_arrays, "t3_rows", 1, 1, 5),
            (_extract_vis_arrays, "vis_rows", 1, 1, 5),
        ],
        ids=["vis2", "t3", "vis"],
    )
    def test_extract_arrays_overflow_guard(
        self, func, row_key, n_limit, n_wav, expected_len
    ):
        """Rows exceeding limit are ignored."""
        if row_key == "vis2_rows":
            rows = [
                {
                    "VIS2DATA": np.array([0.9, 0.8]),
                    "VIS2ERR": np.array([0.01, 0.02]),
                    "FLAG": np.array([False, False]),
                },
                {
                    "VIS2DATA": np.array([0.7, 0.6]),
                    "VIS2ERR": np.array([0.03, 0.04]),
                    "FLAG": np.array([False, True]),
                },
                {
                    "VIS2DATA": np.array([0.5, 0.4]),
                    "VIS2ERR": np.array([0.05, 0.06]),
                    "FLAG": np.array([True, True]),
                },
            ]
        elif row_key == "t3_rows":
            rows = [
                {
                    "T3PHI": np.array([10.0]),
                    "T3PHIERR": np.array([1.0]),
                    "T3AMP": np.array([0.9]),
                    "T3AMPERR": np.array([0.01]),
                    "FLAG": np.array([False]),
                },
                {
                    "T3PHI": np.array([20.0]),
                    "T3PHIERR": np.array([2.0]),
                    "T3AMP": np.array([0.8]),
                    "T3AMPERR": np.array([0.02]),
                    "FLAG": np.array([True]),
                },
            ]
        else:  # vis_rows
            rows = [
                {
                    "VISAMP": np.array([0.9]),
                    "VISAMPERR": np.array([0.01]),
                    "VISPHI": np.array([45.0]),
                    "VISPHIERR": np.array([1.0]),
                    "FLAG": np.array([False]),
                },
                {
                    "VISAMP": np.array([0.7]),
                    "VISAMPERR": np.array([0.02]),
                    "VISPHI": np.array([30.0]),
                    "VISPHIERR": np.array([2.0]),
                    "FLAG": np.array([True]),
                },
            ]

        group_data = {
            "vis2_rows": [],
            "vis_rows": [],
            "t3_rows": [],
        }
        group_data[row_key] = rows

        result = func(group_data, n_limit, n_wav)
        assert len(result) == expected_len
        assert result[0].shape == (n_limit, n_wav)

    @pytest.mark.parametrize(
        ("func", "n_first", "expected_len"),
        [
            (_extract_vis2_arrays, 3, 3),
            (_extract_t3_arrays, 1, 5),
            (_extract_vis_arrays, 3, 5),
        ],
        ids=["vis2_empty", "t3_empty", "vis_empty"],
    )
    def test_extract_arrays_empty_returns_none(
        self, func, n_first, expected_len
    ):
        """Empty rows return tuple of Nones."""
        group_data = {
            "vis2_rows": [],
            "vis_rows": [],
            "t3_rows": [],
        }
        result = func(group_data, n_first, 2)
        assert len(result) == expected_len
        assert all(x is None for x in result)
