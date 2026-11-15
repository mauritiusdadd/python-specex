#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.cubex."""
import os
import json
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from specex.cubex import specex, load_specex_file
from specex.utils import convert_air_to_vac
from test import (
    get_muse_test_cube,
    TEST_DATA_PATH,
    TEST_OUTPUT_PATH,
)


@pytest.fixture
def synthetic_specex_setup(tmp_path):
    cube_path = tmp_path / "test_cube.fits"
    cat_path = tmp_path / "sources.cat"
    outdir = tmp_path / "extracted_spectra"

    # Create 3D data: 30 wavelength channels, 16x16 spatial
    data = np.ones((30, 16, 16), dtype=np.float32)
    # Add a bright artificial source at pixel (8, 8)
    data[:, 8, 8] = 10.0
    var = np.full((30, 16, 16), 0.1, dtype=np.float32)
    mask = np.zeros((30, 16, 16), dtype=np.int16)

    w = WCS(naxis=3)
    w.wcs.crpix = [8.0, 8.0, 1.0]
    w.wcs.cdelt = [-0.0001, 0.0001, 10.0]
    w.wcs.crval = [10.0, 20.0, 5000.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.wcs.cunit = ["deg", "deg", "Angstrom"]
    w.wcs.pc = np.eye(3)
    hdr = w.to_header()

    primary = fits.PrimaryHDU(data=data, header=hdr)
    primary.name = "DATA"
    var_hdu = fits.ImageHDU(data=var, header=hdr, name="STAT")
    mask_hdu = fits.ImageHDU(data=mask, header=hdr, name="MASK")

    hdul = fits.HDUList([primary, var_hdu, mask_hdu])
    hdul.writeto(cube_path)

    # Create a small catalog with RA/DEC matching pixel (8, 8)
    cat = Table({
        "NUMBER": [1],
        "ALPHA_J2000": [10.0],
        "DELTA_J2000": [20.0],
        "A_WORLD": [0.0005],
        "B_WORLD": [0.0005],
        "THETA_WORLD": [0.0],
    })
    cat.write(cat_path, format="ascii.commented_header")

    return str(cube_path), str(cat_path), str(outdir)


@pytest.fixture(scope="session")
def real_cube_file():
    cube = get_muse_test_cube()
    if not cube or not os.path.isfile(cube):
        pytest.skip("MUSE test cube not available.")
    return cube


@pytest.fixture(scope="session")
def test_output_dir():
    outdir = Path(TEST_OUTPUT_PATH)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def test_specex_extraction_catalog(synthetic_specex_setup):
    cube_path, cat_path, outdir = synthetic_specex_setup

    specex_options = [
        "--catalog", cat_path,
        "--mode", "circular_aperture",
        "--aperture-size", "0.5arcsec",
        "--weighting", "none",
        "--spec-hdu", "0",
        "--var-hdu", "1",
        "--outdir", outdir,
        cube_path,
    ]
    specex(options=specex_options)

    outdir_path = os.path.abspath(outdir)
    assert os.path.exists(outdir_path)
    fits_files = [f for f in os.listdir(outdir_path) if f.endswith(".fits")]
    assert len(fits_files) > 0


def test_specex_extraction_whitelight(synthetic_specex_setup):
    cube_path, cat_path, outdir = synthetic_specex_setup

    specex_options = [
        "--catalog", cat_path,
        "--mode", "circular_aperture",
        "--aperture-size", "0.5arcsec",
        "--weighting", "whitelight",
        "--spec-hdu", "0",
        "--var-hdu", "1",
        "--outdir", outdir,
        cube_path,
    ]
    specex(options=specex_options)
    assert os.path.exists(outdir)


def test_specex_catalog_real_cube(real_cube_file, test_output_dir):
    cat_file = os.path.join(TEST_DATA_PATH, "test_sources.cat")
    outdir = test_output_dir / "specex_catalog_real_cube"
    outdir.mkdir(parents=True, exist_ok=True)
    specex_options = [
        "--catalog", cat_file,
        "--mode", "circular_aperture",
        "--aperture-size", "0.8arcsec",
        "--weighting", "whitelight",
        "--no-nans",
        "--outdir", str(outdir),
        real_cube_file,
    ]
    specex(options=specex_options)
    fits_files = list(outdir.glob("spec_*.fits"))
    assert len(fits_files) > 0


def test_specex_regionfile_real_cube(real_cube_file, test_output_dir):
    reg_file = os.path.join(TEST_DATA_PATH, "test_regions.reg")
    outdir = test_output_dir / "specex_regionfile_real_cube"
    outdir.mkdir(parents=True, exist_ok=True)
    specex_options = [
        "--regionfile", reg_file,
        "--mode", "kron_ellipse",
        "--no-nans",
        "--outdir", str(outdir),
        real_cube_file,
    ]
    specex(options=specex_options)
    fits_files = list(outdir.glob("spec_*.fits"))
    assert len(fits_files) > 0


def test_specex_regionfile_wlight_real_cube(real_cube_file, test_output_dir):
    reg_file = os.path.join(TEST_DATA_PATH, "test_regions.reg")
    outdir = test_output_dir / "specex_regionfile_wlight_real_cube"
    outdir.mkdir(parents=True, exist_ok=True)
    specex_options = [
        "--regionfile", reg_file,
        "--mode", "kron_ellipse",
        "--weighting", "whitelight",
        "--no-nans",
        "--outdir", str(outdir),
        real_cube_file,
    ]
    specex(options=specex_options)
    fits_files = list(outdir.glob("spec_*.fits"))
    assert len(fits_files) > 0


def test_specex_regionfile_log_wlight_real_cube(
    real_cube_file, test_output_dir
):
    reg_file = os.path.join(TEST_DATA_PATH, "test_regions.reg")
    outdir = test_output_dir / "specex_regionfile_log_wlight_real_cube"
    outdir.mkdir(parents=True, exist_ok=True)
    specex_options = [
        "--regionfile", reg_file,
        "--mode", "kron_ellipse",
        "--weighting", "log-whitelight",
        "--no-nans",
        "--outdir", str(outdir),
        real_cube_file,
    ]
    specex(options=specex_options)
    fits_files = list(outdir.glob("spec_*.fits"))
    assert len(fits_files) > 0


def _make_wave_cube_and_cat(tmp_path, ctype3="AWAV", tag="air"):
    cube_path = tmp_path / f"cube_{tag}.fits"
    cat_path = tmp_path / f"cat_{tag}.fits"

    data = np.ones((20, 16, 16), dtype=np.float32)
    data[:, 8, 8] = 10.0
    var = np.full((20, 16, 16), 0.1, dtype=np.float32)
    mask = np.zeros((20, 16, 16), dtype=np.int16)

    w = WCS(naxis=3)
    w.wcs.crpix = [8.0, 8.0, 1.0]
    w.wcs.cdelt = [-0.0001, 0.0001, 10.0]
    w.wcs.crval = [10.0, 20.0, 5000.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", ctype3]
    w.wcs.cunit = ["deg", "deg", "Angstrom"]
    w.wcs.pc = np.eye(3)
    hdr = w.to_header()

    primary = fits.PrimaryHDU(data=data, header=hdr)
    primary.name = "DATA"
    var_hdu = fits.ImageHDU(data=var, header=hdr, name="STAT")
    mask_hdu = fits.ImageHDU(data=mask, header=hdr, name="MASK")
    fits.HDUList([primary, var_hdu, mask_hdu]).writeto(cube_path)

    cat = Table({
        "NUMBER": [1],
        "ALPHA_J2000": [10.0],
        "DELTA_J2000": [20.0],
    })
    cat.write(cat_path, overwrite=True)
    return cube_path, cat_path


def test_specex_wave_type_default_air_and_vac(tmp_path):
    air_cube, air_cat = _make_wave_cube_and_cat(
        tmp_path, ctype3="AWAV", tag="air"
    )
    out_air = tmp_path / "out_air"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(air_cat), "--aperture-size", "0.5",
        "--outdir", str(out_air), str(air_cube),
    ])
    air_spec = out_air / "spec_1.fits"
    assert air_spec.exists()

    with fits.open(air_spec) as hdul:
        assert hdul[0].header["WAVETYPE"] == "air"
        assert hdul[0].header["AIRORVAC"] == "air"
        assert hdul[0].header["VACUUM"] is False
        assert "VAC_METH" not in hdul[0].header
        assert hdul["SPECTRUM"].header["UCD1"] == "em.wl;obs.atmos"
        assert hdul["SPECTRUM"].header["TUCD1"] == "em.wl;obs.atmos"
        assert hdul["SPECTRUM"].header["WAVETYPE"] == "air"
        wl = hdul["SPECTRUM"].data["WAVELENGTH"]
        assert np.allclose(wl, 5000.0 + 10.0 * np.arange(20))

    loaded_air = load_specex_file(air_spec)
    assert loaded_air["wave_type"] == "air"

    vac_cube, vac_cat = _make_wave_cube_and_cat(
        tmp_path, ctype3="WAVE", tag="vac"
    )
    out_vac = tmp_path / "out_vac"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(vac_cat), "--aperture-size", "0.5",
        "--outdir", str(out_vac), str(vac_cube),
    ])
    vac_spec = out_vac / "spec_1.fits"
    with fits.open(vac_spec) as hdul:
        assert hdul[0].header["WAVETYPE"] == "vacuum"
        assert hdul[0].header["AIRORVAC"] == "vac"
        assert hdul[0].header["VACUUM"] is True
        assert hdul["SPECTRUM"].header["UCD1"] == "em.wl"
        assert hdul["SPECTRUM"].header["TUCD1"] == "em.wl"

    loaded_vac = load_specex_file(vac_spec)
    assert loaded_vac["wave_type"] == "vacuum"


def test_specex_to_vac_methods_and_overrides(tmp_path):
    air_cube, air_cat = _make_wave_cube_and_cat(
        tmp_path, ctype3="AWAV", tag="air2"
    )
    orig_wave = (5000.0 + 10.0 * np.arange(20)) * u.Angstrom

    # Default --to-vac (Greisen2006), placed right before the FITS cube path
    out_def = tmp_path / "out_to_vac_def"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(air_cat), "--aperture-size", "0.5",
        "--outdir", str(out_def), "--to-vac", str(air_cube),
    ])
    expected_greisen = convert_air_to_vac(
        orig_wave, method="Greisen2006"
    ).value
    with fits.open(out_def / "spec_1.fits") as hdul:
        assert hdul[0].header["WAVETYPE"] == "vacuum"
        assert hdul[0].header["AIRORVAC"] == "vac"
        assert hdul[0].header["VACUUM"] is True
        assert hdul[0].header["VAC_METH"] == "Greisen2006"
        assert hdul["SPECTRUM"].header["UCD1"] == "em.wl"
        assert hdul["SPECTRUM"].header["TUCD1"] == "em.wl"
        wl = hdul["SPECTRUM"].data["WAVELENGTH"]
        assert np.allclose(wl, expected_greisen)

    # Explicit method --to-vac Ciddor1996
    out_ciddor = tmp_path / "out_to_vac_ciddor"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(air_cat), "--aperture-size", "0.5",
        "--to-vac", "Ciddor1996", "--outdir", str(out_ciddor), str(air_cube),
    ])
    expected_ciddor = convert_air_to_vac(
        orig_wave, method="Ciddor1996"
    ).value
    with fits.open(out_ciddor / "spec_1.fits") as hdul:
        assert hdul[0].header["VAC_METH"] == "Ciddor1996"
        wl = hdul["SPECTRUM"].data["WAVELENGTH"]
        assert np.allclose(wl, expected_ciddor)

    # Override AWAV cube with --from-vac and --to-vac: no conversion occurs
    out_from_vac = tmp_path / "out_from_vac"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(air_cat), "--aperture-size", "0.5",
        "--from-vac", "--to-vac",
        "--outdir", str(out_from_vac), str(air_cube),
    ])
    with fits.open(out_from_vac / "spec_1.fits") as hdul:
        assert hdul[0].header["WAVETYPE"] == "vacuum"
        assert "VAC_METH" not in hdul[0].header
        wl = hdul["SPECTRUM"].data["WAVELENGTH"]
        assert np.allclose(wl, orig_wave.value)

    # Override WAVE cube with --from-air (no conversion -> propagates air)
    vac_cube, vac_cat = _make_wave_cube_and_cat(
        tmp_path, ctype3="WAVE", tag="vac2"
    )
    out_from_air = tmp_path / "out_from_air"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(vac_cat), "--aperture-size", "0.5",
        "--from-air", "--outdir", str(out_from_air), str(vac_cube),
    ])
    with fits.open(out_from_air / "spec_1.fits") as hdul:
        assert hdul[0].header["WAVETYPE"] == "air"
        assert hdul["SPECTRUM"].header["UCD1"] == "em.wl;obs.atmos"
        wl = hdul["SPECTRUM"].data["WAVELENGTH"]
        assert np.allclose(wl, orig_wave.value)

    # Override WAVE cube with --from-air and --to-vac: converts to vacuum
    out_from_air_vac = tmp_path / "out_from_air_vac"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(vac_cat), "--aperture-size", "0.5",
        "--from-air", "--to-vac",
        "--outdir", str(out_from_air_vac), str(vac_cube),
    ])
    with fits.open(out_from_air_vac / "spec_1.fits") as hdul:
        assert hdul[0].header["WAVETYPE"] == "vacuum"
        assert hdul[0].header["VAC_METH"] == "Greisen2006"
        wl = hdul["SPECTRUM"].data["WAVELENGTH"]
        assert np.allclose(wl, expected_greisen)

    # Mutually exclusive flags
    with pytest.raises(SystemExit):
        specex(["--from-air", "--from-vac", str(air_cube)])


def test_specex_stdout_modes(synthetic_specex_setup, capsys):
    cube_path, cat_path, outdir = synthetic_specex_setup
    out_root = Path(outdir).parent
    out_json = out_root / "out_json"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(cat_path),
        "--aperture-size", "0.5",
        "--stdout-mode", "json",
        "--outdir", str(out_json),
        cube_path,
    ])
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) > 0
    events = [json.loads(line) for line in lines]
    for entry in events:
        assert set(entry.keys()) == {'progress', 'total', 'description'}
    assert any(e['progress'] == e['total'] and e['total'] > 0 for e in events)

    out_quiet = out_root / "out_quiet"
    specex([
        "--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2",
        "--catalog", str(cat_path),
        "--aperture-size", "0.5",
        "-m", "quite",
        "--outdir", str(out_quiet),
        cube_path,
    ])
    assert capsys.readouterr().out == ''
