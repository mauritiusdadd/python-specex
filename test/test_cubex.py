#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.cubex."""
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from specex.cubex import specex
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
