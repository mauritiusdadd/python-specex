#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.cube."""
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from specex.cube import (
    SpectralCube,
    get_gray_cutout,
    get_rgb_cutout,
    smoothing_main,
    cutout_main,
    get_cube_cutout,
    smooth_cube,
)
from test import (
    get_muse_test_cube,
    get_hst_test_images,
    TEST_DATA_PATH,
    TEST_OUTPUT_PATH,
)

NICMOS_REGFILE_DATA = """
# Region file format: DS9 version 4.1
fk5
box(182.6350214,39.4064356,5.000",3.000",262.21022)
circle(182.6356792,39.4058053,2.500")
ellipse(182.6349624,39.4062180,2.500",1.500",217.21022)
"""


@pytest.fixture
def synthetic_cube(tmp_path):
    cube_path = tmp_path / "synth_cube.fits"

    # Shape: (wavelength=25, y=16, x=16)
    data = np.ones((25, 16, 16), dtype=np.float32)
    var = np.full((25, 16, 16), 0.05, dtype=np.float32)
    mask = np.zeros((25, 16, 16), dtype=np.int16)

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
    return str(cube_path)


@pytest.fixture(scope="session")
def real_or_fallback_cube(tmp_path_factory):
    cube = None
    try:
        cube = get_muse_test_cube()
    except Exception as exc:
        warnings.warn(
            f"Failed to obtain real MUSE cube: {exc}. Falling back to "
            "synthetic cube.",
            UserWarning
        )

    if cube and os.path.isfile(cube):
        return cube, True

    warnings.warn(
        "Real MUSE test cube is not available. Falling back to synthetic "
        "cube.",
        UserWarning
    )

    tmp_dir = tmp_path_factory.mktemp("fallback_cube")
    cube_path = tmp_dir / "synth_cube.fits"
    data = np.ones((25, 16, 16), dtype=np.float32)
    var = np.full((25, 16, 16), 0.05, dtype=np.float32)
    mask = np.zeros((25, 16, 16), dtype=np.int16)

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

    fits.HDUList([primary, var_hdu, mask_hdu]).writeto(cube_path)
    return str(cube_path), False


@pytest.fixture(scope="session")
def test_output_dir():
    outdir = Path(TEST_OUTPUT_PATH)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def test_spectral_cube_open_and_properties(real_or_fallback_cube):
    cube_path, is_real = real_or_fallback_cube
    cube = SpectralCube.open(cube_path, spec_hdu_index="DATA")
    assert cube.spec_hdu is not None
    assert cube.getSpecWCS() is not None
    assert cube.getSpatialSizePixels() is not None

    sc = cube.to_spectral_cube()
    assert sc is not None
    assert len(sc.shape) == 3

    wave = cube.get_spectral_axis()
    assert len(wave) == sc.shape[0]
    assert wave.unit == u.Angstrom

    if is_real:
        center = SkyCoord(16.3867 * u.deg, -24.6464 * u.deg)
        radius = 1.0 * u.arcsec
    else:
        center = SkyCoord(10.0 * u.deg, 20.0 * u.deg)
        radius = 0.0003 * u.deg

    reg = CircleSkyRegion(center=center, radius=radius)
    sub = cube.subcube_from_regions([reg])
    assert sub is not None
    assert sub.shape[0] == sc.shape[0]

    sub_slice = sub.unmasked_data[:5, :5, :5].value
    sm_data, _ = smooth_cube(
        data=sub_slice,
        spatial_size=0.5 * u.arcsec,
        wave_size=2.0 * u.Angstrom,
        cube_wcs=cube.getSpecWCS()
    )
    assert sm_data.shape == sub_slice.shape
    cube.close()


def test_get_gray_cutout():
    img = np.arange(100).reshape(10, 10).astype(float)
    cutout = get_gray_cutout(img, center=(5, 5), size=(4, 4), angle=0)
    assert cutout["data"].shape == (4, 4)


def test_get_rgb_cutout():
    r = np.ones((10, 10))
    g = np.full((10, 10), 2.0)
    b = np.full((10, 10), 3.0)
    cutout = get_rgb_cutout(
        data=(r, g, b),
        center=(5, 5),
        size=(4, 4),
        angle=0
    )
    assert len(cutout["data"]) == 3
    assert cutout["data"][0].shape == (4, 4)


def test_smoothing_main_cli(real_or_fallback_cube, test_output_dir):
    cube_path, is_real = real_or_fallback_cube
    outdir = test_output_dir / "smooth_cli"
    outdir.mkdir(parents=True, exist_ok=True)

    hdu_opts = []
    if not is_real:
        hdu_opts = ["--spec-hdu", "0", "--var-hdu", "1", "--mask-hdu", "2"]

    # 1. Spectral-only smoothing: wave-smooth=1
    out_wave = "smoothed_wave_1.fits"
    smoothing_main([
        "--verbose",
        "--wave-smooth", "1",
        "--outdir", str(outdir),
        "--output", out_wave,
        *hdu_opts,
        cube_path
    ])
    expected_wave = outdir / out_wave
    assert expected_wave.exists()
    assert expected_wave.stat().st_size > 0

    # 2. Spatial-only smoothing: spatial-smooth="1arcsec,0"
    out_spatial = "smoothed_spatial_1arcsec_0.fits"
    smoothing_main([
        "--verbose",
        "--spatial-smooth", "1arcsec,0",
        "--outdir", str(outdir),
        "--output", out_spatial,
        *hdu_opts,
        cube_path
    ])
    expected_spatial = outdir / out_spatial
    assert expected_spatial.exists()
    assert expected_spatial.stat().st_size > 0


def test_cutout_main_cli(real_or_fallback_cube, test_output_dir):
    cube_path, is_real = real_or_fallback_cube
    outdir = test_output_dir / "cutout_cli"
    outdir.mkdir(parents=True, exist_ok=True)

    if is_real:
        cutout_options = [
            "--center", "16.3867deg,-24.6464deg",
            "--size", "4arcsec,4arcsec",
            "--outdir", str(outdir),
            cube_path
        ]
    else:
        cutout_options = [
            "--center", "10.0deg,20.0deg",
            "--size", "0.0005deg,0.0005deg",
            "--data-hdu", "0,1,2",
            "--outdir", str(outdir),
            cube_path
        ]

    cutout_main(cutout_options)
    generated = list(outdir.glob(f"cutout_{Path(cube_path).stem}_*.fits"))
    assert len(generated) > 0


def test_get_cube_cutout_wave_range(real_or_fallback_cube):
    cube_path, is_real = real_or_fallback_cube
    cube = SpectralCube.open(cube_path, spec_hdu_index="DATA")

    if is_real:
        center = SkyCoord(16.3867 * u.deg, -24.6464 * u.deg)
        size = (2 * u.arcsec, 2 * u.arcsec)
        wave_range = (4800 * u.Angstrom, 4820 * u.Angstrom)
    else:
        center = SkyCoord(10.0 * u.deg, 20.0 * u.deg)
        size = (0.0004 * u.deg, 0.0004 * u.deg)
        wave_range = (5050 * u.Angstrom, 5150 * u.Angstrom)

    cutout = get_cube_cutout(
        data=cube.spec_hdu.data,
        center=center,
        size=size,
        data_wcs=cube.getSpecWCS(),
        wave_range=wave_range
    )
    assert cutout["data"].ndim == 3
    assert cutout["wcs"].has_spectral
    cube.close()


def test_cube_cutout_real_cube(real_or_fallback_cube, test_output_dir):
    cube_path, is_real = real_or_fallback_cube
    if not is_real:
        pytest.skip("Real cube not available.")

    outdir = test_output_dir / "cutout_real_cube"
    outdir.mkdir(parents=True, exist_ok=True)
    cutout_main([
        "--verbose",
        "--center", "16.3867deg,-24.6464deg",
        "--size", "6arcsec,3arcsec",
        "--angle", "45deg",
        "--outdir", str(outdir),
        cube_path,
    ])
    generated = list(outdir.glob("cutout_*.fits"))
    assert len(generated) > 0


def test_cube_cutout_regfile_real_cube(
    real_or_fallback_cube, test_output_dir
):
    cube_path, is_real = real_or_fallback_cube
    if not is_real:
        pytest.skip("Real cube not available.")

    reg_file = os.path.join(TEST_DATA_PATH, "test_regions.reg")
    outdir = test_output_dir / "cutout_regfile_real_cube"
    outdir.mkdir(parents=True, exist_ok=True)
    cutout_main([
        "--verbose",
        "--regionfile", reg_file,
        "--outdir", str(outdir),
        cube_path,
    ])
    generated = list(outdir.glob("cutout_*.fits"))
    assert len(generated) > 0


def test_grayscale_cutout_nicmos(test_output_dir):
    hst_imgs = get_hst_test_images()
    if not hst_imgs:
        pytest.skip("HST test images not available.")

    for img in hst_imgs:
        if img.endswith("NICMOSn4hk12010_mos.fits"):
            regfile = os.path.join(
                TEST_DATA_PATH, "NICMOSn4hk12010_mos.reg"
            )
            with open(regfile, "w") as f:
                f.write(NICMOS_REGFILE_DATA)

            outdir = test_output_dir / "cutout_nicmos"
            outdir.mkdir(parents=True, exist_ok=True)
            cutout_main([
                "--verbose",
                "--regionfile", regfile,
                "--outdir", str(outdir),
                img,
            ])
            generated = list(outdir.glob("cutout_*.fits"))
            assert len(generated) > 0
            break
