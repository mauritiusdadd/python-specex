#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.cube."""
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
import pytest

from specex.cube import (
    SpectralCube,
    get_gray_cutout,
    get_rgb_cutout,
    smoothing_main,
    cutout_main,
    get_hdu,
)


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


def test_spectral_cube_open_and_properties(synthetic_cube):
    cube = SpectralCube.open(synthetic_cube, spec_hdu_index="DATA")
    assert cube.spec_hdu is not None
    assert cube.getSpecWCS() is not None
    assert cube.getVarWCS() is not None
    assert cube.getMaskWCS() is not None
    assert cube.getSpatialSizePixels() == (16, 16)

    # Test conversion to spectral_cube.SpectralCube
    sc = cube.to_spectral_cube()
    assert sc is not None
    assert sc.shape == (25, 16, 16)
    assert len(sc.spectral_axis) == 25

    # Test get_spectral_axis
    wave = cube.get_spectral_axis()
    assert len(wave) == 25
    assert wave.unit == u.Angstrom
    assert np.isclose(wave[0].value, 5000.0)

    # Test subcube_from_regions
    from regions import CircleSkyRegion
    reg = CircleSkyRegion(center=SkyCoord(10.0 * u.deg, 20.0 * u.deg),
                          radius=0.0003 * u.deg)
    sub = cube.subcube_from_regions([reg])
    assert sub is not None
    assert sub.shape[0] == 25

    # Test smooth_cube with physical units and mask
    from specex.cube import smooth_cube
    sm_data, sm_mask = smooth_cube(
        data=cube.spec_hdu.data,
        data_mask=cube.mask_hdu.data,
        spatial_size=0.5 * u.arcsec,
        wave_size=20.0 * u.Angstrom,
        cube_wcs=cube.getSpecWCS()
    )
    assert sm_data.shape == (25, 16, 16)
    assert sm_mask.shape == (25, 16, 16)

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


def test_smoothing_main(synthetic_cube, tmp_path):
    smoothing_main([
        "--spatial-smooth", "1,1",
        "--wave-smooth", "1",
        "--spec-hdu", "0",
        "--var-hdu", "1",
        "--mask-hdu", "2",
        synthetic_cube
    ])
    expected_out = "synth_cube_smoothed.fits"
    assert os.path.exists(expected_out)
    os.remove(expected_out)


def test_cutout_main(synthetic_cube, tmp_path):
    cutout_main([
        "--center", "10.0deg,20.0deg",
        "--size", "0.0005deg,0.0005deg",
        "--data-hdu", "0,1,2",
        synthetic_cube
    ])
    expected_out = "cutout_synth_cube_0000.fits"
    assert os.path.exists(expected_out)
    os.remove(expected_out)


def test_get_cube_cutout_wave_range(synthetic_cube):
    from specex.cube import get_cube_cutout
    cube = SpectralCube.open(synthetic_cube, spec_hdu_index="DATA")
    cutout = get_cube_cutout(
        data=cube.spec_hdu.data,
        center=SkyCoord(10.0 * u.deg, 20.0 * u.deg),
        size=(0.0004 * u.deg, 0.0004 * u.deg),
        data_wcs=cube.getSpecWCS(),
        wave_range=(5050 * u.Angstrom, 5150 * u.Angstrom)
    )
    assert cutout["data"].shape[0] == 11
    assert cutout["wcs"].has_spectral
    cube.close()
