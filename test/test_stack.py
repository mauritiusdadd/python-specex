#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.stack."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import pytest

from specex.stack import stack_and_plot, cube_stack


@pytest.fixture
def dummy_fits_cube(tmp_path):
    cube_path = tmp_path / "test_cube.fits"

    # Create 3D data: 20 wavelength slices, 10x10 spatial
    data = np.ones((20, 10, 10), dtype=np.float32)
    for k in range(20):
        data[k] *= (k + 1)

    w = WCS(naxis=3)
    w.wcs.crpix = [5.0, 5.0, 1.0]
    w.wcs.cdelt = [-0.0001, 0.0001, 10.0]  # 10 Angstrom per slice
    w.wcs.crval = [10.0, 20.0, 5000.0]     # starts at 5000 A
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.wcs.cunit = ["deg", "deg", "Angstrom"]

    hdr = w.to_header()
    primary = fits.PrimaryHDU(data=data, header=hdr)
    primary.name = "DATA"
    var_data = np.full((20, 10, 10), 0.1, dtype=np.float32)
    var_hdu = fits.ImageHDU(data=var_data, header=hdr, name="STAT")

    hdul = fits.HDUList([primary, var_hdu])
    hdul.writeto(cube_path)
    return str(cube_path)


def test_stack_and_plot(tmp_path):
    data = np.ones((10, 8, 8), dtype=float)
    w = WCS(naxis=3)
    w.wcs.crpix = [4.0, 4.0, 1.0]
    w.wcs.cdelt = [-0.0001, 0.0001, 5.0]
    w.wcs.crval = [0.0, 0.0, 4000.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.wcs.cunit = ["deg", "deg", "Angstrom"]
    hdr = w.to_header()

    hdu = fits.PrimaryHDU(data=data, header=hdr)
    res = stack_and_plot(
        hdu=hdu,
        basename="sample",
        outdir=str(tmp_path),
        wave_ranges=[[4010, 4030]]
    )
    assert res is not None
    stacked, img_wcs = res
    assert stacked.shape == (8, 8)
    assert (tmp_path / "sample_.png").exists()
    assert (tmp_path / "sample_.fits").exists()


def test_cube_stack_cli(dummy_fits_cube, tmp_path):
    outdir = tmp_path / "stack_output"
    cube_stack([
        dummy_fits_cube,
        "--outdir", str(outdir),
        "--var-hdu", "STAT"
    ])
    assert outdir.exists()
    assert len(list(outdir.glob("*.fits"))) > 0
    assert len(list(outdir.glob("*.png"))) > 0
