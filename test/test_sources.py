#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.sources."""
import numpy as np
import pytest
from specex.sources import (
    rebin_slice_2d,
    get_local_maxima,
    get_dilation_maxima,
    get_spectrum,
)


def test_rebin_slice_2d():
    img = np.ones((10, 10))
    rebinned = rebin_slice_2d(img, bin_size_x=2, bin_size_y=2)
    assert rebinned.shape == (5, 5)
    assert np.allclose(rebinned, 1.0)

    # Uneven dimensions with padding
    img_odd = np.ones((9, 9))
    reb_odd = rebin_slice_2d(img_odd, bin_size_x=2, bin_size_y=2)
    assert reb_odd.shape == (5, 5)

    # Invalid bin sizes
    with pytest.raises(ValueError):
        rebin_slice_2d(img, bin_size_x=0)
    with pytest.raises(ValueError):
        rebin_slice_2d(img, bin_size_x=1, bin_size_y=1)


def test_get_local_maxima():
    img = np.zeros((15, 15))
    img[5, 5] = 10.0
    img[10, 10] = 5.0
    tbl = get_local_maxima(img, threshold=1.0)
    assert len(tbl) >= 1
    assert "Y_IMAGE" in tbl.colnames
    assert "X_IMAGE" in tbl.colnames
    assert "SNR" in tbl.colnames


def test_get_dilation_maxima():
    img = np.zeros((15, 15))
    img[5, 5] = 10.0
    img[10, 10] = 5.0
    tbl = get_dilation_maxima(img, box_sizes=[3], threshold=1.0)
    assert len(tbl) >= 1
    assert "SNR" in tbl.colnames


def test_get_spectrum():
    # 50 spectral channels, 4 spaxels
    flux_spaxels = np.ones((50, 4))
    var_spaxels = np.full((50, 4), 0.25)
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    spec, spec_var = get_spectrum(
        flux_spaxels,
        var_spaxels=var_spaxels,
        weights=weights
    )
    assert spec.shape == (50,)
    assert spec_var.shape == (50,)
    assert np.allclose(spec, 4.0)

    # Without variance spaxels
    spec_novar, var_est = get_spectrum(flux_spaxels, var_spaxels=None)
    assert spec_novar.shape == (50,)
    assert var_est.shape == (50,)

    # Mismatched shape raises ValueError
    with pytest.raises(ValueError, match="different shape"):
        get_spectrum(flux_spaxels, var_spaxels=np.ones((20, 2)))

def test_detect_from_cube(tmp_path):
    from specex.sources import detect_from_cube
    from astropy.io import fits
    from astropy.wcs import WCS

    cube_path = tmp_path / "detect_cube.fits"
    outdir = tmp_path / "detect_out"

    # Datacube 10 slices, 8x8 spatial
    data = np.ones((10, 8, 8), dtype=np.float32)
    # Bright source in the middle
    data[:, 4, 4] = 20.0
    var = np.full((10, 8, 8), 0.1, dtype=np.float32)

    w = WCS(naxis=3)
    w.wcs.crpix = [4.0, 4.0, 1.0]
    w.wcs.cdelt = [-0.0001, 0.0001, 10.0]
    w.wcs.crval = [10.0, 20.0, 5000.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.wcs.cunit = ["deg", "deg", "Angstrom"]
    hdr = w.to_header()

    pri = fits.PrimaryHDU(data=data, header=hdr)
    pri.name = "DATA"
    stat = fits.ImageHDU(data=var, header=hdr, name="STAT")
    fits.HDUList([pri, stat]).writeto(cube_path)

    detect_from_cube([
        str(cube_path),
        "--spec-hdu", "0",
        "--var-hdu", "1",
        "--outdir", str(outdir)
    ])
    assert outdir.exists()
