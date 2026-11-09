#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.plot."""
import os
import glob
import pytest
from specex.plot import plot_spectra


def test_plot_spectra_synthetic(tmp_path):
    spec_files = glob.glob("test/data/specex_sdss_galaxy_00.fits")
    if not spec_files:
        pytest.skip("No SDSS synthetic fits available for testing plot")

    outdir = tmp_path / "plot_out"
    plot_options = [
        "--outdir", str(outdir),
        "--restframe",
        spec_files[0]
    ]
    plot_spectra(options=plot_options)

    assert outdir.exists()
    plots = list(outdir.glob("*.png"))
    assert len(plots) > 0

def test_plot_cube_slice_animation(tmp_path):
    from specex.plot import plot_cube_slice_animation
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    import json

    cube_path = tmp_path / "anim_cube.fits"
    spec_path = tmp_path / "anim_spec.fits"
    out_gif = tmp_path / "test_anim.gif"

    # Datacube 10 slices, 10x10 spatial
    data = np.ones((10, 10, 10), dtype=np.float32)
    w = WCS(naxis=3)
    w.wcs.crpix = [5.0, 5.0, 1.0]
    w.wcs.cdelt = [-0.0001, 0.0001, 10.0]
    w.wcs.crval = [10.0, 20.0, 5000.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.wcs.cunit = ["deg", "deg", "Angstrom"]
    hdr = w.to_header()
    pri = fits.PrimaryHDU(data=data, header=hdr)
    pri.name = "DATA"
    pri.writeto(cube_path)

    # Spectrum file
    pri_hdr = fits.Header()
    pri_hdr["RA"] = 10.0
    pri_hdr["DEC"] = 20.0
    pri_hdr["ID"] = "1"
    pri_hdr["EXT_MODE"] = "circular_aperture"
    pri_hdr["EXT_APER"] = json.dumps(["0.5 arcsec"])
    pri_hdr["FRAME"] = "icrs"
    pri_hdu = fits.PrimaryHDU(header=pri_hdr)

    # WCS 1D for spectrum
    w_spec = WCS(naxis=1)
    w_spec.wcs.crpix = [1.0]
    w_spec.wcs.cdelt = [10.0]
    w_spec.wcs.crval = [5000.0]
    w_spec.wcs.ctype = ["WAVE"]
    w_spec.wcs.cunit = ["Angstrom"]
    s_hdr = w_spec.to_header()
    s_hdr["BUNIT"] = "10**(-20)*erg/s/cm**2/Angstrom"

    flux_hdu = fits.ImageHDU(data=np.ones(10), header=s_hdr, name="SPEC")
    var_hdu = fits.ImageHDU(data=np.full(10, 0.1), header=s_hdr, name="VAR")
    fits.HDUList([pri_hdu, flux_hdu, var_hdu]).writeto(spec_path)

    options = [
        "--outname", str(out_gif),
        "--cutout-size", "2arcsec",
        str(cube_path),
        "5050angstrom",
        "40angstrom",
        str(spec_path)
    ]
    plot_cube_slice_animation(options=options)
    assert out_gif.exists()
