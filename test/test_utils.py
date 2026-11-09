#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.utils."""
import os
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from specex.utils import (
    static_vars,
    find_prog,
    get_pbar,
    loop_print,
    simple_pbar_callback,
    get_mask_intervals,
    nannmad,
    get_spectrum_snr,
    get_spectrum_snr_emission,
    stack,
    log_rebin,
    get_ellipse_skypoints,
    get_vclip,
    get_log_img,
    get_pc_transform_params,
    rotate_data,
    parse_regionfile,
)


def test_static_vars():
    @static_vars(counter=0, label="test")
    def my_fn():
        my_fn.counter += 1
        return my_fn.counter, my_fn.label

    assert my_fn.counter == 0
    assert my_fn.label == "test"
    count, label = my_fn()
    assert count == 1
    assert my_fn.counter == 1


def test_find_prog():
    res = find_prog("python3")
    assert res is not None
    assert os.path.exists(res)

    res_fake = find_prog("non_existent_command_xyz_123")
    assert res_fake is None


def test_get_pbar():
    pbar = get_pbar(0.5, total=1.0, wid=20)
    assert isinstance(pbar, str)
    assert len(pbar) > 0

    pbar_no_total = get_pbar(0.75, wid=20)
    assert isinstance(pbar_no_total, str)


def test_loop_print():
    import io
    buf = io.StringIO()
    loop_print("Line 1", file=buf)
    loop_print("Line 2 longer", file=buf)
    loop_print("Line 3", file=buf, end="\n")
    out = buf.getvalue()
    assert "Line 3" in out


def test_simple_pbar_callback():
    simple_pbar_callback(0, 100, min_interval=0)
    simple_pbar_callback(50, 100, min_interval=0)
    simple_pbar_callback(100, 100, min_interval=0)


def test_get_mask_intervals():
    mask = (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0)
    intervals = get_mask_intervals(mask)
    assert intervals == [(5, 7), (10, 10), (14, 18)]

    # Empty mask
    assert get_mask_intervals([0, 0, 0]) == []


def test_nannmad():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    val = nannmad(data)
    assert not np.isnan(val)
    assert np.isclose(val, 1.48206 * 1.0)


def test_get_spectrum_snr():
    # All NaN
    nan_flux = np.full(50, np.nan)
    assert np.isnan(get_spectrum_snr(nan_flux))

    # Constant signal + noise
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.1, size=200)
    flux = 10.0 + noise
    var = np.full_like(flux, 0.01)
    snr = get_spectrum_snr(flux, var=var, smoothing_window=11)
    assert snr > 0


def test_get_spectrum_snr_emission():
    rng = np.random.default_rng(42)
    flux = rng.normal(10.0, 1.0, 300)
    flux[150] = 50.0  # strong line
    var = np.ones(300)
    snr_em = get_spectrum_snr_emission(flux, var=var, bin_size=50)
    assert snr_em > 0

    all_nan = np.full(300, np.nan)
    assert np.isnan(get_spectrum_snr_emission(all_nan, bin_size=50))


def test_stack():
    cube = np.ones((5, 10, 10))
    for k in range(5):
        cube[k] *= (k + 1)
    # Sum across slices
    stacked = stack(cube, quite=True, average=False)
    assert np.allclose(stacked, 15.0)

    # Average across slices
    avg_stacked = stack(cube, quite=True, average=True)
    assert np.allclose(avg_stacked, 3.0)

    # With wave_mask
    mask = np.array([[True], [True], [False], [False], [False]])
    masked_stack = stack(cube, wave_mask=mask, quite=True, average=True)
    assert np.allclose(masked_stack, 1.5)


def test_log_rebin():
    wave = np.linspace(4000, 6000, 500)
    flux = np.sin(wave / 100.0)
    rebinned, ln_lam, velscale = log_rebin(wave, [flux], oversample=1)
    assert len(rebinned) == 1
    assert rebinned[0].shape == ln_lam.shape
    assert velscale > 0

    # Non-monotonic wave raises ValueError
    with pytest.raises(ValueError, match="monotonically increasing"):
        log_rebin(wave[::-1], [flux])


def test_get_ellipse_skypoints():
    center = SkyCoord(ra=10.0*u.deg, dec=20.0*u.deg)
    points = get_ellipse_skypoints(
        center=center,
        a=10.0*u.arcsec,
        b=5.0*u.arcsec,
        angle=30.0*u.deg,
        n_points=12
    )
    assert len(points) == 12
    for pt in points:
        assert isinstance(pt, SkyCoord)


def test_get_vclip_and_get_log_img():
    img = np.arange(100).reshape((10, 10)).astype(float)
    vmin, vmax = get_vclip(img)
    assert vmin < vmax

    log_img, lvmin, lvmax = get_log_img(img)
    assert log_img.shape == img.shape
    assert lvmin <= lvmax


def test_get_pc_transform_params_and_rotate_data():
    w = WCS(naxis=2)
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [-0.0001, 0.0001]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    sx, sy, rot, shr_y = get_pc_transform_params(w)
    assert np.isclose(abs(sx), 0.0001)

    img = np.ones((10, 10))
    res = rotate_data(img, 45.0*u.deg, data_wcs=w)
    assert "data" in res
    assert "wcs" in res
    assert res["data"].shape[0] > 0


def test_parse_regionfile():
    reg_path = "test/data/NICMOSn4hk12010_mos.reg"
    tbl = parse_regionfile(reg_path)
    assert tbl is not None
    assert len(tbl) == 3
    assert "ALPHA_J2000" in tbl.colnames
    assert "DELTA_J2000" in tbl.colnames


def test_plot_lines_and_plot_spectrum(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from specex.utils import plot_lines, plot_spectrum

    fig, ax = plt.subplots()
    plot_lines(ax, wave_range=[4000, 7000], lines_z=0.1)
    plt.close(fig)

    w = np.linspace(4000, 7000, 300)
    f = np.ones_like(w)
    var = np.full_like(w, 0.1)
    mask = np.zeros_like(w, dtype=bool)
    mask[50:60] = True

    fig, axs = plot_spectrum(
        wavelengths=w,
        flux=f,
        variance=var,
        nan_mask=mask,
        redshift=0.5,
        restframe=True,
        wavelengt_units="angstrom",
        flux_units="erg/s/cm2/A",
        extra_info={"ID": "123", "Z": "0.5"}
    )
    assert fig is not None
    assert len(axs) == 4
    plt.close(fig)
