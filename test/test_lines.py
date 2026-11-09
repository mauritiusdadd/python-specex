#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for specex.lines."""
import numpy as np
import pytest

from specex.lines import (
    get_lines,
    get_spectrum_lines,
    get_redshift_from_lines,
    Emission1D,
    RESTFRAME_LINES,
)


def test_get_lines():
    # All lines
    all_lines = get_lines()
    assert len(all_lines) == len(RESTFRAME_LINES)

    # By name
    oiii_lines = get_lines(name='[OIII]')
    assert len(oiii_lines) == 2
    for w, name, ltype in oiii_lines:
        assert name == '[OIII]'
        assert 'E' in ltype

    # Case insensitive
    halpha = get_lines(name='HALPHA')
    assert len(halpha) == 1
    assert halpha[0][1] == 'Halpha'

    # By line type
    abs_lines = get_lines(line_type='A')
    assert len(abs_lines) > 0
    for _, _, ltype in abs_lines:
        assert 'a' in ltype.lower()

    # With redshift
    z = 1.0
    z_lines = get_lines(name='Halpha', z=z)
    orig_lines = get_lines(name='Halpha', z=0)
    assert np.isclose(z_lines[0][0], orig_lines[0][0] * (1 + z))

    # With wavelength range
    ranged_lines = get_lines(wrange=(4000, 5000))
    for w, _, _ in ranged_lines:
        assert 4000 <= w <= 5000


def test_emission_1d_model():
    candidates = [
        (100, 5000.0, 10.0, 50.0),
        (200, 6563.0, 15.0, 100.0),
    ]
    model = Emission1D(candidates, sigma=5.0, redshift=0.0)
    lam = np.linspace(4000, 7000, 1000)
    flux = model.evaluate(lam, redshift=0.0)
    assert flux.shape == lam.shape
    assert np.all(flux >= 0)
    # Peak should be near 5000 and 6563
    idx_5000 = np.argmin(np.abs(lam - 5000.0))
    assert flux[idx_5000] > 0


def test_get_spectrum_lines():
    # All nan input
    nan_flux = np.full(100, np.nan)
    w = np.linspace(4000, 5000, 100)
    res = get_spectrum_lines(w, nan_flux)
    assert np.isnan(res)

    # Synthetic spectrum with an emission line
    w = np.linspace(4000, 7000, 1000)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, size=len(w))
    line = 5.0 * np.exp(-0.5 * ((w - 5500.0) / 10.0)**2)
    flux = 1.0 + noise + line
    var = np.full_like(flux, 0.01)

    identifications = get_spectrum_lines(
        wavelengths=w,
        flux=flux,
        var=var,
        sigma_threshold=3.0,
        smoothing_window=31,
        smoothing_order=1,
        glob_smoothing=3
    )
    assert isinstance(identifications, list)
    assert len(identifications) >= 1
    # Check that identified line is close to 5500
    closest_w = min(identifications, key=lambda x: abs(x[1] - 5500.0))
    assert abs(closest_w[1] - 5500.0) < 30.0


def test_get_redshift_from_lines():
    # Less than 2 lines should return None
    assert get_redshift_from_lines([]) is None
    assert get_redshift_from_lines([(10, 5000.0, 5.0, 10.0)]) is None

    # Realistic line pairs at z = 0.5
    z_true = 0.5
    halpha_obs = 6562.8 * (1 + z_true)
    oiii_obs = 5006.8 * (1 + z_true)
    hbeta_obs = 4861.3 * (1 + z_true)

    identifications = [
        (100, oiii_obs, 10.0, 50.0),
        (200, hbeta_obs, 10.0, 40.0),
        (300, halpha_obs, 15.0, 80.0),
    ]

    res = get_redshift_from_lines(
        identifications,
        z_min=0.4,
        z_max=0.6,
        z_points=100
    )
    assert res is not None
    z_vals, z_probs = res
    if len(z_vals) > 0:
        assert np.isclose(z_vals[0], z_true, atol=0.05)
