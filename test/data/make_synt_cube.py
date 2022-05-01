#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic spectral datacube that can be used to test spex module.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import numpy as np
from astropy.io import fits


def gen_fake_spectrum(wave_range, template, coeffs, z, wave_step=1):
    """
    Generate a synthetic spectrum using a specific template.

    Parameters
    ----------
    wave_range : tuple or numpy.ndarray
        The range of wavelenght, in Angstrom, the sythetic spectrum should
        cover. Only maximum and minimum values of this parameter are used.
    template : rerdock.template.Template or custom class
        The template used to compute the spectrum. If a custom class is used,
        it should have a method named eval that accetps the following
        parameters eval(coeff, wave, z), where:

            - coeff : list or numpy.ndarray
                Coefficients uesd for generating the spectrum
            - wave : numpy.ndarray
                Wavelenghts, in Angstrom, used to generate spectral data
            - z : float
                Redshift of the spectrum.

        The eval method should return a 1D numpy.ndarray of the same lenght of
        wave and containing the flux of the synthetic spectrum in arbitrary
        units.
    coeffs : list or numpy.ndarray
        Coefficients uesd for generating the spectrum, passed to template.eval
        method.
    z : TYPE
        Redshift of the spectrum.
    wave_step : float, optional
        Resolution of the spectrum in Angstrom. The default is 1.

    Returns
    -------
    wave : numpy.ndarray
        Wavelenghts, in Angstrom, used to generate spectral data
    flux : numpy.ndarray
        Fluxes, in arbitrary units, at the different wavelengths.
    """
    # generate a wavelenght grid
    w_start = np.min(wave_range)
    w_end = np.max(wave_range)
    wave = np.arange(w_start, w_end, step=wave_step)

    flux = template.eval(coeffs, wave, z)

    return wave, flux
