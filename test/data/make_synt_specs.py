#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic spectra that can be used to test spex.rrspex module.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from make_synt_cube import gen_fake_spectrum

from redrock.templates import find_templates, Template


def fake_spectrum_fits(obj_id, wave_range, template, coeffs, z, wave_step=1):
    """
    Generate a fits HDUList containing a synthetic spectrum.

    Parameters
    ----------
    obj_id : integer
        A unique id for this synthetic object. It's reccomended to use positive
        increasing integers when creating multiple spectra that will be used
        at the same time withrrspex.
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
    hdul : astropy.io.fits.HDUList
        A fits HDU list containing three extensions:
            - EXT 0 : PrimaryHDU
                This extension contains only an header with information about
                the spectrum
            - EXT 1 : ImageHDU
                This extension contains the spectrum itself. The fulxes are
                expressed in 10**(-20)*erg/s/cm**2/Angstrom'.
                The header of this extension contains WCS data that can be
                used to compute the wavelength for each pixel of the spectrum.
            - EXT 2 : ImageHDU
                This extension contains the variance of the spectrum.
                Values are expressed in 10**(-20)*erg/s/cm**2/Angstrom'.
                The header of this extension contains WCS data that can be
                used to compute the wavelength for each pixel of the spectrum.
    """
    wave, flux = gen_fake_spectrum(wave_range, template, coeffs, z, wave_step)
    noise = np.random.standard_normal(len(flux)) * 0.1 * np.std(flux)

    spec_header = fits.header.Header()
    spec_header['CRVAL1'] = wave[0]
    spec_header['CRPIX1'] = 1.0
    spec_header['CDELT1'] = wave_step
    spec_header['CUNIT1'] = 'Angstrom'
    spec_header['BUNIT'] = '10**(-20)*erg/s/cm**2/Angstrom'
    spec_header['CTYPE1'] = 'WAVE'
    spec_header['OBJECT'] = 'SYNTHETIC'
    spec_header['PCOUNT'] = 0
    spec_header['GCOUNT'] = 1

    spec_hdu = fits.ImageHDU(data=flux+noise, header=spec_header)
    spec_hdu.name = 'SPECTRUM'

    var = np.ones_like(flux)
    var_hdu = fits.ImageHDU(data=var, header=spec_header)
    var_hdu.name = 'VARIANCE'

    # Smoothing the spectrum to get a crude approximation of the continuum
    smoothed_spec = savgol_filter(flux, 51, 11)

    # Subtract the smoothed spectrum to the spectrum itself to get a
    # crude estimation of the noise
    noise_spec = np.nanstd(flux - smoothed_spec)

    # Get the mean value of the spectrum
    obj_mean_spec = np.nanmean(flux)

    # Get the mean Signal to Noise ratio
    sn_spec = obj_mean_spec / np.nanmean(noise_spec)

    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header['OBJ_Z'] = z
    prim_hdu.header['NPIX'] = 10
    prim_hdu.header['SN'] = sn_spec
    prim_hdu.header['ID'] = obj_id

    hdul = fits.HDUList([prim_hdu, spec_hdu, var_hdu])
    return hdul, wave


def main():
    """
    Generate synthetic spectra.

    Returns
    -------
    None.

    """
    wave_range = [4500, 9000]
    obj_id = 0
    for t_file in find_templates():
        my_temp = Template(t_file)
        f = fits.open(t_file)
        try:
            has_archetypes = f[1].name == 'ARCHETYPE_COEFF'
        except IndexError:
            has_archetypes = False

        if has_archetypes:
            archetype_coeffs = f[1].data
            coeff_indexes = len(archetype_coeffs)
        else:
            coeff_indexes = range(my_temp.nbasis)

        for i in range(5):
            if has_archetypes:
                coeffs = archetype_coeffs[np.random.choice(coeff_indexes)]
            else:
                coeffs = np.zeros_like(coeff_indexes)
                coeffs[np.random.choice(coeff_indexes)] = 1

            # Get a random redshift from the template redshifts range
            z = np.random.choice(my_temp.redshifts)
            hdul, wave = fake_spectrum_fits(
                f'RR{obj_id:04}', wave_range, my_temp, coeffs, z)

            outname = "rrspex_" + my_temp.template_type.lower()
            if my_temp.sub_type:
                outname += f'_{my_temp.sub_type.lower()}'

            hdul.writeto(
                f"{outname}_{i:02d}.fits",
                overwrite=True
            )
            obj_id += 1


if __name__ == '__main__':
    main()
