#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic spectra that can be used to test specex.rrspecex module.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import json
import warnings
import numpy as np
from astropy.io import fits
from astropy import units
from scipy.signal import savgol_filter
from astropy.utils.exceptions import AstropyWarning


from test import TEST_DATA_PATH
from test import make_synt_cube

from specex.utils import get_sdss_spectral_templates, get_sdss_template_data


def fake_spectrum_fits(obj_id, wave_range, template, z, wave_step=1):
    """
    Generate a fits HDUList containing a synthetic spectrum.

    Parameters
    ----------
    obj_id : integer
        A unique id for this synthetic object. It's reccomended to use positive
        increasing integers when creating multiple spectra that will be used
        at the same time withrrspecex.
    wave_range : tuple or numpy.ndarray
        The range of wavelenght, in Angstrom, the sythetic spectrum should
        cover. Only maximum and minimum values of this parameter are used.
    template : dict
        A spectral template dictionary as returned by
        specex.utils.get_sdss_template_data()
    z : float
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
    wave, flux = make_synt_cube.gen_fake_spectrum(
        wave_range, template, z, wave_step
    )

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

    var = 2*np.ones_like(flux)
    delta_flux = (flux[:-1] - flux[1:])**2
    delta_flux -= np.nanmin(delta_flux)
    delta_flux /= np.nanmax(delta_flux)
    var[:-1] = 2 + 100 * delta_flux

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
    if np.isnan(sn_spec):
        sn_spec = 99

    apertures = [
        10 * np.random.rand() * units.arcsec,
        10 * np.random.rand() * units.arcsec,
        360 * np.random.rand() * units.deg
    ]

    apertures_info = [
        x.to_string() for x in apertures
    ]

    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header['OBJ_Z'] = z
    prim_hdu.header['NPIX'] = 10
    prim_hdu.header['SN'] = sn_spec
    prim_hdu.header['ID'] = obj_id
    prim_hdu.header['RA'] = '50.0'
    prim_hdu.header['DEC'] = '50.0'
    prim_hdu.header['Z'] = z
    prim_hdu.header['EXT_MODE'] = 'kron-ellipse'
    prim_hdu.header['EXT_APER'] = json.dumps(apertures_info)

    hdul = fits.HDUList([prim_hdu, spec_hdu, var_hdu])
    return hdul, wave


def main():
    """
    Generate synthetic spectra.

    Returns
    -------
    None.

    """
    warnings.simplefilter('ignore', category=AstropyWarning)

    if not os.path.exists(TEST_DATA_PATH):
        os.makedirs(TEST_DATA_PATH)

    wave_range = [4500, 9000]
    obj_id = 0
    type_counters = {}

    spec_files = []

    for t_dict in get_sdss_spectral_templates(outdir=TEST_DATA_PATH):
        # Get a random redshift from the template redshifts range
        obj_type = t_dict['type'].lower()

        if obj_type not in type_counters:
            type_counters[obj_type] = 0
        else:
            type_counters[obj_type] = type_counters[obj_type] + 1

        z = 0

        if obj_type == 'galaxy':
            while z < 0.2:
                z = np.random.rand() * 1
        elif obj_type == 'qso':
            while z < 1.0:
                z = np.random.rand() * 2

        hdul, wave = fake_spectrum_fits(
            f'RR{obj_id:04}',
            wave_range,
            get_sdss_template_data(t_dict['file']),
            z
        )

        outname = f"specex_sdss_{obj_type}"
        try:
            obj_sub_type = t_dict['sub_type'].lower()
        except KeyError:
            pass
        else:
            outname += f'_{obj_sub_type}'

        out_file_path = os.path.join(
            TEST_DATA_PATH,
            f"{outname}_{type_counters[obj_type]:02d}.fits"
        )
        hdul.writeto(out_file_path, overwrite=True)

        spec_files.append(out_file_path)
        obj_id += 1
    return spec_files


if __name__ == '__main__':
    main()
