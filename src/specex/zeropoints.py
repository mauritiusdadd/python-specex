#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECEX - SPECtra EXtractor.

This module provides functions to calculate the zeropoints of a FITS image.

For more information take a look at the following links:

 - https://www.stsci.edu/hst/wfpc2/Wfpc2_dhb/wfpc2_ch52.html
 - https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints

Copyright (C) 2022-2023  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import sys
import argparse
import numpy as np
from astropy.io import fits
from astropy import units


def get_zeropoint_info(filename, ext=0):
    """
    Compute zero-point information from a fits image.

    For more information about zeropoints take a look at the following links:
    https://www.stsci.edu/hst/wfpc2/Wfpc2_dhb/wfpc2_ch52.html
    https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints

    Parameters
    ----------
    filename : str
        The path of the fits file image.

    ext : int, optional
        The FITS extension containing the image data. The default value is 0.

    Returns
    -------
    zpt_dict : ditct
        A dicttionary containing the zero-point information.
        The dictionary contains the following items:

            - exp_time : float
                The exposure time

            - phot_f_lam : float
                The value of PHOTFLAM. This is the flux of a source with
                constant flux per unit wavelength (in erg s-1 cm-2 A-1) which
                produces a count rate of 1 count per second.

            - phot_p_lam : float
                the pivot wavelength (in Angstrom)

            - zero_point : sloat
                the zero point value

            - zero_point_p : float
                the zero point value plus 2.5 times log10 of the exposure time

            - zero_point_m : float
                the zero point value minus 2.5 times log10 of the exposure time

            - counts_to_flux : float
                Quantity to convert counts to flux units
    """
    hdr = fits.getheader(filename, ext=ext)
    phot_f_lam = hdr['PHOTFLAM']
    phot_p_lam = hdr['PHOTPLAM']
    exp_time = hdr['EXPTIME']

    try:
        science_units = hdr['BUNIT']
    except KeyError:
        science_units = None

    zpt = -2.5 * np.log10(phot_f_lam) - 21.10
    zpt += -5 * np.log10(phot_p_lam) + 18.6921

    acs_zpt_pexp = zpt + 2.5 * np.log10(exp_time)
    acs_zpt_mexp = zpt - 2.5 * np.log10(exp_time)

    if science_units is not None and science_units.lower().endswith('/s'):
        counts_to_flux = phot_f_lam
    else:
        counts_to_flux = phot_f_lam/exp_time

    zpt_dict = {
        "exp_time": exp_time,
        "phot_f_lam": phot_f_lam,
        "phot_p_lam": (phot_p_lam / 10) * units.nm,
        "zero_point": zpt,
        "zero_point_p": acs_zpt_pexp,
        "zero_point_m": acs_zpt_mexp,
        'counts_to_flux': counts_to_flux
    }
    return zpt_dict


def print_zeropoint_info(filename, ext=0):
    """
    Print the zero-points information of a FITS image.

    Parameters
    ----------
    filename : str
        The path of a FITS file image.
    ext : int, optional
        The FITS extension containing the image data. The default value is 0.

    Returns
    -------
    None.

    """
    zpt_dict = get_zeropoint_info(filename)
    print(f"\n{filename}")
    s = "Exp time: {exp_time}\n"
    s += "Pivot wavelenght: {phot_p_lam:.0f}\n"
    s += "Zero point: {zero_point}\n"
    s += "Zero point (+m): {zero_point_p}\n"
    s += "Zero point (-m): {zero_point_m}\n"
    print(s.format(**zpt_dict))


def main(options=None):
    """
    Run the main program of this module.

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser(
        description='Print zero-point ifnormatio of FITS images.'
    )
    parser.add_argument(
        'inp_files', metavar='FITS_IMAGES', type=str, nargs='+',
        help='One or more than one fits file for which you want to view the'
        'zero-point information'
    )
    parser.add_argument(
        '--ext', metavar='EXT', type=int, default=0,
        help='The extension containing the image data.'
     )

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    for fname in args.inp_files:
        try:
            print_zeropoint_info(fname)
        except Exception:
            print(
                f"ERROR: Cannot read zeropoint info for file {fname}",
                file=sys.stderr
            )
            continue


if __name__ == '__main__':
    main()
