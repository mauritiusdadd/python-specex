#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECEX - SPECtra EXtractor.

This module provides functions to stack spectral cubes.

Copyright (C) 2022-2023  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import argparse
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt

from .utils import stack
from .cube import get_hdu


def stack_and_plot(ext, basename, suffix="", is_mask=False, override_wcs=None,
                   dpi=150, wave_ranges=None):
    """
    Stack and plot a spectral datacube.

    Stack a datacube along the spectral axis and plot the result as a
    PNG and a FITS file.

    Parameters
    ----------
    ext : astropy.io.fits.ImageHDU
        The fits exension containing the datacube.
    basename : str
        The name of the output images.
    suffix : str, optional
        An optional suffix for the output image names. The default is "".
    is_mask : bool, optional
        If True, the stacked image is trated as a boolean mask where every
        pixel having non-zero value will be considered as True.
        The default is False.
    override_wcs : astropy.wcs.WCS, optional
        An optional WCS to override the one present in the datacube.
        The default is None.
    dpi : int, optional
        The resolution of the output PNG file. The default is 150.
    wave_range : TYPE, optional
        The optional wavelength range to use, if None all the available
        wavelenghts are used. The default is None.

    Returns
    -------
    new_data : numpy.ndarray
        The stacked image.
    img_wcs : astropy.wcs.WCS
        The WCS of the stacked image.

    """
    if ext.data is None:
        return None

    img_wcs = WCS(ext.header)

    wave_mask = np.zeros(ext.data.shape[0], dtype=bool)
    if wave_ranges and img_wcs.has_spectral:
        for wave_range in wave_ranges:
            wave_index = np.arange(ext.data.shape[0])
            wave_angstrom = img_wcs.spectral.pixel_to_world(wave_index)
            wave_angstrom = wave_angstrom.Angstrom
            mask = wave_angstrom >= np.nanmin(wave_range)
            mask &= wave_angstrom <= np.nanmax(wave_range)
            wave_mask |= mask

    img_height, img_width = ext.data.shape[1], ext.data.shape[2]
    img_figsize = (
        img_width / dpi,
        img_height / dpi
    )

    fig, ax = plt.subplots(
        1, 1,
        figsize=img_figsize,
        subplot_kw={'projection': img_wcs.celestial}
    )

    new_data = stack(ext.data, wave_mask=wave_mask)

    if is_mask:
        new_data = ~(new_data == new_data.max()) * 1.0

    ax.imshow(new_data)
    out_name_png = f"{basename}_{suffix}.png"
    fig.savefig(out_name_png, dpi=dpi)
    plt.close(fig)

    if override_wcs:
        new_header = override_wcs.celestial.to_header()
    else:
        new_header = img_wcs.celestial.to_header()
    out_name_fits = f"{basename}_{suffix}.fits"
    hdu = fits.PrimaryHDU(data=new_data, header=new_header)
    hdu.writeto(out_name_fits, overwrite=True)
    return new_data, img_wcs


def __argshandler(options=None):
    """
    Parse the arguments given by the user.

    Returns
    -------
    args: Namespace
        A Namespace containing the parsed arguments. For more information see
        the python documentation of the argparse module.
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        'input_cube', metavar='SPEC_CUBE', type=str, nargs=1,
        help='The spectral cube in fits format from which spectra will be '
        'extracted.'
    )
    parser.add_argument(
        '--dpi', metavar='DPI', type=int, default=150,
        help='Set the DPI of the output figures. '
        'The default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--spec-hdu', metavar='SPEC_HDU', type=int, default=-1,
        help='The HDU containing the spectral data to use. If this argument '
        'Set this to -1 to automatically detect the HDU containing spectra. '
        'NOTE that this value is zero indexed (i.e. second HDU has index 1).'
    )

    parser.add_argument(
        '--var-hdu', metavar='VAR_HDU', type=int, default=-1,
        help='The HDU containing the variance of the spectral data. '
        'Set this to -1 if no variance data is present in the cube. '
        'The default value is %(metavar)s=%(default)s.'
        'NOTE that this value is zero indexed (i.e. third HDU has index 2).'
    )

    parser.add_argument(
        '--mask-hdu', metavar='MASK_HDU', type=int, default=-1,
        help='The HDU containing the valid pixel mask of the spectral data. '
        'Set this to -1 if no mask is present in the cube. '
        'The default value is %(metavar)s=%(default)s.'
        'NOTE that this value is zero indexed (i.e. fourth HDU has index 3).'
    )

    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None,
        help='Set the directory where extracted spectra will be outputed. '
        'If this parameter is not specified, then a new directory will be '
        'created based on the name of the input cube.'
    )

    parser.add_argument(
        '--wave-ranges', metavar='WAVE_RANGE', type=str, default=None,
        help='Set the wavelength range(s) to stack, in the format '
        'WAVE_RANGE=RANGE1_START-RANGE1_STOP,RANGE2_START-RANGE2_STOP,...'
        ' If not specified, the whole wavelength range is used.'
    )

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def cube_stack(options=None):
    """
    Run the main program of this module.

    Returns
    -------
    None.

    """
    args = __argshandler(options)

    input_file = args.input_cube[0]

    basename = os.path.basename(args.input_cube[0])
    basename = os.path.splitext(basename)[0]

    if args.outdir is not None:
        basename = os.path.join(args.outdir, basename)

    with fits.open(input_file) as hdul:

        spec_hdu = get_hdu(
            hdul,
            hdu_index=args.spec_hdu,
            valid_names=['data', 'spec', 'spectrum', 'spectra'],
            msg_err_notfound="ERROR: Cannot determine which HDU contains "
                             "spectral data, try to specify it manually!",
            msg_index_error="ERROR: Cannot open HDU {} to read specra!"
        )

        var_hdu = get_hdu(
            hdul,
            hdu_index=args.var_hdu,
            valid_names=['stat', 'var', 'variance', 'noise'],
            msg_err_notfound="WARNING: Cannot determine which HDU contains "
                             "the variance data, try to specify it manually!",
            msg_index_error="ERROR: Cannot open HDU {} to read the "
                            "variance!",
            exit_on_errors=False
        )

        mask_hdu = get_hdu(
            hdul,
            hdu_index=args.mask_hdu,
            valid_names=['mask', 'platemask', 'footprint', 'dq'],
            msg_err_notfound="WARNING: Cannot determine which HDU contains "
                             "the mask data, try to specify it manually!",
            msg_index_error="ERROR: Cannot open HDU {} to read the mask!",
            exit_on_errors=False
        )

        if args.wave_ranges is not None:
            wave_ranges = [
                [float(k) for k in x.split('-')]
                for x in args.wave_ranges.split(',')
            ]
        else:
            wave_ranges = None

        dat, dat_wcs = stack_and_plot(
            spec_hdu,
            basename,
            'data',
            wave_ranges=wave_ranges,
            dpi=args.dpi
        )

        if args.var_hdu >= 0:
            var, var_wcs = stack_and_plot(
                var_hdu,
                basename,
                'variance',
                override_wcs=dat_wcs,
                wave_ranges=wave_ranges,
                dpi=args.dpi
            )

        if args.mask_hdu >= 0:
            mask, mask_wcs = stack_and_plot(
                mask_hdu,
                basename,
                'mask',
                is_mask=True,
                override_wcs=dat_wcs,
                wave_ranges=wave_ranges,
                dpi=args.dpi
            )


if __name__ == '__main__':
    cube_stack()
