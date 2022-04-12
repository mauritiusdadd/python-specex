#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import sys
import argparse
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt

from .utils import stack, gethdu


def stackandplot(ext, basename, suffix="", is_mask=False, override_wcs=None,
                 dpi=150, wave_range=None):
    if ext.data is None:
        return None

    img_wcs = WCS(ext.header)

    wave_mask = None
    if wave_range is not None and img_wcs.has_spectral:
        wave_index = np.aragne(ext.data.shape[0])
        wave_angstrom = img_wcs.spectral.pixel_to_world(wave_index).Angstrom
        wave_mask = wave_angstrom >= np.nanmax(wave_range)
        wave_mask &= wave_angstrom <= np.nanmin(wave_range)

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

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def cubestack(options=None):
    """
    Run the main program of this module.

    Returns
    -------
    None.

    """
    args = __argshandler(options)

    input_file = sys.argv[1]

    basename = os.path.basename(args.input_cube)
    basename = os.path.splitext(basename)[0]

    if args.outdir is not None:
        basename = os.path.join(args.outdir, basename)

    hdl = fits.open(input_file)

    spec_hdu = gethdu(
        hdl,
        hdu_index=args.spec_hdu,
        valid_names=['data', 'spec', 'spectrum', 'spectra'],
        msg_err_notfound="ERROR: Cannot determine which HDU contains spectral "
                         "data, try to specify it manually!",
        msg_index_error="ERROR: Cannot open HDU {} to read specra!"
    )

    var_hdu = gethdu(
        hdl,
        hdu_index=args.var_hdu,
        valid_names=['stat', 'var', 'variance', 'noise'],
        msg_err_notfound="ERROR: Cannot determine which HDU contains the "
                         "variance data, try to specify it manually!",
        msg_index_error="ERROR: Cannot open HDU {} to read the "
                        "variance!",
        exit_on_errors=False
    )

    mask_hdu = gethdu(
        hdl,
        hdu_index=args.mask_hdu,
        valid_names=['mask', 'platemask', 'footprint', 'dq'],
        msg_err_notfound="ERROR: Cannot determine which HDU contains the "
                         "mask data, try to specify it manually!",
        msg_index_error="ERROR: Cannot open HDU {} to read the mask!",
        exit_on_errors=False
    )

    dat, dat_wcs = stackandplot(
        spec_hdu,
        basename,
        'data',
        dpi=args.dpi
    )

    if args.var_hdu >= 0:
        var, var_wcs = stackandplot(
            var_hdu,
            basename,
            'variance',
            override_wcs=dat_wcs,
            dpi=args.dpi
        )

    if args.mask_hdu >= 0:
        mask, mask_wcs = stackandplot(
            mask_hdu,
            basename,
            'mask',
            is_mask=True,
            override_wcs=dat_wcs,
            dpi=args.dpi
        )


if __name__ == '__main__':
    cubestack()
