#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

This module provides utility functions and executable to plot spectra.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as apu
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord

from .utils import plot_spectrum, get_pbar, load_rgb_fits
from .cube import get_hdu, get_rgb_cutout, get_gray_cutout


def __argshandler(options=None):
    """
    Parse the arguments given by the user.

    Returns
    -------
    args: Namespace
        A Namespace containing the parsed arguments. For more information see
        the python documentation of the argparse module.
    """
    parser = argparse.ArgumentParser(
        description='Plot spectra extracted with specex.'
    )

    parser.add_argument(
        '--cutouts-image', metavar='IMAGE', type=str, default=None,
        help='The path of a FITS image (grayscale or single/multiext RGB) from'
        'which to extract cutout of the objects. If this option is not '
        'specified then the cutouts will be extracted directly from the '
        'input cube.'
    )

    parser.add_argument(
        '--cutouts-size', metavar='SIZE', type=float, default=10,
        help='Size of the cutouts of the object in pixel or arcseconds. '
        'The default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None,
        help='Set the directory where extracted spectra will be outputed. '
        'If this parameter is not specified, then plots will be saved in the '
        'same directory of the corresponding input spectrum file.'
    )

    parser.add_argument(
        '--zcat', metavar='Z_CAT_FILE', type=str, default=None,
        help='If specified then the catalog %(metavar)s is used to read the '
        'redshift of the spectra. The catalog must contain at least two '
        'columns: one for the id of the objects and one for the redshifts. '
        'The name of the columns can be set using the parameters --key-id and '
        '--key-z. When this option is used, spectra having no matching ID in '
        'the catalog are skipped.'
    )

    parser.add_argument(
        '--key-id', metavar='KEY_ID', type=str, default='ID',
        help='Set the name of the column in the zcat that contains the IDs of '
        'the spectra. See --zcat for more info. If this option is not '
        'specified, then he default value %(metavar)s = %(default)s is used.'
    )

    parser.add_argument(
        '--key-z', metavar='KEY_Z', type=str, default='Z',
        help='Set the name of the column in the zcat that contains the '
        'redshift the spectra. See --zcat for more info. If this option is not'
        ' specified, then he default value %(metavar)s = %(default)s is used.'
    )

    parser.add_argument(
        '--restframe', default=False, action='store_true',
        help='If this option is specified, spectra will be plotted as if they '
        'were in the observer restframe (ie. they are de-redshifted). '
        'In order to use this option, a zcat must be specified.'
    )

    parser.add_argument(
        '--smoothing', metavar='WINDOW_SIZE', type=int,  default=3,
        help='If %(metavar)s >= 0, then plot a smoothed version of the '
        'spectrum alongside the original one.  %(metavar)s = 0 means that '
        'only the original spectrum is plotted. If not specified, the default '
        '%(metavar)s = %(default)s is used.'
    )

    parser.add_argument(
        '--cutout', metavar='SOURCE_IMAGE', type=str, default=None,
        help='path of a FITS image (RGB or grayscale) used to make cutouts '
        'that will be included in the plots. The size of the cutout can be '
        'set using the --cutout-size option.'
    )

    parser.add_argument(
        '--cutout-size', metavar='SIZE', type=str, default='2arcsec',
        help='Set the size of the cutout to %(metavar)s. This option '
        'supports units compatible with astropy (for example "1deg", '
        '"2arcmin", "5arcsec", etc.). If no unit is specified the size is '
        'assumed to be in arcseconds. The default cutout size is %(default)s.'
    )

    parser.add_argument(
        'spectra', metavar='SPECTRUM', type=str, nargs='+',
        help='Input spectra extracted with specex.'
    )

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def plot(options=None):
    """
    Run the plot program.

    Returns
    -------
    None.

    """
    args = __argshandler(options)

    if args.zcat is not None:
        zcat = Table.read(args.zcat)
        if args.key_id not in zcat.colnames:
            print(
                f"ERROR: z catalog does not have id column '{args.key_id}'",
                file=sys.stderr
            )
            sys.exit(1)
        elif args.key_z not in zcat.colnames:
            print(
                f"ERROR: z catalog does not have z column '{args.key_z}'",
                file=sys.stderr
            )
            sys.exit(1)
        zcat = zcat[args.key_id, args.key_z].copy()

        # Remove objects with masked or undefined IDs
        if isinstance(zcat[args.key_id], MaskedColumn):
            zcat = zcat[~zcat[args.key_id].mask]

        zcat.add_index(args.key_id)
    else:
        zcat = None

    if args.cutout is not None:
        try:
            big_image = load_rgb_fits(args.cutout)
        except FileNotFoundError:
            print(f"ERROR: file not found '{args.cutout}'")
            sys.exit(1)
        except IndexError:
            big_image = {
                'data': fits.getdata(args.cutout),
                'wcs': wcs.WCS(fits.getheader(args.cutout)),
                'type': 'gray'
            }

        cutout_size = apu.Quantity(args.cutout_size)
    else:
        big_image = None

    for j, spectrum_fits_file in enumerate(args.spectra):
        progress = j / len(args.spectra)
        sys.stdout.write(f"\r{get_pbar(progress)} {progress:.2%}\r")
        sys.stdout.flush()

        with fits.open(spectrum_fits_file) as hdulist:
            main_header = get_hdu(
                hdulist,
                hdu_index=0,
                valid_names=['PRIMARY', 'primary'],
                msg_index_error="WARNING: No Primary HDU",
                exit_on_errors=False
            ).header
            spec_hdu = get_hdu(
                hdulist,
                valid_names=['SPEC', 'spec', 'SPECTRUM', 'spectrum'],
                msg_err_notfound="WARNING: No spectrum HDU",
                exit_on_errors=False
            )
            var_hdu = get_hdu(
                hdulist,
                valid_names=['VAR', 'var', 'VARIANCE', 'variance'],
                msg_err_notfound="WARNING: No variance HDU",
                exit_on_errors=False
            )

            nan_mask_hdu = get_hdu(
                hdulist,
                valid_names=[
                    'NAN_MASK', 'nan_mask',
                    'NANMASK', 'MASK',
                    'nanmask', 'mask'
                ],
                exit_on_errors=False
            )

            if any(x is None for x in [main_header, spec_hdu, var_hdu]):
                print(f"Skipping file '{spectrum_fits_file}'\n")
                continue

            try:
                object_ra = main_header['RA']
                object_dec = main_header['DEC']
                object_id = main_header['ID']
                extraction_mode = main_header['EXT_MODE']
                specex_apertures = [
                    apu.Quantity(x)
                    for x in json.loads(main_header['EXT_APER'])
                ]
            except KeyError:
                print(
                    f"Skipping file with invalid header: {spectrum_fits_file}"
                )
                continue
            else:
                obj_center = SkyCoord(
                    object_ra, object_dec,
                    unit='deg'
                )

            info_dict = {
                'ID': f"{object_id}",
                'RA': obj_center.ra.to_string(precision=2),
                'DEC': obj_center.dec.to_string(precision=2),
            }
            for key in ['Z', 'SN', 'SN_EMISS']:
                try:
                    info_dict[key] = f"{main_header[key]:.4f}"
                except KeyError:
                    continue

            try:
                flux_units = spec_hdu.header['BUNIT']
            except KeyError:
                flux_units = None

            try:
                wavelenght_units = spec_hdu.header['CUNIT1']
            except KeyError:
                wavelenght_units = None

            if zcat is not None:
                try:
                    object_z = zcat.loc[object_id][args.key_z]
                except KeyError:
                    print(
                        f"WARNING: '{object_id}' not in zcat, skipping...",
                        file=sys.stderr
                    )
                    continue
                else:
                    try:
                        # In case of repeated objects
                        object_z = object_z[0]
                    except IndexError:
                        # Otherwise just go ahead
                        pass
                restframe = args.restframe
                info_dict['Z'] = object_z
            else:
                # If no zcat is provided, check if redshift information is
                # stored in the spectrum itself
                if 'Z' in info_dict:
                    object_z = float(info_dict['Z'])
                    restframe = args.restframe
                else:
                    object_z = None
                    restframe = False

            flux_data = spec_hdu.data
            spec_wcs = wcs.WCS(spec_hdu.header)
            var_data = var_hdu.data

            if nan_mask_hdu is not None:
                nan_mask = nan_mask_hdu.data == 1
            else:
                nan_mask = None

            # NOTE: Wavelenghts must be in Angstrom units
            pixel = np.arange(len(flux_data))
            wavelenghts = spec_wcs.pixel_to_world(pixel).Angstrom

            if big_image is not None:
                if big_image['type'] == 'rgb':
                    cutout_dict = get_rgb_cutout(
                        big_image['data'],
                        position=obj_center,
                        size=cutout_size,
                        data_wcs=big_image['wcs']
                    )
                    cutout = np.array(cutout_dict['data'])
                    cutout_wcs = cutout_dict['wcs'][0]
                else:
                    cutout_dict = get_gray_cutout(
                        big_image['data'],
                        position=obj_center,
                        size=cutout_size,
                        data_wcs=big_image['wcs']
                    )
                    cutout = cutout_dict['data'].transpose(1, 2, 0)
                    cutout_wcs = cutout_dict['wcs']
                cutout_vmin = np.nanmin(big_image['data'])
                cutout_vmax = np.nanmax(big_image['data'])
            else:
                cutout = None
                cutout_wcs = None
                cutout_vmin = None
                cutout_vmax = None

            fig, axs = plot_spectrum(
                wavelenghts,
                flux_data,
                var_data,
                nan_mask=nan_mask,
                redshift=object_z,
                restframe=restframe,
                cutout=cutout,
                cutout_wcs=cutout_wcs,
                cutout_vmin=cutout_vmin,
                cutout_vmax=cutout_vmax,
                flux_units=flux_units,
                wavelengt_units=wavelenght_units,
                smoothing=args.smoothing,
                extra_info=info_dict,
                extraction_info={
                    'mode': extraction_mode,
                    'apertures': specex_apertures,
                    'aperture_ra': object_ra,
                    'aperture_dec': object_dec,
                }
            )

            if args.outdir is None:
                outdir = os.path.dirname(spectrum_fits_file)
            else:
                outdir = args.outdir
                if not os.path.isdir(outdir):
                    os.makedirs(outdir)

            fig_out_name = os.path.basename(spectrum_fits_file)
            fig_out_name = os.path.splitext(fig_out_name)[0]
            fig_out_path = os.path.join(outdir, f"{fig_out_name}.png")
            fig.savefig(fig_out_path, dpi=150)
            plt.close(fig)

    print(f"\r{get_pbar(1)} 100%")


if __name__ == '__main__':
    plot()
