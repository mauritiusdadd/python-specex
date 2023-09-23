#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECEX - SPECtra EXtractor.

This module provides utility functions and executable to plot spectra.

Copyright (C) 2022-2023  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import argparse
import json
import tempfile
import subprocess

from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.ndimage import gaussian_filter

from astropy import units as apu
from astropy.io import fits
from astropy import wcs
from astropy.table import Table, MaskedColumn
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import quantity_support

from .utils import plot_spectrum, get_pbar, load_rgb_fits, find_prog
from .cube import get_hdu, get_rgb_cutout, get_gray_cutout

from .specex import load_specex_file

try:
    import imageio
except ModuleNotFoundError:
    HAS_IMAGEIO = False
else:
    HAS_IMAGEIO = True

FFMPEG_EXC = find_prog('ffmpeg')
if FFMPEG_EXC is None:
    FFMPEG_EXC = find_prog('ffmpeg.exe')


def __plot_spectra_argshandler(options=None):
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
        '--key-id', metavar='KEY_ID', type=str, default='SPECID',
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
        '--key-wrange', metavar='WAVE_RANGE', type=str, default=None,
        help='Set the name of the column in the zcat that contains the range'
        'of wavelength to plot. If not specified then the whole spectrum is '
        'plotted. If specified and the range value is empty no plot is '
        'produced for the object.'
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


def plot_spectra(options=None):
    """
    Run the plot program.

    Returns
    -------
    None.

    """
    quantity_support()
    args = __plot_spectra_argshandler(options)

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

        if big_image is None:
            big_image = {
                'data': fits.getdata(args.cutout),
                'wcs': wcs.WCS(fits.getheader(args.cutout)),
                'type': 'gray'
            }

        cutout_size = apu.Quantity(args.cutout_size)
    else:
        big_image = None

    # Windows prompt does not expand globs, so let's do it
    spectra_list = []
    for globbed_fname in args.spectra:
        for fname in glob(globbed_fname):
            spectra_list.append(fname)

    for j, spectrum_fits_file in enumerate(spectra_list):
        progress = j / len(spectra_list)
        sys.stdout.write(f"\r{get_pbar(progress)} {progress:.2%}\r")
        sys.stdout.flush()

        sp_dict = load_specex_file(spectrum_fits_file)

        if any(
                x is None for x in [
                    sp_dict['main_header'],
                    sp_dict['flux'],
                    sp_dict['variance']
                ]
        ):
            print(f"Skipping file '{spectrum_fits_file}'\n")
            continue

        try:
            object_ra = sp_dict['main_header']['RA']
            object_dec = sp_dict['main_header']['DEC']
            object_id = sp_dict['main_header']['ID']
            extraction_mode = sp_dict['main_header']['EXT_MODE']
            specex_apertures = [
                apu.Quantity(x)
                for x in json.loads(sp_dict['main_header']['EXT_APER'])
            ]
        except KeyError:
            print(
                f"Skipping file with invalid header: {spectrum_fits_file}"
            )
            continue
        else:

            try:
                object_coord_frame = sp_dict['main_header']['FRAME']
            except KeyError:
                object_coord_frame = 'icrs'

            obj_center = SkyCoord(
                object_ra, object_dec,
                unit='deg',
                frame=object_coord_frame
            )

        info_dict = {
            'ID': f"{object_id}",
            'RA': obj_center.ra.to_string(precision=2),
            'DEC': obj_center.dec.to_string(precision=2),
            'FRAME': str(object_coord_frame).upper()
        }
        for key in ['Z', 'SN', 'SN_EMISS']:
            try:
                info_dict[key] = f"{sp_dict['main_header'][key]:.4f}"
            except KeyError:
                continue

        wave_range = None
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

                if args.key_wrange is not None:
                    str_wrange = zcat.loc[object_id][args.key_wrange]
                    try:
                        wave_range = [
                            float(x) for x in str_wrange.split('-')
                        ]
                    except Exception:
                        continue

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

        if big_image is not None:
            if big_image['type'] == 'rgb':
                cutout_dict = get_rgb_cutout(
                    big_image['data'],
                    center=obj_center,
                    size=cutout_size,
                    data_wcs=big_image['wcs']
                )
                cutout = np.asarray(cutout_dict['data']).transpose(1, 2, 0)
                cutout_wcs = cutout_dict['wcs'][0]
            else:
                cutout_dict = get_gray_cutout(
                    big_image['data'],
                    center=obj_center,
                    size=cutout_size,
                    data_wcs=big_image['wcs']
                )
                cutout = np.array(cutout_dict['data'])
                cutout_wcs = cutout_dict['wcs']
            cutout_vmin = np.nanmin(big_image['data'])
            cutout_vmax = np.nanmax(big_image['data'])
        else:
            cutout = None
            cutout_wcs = None
            cutout_vmin = None
            cutout_vmax = None

        fig, axs = plot_spectrum(
            sp_dict['wavelength'],
            sp_dict['flux'],
            sp_dict['variance'],
            nan_mask=sp_dict['nan_mask'],
            redshift=object_z,
            restframe=restframe,
            cutout=cutout,
            cutout_wcs=cutout_wcs,
            cutout_vmin=cutout_vmin,
            cutout_vmax=cutout_vmax,
            flux_units=sp_dict['flux_units'],
            wavelengt_units=sp_dict['wavelength_units'],
            smoothing=args.smoothing,
            extra_info=info_dict,
            extraction_info={
                'mode': extraction_mode,
                'apertures': specex_apertures,
                'aperture_ra': object_ra,
                'aperture_dec': object_dec,
                'frame': object_coord_frame
            },
            wave_range=wave_range
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


def __plot_slice_argshandler(options=None):
    """
    Parse the arguments given by the user.

    Returns
    -------
    args: Namespace
        A Namespace containing the parsed arguments. For more information see
        the python documentation of the argparse module.
    """
    global FFMPEG_EXC

    parser = argparse.ArgumentParser(
        description='Plot spectra extracted with specex.'
    )

    parser.add_argument(
        '--spec-smoothing', metavar='WINDOW_SIZE', type=int,  default=0,
        help='If %(metavar)s >= 0, then plot a smoothed version of the '
        'spectrum alongside the original one.  %(metavar)s = 0 means that '
        'only the original spectrum is plotted. If not specified, the default '
        '%(metavar)s = %(default)s is used.'
    )

    parser.add_argument(
        '--cube-smoothing', metavar='WINDOW_SIZE', type=float,  default=0,
        help='If %(metavar)s >= 0, then plot a smoothed version of the '
        'cube cutout.  %(metavar)s = 0 means that no smoothing is applied.'
        'The default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--cube-vlim', metavar='VMIN,VMAX', type=str, default=None,
        help='Set the minimum and maximum values for the colormap of the '
        'cube animated cutout. If not specified, then vmin and vmax are '
        'computed automatically by matplotlib backend.'
    )

    parser.add_argument(
        '--cutout', metavar='SOURCE_IMAGE', type=str, default=None,
        help='path of a FITS image (RGB or grayscale) used to make cutouts '
        'that will be included in the plots. The size of the cutout can be '
        'set using the --cutout-size option.'
    )

    parser.add_argument(
        '--cutout-size', metavar='SIZE', type=str, default='10arcsec',
        help='Set the size of the cutout to %(metavar)s. This option '
        'supports units compatible with astropy (for example "1deg", '
        '"2arcmin", "5arcsec", etc.). If no unit is specified the size is '
        'assumed to be in arcseconds. The default cutout size is %(default)s.'
    )

    parser.add_argument(
        '--redshift', '-z', metavar='Z', type=float, default=None,
        help='Redshift of the objects. If not specified then no lines will be'
        'plotted on the spectra.'
    )

    parser.add_argument(
        '--plot-wrange', metavar='W_RANGE', type=str,
        help='wavelnght range to plot expressed in the form A,B where A and B '
        'are wavelengths in an astropy compatible format (ie. 100angstrom or '
        '1nm or 1). Adimensional values will be considered expressed in '
        'angstroms.'
    )

    parser.add_argument(
        '--cmap', metavar='CMAP', type=str, default='jet',
        help='Set the colormap to use when plotting the spectral datacube '
        'cutout. Can be the name of any valid matplotlib colormap. The '
        'default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--outname', '-o', metavar='FILE_NAME', type=str, default='anim.gif',
        help='Set the name of the output file. If not specified, then the '
        'default value %(metavar)s=%(default)s is used.'
    )

    parser.add_argument(
        '--out-dpi', metavar='DPI', type=int, default=75,
        help='Set the DPI of the output file. If not specified, then the '
        'default value %(metavar)s=%(default)s is used.'
    )

    parser.add_argument(
        '--out-fps', metavar='FPS', type=int, default=5,
        help='Set the output framerate to %(metaver)s. The default value is '
        '%(metavar)s=%(default)s'
    )

    parser.add_argument(
        '--out-loop', metavar='LOOP_COUNT', type=int, default=0,
        help='Set the number of times the animation is played. If '
        '%(metavar)s=0 then the animation loops indefinitely. The default '
        'value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--ffmpeg-exec', metavar='FFMPEG_PATH', type=str, default=FFMPEG_EXC,
        help='Set the path of the FFMPEG executable. If not specified the '
        'ffmepg executable is searched automatically in your system PATH. '
    )

    parser.add_argument(
        '--ffmpeg', action='store_true', default=False,
        help='Force the use the FFMPEG backend instead of imageio, useful if '
        'you want to generate a video instead of a GIF.'
    )

    parser.add_argument(
        'datacube', metavar='DATA_CUBE', type=str, help='The datacube used '
        'to extract the spectra.'
    )

    parser.add_argument(
        'wavelength', metavar='WAVE', type=str, help='centra wavelnght of '
        'the feature to plot expressed in an astropy compatible format (ie. '
        '100angstrom or 1nm or 1). Adimensional values will be considered '
        'expressed in angstroms.'
    )

    parser.add_argument(
        'size', metavar='SIZE', type=str, help='width of the area used to '
        'generate the animation, expressed in an astropy compatible format '
        '(ie. 100angstrom or 1nm or 1). Adimensional values will be considered'
        ' expressed in angstroms.'
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


def plot_cube_slice_animation(options=None):
    """
    Plot animanted gif of the extracted sources.

    Parameters
    ----------
    options : list of str, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    global HAS_IMAGEIO
    global FFMPEG_EXC

    if not HAS_IMAGEIO:
        if FFMPEG_EXC is None:
            print(
                "\nERROR: ffmpeg not found and imageio not installed!\n\n"
                "This program requires either the python package imageio (the "
                "preferred option) or a working version of FFMPEG. "
                "If you already have FFMPEG try to manually set the executable"
                " path with the --ffmpeg-exec option\n"
            )
            sys.exit(1)

    quantity_support()
    args = __plot_slice_argshandler(options)

    n_spectra = len(args.spectra)

    cutout_size = apu.Quantity(args.cutout_size)

    out_fps = abs(args.out_fps)
    out_loop = abs(args.out_loop)
    ffmpeg_exec = args.ffmpeg_exec

    if args.ffmpeg:
        HAS_IMAGEIO = False

    if args.cube_vlim is not None:
        cube_vlim = [float(x) for x in args.cube_vlim.split(',')]
        cube_vmin = np.min(cube_vlim)
        cube_vmax = np.max(cube_vlim)
    else:
        cube_vmin = None
        cube_vmax = None

    if args.cutout is not None:
        try:
            big_image = load_rgb_fits(args.cutout)
        except FileNotFoundError:
            print(f"ERROR: file not found '{args.cutout}'")
            sys.exit(1)

        if big_image is None:
            big_image = {
                'data': fits.getdata(args.cutout),
                'wcs': wcs.WCS(fits.getheader(args.cutout)),
                'type': 'gray'
            }
    else:
        big_image = None

    print("Loading spectral data...")
    spectra_list = []
    for j, spectrum_fits_file in enumerate(args.spectra):
        progress = (j + 1) / len(args.spectra)
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

                try:
                    object_coord_frame = main_header['FRAME']
                except KeyError:
                    object_coord_frame = 'fk5'

                obj_center = SkyCoord(
                    object_ra, object_dec,
                    unit='deg',
                    frame=object_coord_frame
                )

            try:
                flux_units = spec_hdu.header['BUNIT']
            except KeyError:
                flux_units = None

            try:
                wavelength_units = spec_hdu.header['CUNIT1']
            except KeyError:
                wavelength_units = None

            flux_data = spec_hdu.data.copy()
            spec_wcs = wcs.WCS(spec_hdu.header, fobj=hdulist).copy()
            var_data = var_hdu.data.copy()

            if nan_mask_hdu is not None:
                nan_mask = nan_mask_hdu.data == 1
            else:
                nan_mask = None

            # NOTE: wavelengths must be in Angstrom units
            pixel = np.arange(len(flux_data))
            wavelengths = spec_wcs.pixel_to_world(pixel).Angstrom

            if big_image is not None:
                if big_image['type'] == 'rgb':
                    cutout_dict = get_rgb_cutout(
                        big_image['data'],
                        center=obj_center,
                        size=cutout_size,
                        data_wcs=big_image['wcs']
                    )
                    cutout = np.asarray(cutout_dict['data']).transpose(1, 2, 0)
                    cutout_wcs = cutout_dict['wcs'][0]
                else:
                    cutout_dict = get_gray_cutout(
                        big_image['data'],
                        center=obj_center,
                        size=cutout_size,
                        data_wcs=big_image['wcs']
                    )
                    cutout = np.array(cutout_dict['data'])
                    cutout_wcs = cutout_dict['wcs']
                cutout_vmin = np.nanmin(big_image['data'])
                cutout_vmax = np.nanmax(big_image['data'])
            else:
                cutout = None
                cutout_wcs = None
                cutout_vmin = None
                cutout_vmax = None

            spectra_list.append(
                (
                    object_id,
                    (wavelengths, wavelength_units),
                    (flux_data, flux_units),
                    var_data,
                    nan_mask,
                    obj_center,
                    cutout,
                    cutout_wcs,
                    (cutout_vmin, cutout_vmax),
                    extraction_mode,
                    specex_apertures
                )
            )

    # We assume all the input spectra come from the same cube and have the
    # same dispersion grid
    wavelengths = spectra_list[0][1][0]

    c_wave = apu.Quantity(args.wavelength)
    size = apu.Quantity(args.size)

    print(
        f"Generating animation centered on {c_wave} with a size of {size}..."
    )

    if str(c_wave.unit).strip() != '':
        c_wave = c_wave.to(apu.angstrom).value
    else:
        c_wave = c_wave.value

    if str(size.unit).strip() != '':
        size = size.to(apu.angstrom).value
    else:
        size = size.value

    w_min = c_wave - size/2
    w_max = c_wave + size/2

    play_wav_mask = (wavelengths >= w_min) & (wavelengths <= w_max)
    play_wav_ang = wavelengths[play_wav_mask]

    with fits.open(args.datacube) as cube_hdul:
        cube_spec_hdu = get_hdu(cube_hdul, ['spec', 'data',])
        cube_wcs = wcs.WCS(cube_spec_hdu.header)
        cube_data = cube_spec_hdu.data
        reduced_cube = cube_data[play_wav_mask]

    if args.plot_wrange is None:
        plot_wmin = w_min - 5*size
        plot_wmax = w_max + 5*size
    else:
        plot_w_list = [
            x.to(apu.angstrom).value if x.unit else x.value
            for x in [apu.Quantity(k) for k in args.plot_wrange.split(',')]
        ]
        plot_wmin = np.min(plot_w_list)
        plot_wmax = np.max(plot_w_list)

    # Create the base canvas
    n_spectra = len(spectra_list)

    cell_c_size = 5
    cell_h_frac = 0.3
    cell_h = 5
    cell_w = 5

    fig = plt.figure(figsize=(cell_w * cell_c_size, cell_c_size * n_spectra))
    gs = GridSpec(cell_h*n_spectra, cell_w, figure=fig, hspace=0.5)

    c_dh = int(cell_h * cell_h_frac)

    axs = []
    if big_image:
        cut_cube_idx = -2
    else:
        cut_cube_idx = -1

    for k, spc in enumerate(spectra_list):
        cube_base_cutout = Cutout2D(
            data=reduced_cube[0],
            position=spc[5],
            size=cutout_size,
            wcs=cube_wcs.celestial
        )

        cell_h_k = cell_h*k
        cell_h_k1 = cell_h*(k+1)
        ax_sp = fig.add_subplot(gs[cell_h_k:cell_h_k1-c_dh, :cut_cube_idx])
        ax_vr = fig.add_subplot(
            gs[cell_h_k1-c_dh:cell_h_k1, :cut_cube_idx],
            sharex=ax_sp
        )
        ax_cut_cube = fig.add_subplot(
            gs[cell_h_k:cell_h_k1, cut_cube_idx],
            projection=cube_base_cutout.wcs
        )

        cube_img = ax_cut_cube.imshow(
            cube_base_cutout.data,
            cmap=args.cmap,
            origin='lower',
            aspect='equal',
            vmin=cube_vmin,
            vmax=cube_vmax
        )

        if big_image:
            ax_cut_img = fig.add_subplot(
                gs[cell_h_k:cell_h_k1, -1],
                projection=spc[7]
            )
        else:
            ax_cut_img = None

        plot_spectrum(
            wavelengths=spc[1][0],
            flux=spc[2][0],
            variance=spc[3],
            nan_mask=spc[4],
            redshift=args.redshift,
            wavelengt_units=spc[1][1],
            flux_units=spc[2][1],
            smoothing=args.spec_smoothing,
            cutout=spc[6],
            cutout_wcs=spc[7],
            cutout_vmin=spc[8][0],
            cutout_vmax=spc[8][1],
            wave_range=[plot_wmin, plot_wmax],
            show_legend=False,
            axs=[ax_sp, ax_vr, ax_cut_img, None],
        )

        spec_cursor = ax_sp.axvline(0, color='red', ls='-', lw=2, alpha=0.7)
        var_cursor = ax_vr.axvline(0, color='red', ls='-', lw=2, alpha=0.7)

        row_axs = {
            'spec': ax_sp,
            'spec_cursor': spec_cursor,
            'var': ax_vr,
            'var_cursor': var_cursor,
            'cube_cutout': ax_cut_cube,
            'img_cutout': ax_cut_img,
            'cube_img': cube_img
        }
        ax_cut_cube.axis('off')

        if ax_cut_img:
            # TODO: check condition to invert axis
            ax_cut_img.invert_xaxis()
            ax_cut_img.invert_yaxis()

        axs.append(row_axs)

    bkg = fig.canvas.copy_from_bbox(fig.bbox)

    if HAS_IMAGEIO:
        writer = imageio.get_writer(
            args.outname,
            mode='I',
            loop=out_loop,
            duration=1 / out_fps
        )

    with tempfile.TemporaryDirectory() as tempdir:
        figname = os.path.join(tempdir, 'frame_%06d.png')
        for j, wav in enumerate(play_wav_ang):
            progress = (j + 1) / len(play_wav_ang)
            sys.stdout.write(f"\r{get_pbar(progress)} {progress:.2%}\r")
            sys.stdout.flush()

            fig.canvas.restore_region(bkg)
            for k, (row_axs, spc) in enumerate(zip(axs, spectra_list)):
                row_axs['spec_cursor'].set_xdata([wav, wav])
                row_axs['var_cursor'].set_xdata([wav, wav])

                current_cutout = Cutout2D(
                    data=reduced_cube[j],
                    position=spc[5],
                    size=cutout_size,
                    wcs=cube_wcs.celestial
                )

                if args.cube_smoothing:
                    current_cutout_data = gaussian_filter(
                        current_cutout.data,
                        args.cube_smoothing,
                        mode='constant'
                    )
                else:
                    current_cutout_data = current_cutout.data

                row_axs['cube_img'].set_data(current_cutout_data)

            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
            fig.savefig(
                figname % j,
                bbox_inches='tight',
                dpi=args.out_dpi
            )
            if HAS_IMAGEIO:
                writer.append_data(imageio.v3.imread(figname % j))

        if HAS_IMAGEIO:
            writer.close()
        else:
            print("Encoding with FFMPEG...")
            ffmpeg_cmd = [
                ffmpeg_exec, '-y', '-i', figname,
                '-framerate', str(out_fps),
                '-loop', str(out_loop),
                args.outname
            ]
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE
            )
            print(result)
    plt.close(fig)
