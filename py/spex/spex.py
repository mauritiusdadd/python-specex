#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import argparse

import numpy as np
import multiprocessing

from scipy.signal import savgol_filter
from astropy import wcs
from astropy.table import Table, Column
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .utils import plot_zfit_check, get_cutout, get_log_img, get_hdu, get_pbar
from .utils import stack, plot_scandata

try:
    from .rrspex import rrspex, get_templates
except ImportError:
    HAS_RR = False
else:
    HAS_RR = True


def __argshandler():
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
        '--mode', metavar='WORKING_MODE', type=str, default='auto',
        help='Set the working mode of the program. Can be one of the following'
        ' modes: "auto", "kron_ellipse", "kron_circle", "circular_aperture". '
        'When using "kron_ellipse" mode, the input catalog must contain the '
        'column for objetcs kron radius, semi-major and semi-minor axes and '
        'for the rotation angle of the objects. In this mode all the pixels '
        'whitin this ellipse will be added together to compute the spectra. '
        '"kron_circular" mode requires the same parameters of the previous '
        'mode, but a circular aperture is used instead of an ellipse and the '
        'radius of this aperture is the circularized kron radius (i.e. the '
        'kron darius multiplied by the axis ratio q = minor-axis/major-axis. '
        'In circular_aperture mode, no morphological parameters are required '
        'to be in the input catalog and the aperture size is set using the '
        'input parameter --aperture-size. In "auto" mode, the best working '
        'mode is selected based on the parameters available in the input '
        'catalog.'
    )

    parser.add_argument(
        '--catalog', metavar='CATALOG', type=str, default=None,
        help='A catalog containing the position and the (optional) morphology '
        'of the objects of which you want to extract the spectra. The catalog '
        'must contain at least the RA and DEC coordinates and optionally the '
        'dimensions of the isophotal axes and the rotation angle of the '
        ' isophotal ellipse, as well as the kron radius. The way how these '
        'parameter are used depends on the program working mode (see the '
        '--mode section for more information). The catalog format is'
        ' by default compatible by the one generated by Sextractor, so the '
        'column name used to read the aformentioned qauntities are '
        'respectively ALPHA_J2000, DELTA_J2000, A_IMAGE, B_IMAGE, THETA_IMAGE '
        'and KRON_RADIUS. If the catalog has different column names for these '
        'quantities, they can be overridden with a corresponding --key-* '
        'argument (see below).'
    )

    parser.add_argument(
        '--regionfile', metavar='REGFILE', type=str, default=None,
        help='A region file containing regions from which extract spectra. '
        'The name of the refion will be used as ID for the spectra'
    )

    parser.add_argument(
        '--nspectra', metavar='N', type=int, default=0,
        help='Number of spectra to extract (usefull for debugging purposes).'
    )

    parser.add_argument(
        '--sn-threshold', metavar='SN_THRESH', type=float, default=1.5,
        help='Set the signal to Noise ratio threshold: objects with their '
        'spectrum having a SN ratio lower than threshold will be ignored.'
    )

    parser.add_argument(
        '--no-nans', action='store_true', default=False,
        help="This option tells the program to replace eventual NaNs in the"
        "extracted spectrum with linarly interpolated data. In this case the "
        "output fits files contain an additional extension named 'NAN_MASK' "
        "that identifies the previously bad pixels in the spectrum."
    )

    parser.add_argument(
        '--info-hdu', metavar='INFO_HDU', type=int, default=0,
        help='The HDU containing cube metadata. If this argument '
        'Set this to -1 to automatically detect the HDU containing the info. '
        'NOTE that this value is zero indexed (i.e. firts HDU has index 0).'
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
        '--invert-mask', action='store_true', default=False,
        help='Set whether to use the inverse of the cube mask.'
    )

    parser.add_argument(
        '--aperture-size', metavar='APERTURE', type=float, default=0.8,
        help='Set the radius in arcsecs of a fixed circula aperture used in '
        'the "fixed-aperture" mode. The pixels within this aperture will be '
        'used for spectra extraction. '
        'The default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--cat-pixelscale', metavar='MAS_PIXEL', type=float, default=1,
        help='Set the scale size in mas of the major and minor axes. Useful if'
        'A and B are given in pixels instead of MAS (like in the case of '
        'catalogs extracted by sextractor). NOTE that this is not the pixel '
        'scale of the cube but it is the size of a pixel of the image used to '
        'generate the input catalog. This parameter will be ignored if using '
        'a region file as input.'
        'The default value is %(metavar)s=%(default)s.'
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
        '--key-ra', metavar='RA_KEY', type=str, default='ALPHA_J2000',
        help='Set the name of the column from which to read the RA of the '
        'objects. The default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--key-dec', metavar='DEC_KEY', type=str, default='DELTA_J2000',
        help='Set the name of the column from which to read the RA of the '
        'objects. The default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--key-a', metavar='A_KEY', type=str, default='A_IMAGE',
        help='Set the name of the column from which to read the size of the'
        'semi-major isophotal axis. The default value is '
        '%(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--key-b', metavar='B_KEY', type=str, default='B_IMAGE',
        help='Set the name of the column from which to read the size of the'
        'semi-minor isophotal axis. The default value is '
        '%(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--key-theta', metavar='THETA_KEY', type=str, default='THETA_IMAGE',
        help='Set the name of the column from which to read the rotation angle'
        ' of the isophotal ellipse. The default value is '
        '%(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--key-kron', metavar='KRON_KEY', type=str, default='KRON_RADIUS',
        help='Set the name of the column from which to read the size of the '
        'Kron radius (in units of A or B). The default value is '
        '%(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--key-id', metavar='OBJID_KEY', type=str, default='NUMBER',
        help='Set the name of the column from which to read the id of the '
        'object to which the spectral data belong. If this argument is not '
        'specified then a suitable column will be selected according to its '
        'name or, if none is found, a new progressive id will be generated. '
        'This id will be included in the fits header.'
    )

    parser.add_argument(
        '--outdir', metavar='DIR', type=str, default=None,
        help='Set the directory where extracted spectra will be outputed. '
        'If this parameter is not specified, then a new directory will be '
        'created based on the name of the input cube.'
    )

    parser.add_argument(
        '--check-images', action='store_true', default=False,
        help='Whether or not to generate check images (cutouts, etc.).'
    )

    parser.add_argument(
        "--checkimg-outdir", type=str, default=None, required=False,
        help='Set the directory where check images are saved (when they are '
        'enabled thorugh the appropriate parameter).'
    )

    if HAS_RR:
        rr_help_note = ''
    else:
        rr_help_note = ' (WARNING: this option is unavailable since redrock' \
                       ' seems not to be installed).'

    parser.add_argument(
        "--zbest", type=str, default=None, required=False,
        help='Whether to find redshift from the spectra. ' + rr_help_note
    )

    parser.add_argument(
        "--priors", type=str, default=None, required=False,
        help='optional redshift prior file. ' + rr_help_note
    )

    parser.add_argument(
        "--nminima", type=int, default=3, required=False,
        help="the number of redshift minima to search. " + rr_help_note
    )

    parser.add_argument(
        "--mp", type=int, default=0, required=False,
        help="if not using MPI, the number of multiprocessing processes to use"
        " (defaults to half of the hardware threads). " + rr_help_note
    )

    parser.add_argument(
        "-t", "--templates", type=str, default=None, required=False,
        help="template file or directory. " + rr_help_note
    )

    parser.add_argument(
        "--debug", action='store_true', default=False,
        help="Enable some debug otupout on plots and on the console."
    )

    args = parser.parse_args()

    if args.cutouts_image and not os.path.exists(args.cutouts_image):
        print(f"The file {args.cutouts_image} does not exist!")
        sys.exit(1)

    if args.mp == 0:
        args.mp = int(multiprocessing.cpu_count() / 4)

    return args


# TODO: resample to constant log(lambda)
def get_spplate_fits(cube_header, spec_header, obj_ids, spec_data,
                     var_data=None, and_mask_data=None, or_mask_data=None,
                     wdisp_data=None, sky_data=None):
    spec_hdu = fits.PrimaryHDU()
    ivar_hdu = fits.ImageHDU()
    andmsk_hdu = fits.ImageHDU()
    ormsk_hdu = fits.ImageHDU()
    wdisp_hdu = fits.ImageHDU()
    tbl_hdu = fits.BinTableHDU()
    sky_hdu = fits.ImageHDU()

    spec_hdu.data = spec_data.T
    spec_hdu.header['PLATEID'] = 0

    try:
        spec_hdu.header['MJD'] = cube_header['MJD']
    except KeyError:
        try:
            spec_hdu.header['MJD'] = cube_header['MJD-OBS']
        except KeyError:
            print(
                "WARNING: no MJD found in the cube metadata, using extraction "
                "time instead!",
                file=sys.stderr
            )
            spec_hdu.header['MJD'] = Time.now().mjd

    # See the following link for COEFF0 and COEFF1 definition
    # https://classic.sdss.org/dr7/products/spectra/read_spSpec.html
    try:
        spec_hdu.header["COEFF0"] = spec_header["COEFF0"]
        spec_hdu.header["COEFF1"] = spec_header["COEFF1"]
    except KeyError:
        print(
            "WARING: wavelength binning of the input cube is not compatible "
            "with boss log10 binning data model.\n        Computing the best "
            "values of COEFF0 and COEFF1 for this wavelengt range...",
            file=sys.stderr
        )
        # Compute coeff0 and coeff1 that best approximate wavelenght range
        # of the datacube
        crpix = spec_header["CRPIX3"] - 1
        crval = spec_header["CRVAL3"]
        cd1 = spec_header["CD3_3"]

        w_end = spec_data.shape[1] - crpix

        coeff0 = np.log10(crval + cd1*(crpix))
        coeff1 = np.log10(1 + (w_end*cd1)/crval) / w_end

        spec_hdu.header['COEFF0'] = coeff0
        spec_hdu.header['COEFF1'] = coeff1

    spec_hdu.header["CRVAL1"] = spec_header["CRVAL3"]
    spec_hdu.header["CD1_1"] = spec_header["CD3_3"]
    spec_hdu.header["CDELT1"] = spec_header["CD3_3"]
    spec_hdu.header["CRPIX1"] = spec_header["CRPIX3"]
    spec_hdu.header['BUNIT'] = spec_header['BUNIT']
    spec_hdu.header["CUNIT1"] = spec_header["CUNIT3"]
    spec_hdu.header["CTYPE1"] = spec_header["CTYPE3"]

    if var_data is not None:
        ivar_hdu.data = 1 / var_data.T

    if wdisp_data is None:
        wdisp_hdu.data = np.zeros_like(spec_data.T)
    else:
        wdisp_hdu.data = wdisp_data.T

    info_table = Table()
    info_table.add_column(
        Column(data=obj_ids, name='FIBERID', dtype='>i4')
    )

    tbl_hdu.data = info_table.as_array()

    hdul = fits.HDUList([
        spec_hdu, ivar_hdu, andmsk_hdu, ormsk_hdu, wdisp_hdu, tbl_hdu, sky_hdu
    ])

    return hdul


def get_spsingle_fits(main_header, spec_wcs_header, obj_spectrum,
                      spec_hdu_header, obj_spectrum_var=None, no_nans=False):
    """
    Generate a fits containing the spectral data.

    Parameters
    ----------
    main_header : dict
        Dictionary containing the base information of the spectrum extraction.
        The keys of the dictionaries are the names of the fits CARDS that will
        be written, while the values can be either strings or tuples in the
        form of (cadr value, card comment).
    spec_wcs_header : astropy.io.fits.Header object
        Header containing the WCS information of the spectrum.
    obj_spectrum : numpy.ndarray
        The actual spectral data.
    spec_hdu_header : astropy.io.fits.Header object
        Header of the HDU of the original cube containing the spectral data.
    obj_spectrum_var : numpy.ndarray (default=None)
        The variance of the spectral data.
    no_nans : bool, optional
        If true, nan values in the spectrum are replaced by a linear
        interpolation. In this case an additional extension is added to the
        fits file, containing a mask for identifying the previously bax pixels.

    Returns
    -------
    hdul : astropy.io.fits.HDUList object
        The fits HDUL containing the extracted spectra.

    """
    my_time = Time.now()
    my_time.format = 'isot'

    # The primary hdu contains no data but a header with information on
    # how the spectrum was extracted
    main_hdu = fits.PrimaryHDU()
    for key, val in main_header.items():
        try:
            main_hdu.header[key] = val
        except ValueError:
            print(
                f"Invalid value for spec header: {key} = {val}",
                file=sys.stderr
            )

    spec_header = fits.Header()
    spec_header['BUNIT'] = spec_hdu_header['BUNIT']
    spec_header["OBJECT"] = spec_hdu_header["OBJECT"]
    spec_header["CUNIT1"] = spec_hdu_header["CUNIT3"]
    spec_header["CTYPE1"] = spec_hdu_header["CTYPE3"]
    spec_header["OBJECT"] = spec_hdu_header["OBJECT"]
    spec_header["CRPIX1"] = spec_hdu_header["CRPIX3"]
    spec_header["CRVAL1"] = spec_hdu_header["CRVAL3"]
    spec_header["CDELT1"] = spec_hdu_header["CD3_3"]
    spec_header["OBJECT"] = spec_hdu_header["OBJECT"]

    if no_nans:
        nanmask = np.isnan(obj_spectrum)
        obj_spectrum[nanmask] = 0
        obj_spectrum_var[nanmask] = np.nanmax(obj_spectrum_var) * 100.0
    else:
        nanmask = None

    hdu_list = [
        main_hdu,
        fits.ImageHDU(
            data=obj_spectrum,
            header=spec_header,
            name='SPECTRUM'
        )
    ]

    if obj_spectrum_var is not None:
        var_header = fits.Header()
        var_header['BUNIT'] = spec_hdu_header['BUNIT']
        var_header["OBJECT"] = spec_hdu_header["OBJECT"]
        var_header["CUNIT1"] = spec_hdu_header["CUNIT3"]
        var_header["CTYPE1"] = spec_hdu_header["CTYPE3"]
        var_header["OBJECT"] = spec_hdu_header["OBJECT"]
        var_header["CRPIX1"] = spec_hdu_header["CRPIX3"]
        var_header["CRVAL1"] = spec_hdu_header["CRVAL3"]
        var_header["CDELT1"] = spec_hdu_header["CD3_3"]
        var_header["OBJECT"] = spec_hdu_header["OBJECT"]
        hdu_list.append(
            fits.ImageHDU(
                data=obj_spectrum_var,
                header=var_header,
                name='VARIANCE'
            )
        )

    if nanmask is not None:
        hdu_list.append(
            fits.ImageHDU(
                data=nanmask.astype('uint8'),
                name='NAN_MASK'
            )
        )

    hdul = fits.HDUList(hdu_list)
    return hdul


def parse_regionfile(regionfile, key_ra='ALPHA_J2000', key_dec='DELTA_J2000',
                     key_a='A_IMAGE', key_b='B_IMAGE', key_theta='THETA_IMAGE',
                     key_id='NUMBER', key_kron='KRON_RADIUS'):
    """
    Parse a regionfile and return an asrtopy Table with sources information.

    Note that the only supported shape are 'circle', 'ellipse' and 'box',
    other shapes in the region file will be ignored. Note also that 'box'
    is treated as the bounding box of an ellipse.

    Parameters
    ----------
    regionfile : str
        Path of the regionfile.
    pixel_scale : float or None, optional
        The pixel scale in mas/pixel used to compute the dimension of the
        size of the objects. If None, height and width in the region file will
        be considered already in pixel units.
    key_ra : str, optional
        Name of the column that will contain RA of the objects.
        The default value is 'ALPHA_J2000'.
    key_dec : str, optional
        Name of the column that will contain DEC of the objects
        The default value is 'DELTA_J2000'.
    key_a : str, optional
        Name of the column that will contain the semi major axis.
        The default value is 'A_IMAGE'.
    key_b : str, optional
        Name of the column that will contain the semi minor axis.
        The default value is 'B_IMAGE'.
    key_theta : str, optional
        Name of the column that will contain angle between the major axis and
        the principa axis of the image.
        The default value is 'THETA_IMAGE'.

    Returns
    -------
    sources : astropy.table.Table
        The table containing the sources.

    """
    def degstr2mas(degstr, degsep='°', minsep="'", secsep='"'):
        deg_sep = degstr.find(degsep)
        min_sep = degstr.find(minsep)
        sec_sep = degstr.find(secsep)

        degs = float(degstr[:deg_sep]) if deg_sep >= 0 else 0
        mins = float(degstr[deg_sep+1:min_sep]) if min_sep >= 0 else 0
        secs = float(degstr[min_sep+1:sec_sep]) if sec_sep >= 0 else 0

        return secs + 60*mins + 3600*degs

    with open(regionfile, 'r') as f:
        regions = f.read().splitlines()

    myt = Table(
        names=[
            key_id, key_ra, key_dec, key_a, key_b, key_theta, key_kron
        ],
        dtype=[
            'U16', 'float32', 'float32', 'float32',
            'float32', 'float32', 'float32'
        ]
    )

    for j, reg in enumerate(regions):
        reg = reg.replace(' ', '').lower().split('(')
        if len(reg) < 2:
            continue

        regtype = reg[0]
        regdata = reg[1].split('#')
        regparams = regdata[0][:-1].split(',')
        try:
            regcomments = regdata[1]
            t_start = regcomments.find('text={') + 6
            if t_start < 0:
                obj_name = ''
            else:
                t_end = regcomments.find('}', t_start)
                obj_name = regcomments[t_start:t_end]
        except IndexError:
            obj_name = ''

        if not obj_name:
            obj_name = str(j)

        obj_ra = float(regparams[0])
        obj_dec = float(regparams[1])

        if regtype == "circle":
            obj_a = degstr2mas(regparams[2])
            obj_b = obj_a
            obj_theta = 0
        elif regtype == "ellipse" or regtype == "box":
            obj_a = degstr2mas(regparams[2])
            obj_b = degstr2mas(regparams[3])
            obj_theta = float(regparams[4])
        else:
            print(
                f"WARNING: '{regtype}' region type not supported yet!",
                file=sys.stderr
            )
            continue

        myt.add_row(
            (obj_name, obj_ra, obj_dec, obj_a, obj_b, obj_theta, 1)
        )
    return myt


def spex():
    """
    Run the main program.

    Returns
    -------
    None.

    """
    args = __argshandler()

    if args.catalog is not None:
        sources = Table.read(args.catalog)
        basename_with_ext = os.path.basename(args.catalog)
        pixelscale = args.cat_pixelscale
    elif args.regionfile is not None:
        sources = parse_regionfile(
            args.regionfile,
            key_ra=args.key_ra,
            key_dec=args.key_dec,
            key_a=args.key_a,
            key_b=args.key_b,
            key_theta=args.key_theta,
            key_id=args.key_id,
            key_kron=args.key_kron,
        )
        basename_with_ext = os.path.basename(args.regionfile)
        pixelscale = 1.0
    else:
        print(
            "You must provide at least an input catalog or a region file!",
            file=sys.stderr
        )
        sys.exit(1)

    # Get user input from argument parsers helper

    basename = os.path.splitext(basename_with_ext)[0]

    hdl = fits.open(args.input_cube[0])

    spec_hdu = get_hdu(
        hdl,
        hdu_index=args.spec_hdu,
        valid_names=['data', 'spec', 'spectrum', 'spectra'],
        msg_err_notfound="ERROR: Cannot determine which HDU contains spectral "
                         "data, try to specify it manually!",
        msg_index_error="ERROR: Cannot open HDU {} to read specra!"
    )

    var_hdu = get_hdu(
        hdl,
        hdu_index=args.var_hdu,
        valid_names=['stat', 'var', 'variance', 'noise'],
        msg_err_notfound="ERROR: Cannot determine which HDU contains the "
                         "variance data, try to specify it manually!",
        msg_index_error="ERROR: Cannot open HDU {} to read the "
                        "variance!",
        exit_on_errors=False
    )

    mask_hdu = get_hdu(
        hdl,
        hdu_index=args.mask_hdu,
        valid_names=['mask', 'platemask', 'footprint', 'dq'],
        msg_err_notfound="ERROR: Cannot determine which HDU contains the "
                         "mask data, try to specify it manually!",
        msg_index_error="ERROR: Cannot open HDU {} to read the mask!",
        exit_on_errors=False
    )

    cube_wcs = wcs.WCS(spec_hdu.header)
    celestial_wcs = cube_wcs.celestial
    spectral_wcs = cube_wcs.spectral
    wcs_frame = wcs.utils.wcs_to_celestial_frame(cube_wcs)

    cube_pixelscale = [
        units.Quantity(sc_val, u).to_value('arcsec')
        for sc_val, u in zip(
            wcs.utils.proj_plane_pixel_scales(celestial_wcs),
            celestial_wcs.wcs.cunit
        )
    ]
    cube_pixelscale = np.nanmean(cube_pixelscale)

    ra_unit = sources[args.key_ra].unit
    if ra_unit is None:
        if args.debug:
            print(
                "RA data has no units, assuming values are in degrees!",
                file=sys.stderr
            )
        ra_unit = 'deg'

    dec_unit = sources[args.key_dec].unit
    if dec_unit is None:
        if args.debug:
            print(
                "DEC data has no units, assuming values are in degrees!",
                file=sys.stderr
            )
        dec_unit = 'deg'

    obj_sky_coords = SkyCoord(
        sources[args.key_ra], sources[args.key_dec],
        unit=(ra_unit, dec_unit),
        frame=wcs_frame
    )

    obj_x, obj_y = wcs.utils.skycoord_to_pixel(
        coords=obj_sky_coords,
        wcs=celestial_wcs
    )

    sources.add_column(obj_x, name='CUBE_X_IMAGE')
    sources.add_column(obj_y, name='CUBE_Y_IMAGE')

    img_height, img_width = spec_hdu.data.shape[1], spec_hdu.data.shape[2]

    yy, xx = np.meshgrid(
        np.arange(img_height),
        np.arange(img_width),
        indexing='ij'
    )

    if args.check_images:
        extracted_data = np.zeros((img_height, img_width))

    trasposed_spec = spec_hdu.data.transpose(1, 2, 0)

    if var_hdu is not None:
        trasposed_var = var_hdu.data.transpose(1, 2, 0)

    if mask_hdu is not None:
        cube_footprint = mask_hdu.data.prod(axis=0) == 0
        if args.invert_mask:
            cube_footprint = ~cube_footprint

    else:
        cube_footprint = None

    if args.outdir is None:
        outdir = f'{basename}_spectra'
    else:
        outdir = args.outdir

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if args.mode == 'auto':
        if (
                args.key_a in sources.colnames and
                args.key_b in sources.colnames and
                args.key_kron in sources.colnames
        ):
            mode = 'kron_ellipse'
        else:
            mode = 'circular_aperture'
    else:
        mode = args.mode

    print(f"Extracting spectra with strategy '{mode}'", file=sys.stderr)

    n_objects = len(sources)
    if args.nspectra:
        n_objects = int(args.nspectra)

    if args.key_id is None:
        valid_id_keys = [
            f"{i}{j}"
            for i in ['', 'OBJ', 'OBJ_', 'TARGET', 'TARGET_']
            for j in ['ID', 'NUMBER', 'UID', 'UUID']
        ]

        for key in valid_id_keys:
            if key in sources.colnames:
                key_id = key
                break
        else:
            key_id = None
    else:
        key_id = args.key_id

    out_specfiles = []
    spex_apertures = {}  # these values are in physical units (arcsecs)
    source_ids = []

    valid_sources_mask = np.zeros(len(sources), dtype=bool)

    for i, source in enumerate(sources[:n_objects]):
        progress = (i + 1) / n_objects
        sys.stderr.write(f"\r{get_pbar(progress)} {progress:.2%}\r")
        sys.stderr.flush()

        if key_id is not None:
            obj_id = f"{str(source[key_id]):0>6}"
        else:
            obj_id = f"{i:06}"

        obj_ra = source[args.key_ra]
        obj_dec = source[args.key_dec]
        xx_tr = xx - source['CUBE_X_IMAGE']
        yy_tr = yy - source['CUBE_Y_IMAGE']

        if mode == 'kron_ellipse':
            ang = np.deg2rad(source[args.key_theta])

            x_over_a = xx_tr*np.cos(ang) + yy_tr*np.sin(ang)
            x_over_a /= source[args.key_a]
            x_over_a = x_over_a ** 2

            y_over_b = xx_tr*np.sin(ang) - yy_tr*np.cos(ang)
            y_over_b /= source[args.key_b]
            y_over_b = y_over_b ** 2

            spex_apertures[obj_id] = (
                source[args.key_a] * pixelscale,
                source[args.key_b] * pixelscale,
                source[args.key_theta]
            )

            mask = (x_over_a + y_over_b) < (1.0/source[args.key_kron])

        elif mode == 'kron_circular':
            kron_circular = source[args.key_kron] * source[args.key_b]
            kron_circular /= source[args.key_a]

            spex_apertures[obj_id] = (
                kron_circular * pixelscale,
                kron_circular * pixelscale,
                0
            )

            mask = (xx_tr**2 + yy_tr**2) < (kron_circular ** 2)

        elif mode == 'circular_aperture':
            # NOTE: in this mode, aperture is already in asrcseconds
            spex_apertures[obj_id] = (
                args.aperture_size,
                args.aperture_size,
                0
            )
            mask = (xx_tr**2 + yy_tr**2) < args.aperture_size / cube_pixelscale

        if cube_footprint is not None:
            mask &= cube_footprint

        if np.sum(mask) == 0:
            continue

        obj_spectrum = trasposed_spec[mask].sum(axis=0)

        if np.sum(~np.isnan(obj_spectrum)) == 0:
            if args.debug:
                print(
                    f"WARNING: object {obj_id} has no spectral data, "
                    "skipping...",
                    file=sys.stderr
                )
            continue

        # Smoothing the spectrum to get a crude approximation of the continuum
        smoothed_spec = savgol_filter(obj_spectrum, 51, 11)

        # Subtract the smoothed spectrum to the spectrum itself to get a
        # crude estimation of the noise
        noise_spec = np.nanstd(obj_spectrum - smoothed_spec)

        # Get the mean value of the spectrum
        obj_mean_spec = np.nanmean(obj_spectrum)

        # Get the mean Signal to Noise ratio
        sn_spec = obj_mean_spec / np.nanmean(noise_spec)

        if sn_spec < args.sn_threshold:
            continue

        my_time = Time.now()
        my_time.format = 'isot'

        outname = f"spec_{obj_id}.fits"

        main_header = {
            'CUBE': (basename_with_ext, "Spectral cube used for extraction"),
            'RA': (obj_ra, "Ra of the center of the object"),
            'DEC': (obj_dec, "Dec of the center of the object"),
            'NPIX': (np.sum(mask), 'Number of pixels used for this spectra'),
            'HISTORY': f'Extracted on {str(my_time)}',
            'ID': obj_id,
            'SN': (sn_spec, ""),
        }

        # Copy the spectral part of the WCS into the new FITS
        spec_wcs_header = spectral_wcs.to_header()

        if var_hdu is not None:
            obj_spectrum_var = trasposed_var[mask].sum(axis=0)
            sn_var = obj_mean_spec / np.sqrt(np.nanmean(obj_spectrum_var))

            main_header['SN_VAR'] = (sn_var, "")
        else:
            obj_spectrum_var = None

        hdul = get_spsingle_fits(
            main_header, spec_wcs_header,
            obj_spectrum, spec_hdu.header,
            obj_spectrum_var, no_nans=args.no_nans
        )

        out_file_name = os.path.join(outdir, outname)
        hdul.writeto(
            out_file_name,
            overwrite=True
        )

        # If we arrived here, then the object has a vail spectrum and can be
        # saved. Let's also mark the object as valid and store its ID
        out_specfiles.append(os.path.realpath(out_file_name))
        source_ids.append(obj_id)
        valid_sources_mask[i] = True

        # Add also the extracted pixels to the extraction map
        if args.check_images:
            extracted_data += mask

    # Discard all invalid sources
    sources = sources[valid_sources_mask]
    source_ids = np.array(source_ids)

    if args.debug:
        print(
            f"\nExtracted {len(sources)} valid sources\n",
            file=sys.stderr
        )

    if args.checkimg_outdir is not None:
        check_images_outdir = args.checkimg_outdir
    else:
        check_images_outdir = os.path.join(outdir, 'checkimages')

    if not os.path.isdir(check_images_outdir):
        os.mkdir(check_images_outdir)

    # NOTE: for some weird reason redrock hangs if any figure has been
    #       created using matplotlib. Do all the plottings only after
    #       redrock has finished!
    if args.zbest:
        rrspex_options = [
            '--zbest', args.zbest,
            '--checkimg-outdir', check_images_outdir
        ]

        if args.priors is not None:
            rrspex_options += ['--priors', args.priors]

        if args.nminima is not None:
            rrspex_options += ['--nminima', f'{args.nminima:d}']

        if args.mp is not None:
            rrspex_options += ['--mp', f'{args.mp:d}']

        if args.templates is not None:
            rrspex_options += ['--templates', f'{args.templates}']

        rrspex_options += out_specfiles
        targets, zfit, scandata = rrspex(options=rrspex_options)

        if args.debug or not args.cutouts_image:
            stacked_cube = stack(spec_hdu.data)

        if args.debug:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(projection=celestial_wcs)
            logimg, cvmin, cvmax = get_log_img(stacked_cube)
            ax.imshow(
                logimg,
                origin='lower',
                vmin=cvmin,
                vmax=cvmax,
                cmap='jet'
            )
            fig.savefig("stacked_cube.png")
            plt.tight_layout()
            plt.close(fig)

        if args.cutouts_image:
            cutouts_image = fits.getdata(args.cutouts_image).transpose(1, 2, 0)
            cutouts_wcs = wcs.WCS(fits.getheader(args.cutouts_image))
            cutouts_wcs = cutouts_wcs.celestial
            cutout_wcs_frame = wcs.utils.wcs_to_celestial_frame(cutouts_wcs)
            cutouts_image, cut_vmin, cut_vmax = get_log_img(cutouts_image)
            cut_pixelscale = [
                units.Quantity(sc_val, u).to_value('arcsec')
                for sc_val, u in zip(
                    wcs.utils.proj_plane_pixel_scales(cutouts_wcs),
                    cutouts_wcs.wcs.cunit
                )
            ]
        else:
            cutouts_image = stacked_cube
            cutouts_wcs = None
            cutout_wcs_frame = wcs_frame
            cut_vmin, cut_vmax = None, None
            cut_pixelscale = wcs.utils.proj_plane_pixel_scales(celestial_wcs)
            cut_pixelscale = [
                units.Quantity(sc_val, u).to_value('arcsec')
                for sc_val, u in zip(
                    wcs.utils.proj_plane_pixel_scales(celestial_wcs),
                    celestial_wcs.wcs.cunit
                )
            ]
        cut_pixelscale = np.nanmean(cut_pixelscale)

        if args.debug:
            plot_templates = get_templates(templates=args.templates)
        else:
            plot_templates = None

        print("", file=sys.stderr)

        for i, target in enumerate(targets):
            progress = (i + 1) / len(targets)
            sys.stderr.write(
                f"\rGenerating previews {get_pbar(progress)} {progress:.2%}\r"
            )
            sys.stderr.flush()
            obj = sources[source_ids == target.id][0]

            obj_skycoord = SkyCoord(
                obj[args.key_ra],
                obj[args.key_dec],
                unit=(ra_unit, dec_unit),
                frame=cutout_wcs_frame
            )

            if cutouts_wcs is None:
                obj_position = (obj['CUBE_X_IMAGE'], obj['CUBE_Y_IMAGE'])
            else:
                obj_position = obj_skycoord

            cutout_size = args.cutouts_size / cut_pixelscale
            cutout, scutout_wcs = get_cutout(
                cutouts_image,
                position=obj_position,
                cutout_size=cutout_size,
                wcs=cutouts_wcs,
                vmin=cut_vmin,
                vmax=cut_vmax
            )

            fig, axs = plot_zfit_check(
                target,
                zfit,
                plot_template=plot_templates,
                cutout=cutout,
                wave_units=spec_hdu.header['CUNIT3'],
                flux_units=spec_hdu.header['BUNIT'],
            )

            e_wid = spex_apertures[target.id][0] / cut_pixelscale
            e_hei = spex_apertures[target.id][1] / cut_pixelscale
            e_ang = spex_apertures[target.id][2]

            aperture = Ellipse(
                (cutout_size/2.0, cutout_size/2.0),
                width=e_wid,
                height=e_hei,
                angle=e_ang,
                edgecolor='#0000ff',
                facecolor='none',
                ls='-',
                lw=1,
                alpha=0.8,
                fill=False
            )
            axs[1].add_patch(aperture)
            aperture = Ellipse(
                (cutout_size/2.0, cutout_size/2.0),
                width=e_wid,
                height=e_hei,
                angle=e_ang,
                edgecolor='#00ff00',
                facecolor='none',
                ls='--',
                lw=1,
                alpha=0.8,
                fill=False
            )
            axs[1].add_patch(aperture)

            obj_ra_str = obj_skycoord.ra.to_string(
                unit=units.hourangle, alwayssign=True, precision=4
            )

            obj_dec_str = obj_skycoord.ra.to_string(
                unit=units.deg, alwayssign=True, precision=4
            )

            axs[2].text(
                0, 0.5,
                f"RA: {obj_ra_str}\n"
                f"DEC: {obj_dec_str}\n",
                ha='left',
                va='top',
                transform=axs[2].transAxes,
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'none',
                }
            )

            figname = f'spectrum_{target.id}.png'
            figname = os.path.join(check_images_outdir, figname)
            fig.savefig(figname, dpi=150)
            plt.close(fig)

            if args.debug:
                figname = f'scandata_{target.id}.png'
                figname = os.path.join(check_images_outdir, figname)
                fig, axs = plot_scandata(target, scandata)
                fig.savefig(figname, dpi=150)
                plt.close(fig)

    if args.check_images:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(extracted_data, origin='lower')
        fig.savefig("sext_extraction_map.png")
        plt.close(fig)

    if args.debug:
        import IPython
        IPython.embed()


if __name__ == '__main__':
    spex()
