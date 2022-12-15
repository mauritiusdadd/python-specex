#!/usr/bin/env python
"""Make cutouts of spectral cubes."""

import os
import re
import sys
import argparse
import warnings
from typing import Optional, Union, Callable

import numpy as np

from astropy import units
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D

from .utils import get_pc_transform_params, rotate_data, get_pbar

KNOWN_SPEC_EXT_NAMES = ['spec', 'spectrum', 'flux', 'data', 'sci', 'science']
KNOWN_VARIANCE_EXT_NAMES = ['stat', 'stats', 'var', 'variance', 'noise', 'err']
KNOWN_MASK_EXT_NAMES = ['mask', 'platemask', 'footprint', 'dq']
KNOWN_RGB_EXT_NAMES = ['r', 'g', 'b', 'red', 'green', 'blue']


def __cutout_argshandler(options=None):
    """
    Parse the arguments given by the user.

    Inputs
    ------
    options: list or None
        If none, args are parsed from the command line, otherwise the options
        list is used as input for argument parser.

    Returns
    -------
    args: Namespace
        A Namespace containing the parsed arguments. For more information see
        the python documentation of the argparse module.
    """
    parser = argparse.ArgumentParser(
        description='Generate cutouts of spectral cubes (or fits images, both '
        'grayscale or RGB).'
    )

    parser.add_argument(
        'input_fits', metavar='INPUT_FIST', type=str, nargs=1,
        help='The spectral cube (or image) from which to extract a cutout.'
    )
    parser.add_argument(
        '--regionfile', '-r', metavar='REGION_FILE', type=str, default=None,
        help='The region-file used to identify the locations and sizes of the '
        'cutouts. If multiple regions are present in the region-file, a cutout'
        ' is generated for each region. If the input file is a spectral '
        'datacube, the text field of the region can be used to specify an '
        'optional wavelength range for the cutout of that region. If a region '
        'do not provide a wavelength range information and the --wave-range '
        'option is specified, then the wavelength range specified by the '
        'latter parameter is used, otherwise the cutout will contain the full '
        'wavelength range as the original datacube. wavelength ranges are '
        'ignored for grayscale or RGB images.'
        'If this option is not specified, then the coordinate and the size of '
        'a cutout region must be specified with the options --center, --sizes '
        'and --wave-range.'
    )
    parser.add_argument(
        '--center', '-c', metavar='RA,DEC', type=str, default=None,
        help='Specify the RA and DEC of the center of a single cutout. '
        'Both RA and DEC can be specified with units in a format compatible '
        'with astropy.units (eg. -c 10arcsec,5arcsec). If no no unit is '
        'specified, then the quantity is assumed to be in arcseconds.'
    )
    parser.add_argument(
        '--sizes', '-s', metavar='HEIGHT,WIDTH', type=str, default=None,
        help='Specify the RA and DEC of the center of a single cutout.'
        'Both HEIGHT and WIDTH can be specified with units in a format '
        'compatible with astropy.units (eg. -s 10arcsec,5arcsec).  If no no '
        'unit is specified, then the quantity is assumed to be in arcseconds.'
    )
    parser.add_argument(
        '--wave-range', '-w', metavar='MIN_W,MAX_W', type=str, default=None,
        help='Specify the wavelength range that the extracted cutout will '
        'contains. This option is ignored if the input file is a grayscale or '
        'an RGB image. Both HEIGHT and WIDTH can be specified with units in a '
        'format compatible with astropy.units '
        '(eg. -w 4500angstrom,6500angstrom). If no no unit is specified, then '
        'the quantity is assumed to be in angstrom.'
    )
    parser.add_argument(
        '--data-hdu', metavar='DATA_HDU[,HDU1,HDU2]', type=str,
        default=None, help='Specify which extensions contain data. For an '
        'rgb image more than one HDU can be specified, for example '
        '--data-hdu 1,2,3. If this option is not specified then the program '
        'will try to identify the data type and structure automatically.'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help="Print verbose outout."
    )
    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    inp_fits_file = args.input_fits[0]
    if (inp_fits_file is not None) and (not os.path.isfile(inp_fits_file)):
        print(f"The file {inp_fits_file} does not exist!")
        sys.exit(1)

    if args.regionfile is None:
        if (args.center is None) or (args.sizes is None):
            print(
                "If --regionfile is not specified then both --center and "
                "--sizes must be provided."
            )
            sys.exit(1)
    elif not os.path.isfile(args.regionfile):
        print("The file input regionfile does not exist!")
        sys.exit(1)

    return args


def parse_regionfile(regionfile: str, key_ra: str = 'RA', key_dec: str = 'DEC',
                     key_a: str = 'A_WORLD', key_b: str = 'B_WORLD',
                     key_theta: str = 'THETA_WORLD', key_wmin: str = 'WMIN',
                     key_wmax: str = 'WMAX'):
    """
    Parse a regionfile and return an asrtopy Table with sources information.

    Note that the only supported shape are 'circle', 'ellipse' and 'box',
    other shapes in the region file will be ignored.

    Parameters
    ----------
    regionfile : str
        Path of the regionfile.
    key_ra : str, optional
        Name of the column that will contain RA of the objects.
        The default value is 'RA'.
    key_dec : str, optional
        Name of the column that will contain DEC of the objects
        The default value is 'DEC'.
    key_a : str, optional
        Name of the column that will contain the semi major axis.
        The default value is 'A_WORLD'.
    key_b : str, optional
        Name of the column that will contain the semi minor axis.
        The default value is 'B_WORLD'.
    key_theta : str, optional
        Name of the column that will contain angle between the major axis and
        the principa axis of the image.
        The default value is 'THETA_WORLD'.
    key_wmin : str, optional
        Name of the column that will contain minimum value of the wavelength
        range. The default value is 'WMIN'.
    key_wmax : str, optional
        Name of the column that will contain maximum value of the wavelength
        range. The default value is 'WMAX'.

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

        return (secs + 60*mins + 3600*degs) * units.arcsec

    with open(regionfile, 'r') as f:
        regions = f.read().splitlines()

    myt = Table(
        names=[
            key_ra, key_dec, key_a, key_b, key_theta, key_wmin, key_wmax
        ],
        dtype=[
            'float32', 'float32', 'float32',
            'float32', 'float32', 'float32', 'float32'
        ],
        units=[
            units.deg, units.deg, units.arcsec,
            units.arcsec, units.deg, units.angstrom, units.angstrom
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
                wave_range_data = ['0angstrom', '0angstrom']
            else:
                t_end = regcomments.find('}', t_start)
                wave_range_data = regcomments[t_start:t_end].split(',')

        except IndexError:
            wave_range_data = ['0angstrom', '0angstrom']

        obj_w_min = units.Quantity(wave_range_data[0])
        if not obj_w_min.unit.to_string():
            obj_w_min = obj_w_min * units.angstrom

        obj_w_max = units.Quantity(wave_range_data[1])
        if not obj_w_max.unit.to_string():
            obj_w_max = obj_w_max * units.angstrom

        obj_ra = units.Quantity(regparams[0], units.deg)
        obj_dec = units.Quantity(regparams[1], units.deg)

        if regtype == "circle":
            obj_a = degstr2mas(regparams[2])
            obj_b = obj_a
            obj_theta = units.Quantity(0, units.deg)
        elif regtype == "ellipse":
            obj_a = degstr2mas(regparams[2])
            obj_b = degstr2mas(regparams[3])
            obj_theta = units.Quantity(regparams[4], units.deg)
        elif regtype == "box":
            obj_a = degstr2mas(regparams[2]) / 2
            obj_b = degstr2mas(regparams[3]) / 2
            obj_theta = units.Quantity(regparams[4], units.deg)
        else:
            print(
                f"WARNING: '{regtype}' region type not supported yet!",
                file=sys.stderr
            )
            continue

        myt.add_row(
            (
                obj_ra.to(units.deg), obj_dec.to(units.deg),
                obj_a.to(units.arcsec), obj_b.to(units.arcsec),
                obj_theta.to(units.deg),
                obj_w_min.to(units.angstrom), obj_w_max.to(units.angstrom)
            )
        )
    return myt


def get_gray_cutout(data: np.ndarray,
                    center: Union[SkyCoord, tuple, list],
                    size: Union[tuple, list],
                    angle: Optional[Union[float, units.Quantity]] = 0,
                    data_wcs: Optional[WCS] = None) -> dict:
    """
    Get the cutout for a grayscale image.

    This is a basic wrapper around astropy.nddata.utils.Cutout2D

    Parameters
    ----------
    data : np.ndarray
        The actual image data. Should have only two dimensions (a grayscale
        image has only X and Y corrdinates).
    center : astropy.coordinate.SkyCoord or tuple.
        The center of the cutout. If a SkyCoord is provided then a WCS for the
        image data musto also be specified with the parameter data_wcs.
        If a tuple is provided, then the first two numbers of the tuple are
        interpreted as the X and Y coordinate of the cutout center: in this
        case, if no WCS is specified, the values are assumed to be in pixels,
        else if a WCS is provided then the values are assumed to be in degrees.
    size : tuple or list
        The first two values in the tuple are interpreted as the width and
        height of the cutout. if no WCS is specified, the values are assumed to
        be in pixels, else if a WCS is provided then the values are assumed to
        be in degrees. Astropy.units.Quantity values are also supported.
    angle : float or astropy.units.Quantity, optional
        The rotation angle of the cutout. If it is a float, then it is
        interpreted in degrees. The default is 0.
    data_wcs : astropy.wcs.WCS or None, optional
        A WCS associated with the image data. The default is None.

    Returns
    -------
    cutout_dict: dict
        A dictionary containing the following key: value pairs:
            'data': np.ndarray
                The cutout data.
            'wcs': astropy.wcs.WCS or None
                The wcs for the cutout data.
    """
    angle = units.Quantity(angle, units.deg)
    if data_wcs is not None:
        data_wcs = data_wcs.celestial
        sx, sy, rot, shr_y = get_pc_transform_params(data_wcs)
        angle = angle - rot

    rotated_data = rotate_data(
        data=data,
        angle=angle,
        data_wcs=data_wcs
    )

    cutout = Cutout2D(
        rotated_data['data'],
        center, size,
        mode='partial',
        fill_value=np.nan,
        wcs=rotated_data['wcs'],
        copy=True
    )

    cutout_dict = {
        'data': cutout.data,
        'wcs': cutout.wcs
    }

    return cutout_dict


def get_rgb_cutout(data: Union[tuple, list, np.ndarray],
                   center: Union[SkyCoord, tuple],
                   size: Union[tuple, list],
                   data_wcs: Optional[Union[WCS, list, tuple]] = None,
                   resample_to_wcs: bool = False):
    """
    Get a cutout from a bigger RGB.

    Parameters
    ----------
    data : np.ndarray or tuple or list
        The actual image data. Should have only two dimensions (a grayscale
        image has only X and Y corrdinates).
    center : astropy.coordinate.SkyCoord or tuple.
        The center of the cutout. If a SkyCoord is provided then a WCS for the
        image data musto also be specified with the parameter data_wcs.
        If a tuple is provided, then the first two numbers of the tuple are
        interpreted as the X and Y coordinate of the cutout center: in this
        case, if no WCS is specified, the values are assumed to be in pixels,
        else if a WCS is provided then the values are assumed to be in degrees.
    size : tuple
        The first two values in the tuple are interpreted as the width and
        height of the cutout. if no WCS is specified, the values are assumed to
        be in pixels, else if a WCS is provided then the values are assumed to
        be in degrees. Astropy.units.Quantity values are also supported.
    data_wcs : astropy.wcs.WCS or None, optional
        A WCS associated with the image data. The default is None.
    reample_to_wcs : bool, optional
        If true reample the red, green and blue data to share the same WCS.
        In order to use this option, the WCSs for the input data must be
        provided, otherwise this option will be ignored and a warning message
        is outputed. The default is False.

    Returns
    -------
    cutout_dict: dict
        A dictionary containing the following key: value pairs:
            'data': np.ndarray
                The cutout data.
            'wcs': astropy.wcs.WCS or None
                The wcs for the cutout data.
    """
    # Do some sanity checks on the input parameters
    if isinstance(data, np.ndarray):
        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(
                "Only RGB images are supported: expected shape (N, M, 3) but"
                f"input data has shape {data.shape}."
            )
        if data_wcs is not None:
            data_wcs_r = data_wcs.celestial
            data_wcs_g = data_wcs.celestial
            data_wcs_b = data_wcs.celestial

        data_r = data[..., 0]
        data_g = data[..., 1]
        data_b = data[..., 2]
    elif isinstance(data, Union[tuple, list]):
        if len(data) != 3:
            raise ValueError(
                "'data' parameter only accepts list or tuple containing "
                "exactly 3 elements."
            )
        elif not all([isinstance(x, np.ndarray) for x in data]):
            raise ValueError(
                "All elements of the input tupel or list must be 2D arrays."
            )
        if data_wcs is None:
            if resample_to_wcs:
                warnings.warn(
                    "reample_to_wcs is set to True but no WCS info is provided"
                )
                resample_to_wcs = False
            resample_to_wcs = False
            data_wcs_r = None
            data_wcs_g = None
            data_wcs_b = None
        else:
            if not isinstance(data_wcs, Union[tuple, list]):
                raise ValueError(
                    "When 'data' is a list or a tuple, also data_wcs must be a"
                    "a list or a tuple of WCSs."
                )
            elif not all([isinstance(x, WCS) for x in data_wcs]):
                raise ValueError(
                    "All elements of wcs_data tuple or list must be WCS."
                )
            data_wcs_r = data_wcs[0].celestial
            data_wcs_g = data_wcs[1].celestial
            data_wcs_b = data_wcs[2].celestial
        data_r = data[0]
        data_g = data[1]
        data_b = data[2]
    else:
        raise ValueError(
            "Parameter 'data' only supports ndarray or list/tuple of ndarrays."
        )

    cutout_data_r = get_gray_cutout(data_r, center, size, data_wcs_r)
    cutout_data_g = get_gray_cutout(data_g, center, size, data_wcs_g)
    cutout_data_b = get_gray_cutout(data_b, center, size, data_wcs_b)

    if not resample_to_wcs:
        cutout_dict = {
            'data': (
                cutout_data_r['data'],
                cutout_data_g['data'],
                cutout_data_b['data']
            ),
            'wcs': (
                cutout_data_r['wcs'],
                cutout_data_g['wcs'],
                cutout_data_b['wcs'],
            )
        }

    return cutout_dict


def get_cube_cutout(data: np.ndarray,
                    center: Union[SkyCoord, tuple, list],
                    size: Union[tuple, list],
                    angle: Optional[Union[float, units.Quantity]] = 0,
                    wave_range: Optional[Union[tuple, list]] = None,
                    data_wcs: Optional[WCS] = None,
                    report_callback: Optional[Callable] = None):
    """
    Get a cutout of a spectral datacube.

    Parameters
    ----------
    data : np.ndarray
        The datacube data.
    center : astropy.coordinate.SkyCoord or tuple or list.
        The center of the cutout. If a SkyCoord is provided then a WCS for the
        image data musto also be specified with the parameter data_wcs.
        If a tuple is provided, then the first two numbers of the tuple are
        interpreted as the X and Y coordinate of the cutout center: in this
        case, if no WCS is specified, the values are assumed to be in pixels,
        else if a WCS is provided then the values are assumed to be in degrees.
    size : tuple or list
        The first two values in the tuple are interpreted as the width and
        height of the cutout. Both adimensional values and angular quantities
        are accepted. Adimensional values are interpreted as pixels.
        Angular values are converted to pixel values ignoring any non linear
        distorsion.
    angle : float or astropy.units.Quantity, optional
        The rotation angle of the cutout. If it is a float, then it is
        interpreted in degrees. The default is 0.
    wave_range : tuple or list, optional
        If not None, he first two values in the tuple are interpreted as the
        minimum and maximum value of the wavelength range for the cutout.
        If it is None, the whole wavelength range is used. The default is None.
    data_wcs : astropy.wcs.WCS or None, optional
        A WCS associated with the image data. The default is None..
    report_callback : Callable or None, optional
        A callable that will be execute every time the cutout of a single
        slice of the cube is computed. Must accept in input two arguments:
            - the number of slice processed so far
            - the total number of slices.

    Returns
    -------
    cutout_dict: dict
        A dictionary containing the following key: value pairs:
            'data': np.ndarray
                The cutout data.
            'wcs': astropy.wcs.WCS or None
                The wcs for the cutout data.
    """
    # Do some sanity checks on the input data
    if len(data.shape) != 3:
        raise ValueError("Unsupported datacube shape {data.shape}.")

    if not isinstance(size, Union[list, tuple]):
        raise ValueError(
            "'size' must be a list or a tuple of scalar values or angular "
            "quantities"
        )
    elif not all(
        [isinstance(x, Union[int, float, units.Quantity]) for x in size]
    ):
        raise ValueError(
            "'size' must be a list or a tuple of scalar values or angular "
            "quantities"
        )

    d_a, d_b = size[:2]

    if not isinstance(center, Union[SkyCoord, tuple, list]):
        raise ValueError("'center' must be SkyCoord or tuple or list.")

    angle = units.Quantity(angle, units.deg)

    if data_wcs is not None:
        if not isinstance(data_wcs, WCS):
            raise ValueError(
                "'data_wcs' must be eihter None or a valid WCS object"
            )

        if not data_wcs.has_spectral:
            raise ValueError(
                "The provided WCS does not seem to have a spectral axis"
            )

        celestial_wcs = data_wcs.celestial
        specex_wcs = data_wcs.spectral
    else:
        celestial_wcs = None
        specex_wcs = None

    cutout_data = []
    for k in range(data.shape[0]):
        cutout = get_gray_cutout(
            data=data[k],
            center=center,
            size=size,
            angle=angle,
            data_wcs=celestial_wcs
        )

        cutout_data.append(cutout['data'])
        if report_callback is not None:
            report_callback(k, data.shape[0])

    cutout_data = np.array(cutout_data)

    if celestial_wcs is not None:
        wcs_header = cutout['wcs'].to_header()
        wcs_header['CRPIX3'] = specex_wcs.wcs.crpix[0]
        wcs_header['PC3_3'] = specex_wcs.wcs.get_pc()[0, 0]
        wcs_header['PC1_3'] = 0
        wcs_header['PC2_3'] = 0
        wcs_header['PC3_2'] = 0
        wcs_header['PC3_1'] = 0
        wcs_header['CDELT3'] = specex_wcs.wcs.cdelt[0]
        wcs_header['CUNIT3'] = str(specex_wcs.wcs.cunit[0])
        wcs_header['CTYPE3'] = specex_wcs.wcs.ctype[0]
        wcs_header['CRVAL3'] = specex_wcs.wcs.crval[0]

    else:
        wcs_header = None

    return {
        'data': cutout_data,
        'wcs': WCS(wcs_header)
    }


def _get_fits_data_structure(fits_file):
    data_structure = {
        'type': None,
        'data-ext': [],
        'variance-ext': [],
        'mask-ext': []
    }
    with fits.open(fits_file) as f:
        # If there is only one extension, than it should contain the image data
        if len(f) == 1:
            data_ext = f[0]
            data_structure['data-ext'] = [0, ]
        else:
            # Otherwise, try to identify the extension form its name
            for k, ext in enumerate(f):
                if ext.name.lower() in KNOWN_SPEC_EXT_NAMES:
                    data_ext = ext
                    data_structure['data-ext'] = [k, ]
                    break

                if ext.name.lower() in KNOWN_RGB_EXT_NAMES:
                    data_ext = ext
                    data_structure['data-ext'].append(k)

            # If cannot determine which extensions cointain data,
            # then just use the second extension
            if data_structure['data-ext'] is None:
                data_ext = f[1]
                data_structure['data-ext'] = [1, ]

        data_shape = data_ext.shape
        if len(data_shape) == 2:
            # A 2D image, we should check other extensions to
            # determine if its an RGB multi-extension file
            for k, ext in enumerate(f):
                if k in data_structure['data-ext']:
                    continue

                lower_ext_name = ext.name.strip().lower()

                if ext.data is not None and ext.data.shape == data_shape:
                    if lower_ext_name:
                        if (
                                lower_ext_name in KNOWN_SPEC_EXT_NAMES or
                                lower_ext_name in KNOWN_RGB_EXT_NAMES
                        ):
                            data_structure['data-ext'].append(k)
                        elif lower_ext_name in KNOWN_VARIANCE_EXT_NAMES:
                            data_structure['variance-ext'].append(k)
                        elif lower_ext_name in KNOWN_MASK_EXT_NAMES:
                            data_structure['mask-ext'].append(k)
                        else:
                            continue
                    else:
                        data_structure['data-ext'].append(k)

            if len(data_structure['data-ext']) == 1:
                data_structure['type'] = 'image-gray'
            elif len(data_structure['data-ext']) == 3:
                data_structure['type'] = 'image-rgb'
            else:
                data_structure['type'] = 'unkown'

        elif len(data_shape) == 3:
            # Could be a datacube or an RGB cube or a weird grayscale image,
            # depending on the size of third axis. Only grayscale image will be
            # treated separately, while an RGB cube will be treated as a normal
            # datacube
            if data_shape[2] == 1:
                # A very weird grayscale image.
                data_structure['type'] = 'cube-gray'
            else:
                data_structure['type'] = 'cube'
                for k, ext in enumerate(f):
                    ext_name = ext.name.strip().lower()

                    if k in data_structure['data-ext']:
                        continue
                    elif ext_name in KNOWN_VARIANCE_EXT_NAMES:
                        data_structure['variance-ext'].append(k)
                    elif ext_name in KNOWN_MASK_EXT_NAMES:
                        data_structure['mask-ext'].append(k)

        else:
            # We dont know how to handle weird multidimensional data.
            print(
                "WARNING: cannot handle multidimensional data with shape "
                f"{data_shape}"
            )
            data_structure['type'] = 'unkown'

    if not data_structure['data-ext']:
        data_structure['data-ext'] = None
    if not data_structure['variance-ext']:
        data_structure['variance-ext'] = None
    if not data_structure['mask-ext']:
        data_structure['mask-ext'] = None

    return data_structure


def get_hdu(hdl, valid_names, hdu_index=-1, msg_err_notfound=None,
            msg_index_error=None, exit_on_errors=True):
    """
    Find a valid HDU in a HDUList.

    Parameters
    ----------
    hdl : list of astropy.io.fits HDUs
        A list of HDUs.
    valid_names : list or tuple of str
        A list of possible names for the valid HDU.
    hdu_index : int, optional
        Manually specify which HDU to use. The default is -1.
    msg_err_notfound : str or None, optional
        Error message to be displayed if no valid HDU is found.
        The default is None.
    msg_index_error : str or None, optional
        Error message to be displayed if the specified index is outside the
        HDU list boundaries.
        The default is None.
    exit_on_errors : bool, optional
        If it is set to True, then exit the main program with an error if a
        valid HDU is not found, otherwise just return None.
        The default value is True.

    Returns
    -------
    valid_hdu : astropy.io.fits HDU or None
        The requested HDU.

    """
    valid_hdu = None
    if hdu_index < 0:
        # Try to detect HDU containing spectral data
        for hdu in hdl:
            if hdu.name.lower() in valid_names:
                valid_hdu = hdu
                break
        else:
            if msg_err_notfound:
                print(msg_err_notfound, file=sys.stderr)
            if exit_on_errors:
                sys.exit(1)
    else:
        try:
            valid_hdu = hdl[hdu_index]
        except IndexError:
            if msg_index_error:
                print(msg_index_error.format(hdu_index), file=sys.stderr)
            if exit_on_errors:
                sys.exit(1)
    return valid_hdu


def cutout_main(options=None):
    """
    Run the main program.

    Parameters
    ----------
    options : list or None, optional
        A list of cli input prameters. The default is None.

    Returns
    -------
    None.

    """
    def simple_report_callback(k, total):
        pbar = get_pbar(k / total)
        sys.stderr.write(f"\r{pbar} {k} / {total} \r")
        sys.stderr.flush()

    def updated_wcs_cutout_header(orig_header, cutout_header):
        new_header = orig_header.copy()

        sys.stdout.flush()
        sys.stderr.flush()

        # Delete any existing CD card
        cd_elem_re = re.compile(r'CD[1-9]_[1-9]')
        for k in list(new_header.keys()):
            if cd_elem_re.fullmatch(str(k).strip()):
                new_header.remove(k, ignore_missing=True, remove_all=True)

        # Delete any existing PC card
        pc_elem_re = re.compile(r'PC[1-9]_[1-9]')
        for k in list(new_header.keys()):
            if pc_elem_re.fullmatch(str(k).strip()):
                new_header.remove(k, ignore_missing=True, remove_all=True)

        # Copy new PC cards into the new header
        for k in list(cutout_header.keys()):
            if pc_elem_re.fullmatch(str(k).strip()):
                new_header[k] = cutout_header[k]

        new_header['PC1_1'] = cutout_header['PC1_1']
        new_header['PC2_2'] = cutout_header['PC2_2']
        new_header['CDELT1'] = cutout_header['CDELT1']
        new_header['CDELT2'] = cutout_header['CDELT2']
        new_header['CRVAL1'] = cutout_header['CRVAL1']
        new_header['CRVAL2'] = cutout_header['CRVAL2']
        new_header['CRPIX1'] = cutout_header['CRPIX1']
        new_header['CRPIX2'] = cutout_header['CRPIX2']
        new_header['CUNIT1'] = cutout_header['CUNIT1']
        new_header['CUNIT2'] = cutout_header['CUNIT2']

        try:
            crval3 = units.Quantity(
                cutout_header['CRVAL3'],
                cutout_header['CUNIT3']
            )
            cutout_unit3 = units.Quantity(1, cutout_header['CUNIT3'])
        except KeyError:
            pass
        else:
            c_factor = cutout_unit3.to(orig_header['CUNIT3']).value
            new_header['PC3_3'] = cutout_header['PC3_3'] * c_factor
            new_header['PC1_3'] = 0
            new_header['PC2_3'] = 0
            new_header['PC3_1'] = 0
            new_header['PC3_2'] = 0
            new_header['CDELT3'] = cutout_header['CDELT3']
            new_header['CRVAL3'] = crval3.to(orig_header['CUNIT3']).value
            new_header['CUNIT3'] = orig_header['CUNIT3']

        return new_header

    args = __cutout_argshandler(options)

    if args.regionfile is not None:
        myt = parse_regionfile(args.regionfile)

    if args.verbose:
        print(myt)

    target_data_file = args.input_fits[0]

    fits_base_name = os.path.basename(target_data_file)
    fits_base_name = os.path.splitext(fits_base_name)[0]

    data_structure = _get_fits_data_structure(target_data_file)

    if args.verbose:
        print(
            "\n=== IMAGE INFO ===\n"
            f" Name: {fits_base_name}\n"
            f" Type: {data_structure['type']}\n"
            f" Data EXT: {data_structure['data-ext']}\n"
            f" Var EXT: {data_structure['variance-ext']}\n"
            f" DQ EXT: {data_structure['mask-ext']}\n",
            file=sys.stderr
        )
        report_callback = simple_report_callback
    else:
        report_callback = None

    with fits.open(target_data_file) as hdul:
        for j, cutout_info in enumerate(myt):
            cutout_name = f"cutout_{fits_base_name}_{j:04}.fits"
            cc_ra = cutout_info['RA'] * units.deg
            cc_dec = cutout_info['DEC'] * units.deg

            cutout_sizes = (
                2 * cutout_info['B_WORLD'] * units.arcsec,
                2 * cutout_info['A_WORLD'] * units.arcsec,
            )

            angle_world = cutout_info['THETA_WORLD'] * units.deg

            if data_structure['type'] == 'cube':

                flux_hdu = hdul[data_structure['data-ext'][0]]
                flux_data = flux_hdu.data
                flux_wcs = WCS(flux_hdu.header)
                center_sky_coord = SkyCoord(
                    cc_ra, cc_dec,
                    frame=wcs_to_celestial_frame(flux_wcs)
                )

                var_hdu = get_hdu(
                    hdul,
                    valid_names=KNOWN_VARIANCE_EXT_NAMES,
                    msg_err_notfound="WARNING: Cannot determine which "
                                     "HDU contains the variance data. ",
                    exit_on_errors=False
                )

                mask_hdu = get_hdu(
                    hdul,
                    valid_names=KNOWN_MASK_EXT_NAMES,
                    msg_err_notfound="WARNING: Cannot determine which "
                                     "HDU contains the mask data.",
                    exit_on_errors=False
                )

                if args.verbose:
                    print(
                        "\nComputing flux cutouts...",
                        file=sys.stderr
                    )
                flux_cutout = get_cube_cutout(
                    flux_data,
                    center=center_sky_coord,
                    size=cutout_sizes,
                    angle=angle_world,
                    data_wcs=flux_wcs,
                    report_callback=report_callback
                )

                # Convert specral axis to angtrom units
                flux_header = updated_wcs_cutout_header(
                    flux_hdu.header,
                    flux_cutout['wcs'].to_header()
                )

                cutout_hdul = [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(
                        data=flux_cutout['data'],
                        header=flux_header,
                        name=flux_hdu.name
                    ),
                ]

                if var_hdu is not None:
                    if args.verbose:
                        print(
                            "\nComputing variance cutouts...",
                            file=sys.stderr
                        )
                    var_wcs = WCS(var_hdu.header)
                    var_cutout = get_cube_cutout(
                        var_hdu.data,
                        center=center_sky_coord,
                        size=cutout_sizes,
                        angle=angle_world,
                        data_wcs=var_wcs,
                        report_callback=report_callback
                    )

                    var_header = updated_wcs_cutout_header(
                        var_hdu.header,
                        var_cutout['wcs'].to_header()
                    )

                    cutout_hdul.append(
                        fits.ImageHDU(
                            data=var_cutout['data'],
                            header=var_header,
                            name=var_hdu.name
                        ),
                    )

                if mask_hdu is not None:
                    if args.verbose:
                        print(
                            "\nComputing data mask cutouts...",
                            file=sys.stderr
                        )
                    mask_wcs = WCS(mask_hdu.header)
                    mask_cutout = get_cube_cutout(
                        mask_hdu.data,
                        center=center_sky_coord,
                        size=cutout_sizes,
                        angle=angle_world,
                        data_wcs=mask_wcs,
                        report_callback=report_callback
                    )

                    mask_header = updated_wcs_cutout_header(
                        mask_hdu.header,
                        mask_cutout['wcs'].to_header()
                    )

                    cutout_hdul.append(
                        fits.ImageHDU(
                            data=mask_cutout['data'],
                            header=mask_header,
                            name=mask_hdu.name
                        ),
                    )

                cutout_hdul = fits.HDUList(cutout_hdul)
                cutout_hdul.writeto(cutout_name, overwrite=True)
            elif data_structure['type'].endswith('-gray'):
                if data_structure['type'].startswith('image-'):
                    gray_data = hdul[data_structure['data-ext'][0]].data
                else:
                    gray_data = hdul[data_structure['data-ext'][0]].data
                    gray_data = gray_data[..., 0]

                gray_hdu = hdul[data_structure['data-ext'][0]]
                grey_wcs = WCS(gray_hdu.header)
                center_sky_coord = SkyCoord(
                    cc_ra, cc_dec,
                    frame=wcs_to_celestial_frame(grey_wcs)
                )

                cutout = get_gray_cutout(
                    gray_data,
                    center=center_sky_coord,
                    size=cutout_sizes,
                    angle=angle_world,
                    data_wcs=grey_wcs
                )

                gray_header = updated_wcs_cutout_header(
                    gray_hdu.header,
                    cutout['wcs'].to_header()
                )

                cutout_hdul = fits.HDUList([
                    fits.PrimaryHDU(
                        data=cutout['data'],
                        header=gray_header,
                    ),
                ])
                cutout_hdul.writeto(cutout_name, overwrite=True)
            elif data_structure['type'] == 'image-rgb':
                rgb_data = [hdul[k].data for k in data_structure['data-ext']]
                rgb_wcs = [
                    WCS(hdul[k].header) for k in data_structure['data-ext']
                ]
                center_sky_coord = SkyCoord(
                    cc_ra, cc_dec,
                    frame=wcs_to_celestial_frame(rgb_wcs[0])
                )
                cutout = get_rgb_cutout(
                    rgb_data,
                    center=center_sky_coord,
                    size=cutout_sizes,
                    data_wcs=rgb_wcs
                )

                header_r = cutout['wcs'][0].to_header()
                header_g = cutout['wcs'][1].to_header()
                header_b = cutout['wcs'][2].to_header()

                cutout_hdul = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(
                        data=cutout['data'][0],
                        header=header_r,
                        name='RED',
                    ),
                    fits.ImageHDU(
                        data=cutout['data'][1],
                        header=header_g,
                        name='GREEN',
                    ),
                    fits.ImageHDU(
                        data=cutout['data'][2],
                        header=header_b,
                        name='BLUE',
                    )
                ])
                cutout_hdul.writeto(cutout_name, overwrite=True)
            else:
                print(
                    f"WARNING: not implemente yet [{data_structure['type']}]!",
                    file=sys.stderr
                )


if __name__ == '__main__':
    cutout_main()
