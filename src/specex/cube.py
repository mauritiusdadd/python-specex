#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECEX - SPECtra EXtractor.

This module provides utility functions to handle spectroscopic datacubes.

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import os
import re
import sys
import shutil
import logging
import argparse
import warnings
from pathlib import Path

from typing import Optional, Any
from collections.abc import Sequence, Iterable, Callable

import numpy as np

from rich.progress import Progress

from astropy.units import Quantity

from astropy import units
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.convolution import (
    convolve, Gaussian1DKernel, Gaussian2DKernel,
    Box1DKernel, Box2DKernel, Tophat2DKernel,
)

from specex.utils import get_pc_transform_params, rotate_data
from specex.utils import HAS_REGION, parse_regionfile
from specex.exceptions import get_ipython_embedder

if HAS_REGION:
    import regions

KNOWN_SPEC_EXT_NAMES = ['spec', 'spectrum', 'flux', 'data', 'sci', 'science']
KNOWN_VARIANCE_EXT_NAMES = ['stat', 'stats', 'var', 'variance', 'noise', 'err']
KNOWN_INVAR_EXT_NAMES = ['ivar', 'ivariance']
KNOWN_MASK_EXT_NAMES = ['mask', 'platemask', 'footprint', 'dq', 'nan_mask']
KNOWN_WAVE_EXT_NAMES = ['wave', 'wavelenght', 'lambda', 'lam']
KNOWN_RCURVE_EXT_NAMES = ['r', 'reso', 'resolution', 'rcurve', 'wd']
KNOWN_RGB_EXT_NAMES = ['r', 'g', 'b', 'red', 'green', 'blue']


class SpectralCube:
    """Class to handle datacubes."""

    def __init__(self) -> None:
        self.filename: Path | None = None
        self.hdul: fits.HDUList | None = None
        self.spec_hdu: fits.ImageHDU | None = None
        self.var_hdu: fits.ImageHDU | None = None
        self.mask_hdu: fits.ImageHDU | None = None
        self.wd_hdu: fits.ImageHDU | None = None
        self.spec_wcs: WCS | None = None
        self.var_wcs: WCS | None = None
        self.mask_wcs: WCS | None = None
        self.wd_wcs: WCS | None = None

    def __del__(self) -> None:
        """
        Do cleanup when this object is deleted.

        Returns
        -------
        None.

        """
        self.close()

    def __enter__(self) -> SpectralCube:
        """
        Enter the context manager.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the contex manager.

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.close()

    def close(self) -> None:
        """
        Clean up on closing.

        Returns
        -------
        None.

        """
        if self.hdul is not None:
            self.hdul.close()

    def getSpecWCS(self) -> WCS | None:
        """
        Get the WCS of the spectrum datacube.

        Returns
        -------
        wcs
            The WCS of the spectral HDU.

        """
        if (self.spec_wcs is None) and (self.spec_hdu is not None):
            self.spec_wcs = WCS(self.spec_hdu.header)
        return self.spec_wcs

    def getVarWCS(self) -> WCS | None:
        """
        Get the WCS of the variance datacube.

        Returns
        -------
        wcs
            The WCS of the variance HDU.
        """
        if (self.var_wcs is None) and (self.var_hdu is not None):
            self.var_wcs = WCS(self.var_hdu.header)
            if not (self.var_wcs.has_celestial and self.var_wcs.has_spectral):
                return self.getSpecWCS()
        return self.var_wcs

    def getMaskWCS(self) -> WCS | None:
        """
        Get the WCS of the mask datacube.

        Returns
        -------
        wcs
            The WCS of the mask HDU.
        """
        if (self.mask_wcs is None) and (self.mask_hdu is not None):
            self.mask_wcs = WCS(self.mask_hdu.header)
            if not (self.mask_wcs.has_celestial and self.mask_wcs.has_spectral):
                return self.getSpecWCS()
        return self.mask_wcs

    def getWdWCS(self) -> WCS | None:
        """
        Get the WCS of the variance datacube.

        Returns
        -------
        wcs
            The WCS of the resolution curve HDU.
        """
        if (self.wd_wcs is None) and (self.wd_hdu is not None):
            self.wd_wcs = WCS(self.wd_hdu.header)
            if not (self.wd_wcs.has_celestial and self.wd_wcs.has_spectral):
                return self.getSpecWCS()
        return self.wd_wcs

    def getSpatialSizePixels(
        self
    ) -> list[None | int]:
        """
        Return spatial dimensions in pixels.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.spec_hdu is None:
            return [None, None]
        else:
            return self.spec_hdu.data.shape[1:]

    @classmethod
    def open(cls, filename: str | os.PathLike[str],
             spec_hdu_index: Optional[int | str] = None,
             var_hdu_index: Optional[int | str] = None,
             mask_hdu_index: Optional[int | str] = None,
             wd_hdu_index: Optional[int | str] = None,
             mode: Optional[str] = None) -> SpectralCube:
        """
        Open a datacube from a FITS file.

        Parameters
        ----------
        filename : str
            The file path.
        spec_hdu_index : Optional[int | str], optional
            The index or name of the HDU containing the spectrum.
            The default is None.
        var_hdu_index : Optional[int | str], optional
            The index or name of the HDU containing the variance.
            The default is None.
        mask_hdu_index : Optional[int | str], optional
            The index or name of the HDU containing the nans maks.
            The default is None.
        wd_hdu_index : Optional[int | str], optional
            The index or name of the HDU containing the R matrix.
            The default is None.
        mode : Optional[str], optional
            The mode passed to astropy.io.fits.open

        Returns
        -------
        SpectralCube.

        """
        new_cube = cls()
        new_cube.filename = Path(filename)
        open_kwargs = {}
        if mode is not None:
            open_kwargs['mode'] = mode
        new_cube.hdul = fits.open(filename, **open_kwargs)

        new_cube.spec_hdu = get_hdu(
            new_cube.hdul,
            hdu_index=spec_hdu_index,
            valid_names=KNOWN_SPEC_EXT_NAMES,
            msg_err_notfound=(
                "ERROR: Cannot determine which HDU contains spectral "
                "data, try to specify it manually!"
            ),
            msg_index_error="ERROR: Cannot open HDU {} to read specra!"
        )

        new_cube.var_hdu = get_hdu(
            new_cube.hdul,
            hdu_index=var_hdu_index,
            valid_names=KNOWN_VARIANCE_EXT_NAMES,
            msg_err_notfound=(
                "WARNING: Cannot determine which HDU contains the "
                "variance data, try to specify it manually!"
            ),
            msg_index_error="WARNING: Cannot open HDU {} to read the "
                            "variance!",
            exit_on_errors=False
        )

        new_cube.mask_hdu = get_hdu(
            new_cube.hdul,
            hdu_index=mask_hdu_index,
            valid_names=KNOWN_MASK_EXT_NAMES,
            msg_err_notfound=(
                "WARNING: Cannot determine which HDU contains the "
                "mask data, try to specify it manually!"
            ),
            msg_index_error="WARNING: Cannot open HDU {} to read the "
                            "mask!",
            exit_on_errors=False
        )

        new_cube.wd_hdu = get_hdu(
            new_cube.hdul,
            hdu_index=wd_hdu_index,
            valid_names=KNOWN_RCURVE_EXT_NAMES,
            msg_err_notfound=(
                "WARNING: Cannot determine which HDU contains the "
                "R data, try to specify it manually!"
            ),
            msg_index_error="WARNING: Cannot open HDU {} to read the "
                            "R matrix data!",
            exit_on_errors=False
        )
        return new_cube

    def write(
        self, filename: str,
        output_verify: str = "exception",
        overwrite: bool = False,
        checksum: bool = False
    ) -> None:
        """
        Write the datacube if a FITS file.

        Parameters
        ----------
        filename : str
            The path of the output file.
        kwargs : dict
            Optional arguments passed to astropy.io.fits.HDUList.writeto.
            For example, it can be {'overwrite': True}.

        Returns
        -------
        None

        """
        if self.hdul is None:
            return

        self.hdul.writeto(
            filename,
            output_verify=output_verify,
            overwrite=overwrite,
            checksum=checksum
        )


def __cutout_argshandler(
    options: Optional[Sequence[str]] = None
) -> argparse.Namespace:
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
        'input_fits', metavar='INPUT_FIST', type=str,
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
        'specified, then the quantity is assumed to be in pixels.'
    )
    parser.add_argument(
        '--size', '-s', metavar='WIDTH[,HEIGHT]', type=str, default=None,
        help='Specify the HEIGHT and WIDTH of a single cutout. If HEIGHT is '
        'not specified then it is assumed to be equal to WIDTH. '
        'Both HEIGHT and WIDTH can be specified with units in a format '
        'compatible with astropy.units (eg. -s 10arcsec,5arcsec).  If no no '
        'unit is specified, then the quantity is assumed to be in pixels.'
    )
    parser.add_argument(
        '--angle', '-t', metavar='ANGLE', type=str, default=0,
        help='Specify the rotation angle arounf the center of a single cutout.'
        '%(metavar)s can be specified with units in a format '
        'compatible with astropy.units (eg. -s 10arcsec,5arcsec).  If no no '
        'unit is specified, then the quantity is assumed to be in degrees.'
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
    args = parser.parse_args(options)
    return args


def __smoothing_args_handler(options: Optional[Sequence[str]]=None):
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
        description='Apply a gaussian smoothing kernel to a spectral cubes '
                    'spatially and/or along the spectral axis.'
    )

    parser.add_argument(
        'input_fits_files', metavar='INPUT_FIST', type=str, nargs='+',
        help='The spectral cube (or image) from which to extract a cutout.'
    )
    parser.add_argument(
        '--spatial-smooth', metavar='SIGMA', type=str, default=None,
        help='Set the size of the spatial gaussian kernel. If %(metavar)s '
        'is 0 then no spatial smoothing is applied. If not specified the '
        'default value %(metavar)s=0 is used.'
    )
    parser.add_argument(
        '--spatial-kernel', metavar="KERNEL", type=str, default="gaussian",
        help="Set the kernel to use for spatial smoothing. In can be: gaussian"
        ", top-hat, or box. The default value is %(default)s."
    )
    parser.add_argument(
        '--spatial-supersample', metavar='ZOOM_FACTOR', type=int, default=0,
        help='Set the spatial supersampling factor. If %(metavar)s <= 1 then '
        'no supersampling is applied. %(metavar)s=2 means that the output cube'
        ' will have a doubled width and height, and so on. The default value '
        'is %(metavar)s=%(default)d.'
    )
    parser.add_argument(
        '--wave-supersample', metavar='ZOOM_FACTOR', type=int, default=0,
        help='Set the wavelength supersampling factor. If %(metavar)s <= 1 '
        'then no supersampling is applied. %(metavar)s=2 means that the output'
        'cube will have a doubled spectral resolution, and so on. '
        'The default value is %(metavar)s=%(default)s.'
    )
    parser.add_argument(
        '--wave-smooth', metavar='SIGMA', type=str, default=None,
        help='Set the size of the spectral gaussian kernel. If %(metavar)s '
        'is 0 then no spectral smoothing is applied. If not specified, the '
        'default value %(metavar)s=0 is used.'
    )
    parser.add_argument(
        '--wave-kernel', metavar="KERNEL", type=str, default="gaussian",
        help="Set the kernel used for spectral smoothing. In can be: gaussian"
             " or box. The default value is %(default)s."
    )
    parser.add_argument(
        '--linemap', metavar='WINDOW_SIZE', type=int, default=None,
        help='Perform a running mean along the spectral axis, with a window '
        'of %(metavar)s pixels. If %(metavar)s is 0 then no the running mean '
        'is not applied. If not specified, the default value %(metavar)s=0 '
        'is used.'
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
        '--verbose', action='store_true', default=False,
        help="Print verbose outout."
    )

    parser.add_argument(
        '--debug', action='store_true', default=False,
        help="Start an IPython console for debugging."
    )

    args = parser.parse_args(options)

    for inp_fits_file in args.input_fits_files:
        if (inp_fits_file is not None) and (not os.path.isfile(inp_fits_file)):
            print(f"The file {inp_fits_file} does not exist!")
            sys.exit(1)

    if (
            (args.spatial_smooth is None) and
            (args.wave_smooth is None) and
            (args.linemap  is None)
    ):
        print(
            "Spatial smoothing and spectral smoothing cannot be both disabled,"
            " please set at least one the options --wave-sigma or --linemap or"
            " --spatial-sigma."
        )
        sys.exit(1)

    return args


def get_gray_cutout(
    data: np.ndarray,
    center: SkyCoord | Iterable,
    size: tuple[float | int, float | int] | list[float | int] | units.Quantity,
    angle: float | int | units.Quantity = 0,
    data_wcs: Optional[WCS] = None
) -> dict[str, WCS | np.ndarray]:
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


def get_rgb_cutout(
    data: tuple | list | np.ndarray,
    center: SkyCoord | tuple,
    size: tuple[float | int, float | int] | list[float | int] | units.Quantity,
    angle: float| int | units.Quantity = 0,
    data_wcs: Optional[WCS] = None,
    resample_to_wcs: bool = False
) -> dict[
    str,
    tuple[WCS, WCS, WCS] | tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Get a cutout from a bigger RGB.

    Parameters
    ----------
    data : np.ndarray or tuple or list
        The actual image data.
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
    angle : float or astropy.units.Quantity, optional
        The rotation angle of the cutout. If it is a float, then it is
        interpreted in degrees. The default is 0.
    data_wcs : astropy.wcs.WCS or None, optional
        A WCS associated with the image data. The default is None.
    resample_to_wcs : bool, optional
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
    elif isinstance(data, tuple | list):
        if len(data) != 3:
            raise ValueError(
                "'data' parameter only accepts list or tuple containing "
                "exactly 3 elements."
            )
        elif not all([isinstance(x, np.ndarray) for x in data]):
            raise ValueError(
                "All elements of the input tupel or list must be 2D arrays."
            )

        data_wcs_r = None
        data_wcs_g = None
        data_wcs_b = None

        if data_wcs is None:
            if resample_to_wcs:
                warnings.warn(
                    "reample_to_wcs is set to True but no WCS info is provided"
                )
            resample_to_wcs = False
        else:
            if not isinstance(data, tuple | list):
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

    cutout_data_r = get_gray_cutout(data_r, center, size, angle, data_wcs_r)
    cutout_data_g = get_gray_cutout(data_g, center, size, angle, data_wcs_g)
    cutout_data_b = get_gray_cutout(data_b, center, size, angle, data_wcs_b)

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


def get_cube_cutout(
    data: np.ndarray,
    center: SkyCoord | tuple,
    size: tuple[float | int, float | int] | list[float | int] | units.Quantity,
    angle: float | int | units.Quantity = 0,
    wave_range: Optional[tuple | list] = None,
    data_wcs: Optional[WCS] = None,
    progress: Optional[Progress] = None
) -> dict[str, WCS | np.ndarray]:
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
        Angular values are converted to pixel values ignoring any non-linear
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

    Returns
    -------
    cutout_dict
        A dictionary containing the following key: value pairs:
            'data': np.ndarray
                The cutout data.
            'wcs': astropy.wcs.WCS or None
                The wcs for the cutout data.
    """
    # Do some sanity checks on the input data
    if len(data.shape) != 3:
        raise ValueError("Unsupported datacube shape {data.shape}.")

    if not isinstance(size, list | tuple):
        raise ValueError(
            "'size' must be a list or a tuple of scalar values or angular "
            "quantities"
        )
    elif not all(
        [isinstance(x, float | int | units.Quantity) for x in size]
    ):
        raise ValueError(
            "'size' must be a list or a tuple of scalar values or angular "
            "quantities"
        )

    d_a, d_b = size[:2]

    if not isinstance(center, SkyCoord | tuple | list):
        raise ValueError("'center' must be SkyCoord or tuple or list.")

    angle = units.Quantity(angle, units.deg)

    if data_wcs is not None:
        if not isinstance(data_wcs, WCS):
            raise ValueError(
                "'data_wcs' must be either None or a valid WCS object"
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

    cutout_data_list: list[np.ndarray] = []

    if progress is not None:
        task_id = progress.add_task(
            description="Making cutout:",
            total=data.shape[0]
        )

    for k in range(data.shape[0]):
        cutout = get_gray_cutout(
            data=data[k],
            center=center,
            size=size,
            angle=angle,
            data_wcs=celestial_wcs
        )
        if progress is not None:
            progress.update(task_id, completed=k)

        cutout_data_list.append(cutout['data'])

    if progress is not None:
        progress.update(task_id, completed=data.shape[0])
        progress.remove_task(task_id)

    cutout_data = np.array(cutout_data_list)

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


def _get_fits_data_structure(
    fits_file: str | os.PathLike[str]
) -> dict[str, Any]:
    data_structure: dict[str, Any] = {
        'type': None,
        'data-ext': [],
        'variance-ext': [],
        'mask-ext': []
    }

    with fits.open(fits_file) as f:
        # If there is only one extension, then it should contain the image data
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

            # If cannot determine which extensions contain data,
            # then just use the second extension
            if not data_structure['data-ext']:
                data_ext = f[1]
                data_structure['data-ext'] = [1, ]

        data_shape = data_ext.data.shape
        if len(data_shape) == 2:
            # A 2D image, we should check other extensions to
            # determine if it's an RGB multi-extension file
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


def get_hdu(hdl, valid_names, hdu_index=None, msg_err_notfound=None,
            msg_index_error=None, exit_on_errors=True):
    """
    Find a valid HDU in a HDUList.

    Parameters
    ----------
    hdl : list of astropy.io.fits HDUs
        A list of HDUs.
    valid_names : list or tuple of str
        A list of possible names for the valid HDU.
    hdu_index : int or str, optional
        Manually specify which HDU to use. The default is None.
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
    if hdu_index is None or (isinstance(hdu_index, int) and hdu_index < 0):
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


def cube_tiled_func(
    func: Callable,
    tile_size: int,
    data: np.ndarray,
    weights: Optional[int | float | np.ndarray] = None,
    flat_result: bool = False,
    description: Optional[str] = None,
    progress: Optional[Progress] = None,
    *args, **kwargs
):
    if flat_result:
        data_shape = data.shape[-2:]
    else:
        data_shape = data.shape

    if isinstance(data, np.ma.MaskedArray):
        result = np.ma.zeros(data_shape, dtype=data.dtype)
    else:
        result = np.zeros(data_shape, dtype=data.dtype)

    if progress is not None:
        task_id = progress.add_task(
            description=description if description else "",
            total=data_shape[0]*data_shape[1]
        )

    for j in np.arange(data_shape[0], step=tile_size):
        for k in np.arange(data_shape[1], step=tile_size):
            tile = data[..., j:j + tile_size, k:k + tile_size]

            # skip empty tiles
            if tile.size == 0:
                continue

            if weights is None:
                tile_w: int | float | np.ndarray | None = None
            elif isinstance(weights, int | float):
                tile_w = np.ones_like(tile) * weights
            else:
                tile_w = weights[..., j:j + tile_size, k:k + tile_size]

            if progress is not None:
                progress.update(task_id, completed=j*data_shape[0] + k)

            if not np.isfinite(tile).any():
                result[j:j+tile_size, k:k+tile_size] = np.nan
                try:
                    result[j:j+tile_size, k:k+tile_size].mask = True
                except AttributeError:
                    pass
                continue

            processed_tile = func(tile, tile_w, *args, **kwargs).copy()

            if flat_result:
                result[j:j + tile_size, k:k + tile_size] = processed_tile
                try:
                    result[
                        j:j + tile_size, k:k + tile_size
                    ].mask = processed_tile.mask
                except AttributeError:
                    pass
            else:
                result[..., j:j + tile_size, k:k + tile_size] = processed_tile
                try:
                    result[
                        ..., j:j + tile_size, k:k + tile_size
                    ].mask = processed_tile.mask
                except AttributeError:
                    pass

    print("DONE!!!")
    if progress is not None:
        progress.update(task_id, completed=data_shape[0]*data_shape[1])
        progress.remove_task(task_id)

    return result


def correlate_spaxel(cube_data: np.ndarray,
                     spaxel_data: np.ndarray,
                     similarity_function: Optional[str] = 'rms'):

    if spaxel_data.shape[0] != cube_data.shape[0]:
        raise ValueError(
            "spaxel_data and cube_data must have the same first dimension."
        )

    if len(spaxel_data.shape) == 1:
        spaxel_data = spaxel_data[:, None, None]

    x = cube_data - np.nanmedian(cube_data, axis=0)
    x = x / np.nanmax(x, axis=0)
    y = spaxel_data - np.nanmedian(spaxel_data)
    y = y / np.nanmax(y)

    if similarity_function == 'rms':
        res = np.sqrt(np.nanmean((x - y)**2, axis=0))
        return 1/(1 + res)
    elif similarity_function == 'correlation':
        res = np.nansum(x * y, axis=0)
        return res / (np.nansum(x**2, axis=0) * np.nansum(y**2))

def get_line_map(
        data: np.ndarray,
        window_size_pix: int = 5,
        variance: Optional[np.ndarray] = None,
        data_mask: Optional[np.ndarray] = None,
        tile_size: int = 32,
        progress: Optional[Progress] = None,
) -> np.ndarray:
    def _worker_func(x: np.ndarray, w: Optional[np.ndarray] = None):
        if w is None:
            w = np.ones_like(x)

        return np.array([
            np.nanmean(
                x[j:j+window_size_pix] * w[j:j+window_size_pix],
                axis=0
            ) / np.nanmean(w[j:j+window_size_pix], axis=0)
            for j in range(x.shape[0])
        ])

    weights = None
    if variance is not None:
        valid_var = (variance > 0) & np.isfinite(variance)
        weights = np.zeros_like(variance, dtype=float)
        weights[valid_var] = 1.0 / variance[valid_var]

    line_map = cube_tiled_func(
        _worker_func,
        tile_size,
        data,
        weights,
        flat_result=False,
        description="Building line map",
        progress=progress
    )

    return line_map


def get_continuum_subtracted_slice(
        data: np.ndarray,
        line_wave: int | units.Quantity,
        line_window: int | units.Quantity = 10 * units.angstrom,
        continuum_window: int | units.Quantity = 10 * units.angstrom,
        variance: Optional[np.ndarray] = None,
        data_mask: Optional[np.ndarray] = None,
        cube_wcs: Optional[WCS] = None
) -> np.ndarray:
    """
    Get a continuum subtracted image from a spectral datacube.

    Parameters
    ----------
    data : np.ndarray
        DESCRIPTION.
    line_wave : int | units.Quantity
        DESCRIPTION.
    line_window : int | units.Quantity, optional
        DESCRIPTION. The default is 10 * units.angstrom.
    continuum_window : int | units.Quantity, optional
        DESCRIPTION. The default is 10 * units.angstrom.
    variance : Optional[np.ndarray], optional
        DESCRIPTION. The default is None.
    data_mask : Optional[np.ndarray], optional
        DESCRIPTION. The default is None.
    cube_wcs : WCS, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    param_is_dimensional = [
        isinstance(x, units.Quantity)
        for x in (line_wave, line_window, continuum_window)
    ]

    if any(param_is_dimensional):
        if not all(param_is_dimensional):
            raise ValueError(
                "central_wave, line_window and continuum_window must be all "
                "integer indices or all dimensional quantities."
            )
        elif cube_wcs is None:
            raise ValueError(
                "A valid WCS object must be provided when central_wave, "
                "line_window and continuum_window are dimensional quantities."
            )
        line_wave_pix = cube_wcs.spectral.world_to_pixel(line_wave)

        line_window_pix_low = int(np.round(cube_wcs.spectral.world_to_pixel(
            line_wave - line_window
        )))

        line_window_pix_high = int(np.round(cube_wcs.spectral.world_to_pixel(
            line_wave + line_window
        )))

    else:
        line_wave_pix = line_wave
        line_window_pix_low = int(line_wave_pix - line_window)
        line_window_pix_high = int(line_wave_pix + line_window)

    idx_min = max(0, min(line_window_pix_low, line_window_pix_high))
    idx_max = min(data.shape[0], max(line_window_pix_low, line_window_pix_high))
    line_slice = np.nanmean(data[idx_min:idx_max, ...], axis=0)
    return line_slice


def self_correlate(
    data: np.ndarray,
    data_mask: Optional[np.ndarray] = None,
    similarity_sigma_threshold: Optional[float] = 5,
    tile_size: int = 32,
    block_size: int = 2,
    similarity_function: Optional[str] = 'rms',
    progress: Optional[Progress] = None
) -> np.ndarray:
    if data_mask is not None and data.shape != data_mask.shape:
        raise ValueError("data and data_mask must have the same shape!")

    hei, wid = data.shape[1:]

    sim_table = np.zeros((hei, wid))
    # For each spaxel in the cube
    block_id = 0
    total_blocks = int(wid*hei / (block_size**2))-1
    if progress is not None:
        task_id = progress.add_task(
            description="Correlating spaxels:",
            total=total_blocks
        )

    for h in np.arange(hei, step=block_size):
        for k in np.arange(wid, step=block_size):
            if progress is not None:
                progress.update(task_id, completed=block_id)

            block_id += 1
            if (
                (sim_table[h:h+block_size, k:k+block_size] != 0).any() or
                (
                    data_mask is not None and
                    (data_mask[:, h:h+block_size, k:k+block_size] != 0).any()
                )
            ):
                continue

            spaxel_data = np.nansum(
                data[:, h:h+block_size, k:k+block_size],
                axis=(1, 2)
            )

            if not np.isfinite(spaxel_data).any():
                continue

            similarity_map = cube_tiled_func(
                correlate_spaxel,
                tile_size,
                data,
                spaxel_data=spaxel_data,
                similarity_function=similarity_function
            )

            thresh = np.nanmean(similarity_map)
            thresh += similarity_sigma_threshold * np.nanstd(similarity_map)
            similarity_mask = similarity_map >= thresh
            sim_table[similarity_mask] = block_id

    if progress is not None:
        progress.update(task_id, completed=total_blocks)
        progress.remove_task(task_id)
    return sim_table


def smooth_cube(
    data: np.ndarray, data_mask: Optional[np.ndarray] = None,
    spatial_size: float | int | Sequence[int | float  | units.Quantity] | np.ndarray | units.Quantity = 1.0,
    wave_size: float | int | units.Quantity = 0.0,
    cube_wcs : Optional[WCS] = None,
    progress: Optional[Progress] = None,
    spatial_kernel: str = "gaussian",
    wave_kernel: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Smooth a datacube spatially and/or along the spectral axis.

    Parameters
    ----------
    data : numpy.ndarray
        The spectral datacube.
    data_mask : numpy.ndarray, optional
        The mask for the spectral datacube. The default is None.
    spatial_size : float, optional
        The sigma for the spatial smoothing gaussian kernel.
        The default is 1.0.
    wave_size : float, optional
        The sigma fot the spectral smoothing gaussian kernel.
        The default is 0.0.
    cube_wcs : WCS, optional
        The WCS of the cube
    progress : Callable or None, optional
        A callable that will be executed every time the cutout of a single
        slice of the cube is computed. Must accept in input two arguments:

          * the number of slice processed so far
          * the total number of slices.

    Raises
    ------
    ValueError
        If the shape of data does not match the shape of data_mask.

    Returns
    -------
    smoothed_arr : numpy.ndarray
        The smoothed version of the input data.

    """
    if data_mask is not None and data.shape != data_mask.shape:
        raise ValueError("data and data_mask must have the same shape!")

    smoothed_arr = data.copy()

    if not isinstance(wave_size, float | int):
        if cube_wcs is None:
            raise ValueError(
                "cube_wcs is needed when using dimensional quantities "
                "for wave_sigma"
            )
        elif not cube_wcs.has_spectral:
            raise ValueError(
                "Datacube WCS has no spectral axis!"
            )
        else:
            # Determine the equivalent sigma in pixel space
            p0 = cube_wcs.spectral.pixel_to_world(0).to(units.angstrom)
            wave_size_pix: float | int = cube_wcs.spectral.world_to_pixel(
                p0 + wave_size
            )
    else:
        wave_size_pix = wave_size

    do_wave_smoohting = wave_size_pix > 0

    spatial_size_pix: list[int | float] = [0, 0]
    if isinstance(spatial_size, float | int):
        spatial_size_pix = [spatial_size, spatial_size]
    elif isinstance(spatial_size, Quantity):
        if cube_wcs is None:
            raise ValueError(
                "cube_wcs is needed when using dimensional quantities "
                "for spatial_size"
            )
        elif not cube_wcs.has_celestial:
            raise ValueError(
                "Datacube WCS has no celestial axes!"
            )
        else:
            if spatial_size.ndim == 0:
                spatial_size = Quantity([spatial_size, spatial_size])

            # Determine the equivalent sigma in pixel space
            p0 = cube_wcs.celestial.pixel_to_world(0, 0)
            d_ra = spatial_size[0]
            d_dec = spatial_size[1]
            p1 = p0.spherical_offsets_by(d_ra, d_dec)
            spatial_size_pix = cube_wcs.celestial.world_to_pixel(p1)
    else:
        # Just check if we have mixed quantities and numbers
        if any([isinstance(x, units.Quantity) for x in spatial_size]):
            if cube_wcs is None:
                raise ValueError(
                    "cube_wcs is needed when using dimensional quantities "
                    "for spatial_size"
                )
            elif not cube_wcs.has_celestial:
                raise ValueError(
                    "Datacube WCS has no celestial axes!"
                )
            else:
                # Determine the equivalent sigma in pixel space
                p0 = cube_wcs.celestial.pixel_to_world(0, 0)
                if isinstance(spatial_size[0], Quantity):
                    d_ra = spatial_size[0]
                else:
                    d_ra = 0
                    spatial_size_pix[0] = spatial_size[0]

                if isinstance(spatial_size[1], Quantity):
                    d_dec = spatial_size[1]
                else:
                    d_dec = 0
                    spatial_size_pix[1] = spatial_size[1]

                p1 = p0.spherical_offsets_by(d_ra, d_dec)
                d_pix = cube_wcs.celestial.world_to_pixel(p1)

                if isinstance(spatial_size[0], Quantity):
                    spatial_size_pix[0] = d_pix[0]

                if isinstance(spatial_size[1], Quantity):
                    spatial_size_pix[1] = d_pix[1]
        else:
            spatial_size_pix = [float(x) for x in spatial_size]
            if len(spatial_size_pix) == 1:
                spatial_size_pix = [
                    spatial_size_pix[0], spatial_size_pix[0]
                ]

    do_spatial_smoothing = spatial_size_pix[0] > 0
    do_spatial_smoothing |= spatial_size_pix[1] > 0

    if do_wave_smoohting:
        if progress is not None:
            task_id = progress.add_task(
                description="Smoothing along spectral axis:",
                total=smoothed_arr.shape[1]*smoothed_arr.shape[2]
            )

        if wave_kernel == "gaussian":
            kernel = Gaussian1DKernel(wave_size_pix)
        elif wave_kernel == "box":
            kernel = Box1DKernel(wave_size_pix)

        for h in range(smoothed_arr.shape[1]):
            for k in range(smoothed_arr.shape[2]):
                if progress is not None:
                    progress.update(task_id, advance=1)
                smoothed_spaxel = convolve(
                    smoothed_arr[:, h, k],
                    kernel=kernel,
                    nan_treatment='interpolate',
                    normalize_kernel=True
                )
                smoothed_arr[:, h, k] = smoothed_spaxel

        if progress is not None:
            progress.update(
                task_id, completed=smoothed_arr.shape[1]*smoothed_arr.shape[2]
            )
            progress.remove_task(task_id)

    if do_spatial_smoothing:
        if progress is not None:
            task_id = progress.add_task(
                description="Smoothing along spatial coordinates:",
                total=smoothed_arr.shape[0]
            )

        if spatial_kernel == "gaussian":
            kernel = Gaussian2DKernel(spatial_size_pix[0], spatial_size_pix[1])
        elif spatial_kernel == "box":
            kernel = Box2DKernel(np.max(spatial_size_pix))
        elif spatial_kernel == "top-hat":
            kernel = Tophat2DKernel(np.max(spatial_size_pix))

        for k, data_slice in enumerate(smoothed_arr):
            if progress is not None:
                progress.update(task_id, advance=1)

            smoothed_slice = convolve(
                data_slice, kernel=kernel,
                nan_treatment='interpolate',
                normalize_kernel=True
            )
            smoothed_arr[k] = smoothed_slice

        if progress is not None:
            progress.update(task_id, completed=smoothed_arr.shape[0])
            progress.remove_task(task_id)

    if data_mask is not None:
        smoothed_mask = data_mask.copy().astype(bool)
        if progress is not None:
            task_id = progress.add_task(
                description="Computing data mask:",
                total=smoothed_arr.shape[0]
            )

        for k, mask_slice in enumerate(smoothed_mask):
            if progress is not None:
                progress.update(task_id, advance=1)
            smoothed_mask[k] &= ~np.isfinite(smoothed_arr[k]) | mask_slice

        if progress is not None:
            progress.update(task_id, completed=smoothed_arr.shape[0])
            progress.remove_task(task_id)

        smoothed_mask = smoothed_mask.astype('int8')
    else:
        smoothed_mask = None

    return smoothed_arr, smoothed_mask


def smoothing_main(options: Optional[Sequence[str]] = None) -> None:
    """
    Run the main cutout program.

    Parameters
    ----------
    options : list or None, optional
        A list of cli input prameters. The default is None.

    Returns
    -------
    None.

    """
    args = __smoothing_args_handler(options)

    if args.wave_smooth is None:
        wave_smooth = 0
    else:
        try:
            wave_smooth = float(args.wave_smooth)
        except ValueError:
            wave_smooth = units.Quantity(args.wave_smooth)

    if args.spatial_smooth is not None:
        spatial_smooth_list = args.spatial_smooth.split(',')
        spatial_smooth: list[int | float | units.Quantity] | units.Quantity = []
        for x in spatial_smooth_list:
            try:
                value = float(x)
            except ValueError:
                value = units.Quantity(x)
            spatial_smooth.append(value)

        if all([isinstance(x, units.Quantity) for x in spatial_smooth]):
            spatial_smooth = units.Quantity(spatial_smooth)
    else:
        spatial_smooth = 0

    for target_data_file in args.input_fits_files:
        target_file_path: Path = Path(target_data_file)
        out_fname = f"{target_file_path.stem}_smoothed.fits"

        shutil.copy(target_data_file, out_fname)

        with SpectralCube.open(
            target_data_file, mode='readonly'
        ) as my_cube, Progress() as progress:

            if args.verbose:
                progress.print(f"\n[{target_file_path.stem}]")

            data_mask = None
            if my_cube.mask_hdu is not None:
                data_mask = my_cube.mask_hdu.data

            if args.verbose:
                progress.print(">>> applying smoothing...")
                progress.print(f"  - spatial_size: {spatial_smooth}")
                progress.print(f"  - wave_sigma: {wave_smooth}")

            if my_cube.spec_hdu is None:
                logging.error("No valid spectral HDU!")
                sys.exit(1)

            smoothed_spec, smoothed_mask = smooth_cube(
                data=my_cube.spec_hdu.data,
                data_mask=data_mask,
                spatial_size=spatial_smooth,
                wave_size=wave_smooth,
                spatial_kernel=args.spatial_kernel,
                wave_kernel=args.wave_kernel,
                cube_wcs=my_cube.getSpecWCS(),
                progress=progress
            )
            my_cube.spec_hdu.data = smoothed_spec

            variance: np.ndarray | None = None
            if my_cube.var_hdu is not None:
                variance = my_cube.var_hdu.data
                smoothed_var, _ = smooth_cube(
                    data=variance,
                    spatial_size=spatial_smooth,
                    wave_size=wave_smooth,
                    spatial_kernel=args.spatial_kernel,
                    wave_kernel=args.wave_kernel,
                    cube_wcs=my_cube.getVarWCS(),
                    progress=progress
                )
                my_cube.var_hdu.data = smoothed_var

            if my_cube.mask_hdu is not None:
                my_cube.mask_hdu.data = smoothed_mask

            if  (args.linemap is not None) and (args.linemap > 0):
                line_map = get_line_map(
                    my_cube.spec_hdu.data,
                    window_size_pix=int(args.linemap),
                    variance=variance,
                    progress=progress
                )
                my_cube.spec_hdu.data = line_map

            if args.debug:
                embedder = get_ipython_embedder()
                if embedder is not None:
                    embedder()

            if args.verbose:
                progress.print(f"  - saving to {out_fname}...")

            my_cube.write(out_fname, overwrite=True)


def cutout_main(options: Optional[Sequence[str]] = None) -> None:
    """
    Run the main cutout program.

    Parameters
    ----------
    options : list or None, optional
        A list of cli input prameters. The default is None.

    Returns
    -------
    None.

    """
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

    inp_fits_file: Path = Path(args.input_fits)
    if not inp_fits_file.is_file(follow_symlinks=True):
        print(f"The file {inp_fits_file} does not exist!")
        sys.exit(1)

    regionfile: Path | None = None
    if args.regionfile is None:
        if (args.center is None) or (args.size is None):
            print(
                "If --regionfile is not specified then both --center and "
                "--size must be provided."
            )
            sys.exit(1)
    else:
        regionfile = Path(args.regionfile)
        if not regionfile.is_file(follow_symlinks=True):
            print("The file input regionfile does not exist!")
            sys.exit(1)

    cutout_list = []
    if regionfile is not None:
        if HAS_REGION:
            myt = parse_regionfile(
                regionfile,  key_ra='RA', key_dec='DEC'
            )
        else:
            logging.error("Astropy Regions is needed to handle region files!")
            sys.exit(1)

        if myt is None:
            logging.error(
                "The input region file does not contain any valid sky region!"
            )
            sys.exit(1)

        for row in myt:
            reg = row['region']

            if isinstance(reg, regions.CircleSkyRegion):
                center = reg.center
                sizes = [reg.radius*2, reg.radius*2]
                angle = 0
            elif (
                isinstance(reg, regions.EllipseSkyRegion) or
                isinstance(reg, regions.RectangleSkyRegion)
            ):
                center = reg.center
                sizes = [reg.height, reg.width]
                angle = reg.angle
            elif isinstance(reg, regions.EllipseAnnulusSkyRegion):
                center = reg.center
                sizes = [reg.outer_height, reg.outer_width]
                angle = reg.angle
            elif isinstance(reg, regions.PolygonSkyRegion):
                center = SkyCoord(
                    row['RA'],
                    row['DEC'],
                    unit='deg',
                    frame=reg.vertices[0].frame
                )
                ra_list = [x.ra.to('deg').value for x in reg.vertices]
                dec_list = [x.dec.to('deg').value for x in reg.vertices]

                width = (np.max(ra_list) - np.min(ra_list)) * units.deg
                height = (np.max(dec_list) - np.min(dec_list)) * units.deg
                sizes = [height, width]
                angle = 0 * units.deg

            cutout_list.append((center, sizes, angle))
    else:
        ra_dec = args.center.split(',')

        try:
            ra = float(ra_dec[0])
            dec = float(ra_dec[1])
            center = (ra, dec)
        except ValueError:
            ra = units.Quantity(ra_dec[0])
            dec = units.Quantity(ra_dec[1])
            center = SkyCoord(ra, dec)

        size_list = []
        for size_j in args.size.split(','):
            try:
                size = float(size_j)
            except ValueError:
                size = units.Quantity(size_j)
            size_list.append(size)
        if len(size_list) == 1:
            sizes = (size_list[0], size_list[0])
        else:
            sizes = (size_list[1], size_list[0])

        try:
            angle = float(args.angle) * units.deg
        except ValueError:
            angle = units.Quantity(args.angle)

        cutout_list.append((center, sizes, angle))

    if args.verbose:
        for j, (center, sizes, angle) in enumerate(cutout_list):
            print(f'  - Cutout {j}: {center} {sizes} {angle}')

    data_structure = _get_fits_data_structure(inp_fits_file)

    if args.verbose:
        print(
            "\n=== IMAGE INFO ===\n"
            f" Name: {inp_fits_file.stem}\n"
            f" Type: {data_structure['type']}\n"
            f" Data EXT: {data_structure['data-ext']}\n"
            f" Var EXT: {data_structure['variance-ext']}\n"
            f" DQ EXT: {data_structure['mask-ext']}\n",
            file=sys.stderr
        )

    with fits.open(inp_fits_file) as hdul, Progress() as progress:
        for j, (cut_center, cut_size, cut_angle) in enumerate(cutout_list):
            cutout_name = f"cutout_{inp_fits_file.stem}_{j:04}.fits"

            if data_structure['type'] == 'cube':
                with SpectralCube.open(
                    inp_fits_file,
                    spec_hdu_index=data_structure['data-ext'][0],
                    mode='readonly'
                ) as my_cube:

                    if args.verbose:
                        print(
                            "\nComputing flux cutouts...",
                            file=sys.stderr
                        )

                    if my_cube.spec_hdu is None:
                        logging.error("No spectral HDU!")
                        sys.exit(1)

                    flux_cutout = get_cube_cutout(
                        my_cube.spec_hdu.data,
                        center=cut_center,
                        size=cut_size,
                        angle=cut_angle,
                        data_wcs=my_cube.getSpecWCS(),
                        progress=progress
                    )

                    # Convert specral axis to angtrom units
                    flux_header = updated_wcs_cutout_header(
                        my_cube.spec_hdu.header,
                        flux_cutout['wcs'].to_header()
                    )

                    cutout_hdul = [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(
                            data=flux_cutout['data'],
                            header=flux_header,
                            name=my_cube.spec_hdu.name
                        ),
                    ]

                    if my_cube.var_hdu is not None:
                        if args.verbose:
                            print(
                                "\nComputing variance cutouts...",
                                file=sys.stderr
                            )

                        var_cutout = get_cube_cutout(
                            my_cube.var_hdu.data,
                            center=cut_center,
                            size=cut_size,
                            angle=cut_angle,
                            data_wcs=my_cube.getVarWCS(),
                            progress=progress
                        )

                        var_header = updated_wcs_cutout_header(
                            my_cube.var_hdu.header,
                            var_cutout['wcs'].to_header()
                        )

                        cutout_hdul.append(
                            fits.ImageHDU(
                                data=var_cutout['data'],
                                header=var_header,
                                name=my_cube.var_hdu.name
                            ),
                        )

                    if my_cube.mask_hdu is not None:
                        if args.verbose:
                            print(
                                "\nComputing data mask cutouts...",
                                file=sys.stderr
                            )

                        mask_cutout = get_cube_cutout(
                            my_cube.mask_hdu.data,
                            center=cut_center,
                            size=cut_size,
                            angle=cut_angle,
                            data_wcs=my_cube.getMaskWCS(),
                            progress=progress
                        )

                        mask_header = updated_wcs_cutout_header(
                            my_cube.mask_hdu.header,
                            mask_cutout['wcs'].to_header()
                        )

                        cutout_hdul.append(
                            fits.ImageHDU(
                                data=mask_cutout['data'],
                                header=mask_header,
                                name=my_cube.mask_hdu.name
                            ),
                        )

                    if my_cube.wd_hdu is not None:
                        if args.verbose:
                            print(
                                "\nComputing data R matrix cutouts...",
                                file=sys.stderr
                            )

                        wd_cutout = get_cube_cutout(
                            my_cube.wd_hdu.data,
                            center=cut_center,
                            size=cut_size,
                            angle=cut_angle,
                            data_wcs=my_cube.getWdWCS(),
                            progress=progress
                        )

                        wd_header = updated_wcs_cutout_header(
                            my_cube.wd_hdu.header,
                            wd_cutout['wcs'].to_header()
                        )

                        cutout_hdul.append(
                            fits.ImageHDU(
                                data=wd_cutout['data'],
                                header=wd_header,
                                name=my_cube.wd_hdu.name
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

                cutout = get_gray_cutout(
                    gray_data,
                    center=cut_center,
                    size=cut_size,
                    angle=cut_angle,
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

                cutout = get_rgb_cutout(
                    rgb_data,
                    center=cut_center,
                    size=cut_size,
                    angle=cut_angle,
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
                    f"WARNING: not implemented [{data_structure['type']}]!",
                    file=sys.stderr
                )


if __name__ == '__main__':
    smoothing_main()
