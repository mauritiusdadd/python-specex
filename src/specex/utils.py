#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECEX - SPECtra EXtractor.

This module provides utility functions used by other specex modules.

Copyright (C) 2022-2023  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import tarfile
import logging
import platform
import subprocess
from typing import Optional
from urllib import request

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches
import matplotlib.patheffects as PathEffects

from scipy.signal import savgol_filter
from scipy.ndimage import rotate
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy import wcs as apwcs
from astropy import units as apu
from astropy import coordinates
from astropy import constants
from astropy.table import Table

from .lines import get_lines

try:
    from regions import Regions
except Exception:
    HAS_REGION = False
else:
    HAS_REGION = True


_SDSS_SPECTRAL_TEMPLATES_PACKAGE = "http://classic.sdss.org/dr5/algorithms/spectemplates/spectemplatesDR2.tar.gz"


class ScaleBar:
    """A simple scalebar."""

    def __init__(self, ax, x0=0.025, y0=0.025, length=1, size=0.025,
                 orientation='hor', n_subunits=2, text=None, fontsize=8,
                 units='arcsec', scale_factor=1):

        self.ax = ax
        self.units = units
        self.start_pos = (x0, y0)
        self.scale_factor = scale_factor
        self.size = size

        if orientation.lower().startswith('h'):
            self.end_pos = (x0 + length, y0)
        else:
            self.end_pos = (x0, y0 + length)

        self.orientation = orientation
        self.n_subunits = n_subunits
        self.length = length
        self.scale_bar_elements = []

        if text is None:
            text = f"{length:.1f}"

        length, _ = (
            ax.transData + ax.transAxes.inverted()
        ).transform([length, 0])

        self.text_handler = ax.text(
            x0 + length / 2,
            y0 + 2 * size,
            text,
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes,
            fontsize=fontsize
        )

        self.text_handler.set_path_effects(
            (
                PathEffects.withStroke(
                    linewidth=3, foreground='white'
                ),
            )
        )

        delta = length / n_subunits

        for i in range(n_subunits):
            if orientation.lower().startswith('h'):
                x = x0 + i*delta
                y = y0
                wid = delta
                hei = size
            else:
                x = x0
                y = y0 + i*delta
                wid = size
                hei = delta

            if i % 2:
                facecolor = 'white'
                edgecolor = 'black'
            else:
                facecolor = 'black'
                edgecolor = 'white'

            scale_bar_element = patches.Rectangle(
                (x, y),
                wid,
                hei,
                transform=ax.transAxes,
                facecolor=facecolor,
                edgecolor=edgecolor,
                lw=1,
                ls='-',
            )

            ax.add_artist(scale_bar_element)
            self.scale_bar_elements.append(scale_bar_element)

    def update(self):
        """
        Update the ScaleBar.

        Returns
        -------
        None.

        """
        length, _ = (
            self.ax.transData + self.ax.transAxes.inverted()
        ).transform([self.length, 0])

        delta = length / self.n_subunits

        self.text_handler.set_x(self.start_pos[0] + length / 2)
        self.text_handler.set_y(self.start_pos[1] + 2 * self.size)

        for i, rect in enumerate(self.scale_bar_elements):
            if self.orientation.lower().startswith('h'):
                x = self.start_pos[0] + i*delta
                y = self.start_pos[1]
                wid = delta
                hei = self.size
            else:
                x = self.start_pos[0]
                y = self.start_pos[1] + i*delta
                wid = self.size
                hei = delta

            rect.set_x(x)
            rect.set_y(y)
            rect.set_width(wid)
            rect.set_height(hei)

    def set_wcs(self, wcs):
        """
        Set the WCS.

        Parameters
        ----------
        wcs : astropy.wcs.WCS
            The WCS to use.

        Returns
        -------
        None.

        """
        star_coords = wcs.pixel_to_world(
            *self.start_pos
        )

        end_coords = wcs.pixel_to_world(
            *self.end_pos
        )

        sep = star_coords.separation(end_coords).to(self.units)
        sep /= self.scale_factor

        self.text_handler.set_text(f"{sep:.2f}")
        self.update()


def find_prog(prog):
    """
    Find the path of a pregram in your PATH.

    Parameters
    ----------
    prog : str
        Name of the program.

    Returns
    -------
    str
        The path of the program.

    """
    cmd = "where" if platform.system() == "Windows" else "which"
    try:
        return subprocess.check_output([cmd, prog]).strip().decode()
    except subprocess.CalledProcessError:
        return None


def get_pbar(partial, total=None, wid=32, common_char='\u2588',
             upper_char='\u2584', lower_char='\u2580'):
    """
    Return a nice text/unicode progress bar showing partial and total progress.

    Parameters
    ----------
    partial : float
        Partial progress expressed as decimal value.
    total : float, optional
        Total progress expresses as decimal value.
        If it is not provided or it is None, than
        partial progress will be shown as total progress.
    wid : int , optional
        Width in charachters of the progress bar.
        The default is 32.

    Returns
    -------
    pbar : str
        A unicode progress bar.

    """
    wid -= 2
    prog = int((wid)*partial)
    if total is None:
        total_prog = prog
        common_prog = prog
    else:
        total_prog = int((wid)*total)
        common_prog = min(total_prog, prog)
    pbar_full = common_char*common_prog
    pbar_full += upper_char*(total_prog - common_prog)
    pbar_full += lower_char*(prog - common_prog)
    return (f"\u2595{{:<{wid}}}\u258F").format(pbar_full)


def get_sdss_spectral_templates(outdir: str, use_cached: bool = True) -> list:
    """
    Download spectral templates from SDSS.

    Parameters
    ----------
    outdir : str
        Path where to save the templates.

    use_cached : bool, optional
        If true, do not redownload the package file if it is still present.
        Default is True.

    Returns
    -------
    templates : list
        A a list of dictionaries that have the following structure:

        'file' : str
            The path of the actual template file
        'type' : str
            The type of template. Can be 'star', 'galaxy' or 'qso'
        'sub-type' : str
            The sub-type of the object
    """
    def report_pbar(blocks_count, block_size, total_size):
        downloaded_size = blocks_count * block_size
        progress = downloaded_size / total_size
        pbar = get_pbar(progress)
        report_str = f"\r{pbar} {progress: 6.2%}  "
        report_str += f"{downloaded_size}/{total_size} Bytes\r"
        sys.stderr.write(report_str)
        sys.stderr.flush()

    def same_dest_dir(outdir, member):
        member_path = os.path.join(outdir, member.name)
        abs_dest_dir = os.path.abspath(outdir)
        abs_target_dir = os.path.abspath(member_path)
        prefix = os.path.commonprefix([abs_dest_dir, abs_target_dir])
        return prefix == abs_dest_dir

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    package_name = os.path.basename(_SDSS_SPECTRAL_TEMPLATES_PACKAGE)
    package_out_file = os.path.join(outdir, package_name)

    print(
        f"\nDownloading SDSS spectral templates package {package_name}...",
        file=sys.stderr
    )

    if not (use_cached and os.path.isfile(package_out_file)):
        package_out_file, headers = request.urlretrieve(
            _SDSS_SPECTRAL_TEMPLATES_PACKAGE,
            package_out_file,
            reporthook=report_pbar
        )

    print(
        "\nExtracting template files\n",
        file=sys.stderr
    )

    templates = []
    with tarfile.open(package_out_file, 'r') as gzfile:
        members_list = gzfile.getmembers()
        for j, member in enumerate(members_list):
            progress = j / len(members_list)
            pbar_str = get_pbar(progress)
            sys.stderr.write(f"\r{pbar_str} {progress: 6.2%}  {member.name}\r")
            sys.stdout.flush()
            if not same_dest_dir(outdir, member):
                print(
                    "Attempted Path Traversal in Tar File for member "
                    f"{member.name}",
                    file=sys.stdout
                )
                continue
            gzfile.extract(member, path=outdir, numeric_owner=False)
            t_path = os.path.abspath(os.path.join(outdir, member.name))
            t_type_id = os.path.splitext(os.path.basename(member.name))[0]
            t_type_id = int(t_type_id[-3:])

            if t_type_id <= 23:
                t_type = 'star'
            elif t_type_id <= 29:
                t_type = 'galaxy'
            else:
                t_type = 'qso'

            templates.append({
                'file': t_path,
                'type': t_type,
                'sub-type': ''
            })

    return templates


def get_sdss_template_data(sdss_template_file: str) -> dict:
    """
    Get spectrum from a SDSS spectral template file.

    Parameters
    ----------
    sdss_template_file : str
        Path of a SDSS spectral template file.

    Returns
    -------
    t_dict : dict
        A dictionary contatining the following keys and values:
            'flux' : numpy.ndarray 1D
                The flux data in arbitrary units
            'wavelengths' : numpy.ndarray 1D
                The wavelengths
    """
    try:
        t_spectrum = fits.getdata(sdss_template_file, ext=0)[0]
        t_head = fits.getheader(sdss_template_file, ext=0)
    except (FileNotFoundError, IndexError):
        return {}
    else:
        t_wcs = apwcs.WCS(t_head)

    pixel_coords_x = np.arange(0, len(t_spectrum), 1)
    pixel_coords_y = np.zeros_like(pixel_coords_x)

    t_wavelengths = t_wcs.pixel_to_world_values(pixel_coords_x, pixel_coords_y)
    t_wavelengths = 10 ** t_wavelengths[0]

    t_dict = {
        'flux': t_spectrum,
        'wavelengths': t_wavelengths
    }
    return t_dict


def parse_regionfile(regionfile, key_ra='ALPHA_J2000', key_dec='DELTA_J2000',
                     key_id='NUMBER', file_format='ds9'):
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
    file_format : str, optional
        Format of the input regionfile.
        The default value is 'ds9'.

    Returns
    -------
    sources : astropy.table.Table
        The table containing the sources.
    skyframe : None
        Placeholder.

    """
    global HAS_REGION

    if not HAS_REGION:
        logging.error(
            "astropy regions package is needed to handle regionfiles!"
        )
        return

    myt = Table(
        names=[key_id, key_ra, key_dec, 'region'],
        units=[None, 'deg', 'deg', None],
        dtype=[str, float, float, object]
    )
    for j, reg in enumerate(Regions.read(regionfile, format=file_format)):
        try:
            reg_id = str(reg.meta['text'])
        except Exception:
            reg_id = str(j)

        try:
            center = reg.center
        except AttributeError:
            c_ra = np.mean([x.ra.to('deg').value for x in reg.vertices])
            c_dec = np.mean([x.dec.to('deg').value for x in reg.vertices])
            new_row = [reg_id, c_ra, c_dec, reg]
        else:
            new_row = [reg_id, center.ra, center.dec, reg]
        myt.add_row(new_row)

    return myt, None


def get_aspect(ax):
    """
    Get ratio between y-axis and x-axis of a matplotlib figure.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        Ther axis you want to get the axes ratio.

    Returns
    -------
    ratio : float
        The aspect ratio.

    """
    figW, figH = ax.get_figure().get_size_inches()

    # Axis size on figure
    _, _, w, h = ax.get_position().bounds

    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)

    # Ratio of data units
    data_ratio = (max(*ax.get_ylim()) - min(*ax.get_ylim()))
    data_ratio /= (max(*ax.get_xlim()) - min(*ax.get_xlim()))

    return disp_ratio / data_ratio


def get_vclip(img, vclip=0.25, nsamples=1000):
    """
    Get the clipping values to use with imshow function.

    Parameters
    ----------
    img : numpy.ndarray
        Image data.
    vclip : float, optional
        Contrast parameter. The default is 0.5.

    Returns
    -------
    vmin : float
        median - vclip*std.
    vmax : float
        median + vclip*std.

    """
    img = np.ma.masked_array(img, mask=np.isnan(img))
    zsc = ZScaleInterval(nsamples, contrast=vclip, krej=10)
    vmin, vmax = zsc.get_limits(img)
    return vmin, vmax


def get_log_img(img, vclip=0.5):
    """
    Get the image in log scale.

    Parameters
    ----------
    img : numpy.ndarray
        The image data.
    vclip : float, optional
        Contrast factor. The default is 0.5.

    Returns
    -------
    log_image : numpy.ndarray
        The logarithm of the input image.
    vclip : 2-tuple of floats
        The median +- vclip*std of the image.

    """
    log_img = np.log10(1 + img - np.nanmin(img))
    return log_img, *get_vclip(log_img)


def load_rgb_fits(fits_file, ext_r=1, ext_g=2, ext_b=3):
    """
    Load an RGB image from a FITS file.

    Parameters
    ----------
    fits_file : str
        The path of the fits file.
    ext_r : int, optional
        The index of the extension containing the red channel.
        The default is 1.
    ext_g : int, optional
        The index of the extension containing the green channel.
        The default is 2.
    ext_b : int, optional
        The index of the extension containing the blue channel.
        The default is 3.

    Returns
    -------
    dict
        The dictionary contains the following key: value pairs:
            'data': 3D numpy.ndarray
                The actual image data.
            'wcs': astropy.wcs.WCS
                The WCS of the image.
    """
    try:
        rgb_image = np.array((
            fits.getdata(fits_file, ext=ext_r),
            fits.getdata(fits_file, ext=ext_g),
            fits.getdata(fits_file, ext=ext_b),
        )).transpose(1, 2, 0)
    except IndexError:
        return None

    rgb_image -= np.nanmin(rgb_image)
    rgb_image = rgb_image / np.nanmax(rgb_image)

    rgb_wcs = apwcs.WCS(fits.getheader(fits_file, ext=1))

    return {'data': rgb_image, 'wcs': rgb_wcs, 'type': 'rgb'}


def log_rebin(wave, spec_list, oversample=1, flux_conserving=False):
    """
    Logarithmically rebin a spectrum conserving the flux.

    Parameters
    ----------
    wave : list or numpy.ndarray
        The wavelengths of the originaò dispersion grating.
    spec_list : list
        A list of numpy.ndarrays corrsponding toe the fulx or other quantities
        to be rebinned with the dispersion grating.
    oversample : int, optional
        Whether to oversample the new dispersion grating. The default is 1.
    flux_conserving : bool, optional
        If False conserve the flux density instead of the flux.
        The default is False.

    Raises
    ------
    ValueError
        If wave is not monotonically increasing or if it does not have the
        same dimension of the fluxes.

    Returns
    -------
    rebinned_spec_list : list
        A list containing the rebinned input fluxes.
    log_wave : numpy.ndarray
        The log-reminned disperions grating.
    velscale : float
        The velocity scale of the new dispersion grating.

    """
    wave = np.asarray(wave, dtype=float)
    spec_list = [
        np.asarray(x, dtype=float) for x in spec_list
    ]

    if np.any(np.diff(wave) <= 0):
        raise ValueError("'wave' must be monotonically increasing")

    if (
            (len(wave.shape) != 1) or
            np.any([wave.shape[0] != x.shape[0] for x in spec_list])
    ):
        raise ValueError(
            "'wave' must be an array with the same lenght of the input spectra"
        )

    if wave.shape[0] == 2:
        dlam = np.diff(wave) / (wave.shape[0] - 1)
        lim = wave + [-0.5, 0.5]*dlam
        borders = np.linspace(*lim, wave.shape[0] + 1)
    else:
        lim = 1.5*wave[[0, -1]] - 0.5*wave[[1, -2]]
        borders = np.hstack([lim[0], (wave[1:] + wave[:-1])/2, wave[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    c_km_h = constants.c.to(apu.km / apu.h).value

    m = int(wave.shape[0] * oversample)
    velscale = c_km_h * np.diff(ln_lim) / m
    velscale = velscale.item()

    new_borders = np.exp(ln_lim[0] + velscale / c_km_h * np.arange(m + 1))

    if wave.shape[0] == 2:
        k = ((new_borders - lim[0])/dlam)
        k = k.clip(0, wave.shape[0] - 1).astype(int)
    else:
        k = (np.searchsorted(borders, new_borders) - 1)
        k = k.clip(0, wave.shape[0] - 1).astype(int)

    # Do analytic integral of step function
    rebinned_spec_list = []
    for spec in spec_list:
        spec_rebin = np.add.reduceat((spec.T*dlam).T, k)[:-1]
        # fix for design flaw of reduceat()
        spec_rebin.T[...] *= np.diff(k) > 0
        # Add to 1st dimension
        spec_rebin.T[...] += np.diff(((new_borders - borders[k])) * spec[k].T)

        if not flux_conserving:
            # Divide 1st dimension
            spec_rebin.T[...] /= np.diff(new_borders)

        rebinned_spec_list.append(spec_rebin)

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5*np.log(new_borders[1:]*new_borders[:-1])

    return rebinned_spec_list, ln_lam, velscale


def plot_scandata(target, scandata):
    """
    Plot debugging information about zchi2.

    Parameters
    ----------
    target : redrock.Target
        The target object.
    scandata : dict
        The full scandata as returned by redrock.

    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure of the plot.
    ax : TYPE
        The main axis of the plot.

    """
    try:
        obj_data = scandata[target.id]
    except KeyError:
        return

    template_types = list(obj_data.keys())

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for template_type in template_types:
        ax.plot(
            obj_data[template_type]['redshifts'],
            obj_data[template_type]['zchi2'],
            label=f"{template_type}"
        )
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(left=0.1)
    ax.set_title(f'{target.id}')
    ax.set_xlabel("redshift")
    ax.set_ylabel("Chi2")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def get_ellipse_skypoints(center: coordinates.SkyCoord,
                          a: apu.quantity.Quantity,
                          b: apu.quantity.Quantity,
                          angle: apu.quantity.Quantity = 0*apu.deg,
                          n_points: int = 20) -> list:
    """
    Get points of an ellips on the skyplane.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        DESCRIPTION.
    a : apu.quantity.Quantity
        Angular size of the semi-major axis.
    b : apu.quantity.Quantity
        Angular size of the semi-mino axis.
    angle : apu.quantity.Quantity, optional
        Rotation angle of the semi-major axis respect to the RA axis.
        The default is 0 deg.
    n_points : int, optional
        Number of points to return. The default is 25.

    Returns
    -------
    ellipse_points : list of coordinates.SkyCoord
        List of SkyCoord corresponding to the points of the ellipse.

    Example
    -------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from astropy.io import fits
    >>> from astropy.wcs import WCS
    >>> from astropy.wcs.utils import wcs_to_celestial_frame
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy import units as apu
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> from specex.utils import get_ellipse_skypoints
    >>> fn = get_pkg_data_filename('galactic_center/gc_msx_e.fits')
    >>> image = fits.getdata(fn, ext=0)
    >>> w = WCS(fits.getheader(fn, ext=0))
    >>> ellipse_center = w.pixel_to_world(50, 60)
    >>> a = 10 * apu.arcmin
    >>> b = 5 * apu.arcmin
    >>> angle = 45 * apu.deg
    >>> world_points = get_ellipse_skypoints(
    ...     center=ellipse_center,
    ...     a=a, b=b,
    ...     angle=angle
    ... )
    >>> pp_val = np.array([
    ...     [x.l.value, x.b.value] for x in world_points
    ... ])
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection=w)
    >>> ax.imshow(
    ...     image,
    ...     vmin=-2.e-5,
    ...     vmax=2.e-4,
    ...     origin='lower',
    ...     cmap='plasma'
    ... )
    >>> ax.plot(
    ...     pp_val[..., 0], pp_val[..., 1],
    ...     color='#31cc02',
    ...     lw=2,
    ...     ls='--',
    ...     transform=ax.get_transform(wcs_to_celestial_frame(w))
    ... )
    >>> ax.set_aspect(1)
    >>> plt.show()
    >>> plt.close(fig)
    """
    ellipse_points = []

    # Check if a is actually greater than b, otherwise swap them
    if a < b:
        _tmp = a
        a = b
        b = _tmp

    for theta in np.linspace(0, 2*np.pi, n_points, endpoint=True):
        total_angle = -theta + angle.to(apu.rad).value
        radius = a*b / np.sqrt(
            (a*np.cos(total_angle))**2 + (b*np.sin(total_angle))**2
        )
        new_point = center.directional_offset_by(
            position_angle=apu.Quantity(theta, apu.rad),
            separation=radius
        )

        ellipse_points.append(new_point)
    return ellipse_points


def plot_masked_regions(ax, wavelengths, nan_mask, wave_range=None,
                        label_min_wid=100):
    """
    Plot a mask over a region containing missing data.

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    wavelengths : TYPE
        DESCRIPTION.
    nan_mask : TYPE
        DESCRIPTION.
    wave_range : TYPE, optional
        DESCRIPTION. The default is None.
    label_min_wid : intm optional
        The default value is 100.
    Returns
    -------
    region_patches : dict
        DESCRIPTION.

    """
    region_patches = {
        'background': [],
        'area': [],
        'text': []
    }

    if nan_mask is None:
        return region_patches

    if wave_range is None:
        w_min = np.nanmin(wavelengths)
        w_max = np.nanmax(wavelengths)
    else:
        w_min = np.min(wave_range)
        w_max = np.max(wave_range)

    lam_mask = np.array([
        (wavelengths[m_start], wavelengths[m_end])
        for m_start, m_end in get_mask_intervals(nan_mask)
    ])

    w_min = np.nanmin(wavelengths)
    w_max = np.nanmax(wavelengths)

    for lam_inter in lam_mask:
        rect = patches.Rectangle(
            (lam_inter[0], 0),
            lam_inter[1] - lam_inter[0], 1,
            transform=ax.get_xaxis_transform(),
            linewidth=1,
            fill=True,
            edgecolor='black',
            facecolor='white',
            zorder=10
        )
        ax.add_patch(rect)
        region_patches['background'].append(rect)

        rect = patches.Rectangle(
            (lam_inter[0], 0),
            lam_inter[1] - lam_inter[0], 1,
            transform=ax.get_xaxis_transform(),
            linewidth=1,
            fill=True,
            edgecolor='black',
            facecolor='white',
            hatch='///',
            zorder=11
        )
        ax.add_patch(rect)
        region_patches['area'].append(rect)
        if (
            (lam_inter[1] > w_min + label_min_wid) and
            (lam_inter[0] < w_max - label_min_wid)
        ):
            text = ax.text(
                (lam_inter[0] + lam_inter[1]) / 2, 0.5,
                "MISSING DATA",
                transform=ax.get_xaxis_transform(),
                va='center',
                ha='center',
                rotation=90,
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'black',
                    'boxstyle': 'round,pad=0.5',
                },
                zorder=12,
                clip_on=True
            )
            region_patches['text'].append(text)
    return region_patches


def plot_lines(ax, wave_range, lw=0.7, lines_z=0, label_size='medium',
               colors=['green', 'red', 'yellow'], label_y=0.02,
               alpha=[0.5, 0.75, 0.9], ls=['--', '--', '--'],
               label_rotation=90,):
    """
    Plot emission and/or absorption lines.

    Parameters
    ----------
    ax : matplotlib Axes
        The Axes where to draw the lines.
    wave_range : liat or numpy.ndarray
        wavelength range.
    lw : float, optional
        width of the plotted lines. The default is 0.7.
    lines_z : float, optional
        The redshift. The default is 0.
    label_size : float or str, optional
        The size of the labels. The deafult value is 'medium'.
    colors : list of 3 colors, optional
        The color used to draw the different type of lines. The first color is
        assinged to absorption lines, the second one to emission lines and the
        third one to lines that can be both absoprtion and emission.
        The default value is ['green', 'red', 'yellow'].
    label_y : float, optional
        The position of the labels in axis-coordinates.
        The default value is 0.02
    alpha : list of 3 floats, optional
        The value of the alpha transparence used to draw the lines. The order
        is the same of the colors array. The default value is [0.5, 0.75, 0.9].
    ls : list of 3 line styles, optional
        Styles used to plot lines. The order is the same of the colors array.
        The default value is ['--', '--', '--']
    label_rotation : float, optional
        The rotation in degrees of the labels. The default value is 90
    Returns
    -------
    None.

    """
    w_min = np.min(wave_range)
    w_max = np.max(wave_range)

    absorption_lines = get_lines(
        line_type='A', wrange=[w_min, w_max], z=lines_z
    )
    for line_lam, line_name, line_type in absorption_lines:
        ax.axvline(
            line_lam, color=colors[0], ls='--', lw=lw, alpha=alpha[0],
            label='absorption lines'
        )
        ax.text(
            line_lam, label_y, line_name, rotation=label_rotation,
            transform=ax.get_xaxis_transform(),
            zorder=99,
            fontsize=label_size
        )

    # Plotting emission lines
    emission_lines = get_lines(
        line_type='E', wrange=[w_min, w_max], z=lines_z
    )
    for line_lam, line_name, line_type in emission_lines:
        ax.axvline(
            line_lam, color=colors[1], ls='--', lw=lw, alpha=alpha[1],
            label='emission lines',
            zorder=2
        )
        ax.text(
            line_lam, label_y, line_name, rotation=label_rotation,
            transform=ax.get_xaxis_transform(),
            zorder=99,
            fontsize=label_size
        )

    # Plotting emission/absorption lines
    emission_lines = get_lines(
        line_type='AE', wrange=[w_min, w_max], z=lines_z
    )
    for line_lam, line_name, line_type in emission_lines:
        ax.axvline(
            line_lam, color=colors[2], ls='--', lw=lw, alpha=alpha[2],
            label='emission/absorption lines',
            zorder=3
        )
        ax.text(
            line_lam, label_y, line_name, rotation=label_rotation,
            transform=ax.get_xaxis_transform(),
            zorder=99,
            fontsize=label_size
        )


def plot_spectrum(wavelengths, flux, variance=None, nan_mask=None,
                  restframe=False, cutout=None, cutout_vmin=None,
                  cutout_vmax=None, cutout_wcs=None, redshift=None,
                  smoothing=None, wavelengt_units=None, flux_units=None,
                  extra_info={}, extraction_info={}, wave_range=None,
                  show_legend=True, axs=None):
    """
    Plot a spectrum.

    Parameters
    ----------
    wavelengths : numpy.ndarray 1D
        An array containing the wavelengths.
    flux : numpy.ndarray 1D
        An array containing the fluxes corresponding to the wavelengths.
    variance : numpy.ndarray 1D, optional
        An array containing the variances corresponding to the wavelengths.
        If it is None, then the variance is not plotted. The default is None.
    nan_mask : numpy.ndarray 1D of ndtype=bool, optional
        An array of dtype=bool that contains eventual invalid fluxes that need
        to be masked out (ie. nan_mask[j] = True means that wavelengths[j],
        flux[j] and variance[j] are masked). If it is None, then no mask is
        applyed. The default is None.
    restframe : bool, optional
        If True, then the spectrum is plotted in the observer restframe
        (ie. the spectrum is de-redshifted before plotting). In order to use
        this option, a valid redshift must be specified.
        The default is False.
    cutout : numpy.ndarray 2D or 3D, optional
        A grayscale or RGB image to be shown alongside the spectrum.
        If None, no image is shown. The default is None.
    cutout_wcs : astropy.wcs.WCS, optional
        An optional WCS for the cutout. The default is None.
    cutout_vmin : float, optional
        The value to be interpreted as black in the cutout image.
        If it is None, the value is determined automatically.
        The default is None.
    cutout_vmax : float, optional
        The value to be interpreted as white in the cutout image.
        If it is None, the value is determined automatically.
        The default is None.
    redshift : float, optional
        The redshift of the spectrum. If None then no emission/absorption
        line is plotted and restframe option is ignored. The default is None.
    smoothing : int or None, optional
        If an integer greater than zero is passed to this parameter, then the
        value is used as the window size of scipy.filter.savgol_filter used to
        smooth the flux of the spectum. If this value is 0 or None then no
        smoothing is performed. The default is None.
    wavelengt_units : str, optional
        The units of the wavelengths. The default is None.
    flux_units : str, optional
        The units of the fluxes. The default is None.
    extra_info : dict of {str: str, ...}, optional
        A dictionary containing extra information to be shown in the plot.
        Both keys and values of the dictionary must be strings. This dict is
        rendered as a table of two columns filled with the keys and the values.
        The default is {}.
    extraction_info: dict, optionale
        This dictionary must contain extraction information from specex. If not
        empty or None, extraction information are used to plot the apertures
        used by specex over the cutout (if provided).
        The default is {}.
    wave_range: list, optional
        Range of wavelengths to plot. The default value is None.
    show_legend: bool, optional
        Show the legend if true. Default value it True.
    axs: list of matplotlib.axes._subplots.AxesSubplots, optional
        List of axes to use for plotting. If none a new figure with the
        appropriate axes is created. The default value is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    axs: list of matplotlib.axes._subplots.AxesSubplots
        A list of the axes subplots in the figure.
    """
    if nan_mask is not None:
        if variance is not None:
            var_max = np.nanmax(variance[~nan_mask])
        else:
            var_max = 1
    else:
        if variance is not None:
            var_max = np.nanmax(variance)
        else:
            var_max = 1

    w_min = 1.0e5
    w_max = 0.0

    if restframe and redshift is not None:
        wavelengths = wavelengths / (1 + redshift)
        lines_z = 0
    else:
        wavelengths = wavelengths
        lines_z = redshift

    if wave_range is None:
        w_min = np.nanmin(wavelengths)
        w_max = np.nanmax(wavelengths)
    else:
        w_min = np.min(wave_range)
        w_max = np.max(wave_range)

    if wavelengt_units:
        try:
            x_unit = apu.Unit(wavelengt_units).to_string(
                'latex', fraction='inline'
            )
        except Exception:
            x_label = f'Wavelength [{wavelengt_units}]'
        else:
            x_label = f'Wavelength [{x_unit}]'
    else:
        x_label = 'Wavelength'

    if flux_units:
        try:
            y_unit = apu.Unit(flux_units).to_string(
                'latex', fraction='inline'
            )
        except Exception:
            y_label = f'Flux [{flux_units}]'
        else:
            y_label = f'Flux [{y_unit}]'
    else:
        y_label = 'Flux'

    if axs is None:
        fig = plt.figure(figsize=(15, 5))

        # Make a grid of plots
        gs = GridSpec(6, 6, figure=fig, hspace=0.1)
    else:
        fig = axs[0].figure

    # If variance data are present, then make two plots on the left of the
    # figure. The top one is for the spectrum and the bottom one is for the
    # variance. Otherwise just make a bigger plot on the left only for the
    # spectrum.
    if variance is not None:
        if axs is None:
            ax0 = fig.add_subplot(gs[:4, :-1])
            ax3 = fig.add_subplot(gs[4:, :-1], sharex=ax0)
        else:
            ax0 = axs[0]
            ax3 = axs[1]

        ax3.plot(
            wavelengths, variance,
            ls='-',
            lw=0.5,
            alpha=0.75,
            color='black',
            label='variance',
            zorder=0
        )
        ax3.set_xlabel(x_label)
        ax3.set_ylabel('Variance')

        ax3.set_xlim(w_min, w_max)
        ax3.set_ylim(1, var_max)
        ax3.set_yscale('log')
        ax0.label_outer()
    else:
        if axs is None:
            ax0 = fig.add_subplot(gs[:, :-1])
            ax3 = None
        else:
            ax0 = axs[0]
            ax3 = axs[1]
        ax0.set_xlabel(x_label)

    ax0.set_ylabel(y_label)
    ax0.set_xlim(w_min, w_max)

    # Plot a cutout
    if cutout is not None:
        if axs is None:
            ax1 = fig.add_subplot(gs[:3, -1], projection=cutout_wcs)
            ax2 = fig.add_subplot(gs[3:, -1])
        else:
            ax1 = axs[2]
            ax2 = axs[3]

        ax1.axis('off')
        ax1.imshow(
            cutout,
            origin='lower',
            aspect='equal',
            vmin=cutout_vmin,
            vmax=cutout_vmax,
            zorder=0
        )
        ax1.set_aspect(1)

        # Check if there are info about the specex extraction
        try:
            ext_mode = extraction_info['mode']
            ext_apertures = extraction_info['apertures']
            e_ra = extraction_info['aperture_ra']
            e_dec = extraction_info['aperture_dec']
            e_frame = extraction_info['frame']
        except (TypeError, KeyError):
            # No extraction info present, just ignore
            pass
        else:
            # If there are extraction info, read the information
            e_wid, e_hei, e_ang = ext_apertures

            e_cc = coordinates.SkyCoord(
                e_ra, e_dec,
                unit=('deg', 'deg'),
                frame=e_frame
            )

            # and then draw extraction apertures
            if ext_mode.lower() in [
                    'kron_ellipse', 'kron_circular', 'circular_aperture'
            ]:
                e_world_points = get_ellipse_skypoints(
                    e_cc,
                    a=0.5*e_hei,
                    b=0.5*e_wid,
                    angle=e_ang
                )

                e_world_points_values = np.array([
                    [x.ra.value, x.dec.value]
                    for x in e_world_points
                ])

                ax1.plot(
                    e_world_points_values[..., 0],
                    e_world_points_values[..., 1],
                    color='black',
                    ls='-',
                    lw=1,
                    alpha=0.7,
                    zorder=1,
                    transform=ax1.get_transform(e_frame)
                )
                ax1.plot(
                    e_world_points_values[..., 0],
                    e_world_points_values[..., 1],
                    color='cyan',
                    ls='--',
                    lw=1,
                    alpha=0.7,
                    zorder=2,
                    transform=ax1.get_transform(e_frame)
                )

        scbar = ScaleBar(
            ax1,
            length=cutout.shape[1]/2
        )
        scbar.set_wcs(cutout_wcs)
        scbar.update()
    else:
        if axs is None:
            ax1 = None
            ax2 = fig.add_subplot(gs[:, -1])
        else:
            ax1 = axs[2]
            ax2 = axs[3]

    ax0.set_aspect('auto')

    # Plot only original spectrum or also a smoothed version
    if not smoothing:
        ax0.plot(
            wavelengths, flux,
            ls='-',
            lw=0.5,
            alpha=1,
            color='black',
            label='spectrum',
            zorder=0
        )
    else:
        window_size = 4*smoothing + 1
        smoothed_flux = savgol_filter(flux, window_size, 3)
        ax0.plot(
            wavelengths, flux,
            ls='-',
            lw=1,
            alpha=0.35,
            color='gray',
            label='original spectrum',
            zorder=0
        )
        ax0.plot(
            wavelengths, smoothed_flux,
            ls='-',
            lw=0.4,
            alpha=1.0,
            color='#03488c',
            label='smoothed spectrum',
            zorder=1
        )

    if redshift is not None:
        plot_lines(ax0, [w_min, w_max], lines_z=lines_z)

    # Draw missing data or invalid data regions
    _ = plot_masked_regions(ax0, wavelengths, nan_mask, wave_range)

    if show_legend:
        handles, labels = ax0.get_legend_handles_labels()
        newLabels, newHandles = [], []
        for handle, label in zip(handles, labels):
            if label not in newLabels:
                newLabels.append(label)
                newHandles.append(handle)

        _ = ax0.legend(
            newHandles, newLabels,
            loc='upper center',
            fancybox=True,
            shadow=False,
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(newHandles)
        )

    if ax2 is not None:
        cell_text = [
            [f'{key}', f"{val}"] for key, val in extra_info.items()
        ]

        ax2.axis("off")
        if cell_text:
            tbl = ax2.table(
                cellText=cell_text,
                colWidths=[0.4, 0.6],
                loc='upper center'
            )
            tbl.scale(1, 1.5)

    return fig, [ax0, ax1, ax2, ax3]


def plot_zfit_check(target, zbest, plot_template=None, restframe=False,
                    wavelengt_units='Angstrom', flux_units=''):
    """
    Plot the check images for the fitted targets.

    This function will plot the spectra of the target object along with the
    spectra of the best matching tamplate and some other info.

    Parameters
    ----------
    target : redrock.targets.Target object
        A targets used in redshift estimation process.
    zfit : astropy.table.Table
        A table containing the reshift and other info of the input targets.
    plot_template : list of redrock.templates or None, optional
        If not None, plot the best matching tamplate.
    rest_frame : bool, optional
        Whether to plot the spectrum at restrframe.
    wave_units : str, optional
        The units of the wavelength grid. The default value id 'Angstrom'
    flux_units : str, optional
        The units of the spectum. The default value is ''.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    axs : 2-tuple of matplotlib.axes._subplots.AxesSubplot and/or None
        List of AxesSubplot. If no cutout was passed in input, the list will
        contain only the axes of the plot of the spectrum, otherwise two axes
        will be in the list: the first axes containing the plot of the spectrum
        and the second axes containing the cutout of the object.

    """
    flux_units = flux_units.replace('**', '^')

    t_best_data = zbest[zbest['SPECID'] == target.spec_id][0]

    info_dict = {
        'ID': f"{target.spec_id}",
        'Z': f"z: {t_best_data['Z']:.4f} ± {t_best_data['ZERR']:.2e}\n",
        'Template': f"{t_best_data['SPECTYPE']} {t_best_data['SUBTYPE']}",
        'SNR': f"{t_best_data['SN']:.2f}\n",
        'SNR (EM)': f"{t_best_data['SN_EMISS']:.2f}\n",
        'ZWARN': f"{t_best_data['ZWARN']}"
    }

    lam = target.spectra[0].wave.copy()

    fig, axs = plot_spectrum(
        lam,
        target.spectra[0].flux,
        cutout=None,
        redshift=t_best_data['Z'],
        restframe=restframe,
        wavelengt_units=wavelengt_units,
        flux_units=flux_units,
        extra_info=info_dict
    )

    best_template = None
    if plot_template:
        for t in plot_template:
            if (
                t.template_type == t_best_data['SPECTYPE'] and
                t.sub_type == t_best_data['SUBTYPE']
            ):
                best_template = t
                break

        if best_template:
            try:
                coeffs = t_best_data['COEFF'][:best_template.nbasis]
                template_flux = best_template.eval(
                    coeffs,
                    lam,
                    0 if restframe else t_best_data['Z'],
                )

                axs[0].plot(
                    lam, template_flux,
                    ls='-',
                    lw=1,
                    alpha=0.7,
                    c='red',
                    label=f'best template [{best_template.full_type}]'
                )
            except AssertionError:
                print(
                    f"Template warning for object {target.spec_id}\n"
                    f"  nbasis: {best_template.nbasis}\n"
                    f"  coeffs: {len(coeffs)}",
                    file=sys.stderr
                )

    return fig, axs


def get_mask_intervals(mask):
    """
    Get intervals where mask is True.

    Parameters
    ----------
    mask : numpy.ndarry
        The mask array.

    Returns
    -------
    regions : list of 2-tuples
        List of intervals.

    Example
    -------
    >>> mask = (0, 0, 0, 0 ,0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ,0 ,0)
    >>> get_mask_intervals(mask)

    [(5, 8), (10, 11), (14, 19)]
    """
    regions = []
    r_start = -1
    r_end = 0
    in_region = False
    for i, val in enumerate(mask):
        if val and not in_region:
            r_start = i
            in_region = True
        if in_region and not val:
            r_end = i-1
            in_region = False
            regions.append((r_start, r_end))
    return regions


def stack(data, wave_mask=None, quite=False):
    """
    Stack the spectral cube along wavelength axis.

    Parameters
    ----------
    data : numpy.ndarray
        The spectral datacube.
    wave_mask : 1D np.ndarray, optional
        Optional wavelength mask. Wavelength corresponding to a False will not
        be used in the stacking. The default is None.
    quite : bool, optional
        Whether to avoid printing a progress bar or not.
        The default value is False.

    Returns
    -------
    new_data : numpy.ndarray
        The stacked datacube.

    """
    img_height, img_width = data.shape[1], data.shape[2]
    new_data = np.zeros((img_height, img_width))
    for k, dat in enumerate(data):
        if (not quite) and (k % 10 == 0):
            progress = (k + 1) / len(data)
            sys.stderr.write(
                f"\rstacking cube: {get_pbar(progress)} {progress:.2%}\r"
            )
            sys.stderr.flush()
        if wave_mask is None or wave_mask[k].any():
            new_data = np.nansum(np.array([new_data, dat]), axis=0)
    print("", file=sys.stderr)
    return new_data


def nannmad(x, scale=1.48206, axis=None):
    """
    Compute the MAD of an array.

    Compute the Median Absolute Deviation of an array ignoring NaNs.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    scale : float, optional
        A costant scale factor that depends on the distributuion.
        See https://en.wikipedia.org/wiki/Median_absolute_deviation.
        The default is 1.4826.
    axis : int or None
        The axis along which to compute the MAD.
        The default is None.

    Returns
    -------
    nmad
        The NMAD value.

    """
    x = np.ma.array(x, mask=np.isnan(x))
    x_bar = np.ma.median(x, axis=axis)
    mad = np.ma.median(np.ma.abs(x - x_bar), axis=axis)
    return scale*mad


def get_spectrum_snr(flux: np.ndarray,
                     var: Optional[np.ndarray] = None,
                     smoothing_window: Optional[int] = 51,
                     smoothing_order: Optional[int] = 11):
    """
    Compute the SRN of a spectrum.

    Parameters
    ----------
    flux : numpy.ndarray
        The spectrum itself.
    var : numpy.ndarray, optional
        The variance of the spectrum itself.
        The default value is None.
    smoothing_window : int, optional
        Parameter to be passed to the smoothing function.
        The default is 51.
    smoothing_order : int, optional
        Parameter to be passed to the smoothing function.
        The default is 11.

    Returns
    -------
    sn_spec : float
        The SNR of the spectrum.

    """
    # DER-like SNR but with a true smoothing
    # https://stdatu.stsci.edu/vodocs/der_snr.pdf
    # Smoothing the spectrum to get a crude approximation of the continuum

    if np.isnan(flux).all():
        return np.nan
    else:
        flux = np.ma.array(flux.copy(), mask=np.isnan(flux))

    if var is not None:
        var = np.ma.array(var.copy(), mask=np.isnan(var))
    else:
        var = 1.0

    smoothed_spec = savgol_filter(flux, smoothing_window, smoothing_order)
    smoothed_spec = np.ma.array(smoothed_spec, mask=np.isnan(smoothed_spec))

    # Subtract the smoothed spectrum to the spectrum itself to get a
    # crude estimation of the noise
    noise_spec = flux - smoothed_spec

    # Get the median value of the spectrum, weighted by the variance
    obj_mean_spec = np.ma.sum(smoothed_spec / var) / np.ma.sum(1 / var)

    # Get the mean Signal to Noise ratio
    sn_spec = obj_mean_spec / nannmad(noise_spec)

    if np.isinf(sn_spec):
        sn_spec = 99
    elif np.isnan(sn_spec):
        sn_spec = 0

    return sn_spec


def get_spectrum_snr_emission(flux, var=None, bin_size=150):
    """
    Compute the SRN of a spectrum considering emission lines only.

    Parameters
    ----------
    flux : numpy.ndarray
        The spectrum itself.
    bin_size : int, optional
        Bin size to search for emission lines.
        The default is 50.

    Returns
    -------
    sn_spec : float
        The SNR of the spectrum.

    """
    # Inspired by https://www.aanda.org/articles/aa/pdf/2012/03/aa17774-11.pdf

    # Just ignore negative fluxes!
    flux = flux.copy()
    flux[flux < 0] = 0

    # If we have the variace, we can use it to weight the flux
    if var is not None:
        var = var.copy()
        flux = flux / var

    optimal_width = flux.shape[0] - flux.shape[0] % bin_size
    flux = flux[:optimal_width]

    if np.isnan(flux).all():
        return np.nan
    else:
        flux = np.ma.array(flux, mask=np.isnan(flux))

    if flux.mask.all():
        return np.nan

    # Rebin sub_spec to search for emission features
    sub_spec = flux.reshape(flux.shape[0] // bin_size, bin_size)

    # For each bin we compute the maximum and the median of each bin and
    # get their difference. This is now our "signal": if there is an
    # emission line, the maximum value is greater that the median and this
    # difference will be greater than one
    sub_diff = np.ma.max(sub_spec, axis=1) - np.ma.median(sub_spec, axis=1)

    s_em = sub_diff / 3.0*np.ma.median(sub_diff) - 1
    noise_em = nannmad(sub_diff)

    sn_spec = np.ma.max(s_em / noise_em)

    if np.isinf(sn_spec):
        sn_spec = 99
    elif np.isnan(sn_spec):
        sn_spec = 0

    return sn_spec


def get_pc_transform_params(wcs_object, inverse=False, ftol=1e-6):
    cel_w = wcs_object.celestial
    pcm = cel_w.wcs.get_pc()
    if inverse:
        pcm = np.linalg.inv(pcm)
    cdelt = cel_w.wcs.get_cdelt()

    sx = cdelt[0] * np.sign(pcm[0, 0]) * np.sqrt(pcm[0, 0]**2 + pcm[0, 1]**2)
    sy = cdelt[1] * np.sign(pcm[1, 0]) * np.sqrt(pcm[1, 0]**2 + pcm[1, 1]**2)
    rot = np.arctan2(-pcm[0, 1], pcm[0, 0])
    shr_y = np.arctan2(pcm[1, 1], pcm[1, 0]) - np.pi + rot
    rot = apu.Quantity(rot, apu.rad).to(apu.deg)
    shr_y = apu.Quantity(shr_y, apu.rad).to(apu.deg)

    return (sx, sy, rot, shr_y)


def rotate_data(data: np.ndarray, angle: apu.Quantity,
                data_wcs: Optional[apwcs.WCS] = None):
    """
    Rotate data by angle and update the optionally given WCS data.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be rotated.
    angle : apu.Quantity
        The rotation angle.
    data_wcs : astropy.wcs.WCS or None, optional
        A WCS for the data. The default is None.

    Returns
    -------
    dict
        A dictionary with the the following keys and values.

        * 'data' : numpy.ndarray
            The rotated data
        * 'wcs' : astropy.wcs.WCS or None
            The updated WCS for the rotated data
    """
    if data_wcs is not None:
        # ndimage shape is [height, width] and element [0, 0] has
        # pixel coordinates [1, 1]
        rot_center = (np.array((data.shape[1], data.shape[0])) / 2) + 1
        rr = -angle.to(apu.rad).value
        pcm = data_wcs.celestial.wcs.get_pc()
        crpix = np.array(data_wcs.wcs.crpix)

        rot_matrix = np.array(
            [
                [1 * np.cos(rr), -1 * np.sin(rr)],
                [1 * np.sin(rr), 1 * np.cos(rr)]
            ]
        )

        new_pcm = rot_matrix.dot(pcm)
        new_crpix = rot_matrix.dot(crpix - rot_center) + rot_center

        new_wcs = apwcs.WCS(data_wcs.to_header())
        new_wcs.wcs.pc = new_pcm
        new_wcs.wcs.crpix = new_crpix
    else:
        new_wcs = None

    rotated_data = data.copy().astype('float32')
    nan_mask = np.isnan(rotated_data)

    rotated_data = rotate(
        rotated_data,
        angle.to(apu.deg).value,
        reshape=False,
        prefilter=False,
        order=0,
        cval=np.nan
    )

    rotated_mask = rotate(
        nan_mask,
        angle.to(apu.deg).value,
        reshape=False,
        prefilter=False,
        order=0,
        cval=1
    )

    rotated_data[rotated_mask] = np.nan

    return {'data': rotated_data, 'wcs': new_wcs}
