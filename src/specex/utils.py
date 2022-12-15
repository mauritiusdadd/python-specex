#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

This module provides utility functions used by other specex modules.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
import tarfile
from typing import Optional
from urllib import request

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches

from scipy.signal import savgol_filter
from scipy.ndimage import rotate

from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy import wcs as apwcs
from astropy import units as apu
from astropy import coordinates

from .lines import get_lines


_SDSS_SPECTRAL_TEMPLATES_PACKAGE = "http://classic.sdss.org/dr5/algorithms/spectemplates/spectemplatesDR2.tar.gz"


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
                          n_points: int = 25) -> list:
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


def plot_spectrum(wavelengths, flux, variance=None, nan_mask=None,
                  restframe=False, cutout=None, cutout_vmin=None,
                  cutout_vmax=None, cutout_wcs=None, redshift=None,
                  smoothing=None, wavelengt_units=None, flux_units=None,
                  extra_info={}, extraction_info={}):
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

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    axs: list of matplotlib.axes._subplots.AxesSubplots
        A list of the axes subplots in the figure.
    """
    if nan_mask is not None:
        lam_mask = np.array([
            (wavelengths[m_start], wavelengths[m_end])
            for m_start, m_end in get_mask_intervals(nan_mask)
        ])

        if variance is not None:
            var_max = np.nanmax(variance[~nan_mask])
        else:
            var_max = 1
    else:
        lam_mask = None
        if variance is not None:
            var_max = np.nanmax(variance)
        else:
            var_max = 1

    w_min = 1.0e5
    w_max = 0.0

    if restframe and redshift is not None:
        wavelengths = wavelengths / (1 + redshift)
        if lam_mask is not None:
            lam_mask = lam_mask / (1 + redshift)
        lines_z = 0
    else:
        wavelengths = wavelengths
        lines_z = redshift

    w_min = np.nanmin(wavelengths)
    w_max = np.nanmax(wavelengths)

    if wavelengt_units:
        x_label = f'Wavelenght [{wavelengt_units}]'
    else:
        x_label = 'Wavelenght'

    if flux_units:
        y_label = f'Flux [{flux_units}]'
    else:
        y_label = 'Flux'

    fig = plt.figure(figsize=(15, 5))

    # Make a grid of plots
    gs = GridSpec(6, 6, figure=fig, hspace=0.1)

    # If variance data are present, then make two plots on the left of the
    # figure. The top one is for the spectrum and the bottom one is for the
    # variance. Otherwise just make a bigger plot on the left only for the
    # spectrum.
    if variance is not None:
        ax0 = fig.add_subplot(gs[:4, :-1])
        ax4 = fig.add_subplot(gs[4:, :-1], sharex=ax0)

        ax4.plot(
            wavelengths, variance,
            ls='-',
            lw=0.5,
            alpha=0.75,
            color='black',
            label='variance',
            zorder=0
        )
        ax4.set_xlabel(x_label)
        ax4.set_ylabel('Variance')

        ax4.set_xlim(w_min, w_max)
        ax4.set_ylim(1, var_max)
        ax4.set_yscale('log')
        ax0.label_outer()
    else:
        ax0 = fig.add_subplot(gs[:, :-1])
        ax0.set_xlabel(x_label)

    ax0.set_ylabel(y_label)
    ax0.set_xlim(w_min, w_max)

    # Plot a cutout
    if cutout is not None:
        ax1 = fig.add_subplot(gs[:3, -1], projection=cutout_wcs)
        ax2 = fig.add_subplot(gs[3:, -1])

        if cutout_wcs is not None:
            ra_cunit, dec_cunit = cutout_wcs.celestial.wcs.cunit
        else:
            ra_cunit = apu.deg
            dec_cunit = apu.deg

        ax1.axis('off')
        ax1.imshow(
            cutout,
            origin='lower',
            aspect='equal',
            vmin=cutout_vmin,
            vmax=cutout_vmax
        )
        ax1.set_aspect(1)

        # Check if there are info about the specex extraction
        try:
            ext_mode = extraction_info['mode']
            ext_apertures = extraction_info['apertures']
            e_ra = extraction_info['aperture_ra']
            e_dec = extraction_info['aperture_dec']
        except (TypeError, KeyError):
            # No extraction info present, just ignore
            pass
        else:
            # If there are extraction info, read the information
            e_wid, e_hei, e_ang = ext_apertures

            e_cc = coordinates.SkyCoord(
                e_ra, e_dec,
                unit=(ra_cunit, dec_cunit),
                frame=apwcs.utils.wcs_to_celestial_frame(cutout_wcs)
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
                    [x.ra.value, x.dec.value] for x in e_world_points
                ])

                ax1.plot(
                    e_world_points_values[..., 0],
                    e_world_points_values[..., 1],
                    color='#0000ff',
                    ls='-',
                    lw=0.8,
                    alpha=0.7,
                    transform=ax1.get_transform(
                        ax1.get_transform(
                            apwcs.utils.wcs_to_celestial_frame(cutout_wcs)
                        )
                    )
                )
                ax1.plot(
                    e_world_points_values[..., 0],
                    e_world_points_values[..., 1],
                    color='#00ff00',
                    ls='--',
                    lw=0.8,
                    alpha=0.7,
                    transform=ax1.get_transform(
                        ax1.get_transform(
                            apwcs.utils.wcs_to_celestial_frame(cutout_wcs)
                        )
                    )
                )
    else:
        ax1 = None
        ax2 = fig.add_subplot(gs[:, -1])

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
        # Plotting absorption lines
        absorption_lines = get_lines(
            line_type='A', wrange=wavelengths, z=lines_z
        )
        for line_lam, line_name, line_type in absorption_lines:
            ax0.axvline(
                line_lam, color='green', ls='--', lw=0.7, alpha=0.5,
                label='absorption lines'
            )
            ax0.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax0.get_xaxis_transform(),
                zorder=99
            )

        # Plotting emission lines
        emission_lines = get_lines(
            line_type='E', wrange=wavelengths, z=lines_z
        )
        for line_lam, line_name, line_type in emission_lines:
            ax0.axvline(
                line_lam, color='red', ls='--', lw=0.7, alpha=0.75,
                label='emission lines',
                zorder=2
            )
            ax0.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax0.get_xaxis_transform(),
                zorder=99
            )

        # Plotting emission/absorption lines
        emission_lines = get_lines(
            line_type='AE', wrange=wavelengths, z=lines_z
        )
        for line_lam, line_name, line_type in emission_lines:
            ax0.axvline(
                line_lam, color='yellow', ls='--', lw=0.5, alpha=0.9,
                label='emission/absorption lines',
                zorder=3
            )
            ax0.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax0.get_xaxis_transform(),
                zorder=99
            )

    # Draw missing data or invalid data regions
    if lam_mask is not None:
        for lam_inter in lam_mask:
            rect = patches.Rectangle(
                (lam_inter[0], 0),
                lam_inter[1] - lam_inter[0], 1,
                transform=ax0.get_xaxis_transform(),
                linewidth=1,
                fill=True,
                edgecolor='black',
                facecolor='white',
                zorder=10
            )
            rect = patches.Rectangle(
                (lam_inter[0], 0),
                lam_inter[1] - lam_inter[0], 1,
                transform=ax0.get_xaxis_transform(),
                linewidth=1,
                fill=True,
                edgecolor='black',
                facecolor='white',
                hatch='///',
                zorder=11
            )
            ax0.add_patch(rect)
            if ((lam_inter[1] > w_min + 100) and (lam_inter[0] < w_max - 100)):
                ax0.text(
                    (lam_inter[0] + lam_inter[1]) / 2, 0.5,
                    "MISSING DATA",
                    transform=ax0.get_xaxis_transform(),
                    va='center',
                    ha='center',
                    rotation=90,
                    bbox={
                        'facecolor': 'white',
                        'edgecolor': 'black',
                        'boxstyle': 'round,pad=0.5',
                    },
                    zorder=12
                )

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

    cell_text = [
        [f'{key}', f"{val}"] for key, val in extra_info.items()
    ]

    ax2.axis("off")
    tbl = ax2.table(
        cellText=cell_text,
        colWidths=[0.4, 0.6],
        loc='upper center'
    )
    tbl.scale(1, 1.5)

    return fig, [ax0, ax1, ax2]


def plot_zfit_check(target, zbest, plot_template=None, restframe=True,
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
        The units of the wavelenght grid. The default value id 'Angstrom'
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

    fig, axs = plot_spectrum(
        target.spectra[0].wave,
        target.spectra[0].flux,
        nan_mask=target.lam_mask,
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
                    target.spectra[0].wave,
                    0 if restframe else t_best_data['Z'],
                )

                axs[0].plot(
                    target.spectra[0].wave, template_flux,
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


def stack(data, wave_mask=None):
    """
    Stack the spectral cube along wavelength axis.

    Parameters
    ----------
    data : numpy.ndarray
        The spectral datacube.
    wave_mask : 1D np.ndarray, optional
        Optional wavelength mask. Wavelenght corresponding to a False will not
        be used in the stacking. The default is None.

    Returns
    -------
    new_data : numpy.ndarray
        The stacked datacube.

    """
    img_height, img_width = data.shape[1], data.shape[2]
    new_data = np.zeros((img_height, img_width))
    for k, dat in enumerate(data):
        progress = (k + 1) / len(data)
        sys.stderr.write(
            f"\rstacking cube: {get_pbar(progress)} {progress:.2%}\r"
        )
        sys.stderr.flush()
        if wave_mask is None or wave_mask[k]:
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


def get_spectrum_snr(flux, var=None, smoothing_window=51, smoothing_order=11):
    """
    Compute the SRN of a spectrum.

    Parameters
    ----------
    flux : numpy.ndarray
        The spectrum itself.
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
        'data' : numpy.ndarray
            the rotated data
        'wcs' : astropy.wcs.WCS or None
            the updated WCS for the rotated data
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

    rotated_data = rotate(
        data,
        angle.to(apu.deg).value,
        reshape=False
    )

    return {'data': rotated_data, 'wcs': new_wcs}
