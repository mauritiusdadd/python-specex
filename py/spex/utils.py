#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

This module provides utility functions used by other spex modules.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import patches
from scipy.signal import savgol_filter
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval

from .lines import get_lines


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


def get_cutout(data, position, cutout_size, wcs=None, wave_ranges=None,
               vmin=None, vmax=None):
    valid_shape = True
    is_rgb = True

    if wcs is not None:
        wcs = wcs.celestial

    if len(data.shape) > 3:
        valid_shape = False
    elif len(data.shape) == 3:
        if data.shape[2] == 1:
            # Weired grayscale
            data_gray = data[..., 0]
            is_rgb = False
        elif data.shape[2] == 2:
            # Who knows what is this?
            raise ValueError(
                "Only graiscale, RGB and spectral cube are supported!"
            )
        elif data.shape[2] == 3:
            # RGB image
            data_r = data[..., 0]
            data_g = data[..., 1]
            data_b = data[..., 2]
        else:
            # Probably a spectral datacube
            if wave_ranges is None:
                wave_indexes = np.arange(data.shape[2])
                w_mask_r = wave_indexes > (data.shape[2] * 0.66)
                w_mask_b = wave_indexes < (data.shape[2] * 0.33)
                w_mask_g = ~(w_mask_r | w_mask_b)
                data_r = np.nansum(data[..., w_mask_r], axis=-1)
                data_g = np.nansum(data[..., w_mask_g], axis=-1)
                data_b = np.nansum(data[..., w_mask_b], axis=-1)
    elif len(data.shape) == 2:
        is_rgb = False
        data_gray = data
    else:
        valid_shape = False

    if not valid_shape:
        raise ValueError(
            "Only array with 2 or 3 dimensions are supporter, but input data"
            f" has {len(data.shape)} dimensions!"
        )

    if is_rgb:
        cutout_r = Cutout2D(
            data_r, position, cutout_size, wcs=wcs, copy=True
        )
        cutout_g = Cutout2D(
            data_g, position, cutout_size, wcs=wcs, copy=True
        )
        cutout_b = Cutout2D(
            data_b, position, cutout_size, wcs=wcs, copy=True
        )
        cutout = np.array([
            cutout_r.data,
            cutout_g.data,
            cutout_b.data
        ]).transpose(1, 2, 0)

        if vmin is not None:
            cutout[cutout <= vmin] = vmin

        if vmax is not None:
            cutout[cutout >= vmax] = vmax

        cutout -= np.nanmin(cutout)
        cutout /= np.nanmax(cutout)
        cutout_wcs = cutout_r.wcs
    else:
        cutout = Cutout2D(
            data_gray, position, cutout_size, wcs=wcs, copy=True
        )
        cutout_wcs = cutout.wcs
        cutout = cutout.data

    return cutout, cutout_wcs


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


def plot_zfit_check(target, zbest, plot_template=None, rest_frame=True,
                    cutout=None, wave_units='Angstrom', flux_units=''):
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
    cutout: 2D or 3D numpt.ndarray or None, optional
        If not None, plot the grayscale or RGB image given as numpy.ndarray.
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

    t_best_data = zbest[zbest['TARGETID'] == target.id][0]

    best_template = None
    if plot_template:
        for t in plot_template:
            if (
                t.template_type == t_best_data['SPECTYPE'] and
                t.sub_type == t_best_data['SUBTYPE']
            ):
                best_template = t
                break

    w_min = 1.0e5
    w_max = 0.0

    f_min = 1.0e5
    f_max = 0.0

    fig = plt.figure(figsize=(15, 5))

    gs = GridSpec(2, 6, figure=fig)

    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[0, -1])
    ax2 = fig.add_subplot(gs[1, -1])

    ax0.set_title(f"object: {target.id}")
    ax0.set_aspect('auto')

    try:
        lam_mask = target.lam_mask
    except AttributeError:
        lam_mask = None

    for target_spec in target.spectra:
        # If we plot the spectrum at restframe wavelenghts, then we must use
        # rest frame wavelengths also for the lines.

        if rest_frame:
            wavelenghts = target_spec.wave / (1 + t_best_data['Z'])
            if lam_mask is not None:
                lam_mask = lam_mask / (1 + t_best_data['Z'])
            lines_z = 0
        else:
            wavelenghts = target_spec.wave
            lines_z = t_best_data['z']

        w_min = np.minimum(w_min, np.nanmin(wavelenghts))
        w_max = np.maximum(w_max, np.nanmax(wavelenghts))

        f_min = np.minimum(f_min, np.nanmin(target_spec.flux))
        f_max = np.maximum(f_max, np.nanmax(target_spec.flux))

        ax0.plot(
            wavelenghts, target_spec.flux,
            ls='-',
            lw=2,
            alpha=1,
            label='spectrum'
        )

        ax0.plot(
            wavelenghts, savgol_filter(target_spec.flux, 51, 11),
            ls='-',
            lw=1,
            alpha=0.8,
            label='smoothed spectrum'
        )

        if plot_template and best_template:
            try:
                coeffs = t_best_data['COEFF'][:best_template.nbasis]
                template_flux = best_template.eval(
                    coeffs,
                    wavelenghts,
                    lines_z,
                )

                ax0.plot(
                    wavelenghts, template_flux,
                    ls='-',
                    lw=1,
                    alpha=0.7,
                    c='red',
                    label=f'best template [{best_template.full_type}]'
                )
            except AssertionError:
                print(
                    f"Template warning for object {target.id}\n"
                    f"  nbasis: {best_template.nbasis}\n"
                    f"  coeffs: {len(coeffs)}",
                    file=sys.stderr
                )

        # Plotting absorption lines
        absorption_lines = get_lines(
            line_type='A', wrange=wavelenghts, z=lines_z
        )
        for line_lam, line_name, line_type in absorption_lines:
            ax0.axvline(
                line_lam, color='green', ls='--', lw=1, alpha=0.5,
                label='absorption lines'
            )
            ax0.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax0.get_xaxis_transform(),
            )

        # Plotting emission lines
        emission_lines = get_lines(
            line_type='E', wrange=wavelenghts, z=lines_z
        )
        for line_lam, line_name, line_type in emission_lines:
            ax0.axvline(
                line_lam, color='red', ls='--', lw=1, alpha=0.5,
                label='emission lines'
            )
            ax0.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax0.get_xaxis_transform(),
            )

        # Plotting emission/absorption lines
        emission_lines = get_lines(
            line_type='AE', wrange=wavelenghts, z=lines_z
        )
        for line_lam, line_name, line_type in emission_lines:
            ax0.axvline(
                line_lam, color='yellow', ls='--', lw=1, alpha=0.5,
            )
            ax0.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax0.get_xaxis_transform(),
            )

    ax0.set_xlim(w_min, w_max)
    ax0.set_ylim(f_min, f_max)
    ax0.set_xlabel(f'Wavelenght [{wave_units}]')
    ax0.set_ylabel(f'Wavelenght [{flux_units}]')

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
                hatch='///'
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
                    }
                )

    if cutout is not None:
        ax1.imshow(
            cutout,
            origin='lower',
            aspect='auto',
        )
    else:
        ax1.text(
            0.5, 0.25,
            "NO IMAGE",
            ha='center',
            va='center',
            transform=ax1.transAxes,
            bbox={
                'facecolor': 'white',
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.5',
            }
        )
    ax1.axis('off')
    ax2.axis('off')

    splabel = f"z: {t_best_data['Z']:.4f} ± {t_best_data['ZERR']:.2e}\n"
    splabel += f"best template: {t_best_data['SPECTYPE']} "
    splabel += f"{t_best_data['SUBTYPE']}\n" if t_best_data['SUBTYPE'] else'\n'
    try:
        splabel += f"SN ratio: {t_best_data['SN']:.2f}\n"
    except KeyError:
        pass
    splabel += f"z-fit warn: {t_best_data['ZWARN']}"

    ax2.text(
        0, 0.9,
        splabel,
        ha='left',
        va='top',
        transform=ax2.transAxes,
        bbox={
            'facecolor': 'white',
            'edgecolor': 'none',
        }
    )

    handles, labels = ax0.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    _ = ax0.legend(newHandles, newLabels, loc='upper left')
    plt.tight_layout()
    return fig, [ax0, ax1, ax2]


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
