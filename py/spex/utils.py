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

import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D

from .lines import getlines


def getaspect(ax):
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


def getvclip(img, vclip=0.5):
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
    vmin = np.nanmedian(img) - vclip*np.nanstd(img)
    vmax = np.nanmedian(img) + vclip*np.nanstd(img)
    return vmin, vmax


def getlogimg(img, vclip=0.5):
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
    return log_img, *getvclip(log_img)


def getcutout(data, position, cutout_size, wcs=None, wave_ranges=None,
              vmin=None, vmax=None):
    valid_shape = True
    is_rgb = True

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
            data_r, position, cutout_size, wcs=wcs.celestial, copy=True
        )
        cutout_g = Cutout2D(
            data_g, position, cutout_size, wcs=wcs.celestial, copy=True
        )
        cutout_b = Cutout2D(
            data_b, position, cutout_size, wcs=wcs.celestial, copy=True
        )
        cutout = np.array([
            cutout_r.data,
            cutout_g.data,
            cutout_b.data
        ]).transpose(1, 2, 0)

        if vmin is not None:
            cutout[cutout < vmin] = vmin

        if vmax is not None:
            cutout[cutout > vmax] = vmax

        cutout -= np.nanmin(cutout)
        cutout /= np.nanmax(cutout)
        print(vmin, vmax)
    else:
        cutout = Cutout2D(
            data_gray, position, cutout_size, wcs=wcs.celestial, copy=True
        ).data

    return cutout


def plot_zfit_check(target, zfit, plot_template=None, rest_frame=True,
                    cutout=None):
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
        If None, plot the grayscale or RGB image given as numpy.ndarray.

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
    zbest = zfit[zfit['znum'] == 0]

    t_best_data = zbest[zbest['targetid'] == target.id][0]

    best_template = None
    if plot_template:
        for t in plot_template:
            if (
                t.template_type == t_best_data['spectype'] and
                t.sub_type == t_best_data['subtype']
            ):
                best_template = t
                break

    w_min = 1.0e5
    w_max = 0.0

    f_min = 1.0e5
    f_max = 0.0

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_title(f"object: {target.id}")
    ax.set_aspect('auto')

    axs = [ax, ]

    for target_spec in target.spectra:
        # If we plot the spectrum at restframe wavelenghts, then we must use
        # rest frame wavelengths also for the lines.
        if rest_frame:
            wavelenghts = target_spec.wave / (1 + t_best_data['z'])
            lines_z = 0
        else:
            wavelenghts = target_spec.wave
            lines_z = t_best_data['z']

        w_min = np.minimum(w_min, np.nanmin(wavelenghts))
        w_max = np.maximum(w_max, np.nanmax(wavelenghts))

        f_min = np.minimum(f_min, np.nanmin(target_spec.flux))
        f_max = np.maximum(f_max, np.nanmax(target_spec.flux))

        ax.plot(
            wavelenghts, target_spec.flux,
            ls='-',
            lw=2,
            alpha=1,
            label='spectrum'
        )

        if plot_template and best_template:
            template_flux = best_template.eval(
                t_best_data['coeff'],
                wavelenghts,
                lines_z,
            )

            ax.plot(
                wavelenghts, template_flux,
                ls='-',
                lw=2,
                alpha=0.7,
                label=f'best template [{best_template.full_type}]'
            )

        # Plotting absorption lines lines
        lines_to_plot = getlines(
            line_type='A', wrange=wavelenghts, z=lines_z
        )
        for line_lam, line_name, line_type in lines_to_plot:
            ax.axvline(
                line_lam, color='green', ls='--', lw=1, alpha=0.7
            )
            ax.text(
                line_lam, 0.02, line_name, rotation=90,
                transform=ax.get_xaxis_transform(),
            )

    ax.set_xlim(w_min, w_max)
    ax.set_ylim(f_min, f_max)

    if cutout is not None:
        # Place the image in the upper-right corner of the figure
        # ---------------------------------------------------------------------
        # We're specifying the position and size in _figure_ coordinates, so
        # the image will shrink/grow as the figure is resized.
        # Remove "zorder=-1" to place the image in front of the axes.
        #
        # See: https://stackoverflow.com/questions/3609585/how-to-insert-a-small-image-on-the-corner-of-a-plot-with-matplotlib
        figW, figH = ax.get_figure().get_size_inches()
        fig_ratio = figH / figW

        cut_ax = fig.add_axes(
            [0.85, 0.15, 0.1, 0.1 / fig_ratio],
            anchor='SE',
            zorder=5
        )
        cut_ax.imshow(
            cutout,
            origin='lower',
            aspect='auto',
        )
        cut_ax.get_xaxis().set_visible(False)
        cut_ax.get_yaxis().set_visible(False)
        axs.append(cut_ax)
    else:
        axs.append(None)

    _ = ax.legend(loc='upper left')
    plt.tight_layout()
    return fig, axs
