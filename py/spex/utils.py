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


def plot_zfit_check(target, zfit, plot_template=None, cutout=None):
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

    Returns
    -------
    None.

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

    for target_spec in target.spectra:
        w_min = np.minimum(w_min, np.nanmin(target_spec.wave))
        w_max = np.maximum(w_max, np.nanmax(target_spec.wave))

        f_min = np.minimum(f_min, np.nanmin(target_spec.flux))
        f_max = np.maximum(f_max, np.nanmax(target_spec.flux))

        ax.plot(
            target_spec.wave, target_spec.flux,
            ls='-',
            lw=2,
            alpha=1,
            label='spectrum'
        )

        if plot_template and best_template:
            template_flux = best_template.eval(
                t_best_data['coeff'],
                target_spec.wave,
                t_best_data['z'],
            )

            ax.plot(
                target_spec.wave, template_flux,
                ls='-',
                lw=2,
                alpha=0.7,
                label=f'best template [{best_template.full_type}]'
            )

        # Plotting absorption lines lines
        lines_to_plot = getlines(
            line_type='A', wrange=target_spec.wave, z=t_best_data['z']
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
        c_w = 0.1
        c_x = 0.85
        c_y = 0.1
        c_h = c_w * getaspect(ax)
        ax.imshow(
            cutout,
            extent=(c_x, c_x + c_w, c_y, c_y + c_h),
            origin='lower',
            transform=ax.transAxes,
            aspect='auto'
        )

    _ = ax.legend(loc='upper left')
    plt.tight_layout()
    return fig, ax
