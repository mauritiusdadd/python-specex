#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

redrock wrapper tools for python-specex spectra.
This program is based on the structure of redrock.external.boss function.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import traceback

import argparse

import numpy as np
from scipy import sparse

from astropy.io import fits
from astropy.table import Table, join
import astropy.wcs as wcs
import matplotlib.pyplot as plt

from redrock.utils import elapsed, get_mp
from redrock.targets import Spectrum, Target, DistTargetsCopy
from redrock.templates import load_dist_templates, find_templates, Template
from redrock.results import write_zscan
from redrock.zfind import zfind
from redrock._version import __version__
from redrock.archetypes import All_archetypes

from .utils import plot_zfit_check, get_mask_intervals, plot_scandata


def get_templates(template_types=[], filepath=False, templates=None):
    """
    Get avilable templates.

    Parameters
    ----------
    template_types : list of str, optional
        List of template types to retrieve. If it's empty all available
        templates will be returned.
        The default is [].
    filepath : boot, optional
        If it's true then return the file paths instead of actual templates.
    templates : str, optional
        The path of a template file or of a directory containing templates
        files. If None, templates are searched in the default redrock path.
        The default value is None.

    Returns
    -------
    available_templates : list of redrock.templates.Template or file paths
        The available templates or the corresponding file paths.

    """
    if templates is not None and os.path.isfile(templates):
        return [Template(templates), ]

    available_templates = []
    for t in find_templates(templates):
        templ = Template(t)
        if not template_types or templ.template_type in template_types:
            if filepath:
                available_templates.append(t)
            else:
                available_templates.append(templ)

    return available_templates


def get_template_types():
    """
    Get the available types of templates.

    Returns
    -------
    types : list of str
        List of types of available templates.

    """
    templates = [
        t.template_type
        for t in get_templates()
    ]
    types = set(templates)
    return types


def write_zbest(outfile, zbest, template_version, archetype_version):
    """
    Write zbest Table to outfile.

    Parameters
    ----------
        outfile : str
            The output file path.
        zbest : astropy.table.Table
            The output redshift fitting results.

    """
    header = fits.Header()
    header['RRVER'] = (__version__, 'Redrock version')

    for i, fulltype in enumerate(template_version.keys()):
        header['TEMNAM'+str(i).zfill(2)] = fulltype
        header['TEMVER'+str(i).zfill(2)] = template_version[fulltype]

    if archetype_version is not None:
        for i, fulltype in enumerate(archetype_version.keys()):
            header['ARCNAM'+str(i).zfill(2)] = fulltype
            header['ARCVER'+str(i).zfill(2)] = archetype_version[fulltype]

    zbest.meta['EXTNAME'] = 'ZBEST'

    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU(header=header))
    hx.append(fits.convenience.table_to_hdu(zbest))
    hx.writeto(os.path.expandvars(outfile), overwrite=True)
    return


def read_spectra(spectra_fits_list, spec_hdu=None, var_hdu=None, wd_hdu=None):
    """
    Read input spectra fits files.

    Parameters
    ----------
    spectra_fits_list : list
        List of fits files containing the input spectra.
    spec_hdu : int or None, optional
        The index of the HDU that contains the spectral data itself.
        If it is None, then the index is determined automatically by the name
        of the HDU. If this operation fails ValueError exception is raised.
        The default value is None.
    var_hdu : int or None, optional
        The index of the HDU that contains the  variance of the spectral data.
        If it is None, then the index is determined automatically by the name
        of the HDU. If this operation fails ValueError exception is raised.
        The default value is None.
    wd_hdu : int or None, optional
        The index of the HDU that contains the wavelength.
        If it is None, then the index is determined automatically by the name
        of the HDU. If this operation fails, no wavelenght dispersion will be
        used and the spectra will be considered having a uniform resolution.
        The default value is None.

    Raises
    ------
    ValueError
        If cannot automatically determine the HDU containing the specral data
        or its variance.

    Returns
    -------
    targets : list of redrock.targets.Target
        The target spectra for which redshift will be computed.
    metatable : astropy.table.Table
        A table containing metadata.

    """
    targets = []
    targetids = []
    specids = []
    sn_vals = []
    sn_var_vals = []
    sn_em_vals = []
    for j, fits_file in enumerate(spectra_fits_list):
        hdul = fits.open(fits_file)

        valid_id_keys = [
            f"{i}{j}"
            for i in ['', 'OBJ', 'OBJ_', 'TARGET', 'TARGET_']
            for j in ['ID', 'NUMBER', 'UID', 'UUID']
        ]

        target_id = f"{j:09}"
        spec_id = target_id
        for hdu in hdul:
            for key in valid_id_keys:
                try:
                    spec_id = hdu.header[key]
                except KeyError:
                    continue
                else:
                    break

        if spec_hdu is None:
            for hdu in hdul:
                if hdu.name.lower() in ['spec', 'spectrum', 'data']:
                    flux = hdu.data
                    spec_wcs = wcs.WCS(hdu.header)
                    break
            else:
                raise ValueError(
                    "Cannot determine the HDU containing spectral data: "
                    f"'{fits_file}'"
                )
        else:
            flux = hdul[spec_hdu].data
            spec_wcs = wcs.WCS(hdul[spec_hdu].header)

        for hdu in hdul:
            if hdu.name.lower() in ['mask', 'nanmask', 'nan_mask', 'nans']:
                nanmask = hdu.data.astype(bool)
                break
        else:
            nanmask = None

        if var_hdu is None:
            for hdu in hdul:
                if hdu.name.lower() in ['var', 'variance', 'stat']:
                    ivar = 1 / hdu.data
                    break
                elif hdu.name.lower in ['ivar', 'ivariance']:
                    ivar = hdu.data
                    break
            else:
                print(
                    "WARNING: Cannot determine the HDU containing variance "
                    f"data in '{fits_file}'! Using dumb constan variance...",
                )
                ivar = np.ones_like(flux)
        else:
            ivar = 1 / hdul[var_hdu].data

        if wd_hdu is None:
            for hdu in hdul:
                if hdu.name.lower() in ['wd', 'dispersion', 'resolution']:
                    wd = hdu.data
                    break
            else:
                wd = None
        else:
            wd = hdul[wd_hdu].data

        if flux.shape != ivar.shape:
            raise ValueError(
                f"'{fits_file}' - "
                "Spectral data invalid or corruptede: Flux data shape "
                "do not match variance data one!"
            )

        main_header = hdul[0].header

        # NOTE: Wavelenghts must be in Angstrom units
        pixel = np.arange(len(flux))
        lam = spec_wcs.pixel_to_world(pixel).Angstrom

        flux = flux.astype('float32')

        if nanmask is None:
            flux_not_nan_mask = ~np.isnan(flux)
        else:
            flux_not_nan_mask = ~nanmask

        lam_mask = np.array([
            (lam[m_start], lam[m_end])
            for m_start, m_end in get_mask_intervals(~flux_not_nan_mask)
        ])

        flux = flux[flux_not_nan_mask]
        ivar = ivar[flux_not_nan_mask]
        lam = lam[flux_not_nan_mask]
        if wd is not None:
            wd = wd[flux_not_nan_mask]

        # If now wavelenght dispersion information is present, then fallback
        # to a uniform resolution model (i.e. an identiry matrix)
        if wd is None:
            R = np.eye(lam.shape[0])
            R = sparse.dia_matrix(R)

        try:
            s_n = main_header['SN']
        except KeyError:
            s_n = -1

        try:
            s_n_var = main_header['SN_VAR']
        except KeyError:
            s_n_var = -1

        try:
            s_n_em = main_header['SN_EMISS']
        except KeyError:
            s_n_em = -1

        rrspec = Spectrum(lam, flux, ivar, R, None)
        target = Target(target_id, [rrspec])
        target.input_file = fits_file
        target.lam_mask = flux_not_nan_mask
        target.spec_id = spec_id
        targets.append(target)
        targetids.append(target_id)
        specids.append(spec_id)
        sn_vals.append(s_n)
        sn_var_vals.append(s_n_var)
        sn_em_vals.append(s_n_em)

    metatable = Table()
    metatable['TARGETID'] = targetids
    metatable['SPECID'] = specids
    metatable['SN'] = sn_vals
    metatable['SN_VAR'] = sn_var_vals
    metatable['SN_EMISS'] = sn_em_vals

    return targets, metatable


def __argshandler(options=None):
    """
    Handle input arguments.

    Parameters
    ----------
    options : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    args : TYPE
        DESCRIPTION.

    """
    parser = argparse.ArgumentParser(
        description="Estimate redshifts for spectra extracted with"
        "python-specex using redrock interface."
    )

    parser.add_argument(
        "spectra", metavar='SPECTRA_FITS', type=str, nargs='+',
        help="input spectra fits files"
    )

    parser.add_argument(
        "-t", "--templates", type=str, default=None, required=False,
        help="template file or directory"
    )

    parser.add_argument(
        "--archetypes", type=str, default=None, required=False,
        help="archetype file or directory for final redshift comparisons"
    )

    parser.add_argument(
        "-o", "--output", type=str, default=None, required=False,
        help="output file"
    )

    parser.add_argument(
        "--zbest", type=str, default=None, required=False,
        help="output zbest FITS file"
    )

    parser.add_argument(
        "--priors", type=str, default=None, required=False,
        help="optional redshift prior file"
    )

    parser.add_argument(
        "--chi2-scan", type=str, default=None, required=False,
        help="Load the chi2-scan from the input file"
    )

    parser.add_argument(
        "--nminima", type=int, default=3, required=False,
        help="the number of redshift minima to search"
    )

    parser.add_argument(
        "--mp", type=int, default=0, required=False,
        help="if not using MPI, the number of multiprocessing processes to use"
        " (defaults to half of the hardware threads)"
    )

    parser.add_argument(
        "--debug", default=False, action="store_true", required=False,
        help="debug with ipython (only if communicator has a single process)"
    )

    parser.add_argument(
        "--plot-zfit", action="store_true", default=False, required=False,
        help="Generate plots of the spectra with infomrazion about the "
        "result of the template fitting (i.e. the best redshift, the position "
        "of most important lines, the best matching template, etc...)."
    )

    parser.add_argument(
        "--checkimg-outdir", type=str, default='checkimages', required=False,
        help='Set the directory where check images are saved (when they are '
        'enabled thorugh the appropriate parameter).'
    )

    parser.add_argument(
        "--quite", action='store_true', default=False,
        help="Reduce program output at bare minimum. Do not print fitting "
        "result,"
    )

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def rrspecex(options=None, comm=None):
    """
    Estimate redshifts for spectra extracted with python-specex using redrock.

    This loads targets serially and copies them into a DistTargets class.
    It then runs redshift fitting and writes the output to a catalog.

    Parameters
    ----------
    options : list, optional
        lLst of commandline options to parse. The default is None.
    comm : mpi4py.Comm, optional
        MPI communicator to use. The default is None.

    Returns
    -------
    targets : list of redrock.targets.Target objects
        list of target spectra.
    zfit : astropy.table.Table
        Table containing the fit results.
    scandata : dict
        A dictionary containing the redshift scanning information for each
        target
    """
    global_start = elapsed(None, "", comm=comm)
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.size
        comm_rank = comm.rank

    args = __argshandler(options)

    # Check arguments - all processes have this, so just check on the first
    # process
    if comm_rank == 0:
        if args.debug and comm_size != 1:
            print(
                "--debug can only be used if the communicator has one process"
            )
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

        if (args.output is None) and (args.zbest is None):
            print("--output or --zbest required")
            sys.stdout.flush()
            if comm is not None:
                comm.Abort()

    # Multiprocessing processes to use if MPI is disabled.
    mpprocs = 0
    if comm is None:
        mpprocs = get_mp(args.mp)
        print("Running with {} processes".format(mpprocs))
        if "OMP_NUM_THREADS" in os.environ:
            nthread = int(os.environ["OMP_NUM_THREADS"])
            if nthread != 1:
                print("WARNING:  {} multiprocesses running, each with "
                      "{} threads ({} total)".format(
                          mpprocs, nthread, mpprocs*nthread
                      ))
                print("WARNING:  Please ensure this is <= the number of "
                      "physical cores on the system")
        else:
            print("WARNING:  using multiprocessing, but the OMP_NUM_THREADS")
            print("WARNING:  environment variable is not set- your system may")
            print("WARNING:  be oversubscribed.")
        sys.stdout.flush()
    elif comm_rank == 0:
        print("Running with {} processes".format(comm_size))
        sys.stdout.flush()

    try:
        # Load and distribute the targets
        if comm_rank == 0:
            print("Loading targets...")
            sys.stdout.flush()

        start = elapsed(None, "", comm=comm)

        # Read the spectra on the root process.  Currently the "meta" Table
        # returned here is not propagated to the output zbest file.  However,
        # that could be changed to work like the DESI write_zbest() function.
        # Each target contains metadata which is propagated to the output zbest
        # table though.
        targets, meta = read_spectra(args.spectra)

        _ = elapsed(
            start, "Read of {} targets".format(len(targets)), comm=comm
        )

        # Distribute the targets.

        start = elapsed(None, "", comm=comm)

        dtargets = DistTargetsCopy(targets, comm=comm, root=0)

        # Get the dictionary of wavelength grids
        dwave = dtargets.wavegrids()

        _ = elapsed(
            start,
            "Distribution of {} targets".format(len(dtargets.all_target_ids)),
            comm=comm
        )

        # Read the template data

        dtemplates = load_dist_templates(
            dwave,
            templates=args.templates,
            comm=comm,
            mp_procs=mpprocs
        )

        # Compute the redshifts, including both the coarse scan and the
        # refinement.  This function only returns data on the rank 0 process.

        start = elapsed(None, "", comm=comm)

        scandata, zfit = zfind(
            dtargets,
            dtemplates,
            mpprocs,
            nminima=args.nminima,
            archetypes=args.archetypes,
            priors=args.priors,
            chi2_scan=args.chi2_scan
        )

        _ = elapsed(start, "Computing redshifts took", comm=comm)

        # Change to upper case like DESI
        for colname in zfit.colnames:
            if colname.islower():
                zfit.rename_column(colname, colname.upper())

        zfit = join(zfit, meta, keys=['TARGETID'], join_type='left')
        zfit.remove_column('TARGETID')

        if args.plot_zfit:
            if comm_rank == 0:
                available_templates = get_templates(
                    templates=args.templates
                )
                for target in targets:
                    fig, ax = plot_zfit_check(
                        target,
                        zfit,
                        plot_template=available_templates
                    )
                    figname = f'rrspecex_{target.id}.png'
                    if args.checkimg_outdir is not None:
                        figname = os.path.join(args.checkimg_outdir, figname)
                    fig.savefig(figname, dpi=150)
                    plt.close(fig)

                    if args.debug:
                        figname = f'rrspecex_scandata_{target.id}.png'
                        figname = os.path.join(args.checkimg_outdir, figname)
                        fig, axs = plot_scandata(target, scandata)
                        fig.savefig(figname, dpi=150)
                        plt.close(fig)

        # Write the outputs
        if args.output is not None:
            start = elapsed(None, "", comm=comm)
            if comm_rank == 0:
                write_zscan(args.output, scandata, zfit, clobber=True)
            _ = elapsed(start, "Writing zscan data took", comm=comm)

        zbest = None
        if comm_rank == 0:
            zbest = zfit[zfit['ZNUM'] == 0]

        if args.zbest:
            start = elapsed(None, "", comm=comm)
            if comm_rank == 0:

                # Remove extra columns not needed for zbest
                # zbest.remove_columns(['zz', 'zzchi2', 'znum'])
                # zbest.remove_columns(['ZNUM'])

                template_version = {
                    t._template.full_type: t._template._version
                    for t in dtemplates
                }

                archetype_version = None
                if args.archetypes is not None:
                    archetypes = All_archetypes(
                        archetypes_dir=args.archetypes
                    ).archetypes
                    archetype_version = {
                        name: arch._version
                        for name, arch in archetypes.items()
                    }

                write_zbest(
                    args.zbest, zbest, template_version, archetype_version
                )

            _ = elapsed(start, "Writing zbest data took", comm=comm)

        if comm_rank == 0:
            if (not args.zbest) and (args.output is None) and (not args.quite):
                print("")
                print(zbest)
                print("")

    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [
            "Proc {}: {}".format(comm_rank, x)
            for x in lines
        ]
        print("".join(lines))
        sys.stdout.flush()
        if comm is not None:
            comm.Abort()

    _ = elapsed(global_start, "Total run time", comm=comm)

    if args.debug:
        import IPython
        IPython.embed()

    return targets, zbest, scandata


if __name__ == '__main__':
    _ = rrspecex()
