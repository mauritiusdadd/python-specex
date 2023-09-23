#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPECEX - SPECtra EXtractor.

redrock wrapper tools for python-specex spectra.
This program is based on the structure of redrock.external.boss function.

Copyright (C) 2022-2023  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import traceback
import argparse
from glob import glob

import numpy as np
from scipy import sparse

from astropy.io import fits
from astropy.table import Table, join
import astropy.wcs as wcs
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

try:
    from redrock.utils import elapsed, get_mp
    from redrock.targets import Spectrum, Target, DistTargetsCopy
    from redrock.templates import load_dist_templates, find_templates, Template
    from redrock.results import write_zscan
    from redrock.zfind import zfind
    from redrock._version import __version__
    from redrock.archetypes import All_archetypes
except (ModuleNotFoundError, Exception):
    HAS_REDROCK = False
else:
    HAS_REDROCK = True

from .exceptions import exception_handler
from .utils import (
    plot_zfit_check, plot_scandata, get_pbar
)

from .cube import (
    KNOWN_SPEC_EXT_NAMES,
    KNOWN_VARIANCE_EXT_NAMES,
    KNOWN_INVAR_EXT_NAMES,
    KNOWN_MASK_EXT_NAMES,
    KNOWN_RCURVE_EXT_NAMES
)

MUSE_RESOLUTION_ANG = 2.51


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


def read_spectra(spectra_fits_list, spec_hdu=None, var_hdu=None, wd_hdu=None,
                 memmap=True, resolution=2.51, smoothing=0, quite=False):
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
        The index of the HDU that contains the wavelength dispersion in pixels.
        If it is None, then the index is determined automatically by the name
        of the HDU. If this operation fails, no wavelenght dispersion will be
        used and the spectra will be considered having a uniform resolution.
        The default value is None.
    memmap : bool, optional
        Whether to use memmap or not. The default value is False.
    resolution : float, optional
        A fixed spectral resolution in Angrstrom to be used if no dispersion
        information is available from the spectrum file itself. The default
        value is 2.51 (the MUSE resolution).
    smoothing : int, optional
        A smooting to be applyed before redshift estimation.
        A value of 0 means no smoothing. The default value is 0.
    quite : bool, optional
        Reduce the verbosity of the output.

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
    target_file_names = []
    specids = []
    sn_vals = []
    sn_var_vals = []
    sn_em_vals = []

    # crop to templates limits
    lmin = 3500.
    lmax = 10000.

    for j, fits_file in enumerate(spectra_fits_list):
        if not quite:
            progress = (j + 1) / len(spectra_fits_list)
            sys.stderr.write(f"\r{get_pbar(progress)} {progress:.2%}\r")
            sys.stderr.flush()
        hdul = fits.open(fits_file, memmap=memmap)

        valid_id_keys = [
            f"{i}{j}"
            for i in ['', 'OBJ', 'OBJ_', 'TARGET', 'TARGET_']
            for j in ['ID', 'NUMBER', 'UID', 'UUID']
        ]

        target_id = f"{j:09}"
        target_file_names.append(fits_file)
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
                if hdu.name.lower() in KNOWN_SPEC_EXT_NAMES:
                    flux = hdu.data
                    if smoothing > 0:
                        window_size = 4*smoothing + 1
                        flux = savgol_filter(flux, window_size, 3)
                    spec_header = hdu.header
                    spec_wcs = wcs.WCS(spec_header)
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
            if hdu.name.lower() in KNOWN_MASK_EXT_NAMES:
                nanmask = hdu.data.astype(bool)
                break
        else:
            nanmask = None

        if var_hdu is None:
            for hdu in hdul:
                if hdu.name.lower() in KNOWN_VARIANCE_EXT_NAMES:
                    ivar = 1 / hdu.data
                    break
                elif hdu.name.lower() in KNOWN_INVAR_EXT_NAMES:
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
                if hdu.name.lower() in KNOWN_RCURVE_EXT_NAMES:
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
        if spec_wcs.has_spectral:
            lam = spec_wcs.pixel_to_world(pixel).Angstrom
        else:
            try:
                coeff0 = spec_header["COEFF0"]
                coeff1 = spec_header["COEFF1"]
            except KeyError:
                continue
            lam = 10**(coeff0 + coeff1*pixel)
        flux = flux.astype('float32')

        if nanmask is None:
            flux_not_nan_mask = ~np.isnan(flux)
        else:
            flux_not_nan_mask = ~nanmask

        flux = flux[flux_not_nan_mask]
        ivar = ivar[flux_not_nan_mask]
        lam = lam[flux_not_nan_mask]
        if wd is not None:
            wd = wd[flux_not_nan_mask]
        else:
            # If now wavelenght dispersion information is present, then
            # compute it using the wavelenght
            delta_lambda = np.ones_like(lam)
            delta_lambda[1:] = (lam[1:] - lam[:-1])
            wd = resolution / delta_lambda
            wd[0] = wd[1]
        wd[wd < 1e-3] = 2.

        imin = abs(lam-lmin).argmin()
        imax = abs(lam-lmax).argmin()

        lam = lam[imin:imax]
        flux = flux[imin:imax]
        ivar = ivar[imin:imax]
        wd = wd[imin:imax]

        ndiag = int(4*np.ceil(wd.max())+1)
        nbins = wd.shape[0]

        ii = np.arange(lam.shape[0])
        di = ii-ii[:, None]
        di2 = di**2

        # build resolution from wdisp
        reso = np.zeros([ndiag, nbins])

        for idiag in range(ndiag):
            offset = ndiag//2-idiag
            d = np.diagonal(di2, offset=offset)
            if offset < 0:
                reso[idiag, :len(d)] = np.exp(-d/2/wd[:len(d)]**2)
            else:
                reso[idiag, nbins-len(d):nbins] = np.exp(
                    -d/2/wd[nbins-len(d):nbins]**2
                )

        reso /= np.sum(reso, axis=0)
        offsets = ndiag//2 - np.arange(ndiag)
        nwave = reso.shape[1]
        R = sparse.dia_matrix((reso, offsets), (nwave, nwave))

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
    metatable['FILE'] = target_file_names
    metatable['SN'] = sn_vals
    metatable['SN_VAR'] = sn_var_vals
    metatable['SN_EMISS'] = sn_em_vals

    if not quite:
        print("", file=sys.stderr)

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

    parser.add_argument(
        "--no-memmap", action='store_true', default=False,
        help='Disable memmapping to speed the computation at a cost of an '
        'increased memory usage.'
    )

    parser.add_argument(
        "--resolution", type=float, default=2.51, help="A fixed resolution of "
        "the spectrograph to be used if no resolution curve is present in the "
        "input spectrum itself."
    )

    parser.add_argument(
        '--smoothing', metavar='WINDOW_SIZE', type=int,  default=0,
        help='If %(metavar)s >= 0, then the spectrua are smoothed before the '
        'redshift estimation procedure is run.  %(metavar)s = 0 means that '
        'no smoothing is applyed. If not specified, the default '
        '%(metavar)s = %(default)s is used.'
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
    global HAS_REDROCK
    global HAS_IPYTHON

    global_start = elapsed(None, "", comm=comm)
    comm_size = 1
    comm_rank = 0
    if comm is not None:
        comm_size = comm.size
        comm_rank = comm.rank

    args = __argshandler(options)

    if args.debug:
        sys.excepthook = exception_handler

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

        # Windows prompt does not expand globs, so let's do it
        spectra_list = []
        for globbed_fname in args.spectra:
            for fname in glob(globbed_fname):
                spectra_list.append(fname)

        targets, meta = read_spectra(
            spectra_list,
            memmap=not args.no_memmap,
            resolution=args.resolution,
            smoothing=args.smoothing,
            quite=args.quite
        )

        if len(targets) == 0:
            raise ValueError("No spectra were loaded!")
        else:
            print(f"Loaded {len(targets)} spectra")

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
            mp_procs=mpprocs,
            # use_gpu=True,
            # gpu_mode=True
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

        # Write the outputs
        if args.output is not None:
            start = elapsed(None, "", comm=comm)
            if comm_rank == 0:
                write_zscan(args.output, scandata, zfit, clobber=True)
            _ = elapsed(start, "Writing zscan data took", comm=comm)

        # Change to upper case like DESI
        for colname in zfit.colnames:
            if colname.islower():
                zfit.rename_column(colname, colname.upper())

        matched_zfit = join(zfit, meta, keys=['TARGETID'], join_type='left')

        zbest = None
        if comm_rank == 0:
            zbest = matched_zfit[matched_zfit['ZNUM'] == 0]
            zbest.add_index('TARGETID')

            if args.plot_zfit:

                if not os.path.isdir(args.checkimg_outdir):
                    os.makedirs(args.checkimg_outdir)

                available_templates = get_templates(
                    templates=args.templates
                )
                for target in targets:
                    orig_file_name = os.path.basename(
                        zbest.loc[target.id]['FILE']
                    )

                    fig, ax = plot_zfit_check(
                        target,
                        zbest,
                        plot_template=available_templates
                    )
                    figname = f'rrspecex_{orig_file_name}.png'
                    if args.checkimg_outdir is not None:
                        figname = os.path.join(args.checkimg_outdir, figname)
                    fig.savefig(figname, dpi=150)
                    plt.close(fig)

                    if args.debug:
                        figname = f'rrspecex_scandata_{orig_file_name}.png'
                        figname = os.path.join(args.checkimg_outdir, figname)
                        fig, axs = plot_scandata(target, scandata)
                        fig.savefig(figname, dpi=150)
                        plt.close(fig)

            zbest.remove_column('TARGETID')

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

    return targets, zbest, scandata


if __name__ == '__main__':
    _ = rrspecex()
