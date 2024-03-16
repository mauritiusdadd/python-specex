"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes and find their redshift.

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

import os
import sys
import time
from urllib import request
import pathlib

from specex.utils import get_pbar

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")


def downloadFiles(url_list: tuple, out_dir: str = TEST_DATA_PATH,
                  use_cached: bool = True, report_interval=1):
    """
    Download a set of files with a progressbar.

    Parameters
    ----------
    url_list : tuple
        DESCRIPTION.
    out_dir : str, optional
        DESCRIPTION. The default is TEST_DATA_PATH.
    use_cached : bool, optional
        DESCRIPTION. The default is True.
    report_interval : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    outfile_list : TYPE
        DESCRIPTION.

    """

    def report_pbar(blocks_count, block_size, total_size):
        if (time.perf_counter() - report_pbar.last_time) > report_interval:
            report_pbar.last_time = time.perf_counter()
            downloaded_size = blocks_count * block_size
            progress = downloaded_size / total_size
            pbar = get_pbar(progress)
            report_str = f"\r{pbar} {progress: 6.2%}  "
            report_str += f"{downloaded_size}/{total_size} Bytes\r"
            sys.stdout.write(report_str)
            sys.stdout.flush()
    report_pbar.last_time = time.perf_counter()

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    outfile_list = []
    for target_url in url_list:
        target_file = os.path.basename(target_url)
        out_file = os.path.join(out_dir, target_file)
        if not (use_cached and os.path.isfile(out_file)):
            print(f"Downloading {target_file}...")
            try:
                package_out_file, headers = request.urlretrieve(
                    target_url,
                    out_file,
                    reporthook=report_pbar
                )
            except Exception as exc:
                print(
                    f"An exception occurred while downloading {target_file}:",
                    str(exc),
                    file=sys.stderr
                )
                continue
        else:
            print(f"Using cached {target_file}...")
        outfile_list.append(os.path.realpath(out_file))
    return outfile_list


def get_muse_test_cube(out_dir: str = TEST_DATA_PATH,
                       use_cached: bool = True):
    TEST_CUBE_URL = 'https://dataportal.eso.org/dataportal_new/file//ADP.2023-09-01T12:56:41.595'
    return downloadFiles(
        [TEST_CUBE_URL,],
        out_dir=out_dir,
        use_cached=use_cached,
        report_interval=10
    )[0]


def get_hst_test_images(out_dir: str = TEST_DATA_PATH,
                        use_cached: bool = True):
    """
    Download an HST test files.

    Parameters
    ----------
    out_dir : str, optional
        The Directory whre to save the files.
        The default is test.TEST_DATA_PATH.
    use_cached : bool, optional
        If True, do not download the file if a cached version exists.
        The default is True.

    Returnsoutfile_list
    -------
    outfile : str
        The absolute path of the downloaded image.
    """
    # See: https://fits.gsfc.nasa.gov/fits_samples.html
    HST_BASE_URL = 'https://fits.gsfc.nasa.gov/samples'

    HST_TEST_FITS = [
        f'{HST_BASE_URL}/NICMOSn4hk12010_mos.fits',
        f'{HST_BASE_URL}/WFPC2ASSNu5780205bx.fits',
        f'{HST_BASE_URL}/WFPC2u5780205r_c0fx.fits',
    ]

    return downloadFiles(
        HST_TEST_FITS,
        out_dir=out_dir,
        use_cached=use_cached,
        report_interval=1
    )
