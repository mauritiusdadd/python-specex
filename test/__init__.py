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
from typing import Optional
from collections.abc import Iterable
from urllib import request
import pathlib

from rich.progress import Progress
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")

DOWNLOAD_HEADERS = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36',
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate"
}

def download_files(
    url_list: Iterable[str],
    out_dir: str = TEST_DATA_PATH,
    use_cached: bool = True,
    report_interval: int = 1
) -> list[str]:
    """
    Download a set of files with a progressbar.

    :param url_list:
    :param out_dir:
    :param use_cached:
    :param report_interval:
    :return:
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    outfile_list = []
    with Progress() as progress:
        for target_url in url_list:
            target_file = os.path.basename(target_url)
            out_file = os.path.join(out_dir, target_file)
            if not (use_cached and os.path.isfile(out_file)):
                task_id = progress.add_task(
                    f"[yellow]Downloading {target_file}...",
                    total=1000
                )

                def report_hook(
                        block_num: int = 1,
                        block_size: int = 1,
                        total_size: Optional[int] = None) -> None:
                    if total_size is not None:
                        progress.update(task_id, total=total_size)

                    progress.update(task_id, completed=block_num * block_size)

                try:
                    opener = request.build_opener()
                    opener.addheaders = list(DOWNLOAD_HEADERS.items())
                    request.install_opener(opener)
                    package_out_file, headers = request.urlretrieve(
                        target_url,
                        out_file,
                        reporthook=report_hook
                    )
                except Exception as exc:
                    print(
                        "An exception occurred while downloading "
                        f"{target_file}: {str(exc)}",
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
    return download_files(
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

    return download_files(
        HST_TEST_FITS,
        out_dir=out_dir,
        use_cached=use_cached,
        report_interval=1
    )
