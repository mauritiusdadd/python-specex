#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
import os
import sys
from urllib import request
import unittest

from spex.zeropoints import main as zpinfo
from spex.utils import get_pbar


# See: https://fits.gsfc.nasa.gov/fits_samples.html
HST_TEST_FITS = 'https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits'


def get_hst_test_image(out_file: str = 'hst_test.fits',
                       use_cached: bool = True):
    """
    Downloads an HST test file.

    Parameters
    ----------
    out_file : str, optional
        The output file name. The default is 'hst_test.fits'.
    use_cached : bool, optional
        If True, do not download the file if a cached version exists.
        The default is True.

    Returns
    -------
    outfile : str
        The absolute path of the downloaded image.
    """
    def report_pbar(blocks_count, block_size, total_size):
        downloaded_size = blocks_count * block_size
        progress = downloaded_size / total_size
        pbar = get_pbar(progress)
        report_str = f"\r{pbar} {progress: 6.2%}  "
        report_str += f"{downloaded_size}/{total_size} Bytes\r"
        sys.stderr.write(report_str)
        sys.stderr.flush()

    print("Downloading HST test file...")
    if not (use_cached and os.path.isfile(out_file)):
        package_out_file, headers = request.urlretrieve(
            HST_TEST_FITS,
            out_file,
            reporthook=report_pbar
        )
    return os.path.realpath(out_file)


class TestZeropointInfo(unittest.TestCase):

    test_hst_img = get_hst_test_image()

    def test_zeropoint_info(self):
        zpinfo_options = [
            self.test_hst_img
        ]
        zpinfo(zpinfo_options)


if __name__ == '__main__':
    mytest = TestZeropointInfo()
    mytest.test_zeropoint_info()
