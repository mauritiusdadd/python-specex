#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import unittest

from specex.cube import cutout_main, smoothing_main
from specex.stack import cube_stack
from specex.sources import detect_from_cube
from specex.zeropoints import main as zpinfo
from specex.cubex import specex
from specex.plot import plot_spectra

from test import get_hst_test_images, TEST_DATA_PATH
from test import make_synt_specs, get_muse_test_cube


NICMOS_REGFILE_DATA = """
# Region file format: DS9 version 4.1
fk5
box(182.6350214,39.4064356,5.000",3.000",262.21022)
circle(182.6356792,39.4058053,2.500")
ellipse(182.6349624,39.4062180,2.500",1.500",217.21022)
"""


class MyTests(unittest.TestCase):

    test_hst_imgs = get_hst_test_images()
    cube_file = get_muse_test_cube()
    spec_files = make_synt_specs.main()
    reg_file = os.path.join(TEST_DATA_PATH, 'test_regions.reg')
    cat_file = os.path.join(TEST_DATA_PATH, 'test_sources.cat')

    def test_zeropoint_info(self):
        print(">>> Testing zeropoint_info...\n")
        if not self.test_hst_imgs:
            print(
                "Failed to download HST test images, skipping this test...",
                file=sys.stderr
            )
            return True
        zpinfo(self.test_hst_imgs)

    def test_cube_stack(self):
        print(">>> Testing cube_stack...\n")
        print(self.cube_file)
        cube_stack_options = [
            self.cube_file,
            '--var-hdu', 'STAT'
        ]
        cube_stack(cube_stack_options)

    def test_cube_smoothing_phisical_units(self):
        print(">>> Testing smoothing...\n")
        smoohting_options = [
            '--verbose',
            '--spatial-smooth', '0.5arcsec,0.25arcsec',
            '--wave-smooth', '1angstrom',
            self.cube_file
        ]
        smoothing_main(smoohting_options)

    def test_cube_smoothing_pixel_units(self):
        print(">>> Testing smoothing...\n")
        smoohting_options = [
            '--verbose',
            '--spatial-smooth', '3,2',
            '--wave-smooth', '2',
            self.cube_file
        ]
        smoothing_main(smoohting_options)

    def test_grayscale_cutout(self):
        print(">>> Testing grayscale cutout...\n")
        if not self.test_hst_imgs:
            print(
                "Failed to download HST test images, skipping this test...",
                file=sys.stderr
            )
            return True
        for img in self.test_hst_imgs:
            if img.endswith('NICMOSn4hk12010_mos.fits'):
                regfile = os.path.join(
                    TEST_DATA_PATH, 'NICMOSn4hk12010_mos.reg'
                )

                with open(regfile, 'w') as f:
                    f.write(NICMOS_REGFILE_DATA)

                cutout_options = [
                    '--verbose',
                    '--regionfile', regfile,
                    img
                ]
                cutout_main(cutout_options)
                break

    def test_cube_cutout(self):
        print(">>> Testing cube cutout...\n")

        cutout_options = [
            '--verbose',
            '--center', '16.3867deg,-24.6464deg',
            '--size', '6arcsec,3arcsec',
            '--angle', '45deg',
            self.cube_file
        ]
        cutout_main(cutout_options)

    def test_cube_cutout_regfile(self):
        print(">>> Testing cube cutout...\n")

        cutout_options = [
            '--verbose',
            '--regionfile', self.reg_file,
            self.cube_file
        ]
        cutout_main(cutout_options)

    def test_extract_sources(self):
        print(">>> Testing sources detection from datacube...\n")
        detect_from_cube([self.cube_file])

    def test_specex_catalog(self):
        print(">>> Testing specex catalog...\n")
        specex_options = [
            '--catalog', self.cat_file,
            '--mode', 'circular_aperture',
            '--aperture-size', '0.8arcsec',
            '--weighting', 'whitelight',
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)

    def test_specex_regionfile(self):
        print(">>> Testing specex regionfile...")
        specex_options = [
            '--regionfile', self.reg_file,
            '--mode', 'kron_ellipse',
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)


    def test_specex_regionfile_wlight(self):
        print(">>> Testing specex regionfile (cube whitelight)...")
        specex_options = [
            '--regionfile', self.reg_file,
            '--mode', 'kron_ellipse',
            '--weighting', 'whitelight',
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)

    def test_specex_regionfile_log_wlight(self):
        print(">>> Testing specex regionfile (cube log-whitelight)...")
        specex_options = [
            '--regionfile', self.reg_file,
            '--mode', 'kron_ellipse',
            '--weighting', 'log-whitelight',
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)

    def test_specex_regionfile_wimg(self):
        print(">>> Testing specex regionfile (ext. image)...")
        specex_options = [
            '--regionfile', self.reg_file,
            '--mode', 'kron_ellipse',
            '--weighting', "ADP.2023-09-01T12:56:41_data.fits",
            '--no-nans', self.cube_file
        ]
        specex(options=specex_options)

    def test_plot(self):
        print(">>> Testing specex-plot...")

        plot_options = [
            '--restframe',
            '--outdir', 'test_plot_out',
            *self.spec_files
        ]

        plot_spectra(options=plot_options)


if __name__ == '__main__':
    tests = MyTests()
    tests.test_zeropoint_info()

    tests.test_cube_stack()

    tests.test_grayscale_cutout()
    tests.test_cube_cutout()

    tests.test_extract_sources()

    tests.test_specex_catalog()
    tests.test_specex_regionfile()
    tests.test_specex_regionfile_wlight()
    tests.test_specex_regionfile_log_wlight()
    tests.test_specex_regionfile_wimg()

    tests.test_plot()
