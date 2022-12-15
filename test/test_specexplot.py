#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import unittest

from specex.plot import plot

from test import make_synt_specs


class TestSpexplot(unittest.TestCase):

    def test_plot_success(self):
        spec_files = make_synt_specs.main()

        plot_options = [
            '--restframe',
            '--outdir', 'test_plot_out',
            *spec_files
        ]

        plot(options=plot_options)


if __name__ == '__main__':
    mytest = TestSpexplot()
    mytest.test_plot_success()
