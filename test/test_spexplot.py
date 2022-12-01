#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import unittest

from spex.spexplot import spexplot

from test import make_synt_specs


class TestSpexplot(unittest.TestCase):

    def test_spexplot_success(self):
        spec_files = make_synt_specs.main()

        spexplot_options = [
            '--restframe',
            '--outdir', 'test_spexplot_out',
            *spec_files
        ]

        spexplot(options=spexplot_options)


if __name__ == '__main__':
    mytest = TestSpexplot()
    mytest.test_spexplot_success()
