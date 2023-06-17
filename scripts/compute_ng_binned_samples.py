#!/usr/bin/env python

"""compute_ng_binned_samples.py

Compute GGL (ng-correlation) between two (binned) input catalogues.

:Authors: Martin Kilbinger, Elisa Russier

:Date: 2022
"""

import sys
import os

import numpy as np

from astropy.io import fits
from astropy import units

import treecorr

from unions_wl.run import Compute_NG
from unions_wl.stack_ng import ng_essentials, ng_stack


#def compute_ng(*argv):
def compute_ng(obj):

    params = obj._params

    data = obj._data

    coord_units = obj._coord_units
    sep_units = obj._sep_units

    cats = obj._cats

    n_theta = params['n_theta']

    TreeCorrConfig = self._TreeCorrConfig

    # Fix missing keywords, to prevent subsequent treecorr read error
    if params['stack'] != 'cross':
        hdu_list = fits.open(out_path)
        hdu_list[1].header['COORDS'] = 'spherical'
        hdu_list[1].header['metric'] = 'Euclidean'
        hdu_list.writeto(out_path, overwrite=True)
        if ng_jk:
            hdu_list = fits.open(out_path_jk)
            hdu_list[1].header['COORDS'] = 'spherical'
            hdu_list[1].header['metric'] = 'Euclidean'
            hdu_list.writeto(out_path_jk, overwrite=True)

    return 0


def main(argv=None):
    """Main

    Main program

    """
    # Create instance for number-shear correlation computation
    obj = Compute_NG()

    obj.set_params_from_command_line(argv)

    obj.run()
    # MKDEBUG TODO: move everything from
    compute_ng(obj)
    # to Compute_NG.run()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
