# -*- coding: utf-8 -*-

"""CATALOGUE MODULE.

This module provides functions to handle samples of objects and
catalogues.

"""

import numpy as np
from astropy.io import ascii

import treecorr


def y_equi(cdf, n):
    """ Y EQUI

    Split sample into n equi-populated bins, return bin boundaries.

    Parameters
    ----------
    cdf : statsmodels.distributions.empirical_distribution.ECDF
        distribution data
    n : int
        number of splits

    Returns
    -------
    list
        bin boundaries of samples

    """
    x_list = []
    for denom in range(1, n):
        idx =  np.where(cdf.y >= denom/n)[0][0]
        x_list.append(cdf.x[idx])

    return x_list


def bin_edges2centers(bin_edges):
    """BIN EDGES TO CENTERS

    Transform bin edge values to central values

    Parameters
    ----------
    bin_edges : list
        bin edge values

    Returns
    -------
    list
        bin central values

    """
    bin_means = 0.5 * (
        bin_edges[1:] + bin_edges[:-1]
    )

    return bin_means


def read_dndz(file_path):
    """Read Dndz.

    Read redshift histogram from file.

    Parameters
    ----------
    file_path : str
        input file path

    Returns
    -------
    list :
        redshift bin centers
    list :
        number densities
    list :
        redshift bin edges

    """
    dat = ascii.read(file_path, format='commented_header')
    # Remove last n(z) value which is zero, to match bin centers
    nz = dat['dn_dz'][:-1]
    z_edges = dat['z']
    z_centers = bin_edges2centers(z_edges)

    return z_centers, nz, z_edges


def get_ngcorr_data(path):
    """Get Corr Data.

    Return correlation data from file, computed by treecorr

    Parameters
    ----------
    path : str
        input file path

    Returns
    -------
    treecorr.ngcorrelation.NGCorrelation :
        correlation information

    """
    # Dummy treecorr initialisation
    coord_units = 'deg'
    sep_units = 'arcmin'

    TreeCorrConfig = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'min_sep': 1.0,
        'max_sep': 2.0,
        'sep_units': sep_units,
        'nbins': 1,
    }

    ng = treecorr.NGCorrelation(TreeCorrConfig)
    ng.read(path)

    return ng
