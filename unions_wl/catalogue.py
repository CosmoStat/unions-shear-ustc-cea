# -*- coding: utf-8 -*-

"""CATALOGUE MODULE.

This module provides functions to handle samples of objects and
catalogues.

"""

import numpy as np
from astropy.io import ascii
from astropy.table import Table

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


def write_ascii(out_path, data, names):
    """Write Ascii.

    Write data to an ascii file.

    Parameters                                                                  
    ----------                                                                  
    out_path : str                                                              
        output file path                                                        
    data : tupel of arrays
        data for output
    names : tupel
        column names

    """
    t = Table(data, names=names)                                       
    with open(out_path, 'w') as f:                                              
        ascii.write(t, f, delimiter='\t', format='commented_header') 


def write_dndz(out_path, dndz):                                                 
    """Write Dndz.                                                              
                                                                                
    Write redshift distribution (histogram) to file.                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    out_path : str                                                              
        output file path                                                        
    dndz : tupel of arrays                                                      
        redshift distribution histogram (z, n(z))                               
                                                                                
    """                                                                         
    write_ascii(out_path, dndz, ('z', 'dn_dz'))


def write_cls(out_path, ell, cls):                                                 
    """Write Cls.                                                              
                                                                                
    Write angular power spectrum to file.                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    out_path : str                                                              
        output file path                                                        
    ell : array                                                      
        2D Fourier modes
    cls : array
        angular power spectrum
                                                                                
    """                                                                         
    write_ascii(out_path, (ell, cls), ('ell', 'C_ell'))


def get_ngcorr_data(
    path,
    theta_min=1,
    theta_max=100,
    n_theta=10,
):
    """Get Corr Data.

    Return correlation data from file, computed by treecorr

    Parameters
    ----------
    path : str
        input file path
    theta_min : int, optional
        smallest angular scale in arcmin, default is 1
    theta_max : int, optional
        largest angular scale in arcmin, default is 100
    n_theta : int, optional
        number of angular scales, default is 10

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
        'min_sep': theta_min,
        'max_sep': theta_max,
        'sep_units': sep_units,
        'nbins': n_theta,
    }

    ng = treecorr.NGCorrelation(TreeCorrConfig)
    try:
        ng.read(path)
    except:
        print(f'Error while reading treecorr input file {path}')
        raise

    return ng
