# -*- coding: utf-8 -*-

"""CATALOGUE MODULE.

This module provides functions to handle samples of objects and
catalogues.

"""

import numpy as np
import healpy as hp

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

    print("MKDEBUG")
    import pdb
    pdb.set_trace()

    return ng


def read_hp_mask(input_name, verbose=False):

    if verbose:                                                       
        print(f'Reading mask {input_name}...')                        

    nest = False                                                                

    # Open input mask                                                           
    mask, header= hp.read_map(                                                  
        input_name,                                                   
        h=True,                                                                 
        nest=nest,                                                              
    )                                                                           
    for (key, value) in header:                                                 
        if key == 'ORDERING':                                                   
            if value == 'RING':                                                 
                if nest:                                                        
                    raise ValueError(                                           
                        'input mask has ORDENING=RING, set nest to False'       
                    )                                                           
            elif value == 'NEST':                                               
                if not nest:                                                    
                    raise ValueError(                                           
                        'input mask has ORDENING=NEST, set nest to True'        
                    )                                                           
                                                                                
    # Get nside from header                                                     
    nside = None                                                                
    for key, value in header:                                                   
        if key == 'NSIDE':                                                      
            nside = int(value)                                                  
    if not nside:                                                               
        raise KeyError('NSIDE not found in FITS mask header')

    return mask, nest, nside


def get_length(dat):
    """GET LENGTH.

    Return length of columns in data dictionary
    
    Parameters
    ----------
    dat : dict
        input data columns

    Returns
    -------
    int
        column length

    """
    return len(dat[[key for key in dat][0]])


def cut_data(dat, key, value, operator, verbose=False):
    """CUT DATA.

    Cut catalogue.

    Parameters
    ----------
    dat : dict
        data catalogue
    key : str
        column name on which to cut
    value : float
        threshold value for cut
    operator : str
        allowed are '<', '>'
    verbose : bool, optional
        verbose output if ``True``, default is ``False``

    Returns
    -------
    dict
        cut data catalogue

    """
    if value is not None:
        if verbose:
            print(f'cut_data: keep {key} {operator} {value}')
        n_all = get_length(dat)

        # Get indices to keep
        if operator == '>':
            w = dat[key] > value
        if operator == '<':
            w = dat[key] < value

        # Cut data for all columns
        for col in dat:
            dat[col] = dat[col][w]

        n_keep = get_length(dat)
        n_cut = n_all - n_keep
        if verbose:
            print(f'removed/kept/all = {n_cut}/{n_keep}/{n_all} objects')

        return dat
