"""DEFAULTS MODULE.

:Description: This module provides default values for various quantities.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import pyccl as ccl


# Set ell_max to large value, for spline interpolation (in integral over    
# C_ell to get real-space correlation functions). Avoid aliasing            
# (oscillations)                                                            
ccl.spline_params.ELL_MAX_CORR = 10_000_000
# was 500_000 in fit_all...

ccl.spline_params.N_ELL_CORR = 5_000


def get_cosmo_default():
    """Get Cosmo Default.

    Return default cosmology.

    Returns
    -------
    Cosmology
        pyccl cosmology objecy

    """
    cosmo = ccl.Cosmology(                                                      
        Omega_c=0.27,                                                           
        Omega_b=0.045,                                                          
        h=0.67,                                                                 
        sigma8=0.83,                                                            
        n_s=0.96,                                                               
    )

    return cosmo
