"""FIT MODULE.

:Description: This module provides functions for parameter fitting.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>

"""


from joblib import Parallel, delayed                                            
from lmfit import minimize

from . import theory


def loss(params, x_data, y_data, err, extra):                                   
    """Loss function                                                            
                                                                                
    Loss function for tangential shear fit                                      
                                                                                
    Parameters                                                                  
    ----------                                                                  
    params : lmfit.Parameters                                                   
        fit parameters                                                          
    x_data : numpy.array                                                        
        x-values of the data                                                    
    y_data : numpy.array                                                        
        y-values of the data                                                    
    err : numpy.array                                                           
        error values of the data                                                
    extra : dict                                                                
        additional parameters                                                   

    Raises
    ------
    ValuerError
        if err contains at least one zero
                                                                                
    Returns                                                                     
    -------                                                                     
    numpy.array                                                                 
        residuals                                                               
                                                                                
    """                                                                         

    if any(err == 0):
        raise ValueError('Division by zero')

    y_model = theory.g_t_model(params, x_data, extra)                                  
                                                                                
    residuals = (y_model - y_data) / err                                        
                                                                                
    return residuals


def do_minimize(idx, loss, fit_params, args):                                   
    """Do Minimize.                                                             
                                                                                
    Call minimize for task `idx`, and return result.                            
    Can be called with `Parallel` with `idx` as iterating index.                
                                                                                
    """                                                                         
    return minimize(loss, fit_params, args=args[idx])


def fit(args, fit_params, n_cpu, verbose):                                      
                                                                                
    # Number of jobs to do                                                      
    n_fit = len(args)                                                           
                                                                                
    if verbose:                                                                 
        print(f'Fit {n_fit} models on {n_cpu} CPUs...')                         
                                                                                
    # Fit models in parallel                                                    
    res_arr = Parallel(n_jobs=n_cpu, verbose=13)(                               
        delayed(do_minimize)(                                                   
            idx,                                                                
            loss,                                                               
            fit_params,                                                         
            args                                                                
        ) for idx in range(n_fit)                                               
    )                                                                           
                                                                                
    return res_arr
