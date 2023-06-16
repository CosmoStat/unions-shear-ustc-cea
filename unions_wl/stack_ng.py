"""STACK_NG MODULE.

:Description: This module provides classes and methods for
    stacking of umber - shear correlations.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>

"""


import math
import numpy as np

from scipy.interpolate import interp1d

from astropy.stats import jackknife_stats                                       
from astropy import units

import treecorr


class ng_essentials(object):
    """Ng essential.

    This class manipulates treecorr.NGCorrelation (number-shear)
    correlation information.

    Parameters
    ----------
    n_bins : int
        number of angular bins

    """

    def __init__(self, n_bin):
        self.meanr = np.zeros(n_bin)
        self.meanlogr = np.zeros(n_bin)
        self.xi = np.zeros(n_bin)
        self.xi_im = np.zeros(n_bin)
        self.varxi = np.zeros(n_bin)
        self.weight = np.zeros(n_bin)
        self.npairs = np.zeros(n_bin)

        # For jackknife resamples
        self.xi_jk = np.zeros(n_bin)
        self.xi_im_jk = np.zeros(n_bin)
        self.varxi_jk = np.zeros(n_bin)
        self.xi_jk_arr = []
        self.xi_im_jk_arr = []

    def copy_from(self, ng):
        """Copy From.

        Copy information from an NGCorrelation instance.

        Parameters
        ----------
        ng : treecorr.NGCorrelation
            number-shear correleation information

        """
        for jdx in range(len(self.meanr)):
            self.meanr[jdx] = ng.meanr[jdx]
            self.meanlogr[jdx] = ng.meanlogr[jdx]
            self.xi[jdx] = ng.xi[jdx]
            self.xi_im[jdx] = ng.xi_im[jdx]
            self.weight[jdx] = ng.weight[jdx]
            self.npairs[jdx] = ng.npairs[jdx]

    def copy_to(self, ng, jackknife=False):
        """Copy To.

        Copy class content to an NGCorrelation instance.

        Parameters
        ----------
        ng : treecorr.NGCorrelation
            number-shear correleation information
        jackknife : bool, optional
            uses jackknife mean and std if True; default is False

        """
        for jdx in range(len(ng.meanr)):
            ng.meanr[jdx] = self.meanr[jdx]
            ng.meanlogr[jdx] = self.meanlogr[jdx]
            if not jackknife:
                ng.xi[jdx] = self.xi[jdx]
                ng.xi_im[jdx] = self.xi_im[jdx]
                ng.varxi[jdx] = self.varxi[jdx]
            else:
                ng.xi[jdx] = self.xi_jk[jdx]
                ng.xi_im[jdx] = self.xi_im_jk[jdx]
                ng.varxi[jdx] = self.varxi_jk[jdx]
            ng.weight[jdx] = self.weight[jdx]
            ng.npairs[jdx] = self.npairs[jdx]

    def difference(self, ng_min, ng_sub):
        """Difference.

        Compute difference between two correlations.

        Parameters
        ----------
        ng_min : NGcorrelation of ng_essential
            minuend (term before the minus sign)
        ng_sub : NGcorrelation of ng_essential
            subtrahend (term after the minus sign)

        """
        # If ng_min and ng_sub are two subsequent outputs of
        # treecorr.proess_cross, these are weighted quantities.
        # This is because the cumulative processing adds
        # weighted results.
        for jdx in range(len(self.meanr)):
            self.weight[jdx] = ng_min.weight[jdx] - ng_sub.weight[jdx]

            self.meanr[jdx] = ng_min.meanr[jdx] - ng_sub.meanr[jdx]
            self.meanlogr[jdx] = ng_min.meanlogr[jdx] - ng_sub.meanlogr[jdx]

            self.xi[jdx] = ng_min.xi[jdx] - ng_sub.xi[jdx]
            self.xi_im[jdx] = ng_min.xi_im[jdx] - ng_sub.xi_im[jdx]
            self.npairs[jdx] = ng_min.npairs[jdx] - ng_sub.npairs[jdx]

        # Remove weight for angular scales, such that the latter
        # are unweighted scales. Required for stacking on physical
        # scales, where we need unweighted scale for each correlation
        for jdx in range(len(self.meanr)):
            if self.meanr[jdx] > 0:
                self.meanr[jdx] = self.meanr[jdx] / self.weight[jdx]

    def add(self, ng_sum):
        """Add.

        Add number-shear correlation to class content, stacking on
        angular coordinates.

        Parameters
        ----------
        ng_sum : NGcorrelation of ng_essential
            summand

        """
        # Add weighted angular scales
        for jdx in range(len(self.meanr)):
            self.meanr[jdx] += ng_sum.meanr[jdx] * ng_sum.weight[jdx]

        self.meanlogr += ng_sum.meanlogr
        self.xi += ng_sum.xi
        self.xi_im += ng_sum.xi_im
        self.weight += ng_sum.weight
        self.npairs += ng_sum.npairs

        self.xi_jk_arr.append(ng_sum.xi)
        self.xi_im_jk_arr.append(ng_sum.xi_im)

    def add_physical(self, ng, r, d_ang):
        """Add Physical.

        Add number-shear correlation to class content, stacking on
        physical coordinates.

        Parameters
        ----------
        ng : NGcorrelation of ng_essential
            summand
        r : numpy.array
            physical coordinates
        d_ang : float
            angular diameter distance, interpreted in same units as r

        """
        # Original angular x values [rad]
        x = ng.meanr

        # New x values: transfer from physical [Mpc] to angular [rad]
        x_new = r / d_ang

        # Re-bin to new angular coordinates and add (= stack)

        # Angular scales: individual ones were not weighted, add weight
        # back here
        self.meanr += get_interp(x_new, x, ng.meanr * ng.weight)
        self.meanlogr += get_interp(x_new, x, ng.meanlogr)

        xi_new = get_interp(x_new, x, ng.xi)
        self.xi += xi_new

        xi_im_new = get_interp(x_new, x, ng.xi_im)
        self.xi_im += xi_im_new

        self.weight += get_interp(x_new, x, ng.weight)
        self.npairs += get_interp(x_new, x, ng.npairs)

        # Jackknife array
        self.xi_jk_arr.append(xi_new)
        self.xi_im_jk_arr.append(xi_im_new)

    def normalise(self):
        """Normalise.

        Normalise class content.

        """
        sw = self.weight

        for jdx in range(len(self.meanr)):
            self.meanr[jdx] = self.meanr[jdx] / sw[jdx]
            self.meanlogr[jdx] = self.meanlogr[jdx] / sw[jdx]
            self.xi[jdx] = self.xi[jdx] / sw[jdx]
            self.xi_im[jdx] = self.xi_im[jdx] / sw[jdx]

    def jackknife(self, all_ng):
        """Jackknife.

        Compute jackknife mean and standard deviation of number-shear
        correlation.

        Parameters
        ----------
        all_ng : list
            number-shear correlations for all foreground objects

        """
        test_statistic = lambda x: (np.mean(x), np.var(x))
        xi_jk_arr_all = np.array(self.xi_jk_arr)
        xi_im_jk_arr_all = np.array(self.xi_im_jk_arr)

        for jdx in range(len(self.meanr)):

            # The following lines correct xi by the already included weights
            my_xi = xi_jk_arr_all[:, jdx] / self.weight[jdx]
            my_xi *= len(my_xi)

            estimate, bias, stderr, conf_interval = jackknife_stats(
                my_xi,
                lambda x: (np.mean(x), np.var(x)),
            )
            # std of samples -> std of mean
            estimate[1] /= len(my_xi)

            # Assign Jackknife estimates
            self.xi_jk[jdx] = estimate[0]
            self.varxi_jk[jdx] = estimate[1]

            my_xi_im = xi_im_jk_arr_all[:, jdx] / self.weight[jdx]
            my_xi_im *= len(my_xi_im)

            # Only compute mean; ignore var_jk(xi_im), which is
            # very close to var_jk(xi)
            estimate, bias, stderr, conf_interval = jackknife_stats(
                my_xi_im,
                np.mean,
            )
            self.xi_im_jk[jdx] = estimate

    def set_units_scales(self, sep_units):

        # Angular scales: coordinates need to be attributed at the end
        self.meanr = (self.meanr * units.rad).to(sep_units).value

        # Log-scales: TODO


def ng_stack(TreeCorrConfig, all_ng, all_d_ang):
    """NG Stack.

    Stack number-shear correlations.

    Parameters
    ----------
    TreeCorrConfig : dict
        treecorr configuration information
    all_ng : list
        individual number-shear correlations
    all_d_ang : list
        angular diameter distance to objects from all_ng

    Returns
    -------
    treecorr.NGCorrelation
        stacked number-shear correlation
    treecorr.NGCorrelation
        stacked number-shear correlation with Jackknife errors

    """
    # Initialise combined correlation objects
    ng_comb = treecorr.NGCorrelation(TreeCorrConfig)
    ng_comb_jk = treecorr.NGCorrelation(TreeCorrConfig)

    n_bins = len(ng_comb.rnom)
    sep_units = ng_comb.sep_units

    ng_final = ng_essentials(n_bins)

    if all_d_ang is not None:

        # New x values to interpolate on [Mpc]
        r = ng_comb.rnom

        # Add up all individual correlations on physical coordinates
        for ng, d_ang in zip(all_ng, all_d_ang):
            ng_final.add_physical(ng, r, d_ang)

    else:

        # Add up all individual correlations on angular coordinates
        for ng in all_ng:
            ng_final.add(ng)

    ng_final.set_units_scales(sep_units)
    ng_final.normalise()
    ng_final.jackknife(all_ng)

    # Copy results to NGCorrelation instances
    ng_final.copy_to(ng_comb)
    ng_final.copy_to(ng_comb_jk, jackknife=True)

    # Angular scales: coordinates need to be attributed at the end
    for this_ng in [ng_comb, ng_comb_jk]:
        this_ng.meanlogr = (
            (np.exp(this_ng.meanlogr) * units.rad).to(sep_units).value
        )
        this_ng.meanlogr = np.log(this_ng.meanlogr)

    return ng_comb, ng_comb_jk


def get_interp(x_new, x, y):                                                    
    """Get Interp.

    Return values of the function y(x) interpolated to x_new.

    Parameters
    ----------
    x_new : numpy.array
        new x-values
    x : numpy.array
        existing x-values
    y : numpy.array
        existing y-values

    Returns
    -------
    numpy.array
        new y-values

    """                                                                                
    y_new = np.zeros_like(x_new)                                                
                                                                                
    # Compute upper limit (assuming logarithmic bins)                           
    log_x_upper_new = np.log(x_new[-1]) + np.log(x_new[-1]) - np.log(x_new[-2]) 
    x_upper_new = np.exp(log_x_upper_new)                                       
                                                                                
    # Loop over original x-bins                                                 
    n_bins = len(x)                                                             
    for idx, x_val in enumerate(x):                                             
                                                                                
        # Zero value indicates no data in this bin                              
        if x[idx] == 0:                                                         
            continue                                                            
                                                                                
        # Issue warning if out of range and not first or last bins.             
        if (                                                                    
            (idx != 0 and x[idx] < x_new[0])                                    
            or (idx != n_bins - 1 and x[idx] > x_upper_new)                     
        ):                                                                      
            continue                                                            
                                                                                
        idx_tmp = np.searchsorted(x_new, x_val, side='left')                    
        if idx_tmp == len(x_new): continue                                      
        if (                                                                    
            (idx_tmp > 0)                                                       
            and (                                                               
                idx_tmp == len(x_new)                                           
                or (                                                            
                    math.fabs(x_val - x_new[idx_tmp - 1])                       
                    < math.fabs(x_val - x_new[idx_tmp])                         
                )                                                               
            )                                                                   
        ):                                                                      
            idx_new = idx_tmp - 1                                               
        else:                                                                   
            idx_new = idx_tmp                                                   
                                                                                
        # Place y value                                                         
        y_new[idx_new] = y[idx]                                                 
                                                                                
    return y_new
