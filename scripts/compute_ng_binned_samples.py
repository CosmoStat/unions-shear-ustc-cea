#!/usr/bin/env python

"""compute_ng_binned_samples.py

Compute GGL (ng-correlation) between two (binned) input catalogues.

:Authors: Martin Kilbinger, Elisa Russier

:Date: 2022
"""

import sys
import os

from copy import copy

from tqdm import tqdm

from optparse import OptionParser

import numpy as np
import math
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy import units
from astropy.stats import jackknife_stats

import treecorr

from unions_wl import defaults

from cs_util import logging


class ng_essentials(object):

    def __init__(self, n_bin):
        # Variance is not set during process_corr
        self.meanr = np.zeros(n_bin)
        self.meanlogr = np.zeros(n_bin)
        self.xi = np.zeros(n_bin)
        self.xi_im = np.zeros(n_bin)
        self.varxi = np.zeros(n_bin)
        self.weight = np.zeros(n_bin)
        self.npairs = np.zeros(n_bin)

        # For jackknife resamples
        self.xi_jk = np.zeros(n_bin)
        self.varxi_jk = np.zeros(n_bin)
        self.xi_jk_arr = []

    def copy_from(self, ng):

        for jdx in range(len(self.meanr)):
            self.meanr[jdx] = ng.meanr[jdx]
            self.meanlogr[jdx] = ng.meanlogr[jdx]
            self.xi[jdx] = ng.xi[jdx]
            self.xi_im[jdx] = ng.xi_im[jdx]
            self.weight[jdx] = ng.weight[jdx]
            self.npairs[jdx] = ng.npairs[jdx]

    def copy_to(self, ng, jackknife=False):

        for jdx in range(len(ng.meanr)):
            ng.meanr[jdx] = self.meanr[jdx]
            ng.meanlogr[jdx] = self.meanlogr[jdx]
            if not jackknife:
                ng.xi[jdx] = self.xi[jdx]
                ng.varxi[jdx] = self.varxi[jdx]
            else:
                ng.xi[jdx] = self.xi_jk[jdx]
                ng.varxi[jdx] = self.varxi_jk[jdx]
            # MKDEBUG TODO xi_im
            ng.xi_im[jdx] = self.xi_im[jdx]
            ng.weight[jdx] = self.weight[jdx]
            ng.npairs[jdx] = self.npairs[jdx]

    def difference(self, ng_min, ng_sub):

        # Compute difference between both correlations.
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

        for jdx in range(len(self.meanr)):
            self.meanr[jdx] += ng_sum.meanr[jdx] * ng_sum.weight[jdx]

        self.meanlogr += ng_sum.meanlogr
        self.xi += ng_sum.xi
        self.xi_im += ng_sum.xi_im
        self.weight += ng_sum.weight
        self.npairs += ng_sum.npairs

        self.xi_jk_arr.append(ng_sum.xi)

    def add_physical(self, ng, r, d_ang, sep_units):

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
        self.xi_im += get_interp(x_new, x, ng.xi_im)
        self.weight += get_interp(x_new, x, ng.weight)
        self.npairs += get_interp(x_new, x, ng.npairs)

        # Jackknife array
        # The following gives very biased mean, maybe because
        # of many zeros in weights?
        #self.xi_jk_arr.append(np.nan_to_num(xi_new / w_new))

        self.xi_jk_arr.append(xi_new)

    def normalise(self, n_bin_fac=1):

        sw = self.weight

        for jdx in range(len(self.meanr)):
            self.meanr[jdx] = self.meanr[jdx] / sw[jdx]
            self.meanlogr[jdx] = self.meanlogr[jdx] / sw[jdx]
            self.xi[jdx] = self.xi[jdx] / sw[jdx]
            self.xi_im[jdx] = self.xi_im[jdx] / sw[jdx]

        self.weight *= n_bin_fac
        self.npairs *= n_bin_fac

    def jackknife(self, all_ng):

        xi_jk_arr_all = np.array(self.xi_jk_arr)
        for jdx in range(len(self.meanr)):

            my_xi = xi_jk_arr_all[:, jdx] / self.weight[jdx]
            my_xi *= len(my_xi)
            test_statistic = lambda x: (np.mean(x), np.var(x))
            estimate, bias, stderr, conf_interval = jackknife_stats(
                my_xi,
                test_statistic,
            )

            print("Jackknife mean, std, sqrt(std) at bin #", jdx, estimate[0], estimate[1], np.sqrt(estimate[1]))
            self.xi_jk[jdx] = estimate[0]
            self.varxi_jk[jdx] = estimate[1]

    def set_units_scales(self, sep_units):

        # Angular scales: coordinates need to be attributed at the end
        self.meanr = (self.meanr * units.rad).to(sep_units).value

        # Log-scales: more complicated


def params_default():
    """PARAMS DEFAULT

    Return default parameter values and additional information
    about type and command line options.

    Returns
    -------
    list :
        parameter dict
        types if not default (``str``)
        help string dict for command line option
        short option letter dict

    """
    # Specify all parameter names and default values
    params = {
        'input_path_fg': 'unions_sdss_matched_lens.fits',
        'input_path_bg': 'unions_sdss_matched_source.fits',
        'key_ra_fg': 'RA',
        'key_dec_fg': 'DEC',
        'key_ra_bg': 'RA',
        'key_dec_bg': 'DEC',
        'key_w_fg': None,
        'key_w_bg': None,
        'key_e1': 'e1',
        'key_e2': 'e2',
        'sign_e1': +1,
        'sign_e2': +1,
        'key_z': 'z',
        'theta_min': 0.1,
        'theta_max': 200,
        'n_theta': 10,
        'scales' : 'angular',
        'stack': 'auto',
        'out_path' : './ggl_unions_sdss_matched.txt',
        'out_path_jk' : None,
        'n_cpu': 1,
        'verbose': False,
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'sign_e1': 'int',
        'sign_e2': 'int',
        'n_theta': 'int',
        'n_cpu': 'int',
    }

    # Parameters which can be specified as command line option
    help_strings = {
        'input_path_fg': 'background catalogue input path, default={}',
        'input_path_bg': 'foreground catalogue input path, default={}',
        'key_ra_fg': 'foreground right ascension column name, default={}',
        'key_dec_fg': 'foreground declination column name, default={}',
        'key_ra_bg': 'background right ascension column name, default={}',
        'key_dec_bg': 'background declination column name, default={}',
        'key_w_fg': 'foreground weight column name, default={}',
        'key_w_bg': 'background weight column name, default={}',
        'key_e1': 'first ellipticity component column name, default={}',
        'key_e2': 'second ellipticity component column name, default={}',
        'sign_e1': 'first ellipticity multiplier (sign), default={}',
        'sign_e2': 'first ellipticity multiplier (sign), default={}',
        'key_z': 'foreground redshift column name (if scales=physical), default={}',
        'theta_min': 'minimum angular scale, default={}',
        'theta_max': 'maximum angular scale, default={}',
        'n_theta': 'number of angular scales, default={}',
        'scales' : '2D coordinates (scales) are angular (arcmin) or physical [Mpc], default={}',
        'stack' : 'allowed are auto, cross, post, default={}',
        'out_path' : 'output path, default={}',
        'out_path_jk' : 'output path, default=<out_path>_jk.<ext>',
        'n_cpu' : 'number of CPUs for parallel processing, default={}',
    }

    # Options which have one-letter shortcuts
    short_options = {
    }

    return params, short_options, types, help_strings


def parse_options(p_def, short_options, types, help_strings):
    """Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    help_strings : dict
        help strings for options

    Returns
    -------
    options: tuple
        Command line options
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    for key in p_def:
        if key in help_strings:

            if key in short_options:
                short = short_options[key]
            else:
                short = ''

            if key in types:
                typ = types[key]
            else:
                typ = 'string'

            if typ == 'bool':
                parser.add_option(
                    f'{short}',
                    f'--{key}',
                    dest=key,
                    default=False,
                    action='store_true',
                    help=help_strings[key].format(p_def[key]),
                )
            else:
                parser.add_option(
                    short,
                    f'--{key}',
                    dest=key,
                    type=typ,
                    default=p_def[key],
                    help=help_strings[key].format(p_def[key]),
                )

    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help=f'verbose output'
    )

    options, args = parser.parse_args()

    return options


def create_treecorr_catalogs(
    positions,
    sample,
    key_ra,
    key_dec,
    g1,
    g2,
    w,
    coord_units,
    split,
):
    """Create Treecorr Catalogs.

    Return treecorr catalog(s).

    Parameters
    ----------
    positions : dict
        input positions
    sample : str
        sample string
    key_ra : str
        data key for right ascension
    key_dec : str
        data key for declination
    g1 : dict
        first shear component
    g2 : dict
        second shear component
    w : dict
        weight
    coord_units : str
        coordinate unit string
    split : bool
        if True split foreground sample into individual objects

    Returns
    -------
    list
        treecorr Cataloge objects

    """
    cat = []
    if not split:
        my_cat = treecorr.Catalog(
            ra=positions[sample][key_ra],
            dec=positions[sample][key_dec],
            g1=g1[sample],
            g2=g2[sample],
            w=w[sample],
            ra_units=coord_units,
            dec_units=coord_units,
        )
        cat = [my_cat]
    else:
        n_obj = len(positions[sample][key_ra])
        for idx in range(n_obj):
            if not g1[sample]:
                my_g1 = None
                my_g2 = None
            else:
                my_g1 = g1[sample][idx:idx+1]
                my_g2 = g2[sample][idx:idx+1]

            my_cat = treecorr.Catalog(
                ra=positions[sample][key_ra][idx:idx+1],
                dec=positions[sample][key_dec][idx:idx+1],
                g1=my_g1,
                g2=my_g2,
                w=w[sample][idx:idx+1],
                ra_units=coord_units,
                dec_units=coord_units,
            )
            cat.append(my_cat)

    return cat


def rad_to_unit(value, unit):

    return (value * units.rad).to(unit).value
 

def unit_to_rad(value, unit):

    return value * units.Unit(units).to('rad')


def get_theta_min_max(r_min, r_max, d_ang_arr, sep_units):

    # Min and max angular distance at object redshift
    d_ang_min = min(d_ang_arr)
    d_ang_max = max(d_ang_arr)
    d_ang_mean = np.mean(d_ang_arr)

    # Transfer physical to angular scales

    theta_min = 1e30
    theta_max = -1
    for d_ang in d_ang_arr:
        th_min = float(r_min) / d_ang
        if th_min < theta_min:
            theta_min = th_min
        th_max = float(r_max) / d_ang
        if th_max > theta_max:
            theta_max = th_max

    # Enlarge range to account for actual scale different from nominal
    # scales, and outside of range
    theta_min = theta_min / 1.5
    theta_max = theta_max * 1.5

    print(f'physical to angular scales, r = {r_min}  ... {r_max:} Mpc')
    print(f'physical to angular scales, d_ang = {min(d_ang_arr):.2f}  ... {max(d_ang_arr):.2f} (mean {d_ang_mean:.2f}) Mpc')
    print(f'physical to angular scales, theta = {theta_min:.2g}  ... {theta_max:.2g} rad')

    theta_min = rad_to_unit(theta_min, sep_units)
    theta_max = rad_to_unit(theta_max, sep_units)

    print(f'physical to angular scales, theta = {theta_min:.2g}  ... {theta_max:.2f} arcmin')

    return theta_min, theta_max


def create_treecorr_config(
    coord_units,
    scale_min,
    scale_max,
    sep_units,
    n_theta,
    n_cpu,
):

    TreeCorrConfig = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'min_sep': scale_min,
        'max_sep': scale_max,
        'sep_units': sep_units,
        'nbins': n_theta,
        'num_threads': n_cpu,
    }

    return TreeCorrConfig


def get_interp(x_new, x, y):

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
            #print(f'Warning: x[{idx}]={x[idx]:.3g} outside range {x_new[0]:.3g} ... {x_upper_new:.3g}')
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


def ng_stack(TreeCorrConfig, all_ng, all_d_ang, n_bin_fac=1):

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
            ng_final.add_physical(ng, r, d_ang, sep_units)

    else:

        # Add up all individual correlations on angular coordinates
        for ng in all_ng:
            ng_final.add(ng)

    ng_final.set_units_scales(sep_units)
    ng_final.normalise(n_bin_fac=n_bin_fac)
    ng_final.jackknife(all_ng)

    # Copy results to NGCorrelation instances
    ng_final.copy_to(ng_comb)
    ng_final.copy_to(ng_comb_jk, jackknife=True)

    # Angular scales: coordinates need to be attributed at the end
    for this_ng in [ng_comb, ng_comb_jk]:
        this_ng.meanlogr = (np.exp(this_ng.meanlogr) * units.rad).to(sep_units).value
        this_ng.meanlogr = np.log(this_ng.meanlogr)

    return ng_comb, ng_comb_jk


def main(argv=None):
    """Main

    Main program

    """

    params, short_options, types, help_strings = params_default()

    options = parse_options(params, short_options, types, help_strings)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    # Save calling command
    logging.log_command(argv)

    # Open input catalogues
    data = {}
    for sample in ('fg', 'bg'):
        input_path = params[f'input_path_{sample}']
        if params['verbose']:
            print(f'Reading catalogue {input_path}')
        data[sample] = fits.getdata(input_path)

    #print('MKDEBUG cut to 0.4M')
    #data['bg'] = data['bg'][400_000:600_000]

    print(f'scales={params["scales"]}, stack={params["stack"]}')

    n_bin_fac = 1
    print('n_bin_fac = ', n_bin_fac)

    # Set treecorr catalogues
    coord_units = 'degrees'
    cats = {}
    if (
        params['verbose']
        and params['sign_e1'] != +1
        and params['sign_e2'] != +1
    ):
        print(
            'Non-standard signs for ellipticity components ='
            + f' ({params["sign_e1"]:+d}, {params["sign_e2"]:+d})'
        )

    # Set fg and bg sample data columns
    # Shear components g1, g2: Set `None` for foreground
    g1 = {
        'fg': None,
        'bg': data[sample][params['key_e1']] * params['sign_e1']
    }
    g2 = {
        'fg': None,
        'bg': data[sample][params['key_e2']] * params['sign_e2']
    }
    w = {}
    for sample in ['fg', 'bg']:
        n = len(data[sample][params[f'key_ra_{sample}']])

        # Set weight
        if params[f'key_w_{sample}'] is None:
            w[sample] = [1] * n
            if params['verbose']:
                print(f'Not using weights for {sample} sample')
        else:
            w[sample] = data[sample][params[f'key_w_{sample}']]
            if params['verbose']:
                print(f'Using catalog weights for {sample} sample')

    if params['scales'] == 'physical':
        cosmo = defaults.get_cosmo_default()
        n_theta = params['n_theta'] * n_bin_fac

    else:
        cosmo = None
        n_theta = params['n_theta']

    # Create treecorr catalogues
    for sample in ('fg', 'bg'):

        # Split cat into single objects if fg and physical
        if sample == 'fg' and (params['scales'] == 'physical' or params['stack'] != 'auto'):
            split = True
        else:
            split = False

        cats[sample] = create_treecorr_catalogs(
            data,
            sample,
            params[f'key_ra_{sample}'],
            params[f'key_dec_{sample}'],
            g1,
            g2,
            w,
            coord_units,
            split,
        )

    # Set treecorr config info for correlation

    sep_units = 'arcmin'
    if params['scales'] == 'physical':

        # Angular distances to all objects
        a_arr = 1 / (1 + data['fg'][params['key_z']]) 
        d_ang = cosmo.angular_diameter_distance(a_arr)

        theta_min, theta_max = get_theta_min_max(
            params['theta_min'],
            params['theta_max'],
            d_ang,
            sep_units,
        )
    else:
        d_ang = None

        theta_min = params['theta_min']
        theta_max = params['theta_max']

    TreeCorrConfig = create_treecorr_config(
        coord_units,
        theta_min,
        theta_max,
        sep_units,
        n_theta,
        params['n_cpu'],
    )

    n_fg = len(cats['fg'])
    if params['verbose']:
        print(f'Correlating 1 bg with {n_fg} fg catalogues...')

    # Compute correlation(s)
    ng = treecorr.NGCorrelation(TreeCorrConfig)
    n_corr = 0
    if len(cats['fg']) > 1:

        ng_prev = ng_essentials(n_theta)
        all_ng = []

        # More than one foreground catalogue: run individual correlations
        for idx, cat_fg in tqdm(
            enumerate(cats['fg']),
            total=len(cats['fg']),
            disable=not params['verbose'],
        ):

            # Save previous cumulative correlations (empty if first)
            ng_essentials.copy_from(ng_prev, ng)

            # Perform correlation
            ng.process_cross(cat_fg, cats['bg'][0], num_threads=params['n_cpu'])

            # Last correlation = difference betwen two cumulative results
            ng_diff = ng_essentials(n_theta)
            ng_diff.difference(ng, ng_prev)

            # Count (and add) correlations
            n_corr += 1

            all_ng.append(ng_diff)
            # Update previous cumulative correlations
            ng_essentials.copy_from(ng_prev, ng)

        if params['stack'] == 'cross':
            if params['verbose']:
                print('Cross (treecorr process_cross) stacking of fg objects')
            varg = treecorr.calculateVarG(cats['bg'])
            ng.finalize(varg)

        if n_corr == 0:
            raise ValueError('No correlations computed')
        print(f'Computed {n_corr} correlations')

    else:

        # One foreground catalogue: run single simultaneous correlation
        if params['verbose']:
            print('Automatic (treecorr process) stacking of fg objects')
        ng.process(cats['fg'][0], cats['bg'][0], num_threads=params['n_cpu'])

    if params['scales'] == 'physical':

        # Create new config for correlations stacked on physical scales.
        # Use (command-line) input scales but interpret in Mpc
        TreeCorrConfig = create_treecorr_config(
            'deg',
            params['theta_min'],
            params['theta_max'],
            sep_units,
            params['n_theta'],
            1,
        )

    if len(cats['fg']) > 1 and not params['stack'] == 'cross':
        # Stack now (in post-processing) if more than one fg catalogue,
        # and not cross stacking done
        if params['verbose']:
            print('Post-process (this script) stacking of fg objects')
        ng, ng_jk = ng_stack(TreeCorrConfig, all_ng, d_ang, n_bin_fac=n_bin_fac)
    else:
        ng_jk = None

    # Write stack to file
    out_path = params['out_path']
    if params['verbose']:
        print(f'Writing output file {out_path}')
    ng.write(
        out_path,
        rg=None,
        file_type=None,
        precision=None,
    )

    # Write stack with jackknife resamples summaries to file
    if ng_jk:
        if not params['out_path_jk']:
            base, ext = os.path.splitext(params['out_path'])
            out_path_jk = f'{base}_jk{ext}'
        else:
            out_path_jk = params['out_path_jk']
        if params['verbose']:
            print(f'Writing output file {out_path_jk}')
        ng_jk.write(
            out_path_jk,
            rg=None,
            file_type=None,
            precision=None,
        )

    # Fix missing keywords, to prevent subsequent treecorr read error
    if params['stack'] != 'cross':
        hdu_list = fits.open(out_path)
        hdu_list[1].header['COORDS'] = 'spherical'
        hdu_list[1].header['metric'] = 'Euclidean'
        hdu_list.writeto(out_path, overwrite=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
