#!/usr/bin/env python

"""compute_ng_binned_samples.py

Compute GGL (ng-correlation) between two (binned) input catalogues.

:Authors: Martin Kilbinger, Elisa Russier

:Date: 2022
"""

import sys

from copy import copy

from tqdm import tqdm

from optparse import OptionParser

import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy import units

import pyccl as ccl

import treecorr

from cs_util import logging


class ng_essentials(object):

    def __init__(self, n_bin):
        # Variance is not set during process_corr
        self.meanr = np.zeros(n_bin)
        self.meanlogr = np.zeros(n_bin)
        self.xi = np.zeros(n_bin)
        self.xi_im = np.zeros(n_bin)
        self.weight = np.zeros(n_bin)
        self.npairs = np.zeros(n_bin)

    def copy(self, ng):

        for jdx in range(len(self.meanr)):
            self.meanr[jdx] = ng.meanr[jdx]
            self.meanlogr[jdx] = ng.meanlogr[jdx]
            self.xi[jdx] = ng.xi[jdx]
            self.xi_im[jdx] = ng.xi_im[jdx]
            self.weight[jdx] = ng.weight[jdx]
            self.npairs[jdx] = ng.npairs[jdx]

    def difference(self, ng_min, ng_sub):

        # Compute difference between both correlations.
        # These are weighted quantities
        for jdx in range(len(self.meanr)):
            self.weight[jdx] = ng_min.weight[jdx] - ng_sub.weight[jdx]

            self.meanr[jdx] = ng_min.meanr[jdx] - ng_sub.meanr[jdx]
            self.meanlogr[jdx] = ng_min.meanlogr[jdx] - ng_sub.meanlogr[jdx]


            self.xi[jdx] = ng_min.xi[jdx] - ng_sub.xi[jdx]
            self.xi_im[jdx] = ng_min.xi_im[jdx] - ng_sub.xi_im[jdx]
            self.npairs[jdx] = ng_min.npairs[jdx] - ng_sub.npairs[jdx]

        # Remove weight for angular scales
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

    def normalise_scales(self, sum_w=None):

        if sum_w is not None:
            sw = sum_w
        else:
            sw = self.weight

        for jdx in range(len(self.meanr)):
            self.meanr[jdx] = self.meanr[jdx] / sw[jdx]
            self.meanlogr[jdx] = self.meanlogr[jdx] / sw[jdx]

    def normalise_xi(self, sum_w=None):

        if sum_w is not None:
            sw = sum_w
        else:
            sw = self.weight

        for jdx in range(len(self.meanr)):
            self.xi[jdx] = self.xi[jdx] / sw[jdx]
            self.xi_im[jdx] = self.xi_im[jdx] / sw[jdx]

    def set_units_scales(self, sep_units):

        #if all(self.meanr) > 0:
            #print('set_units 1', self.meanr[0], self.meanr[-1])

        # Angular scales: coordinates need to be attributed at the end
        self.meanr = (self.meanr * units.rad).to(sep_units).value
        #if all(self.meanr) > 0:
            #print('set_units 2', self.meanr[0], self.meanr[-1])

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
        'physical' : False,
        'out_path' : './ggl_unions_sdss_matched.txt',
        'n_cpu': 1,
        'verbose': False,
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'sign_e1': 'int',
        'sign_e2': 'int',
        'physical': 'bool',
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
        'key_z': 'foreground redshift column name (if physical), default={}',
        'theta_min': 'minimum angular scale, default={}',
        'theta_max': 'maximum angular scale, default={}',
        'n_theta': 'number of angular scales, default={}',
        'physical' : '2D coordinates are physical [Mpc]',
        'out_path' : 'output path, default={}',
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


def create_treecorr_config(
    coord_units,
    scale_min,
    scale_max,
    sep_units,
    n_theta,
    n_cpu,
    physical,
    cosmo,
    z_arr,
):

    if physical:

        print('Create config physical')
        # Min and max angular distance at object redshift
        a_max = 1 / (1 + min(z_arr))
        a_min = 1 / (1 + max(z_arr))
        d_ang_min = ccl.angular_diameter_distance(cosmo, a_max)
        d_ang_max = ccl.angular_diameter_distance(cosmo, a_min)

        print('z = ', min(z_arr), ' ... ', max(z_arr))
        print('d_ang = ', d_ang_min, ' ... ', d_ang_max, ' Mpc')
        print('scale = ', scale_min, ' ... ', scale_max, ' Mpc')

        # Transfer physical to angular scales
        theta_min = rad_to_unit(float(scale_min) / d_ang_max, sep_units)
        theta_max = rad_to_unit(float(scale_max) / d_ang_min, sep_units)

        print('theta = ', theta_min, ' ... ', theta_max, ' arcmin')
        print('theta = ', theta_min*0.0002908882086657216, ' ... ', theta_max*0.0002908882086657216, ' rad')

    else:

        print('Create config not physical')
        # Interpret scale_{min, max} as angular scales
        theta_min = scale_min
        theta_max = scale_max

    TreeCorrConfig = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'min_sep': theta_min,
        'max_sep': theta_max,
        'sep_units': sep_units,
        'nbins': n_theta,
        'num_threads': n_cpu,
    }

    t_min = theta_min * units.Unit(sep_units)
    t_max = theta_max * units.Unit(sep_units)
    print('Min and max angular scales = ', t_min, t_max)
    print('Min and max angular scales = ', t_min.to('rad'), t_max.to('rad'))

    return TreeCorrConfig


def ng_copy(ng):

    ng_c = treecorr.NGCorrelation(ng.config)

    ng_c.meanr = ng.meanr
    ng_c.meanlogr = ng.meanlogr
    ng_c.xi = ng.xi
    ng_c.xi_im = ng.xi_im
    ng_c.varxi = ng.varxi
    ng_c.weight = np.zeros_like(ng.weight)
    for idx in range(len(ng.weight)):
        ng_c.weight[idx] = ng.weight[idx]
    ng_c.npairs = ng.npairs

    return ng_c


def ng_diff(ng_min, ng_sub):

    ng_dif = ng_copy(ng_min)

    ng_dif.meanr = ng_min.meanr - ng_sub.meanr
    ng_dif.meanlogr = ng_min.meanlogr - ng_sub.meanlogr
    ng_dif.xi = ng_min.xi - ng_sub.xi
    ng_dif.xi_im = ng_min.xi_im - ng_sub.xi_im
    ng_dif.varxi = ng_min.varxi - ng_sub.varxi
    ng_dif.weight = ng_min.weight - ng_sub.weight
    ng_dif.npairs = ng_min.npairs - ng_sub.npairs

    return ng_dif


def get_interp(x_new, x, y):

    # Create the interpolation function y(x)
    f_interp = interp1d(x, y)
    try:
        # Interpolate y to different coordinates x_new
        y_new = f_interp(x_new)
    except ValueError:
        print('Failed: Interpolating from: ', x[0], ' ... ', x[-1])
        print(' to: ', x_new[0], ' ... ', x_new[-1])
        raise

    return y_new


def ng_stack_angular(TreeCorrConfig, all_ng, n_corr):

    # Initialise combined correlation objects
    ng_comb = treecorr.NGCorrelation(TreeCorrConfig)                 

    n_bins = len(ng_comb.meanr)
    sep_units = ng_comb.sep_units

    # Add up all individual correlations
    ng_final = ng_essentials(n_bins)

    sum_w = 0
    for ng in all_ng:
        sum_w += ng.weight

    for ng in all_ng:
        ng.normalise_scales(sum_w=sum_w)
        ng.set_units_scales(sep_units)

    for ng in all_ng:
        ng_final.add(ng)

    ng_final.normalise_xi(sum_w=sum_w)

    for jdx in range(n_bins):
        ng_comb.meanr[jdx] = ng_final.meanr[jdx]
        ng_comb.meanlogr[jdx] = ng_final.meanlogr[jdx]
        ng_comb.xi[jdx] = ng_final.xi[jdx]
        ng_comb.xi_im[jdx] = ng_final.xi_im[jdx]
        ng_comb.weight[jdx] = ng_final.weight[jdx]
        ng_comb.npairs[jdx] = ng_final.npairs[jdx]

    # Angular scales: coordinates need to be attributed at the end
    ng_comb.meanlogr = (np.exp(ng_comb.meanlogr) * units.rad).to(sep_units).value
    ng_comb.meanlogr = np.log(ng_comb.meanlogr)

    return ng_comb


def ng_stack_physical(TreeCorrConfig, all_ng, cosmo, z_arr):

    # Initialise combined correlation objects
    ng_comb = treecorr.NGCorrelation(TreeCorrConfig)                 

    n_bins = len(ng_comb.rnom)
    sep_units = ng_comb.sep_units
    print('sep_units = ', sep_units)

    ng_final = ng_essentials(n_bins)

    sum_w = 0
    for ng in all_ng:
        sum_w += ng.weight

    for ng in all_ng:
        #ng.normalise_scales(sum_w=sum_w)
        ng.set_units_scales(sep_units)

    for idx, ng in enumerate(all_ng):
        if all(ng.meanr) > 0:
            print('ng r test ', idx, ng.meanr)
            break

    # New x values to interpolate on [Mpc]
    r = ng_comb.rnom

    d_ang = cosmo.angular_diameter_distance(z_arr)

    sum_w_new = 0
    for idx, ng in enumerate(all_ng):

        # Original angular x values [rad]
        x = ng.meanr
        if any(x < 1e-10):
            continue

        print(idx, x[0], x[-1], 'arcmin', z_arr[idx])

        # New x values: transfer from physical [Mpc] to angular [sep_units]
        x_new = r / d_ang[idx]
        x_new = (x_new * units.rad).to(sep_units).value
        print('idx, d_ang, r = ', idx, d_ang[idx], x_new[0], x_new[-1], ' arcmin')

        # gamma_t

        # Interpolate to new angular coordinates and add (= stack)
        y = ng.xi
        y_new = get_interp(x_new, x, y)
        ng_final.xi += y_new

        # Weights
        y = ng.weight
        y_new = get_interp(x_new, x, y)
        sum_w_new += y_new

    ng_final.normalise_xi(sum_w=sum_w_new)

    for jdx in range(n_bins):
        ng_comb.meanr[jdx] = ng_final.meanr[jdx]
        ng_comb.meanlogr[jdx] = ng_final.meanlogr[jdx]
        ng_comb.xi[jdx] = ng_final.xi[jdx]
        ng_comb.xi_im[jdx] = ng_final.xi_im[jdx]
        ng_comb.weight[jdx] = ng_final.weight[jdx]
        ng_comb.npairs[jdx] = ng_final.npairs[jdx]

    # Angular scales: coordinates need to be attributed at the end
    ng_comb.meanlogr = (np.exp(ng_comb.meanlogr) * units.rad).to(sep_units).value
    ng_comb.meanlogr = np.log(ng_comb.meanlogr)


    return ng_comb 


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

    print('MKDEBUG cut to 10M')
    data['bg'] = data['bg'][:10_000_000]

    physical_test = False
    if physical_test:
        print('MKDEBUG physical test')

    #import matplotlib.pylab as plt
    #plt.plot(data['bg']['RA'], data['bg']['DEC']


    # Set treecorr catalogues
    coord_units = 'degrees'
    cats = {}
    if params['verbose']:
        print(
            'Signs for ellipticity components ='
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

    if params['physical']:
        cosmo = ccl.Cosmology(
            Omega_c=0.27,
            Omega_b=0.045,
            h=0.67,
            sigma8=0.83,
            n_s=0.96,
        )

        if physical_test:
            n_theta = params['n_theta']
        else:
            n_theta = 100
    else:
        cosmo = None

        n_theta = params['n_theta']

    # Create treecorr catalogues
    for sample in ('fg', 'bg'):

        # Split cat into single objects if fg and physical
        if sample == 'fg' and params['physical']:
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

    # Set treecorr config info (array) for correlation.
    sep_units = 'arcmin'
    TreeCorrConfig = create_treecorr_config(
        coord_units,
        params['theta_min'],
        params['theta_max'],
        sep_units,
        n_theta,
        params['n_cpu'],
        not physical_test,
        cosmo,
        data['fg'][params['key_z']],
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
            ng_essentials.copy(ng_prev, ng)

            # Perform correlation
            ng.process_cross(cat_fg, cats['bg'][0], num_threads=params['n_cpu'])

            # Last correlation = difference betwen two cumulative results
            ng_diff = ng_essentials(n_theta)
            ng_diff.difference(ng, ng_prev)
            all_ng.append(ng_diff)

            if all(ng_diff.weight) > 1e-10:
                n_corr += 1
                if n_corr < 5:
                    print('MKDEBUG ', idx, ng_diff.meanr[4] / 0.0002908882086657216, ng.rnom[4])

            # Update previous cumulative correlations
            ng_essentials.copy(ng_prev, ng)

        varg = treecorr.calculateVarG(cats['bg'])
        ng.finalize(varg)

        print(f'Computed {n_corr} non-zero correlations')

    else:

        # One foreground catalogue: run single simultaneous correlation
        ng.process(cats['fg'][0], cats['bg'][0], num_threads=params['n_cpu'])

    if params['physical']:

        if physical_test:
            ng_comb = ng_stack_angular(TreeCorrConfig, all_ng, n_corr)
        else:
            # Create config for correlations stacked on physical scales
            TreeCorrConfig = create_treecorr_config(
                'deg',
                params['theta_min'],
                params['theta_max'],
                sep_units,
                params['n_theta'],
                1,
                False,
                None,
                None,
            )
            ng_comb = ng_stack_physical(TreeCorrConfig, all_ng, cosmo, data['fg'][params['key_z']])


        # Without the following line the ng correlation internally
        # combined by treecorr will be used
        ng = ng_comb

    # Write to file
    out_path = params['out_path']
    if params['verbose']:
        print(f'Writing output file {out_path}')
    ng.write(
        out_path,
        rg=None,
        file_type=None,
        precision=None
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
