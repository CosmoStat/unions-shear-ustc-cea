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

        # Min and max angular distance at object redshift
        a_max = 1 / (1 + min(z_arr))
        a_min = 1 / (1 + max(z_arr))
        d_ang_min = ccl.angular_diameter_distance(cosmo, a_max)
        d_ang_max = ccl.angular_diameter_distance(cosmo, a_min)

        # Transfer physical to angular scales
        theta_min = rad_to_unit(float(scale_min) / d_ang_max, sep_units)
        theta_max = rad_to_unit(float(scale_max) / d_ang_min, sep_units)

    else:

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

    print('Min and max angular scales = ', theta_min, theta_max, sep_units)

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
        raise

    return y_new


def ng_stack(ng_step, cosmo, z_arr, r_min, r_max, n_r, sep_units):

    # Angular distance at object redshift
    d_ang = ccl.angular_diameter_distance(cosmo, 1 / (1 + z_arr))

    TreeCorrConfig = create_treecorr_config(
        'deg',
        r_min,
        r_max,
        sep_units,
        n_r,
        1,
        False,
        cosmo,
        None,
    )

    ng_comb = treecorr.NGCorrelation(TreeCorrConfig)
    for idx in range(len(ng_step)):

        #if ng_step[idx].weight.sum() == 0:
            #continue

        # Transfer min and max physical to angular scales
        #theta_min = rad_to_unit(float(r_min) / d_ang[idx], sep_units)
        #theta_max = rad_to_unit(float(r_max) / d_ang[idx], sep_units)
        
        # Create array with new angular scales
        #theta = np.geomspace(theta_min, theta_max, n_r)

        #x = ng_step[idx].meanr

        # Stack on physical coordinates
        #ng_comb.xi += get_interp(theta, x, ng_step[idx].xi * ng_step[idx].weight)
        #ng_comb.weight += get_interp(theta, x, ng_step[idx].weight)

        # MKDEBUG angular coordinates
        ng_comb.meanr += ng_step[idx].meanr * ng_step[idx].weight
        ng_comb.meanlogr += ng_step[idx].meanlogr * ng_step[idx].weight
        ng_comb.xi += ng_step[idx].xi # * ng_step[idx].weight
        ng_comb.xi_im += ng_step[idx].xi_im * ng_step[idx].weight
        ng_comb.varxi += ng_step[idx].varxi * ng_step[idx].weight
        ng_comb.weight += ng_step[idx].weight
        ng_comb.npairs += ng_step[idx].npairs

    #ng_comb.meanr = theta
    ng_comb.meanr /= ng_comb.weight
    ng_comb.meanlogr /= ng_comb.weight
    ng_comb.xi /=  len(ng_step) # ng_comb.weight
    ng_comb.xi_im /= ng_comb.weight
    ng_comb.varxi /= ng_comb.weight
        
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

        #n_theta = 100
        n_theta = params['n_theta']
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
        False,
        cosmo,
        data['fg'][params['key_z']],
    )
    # MKDEBUG above False instead of:
        #params['physical'],

    # Compute correlation

    # Variance is not set during process_corr
    r_prev = np.zeros(params['n_theta'])
    logr_prev = np.zeros(params['n_theta'])
    xi_prev = np.zeros(params['n_theta'])
    xi_im_prev = np.zeros(params['n_theta'])
    w_prev = np.zeros(params['n_theta'])
    np_prev = np.zeros(params['n_theta'])

    all_r = []
    all_logr = []
    all_xi = []
    all_xi_im = []
    sum_w = np.zeros(params['n_theta'])
    sum_np = np.zeros(params['n_theta'])

    n_fg = len(cats['fg'])
    if params['verbose']:
        print(f'Correlating 1 bg with {n_fg} fg catalogues...')

    ng = treecorr.NGCorrelation(TreeCorrConfig)
    if len(cats['fg']) > 1:

        # More than one foreground catalogue: run individual correlations
        for idx, cat_fg in tqdm(
            enumerate(cats['fg']),
            total=len(cats['fg']),
            disable=not params['verbose'],
        ):

            # Save previous cumulative correlations (empty if first)
            for jdx in range(params['n_theta']):
                r_prev[jdx] = ng.meanr[jdx]
                logr_prev[jdx] = ng.meanlogr[jdx]
                xi_prev[jdx] = ng.xi[jdx]
                xi_im_prev[jdx] = ng.xi_im[jdx]
                w_prev[jdx] = ng.weight[jdx]
                np_prev[jdx] = ng.npairs[jdx]

            # Perform correlation
            ng.process_cross(cat_fg, cats['bg'][0], num_threads=params['n_cpu'])

            # Last correlation = difference betwen two cumulative results
            sum_w += ng.weight - w_prev
            sum_np += ng.npairs - np_prev
            all_r.append(ng.meanr - r_prev)
            all_logr.append(ng.meanlogr - logr_prev)
            all_xi.append(ng.xi - xi_prev)
            all_xi_im.append(ng.xi_im - xi_im_prev)

            # Update previous cumulative correlations
            for jdx in range(params['n_theta']):
                r_prev[jdx] = ng.meanr[jdx]
                logr_prev[jdx] = ng.meanlogr[jdx]
                xi_prev[jdx] = ng.xi[jdx]
                xi_im_prev[jdx] = ng.xi_im[jdx]
                w_prev[jdx] = ng.weight[jdx]
                np_prev[jdx] = ng.npairs[jdx]

        varg = treecorr.calculateVarG(cats['bg'])
        ng.finalize(varg)

    else:

        # One foreground catalogue: run single simultaneous correlation
        ng.process(cats['fg'][0], cats['bg'][0], num_threads=params['n_cpu'])

    if params['physical']:

        ng_comb = treecorr.NGCorrelation(TreeCorrConfig)                 
        xi_final = np.zeros(params['n_theta'])
        xi_im_final = np.zeros(params['n_theta'])

        for idx in range(len(all_r)):
            ng_comb.meanr += all_r[idx]
            ng_comb.meanlogr += all_logr[idx]
            xi_final += all_xi[idx]
            xi_im_final += all_xi_im[idx]

        ng_comb.meanr = ((ng_comb.meanr / sum_w) * units.rad).to(sep_units).value

        # Log scales
        logr_tmp = ng_comb.meanlogr / sum_w
        r_tmp = (np.exp(logr_tmp) * units.rad).to(sep_units).value
        ng_comb.meanlogr = np.log(r_tmp)

        for jdx in range(params['n_theta']):
            ng_comb.xi[jdx] = xi_final[jdx] / sum_w[jdx]
            ng_comb.xi_im[jdx] = xi_im_final[jdx] / sum_w[jdx]
            ng_comb.weight[jdx] = sum_w[jdx]
            ng_comb.npairs[jdx] = sum_np[jdx]

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
