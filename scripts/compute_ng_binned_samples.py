#!/usr/bin/env python

"""compute_ng_binned_samples.py

Compute GGL (ng-correlation) between two (binned) input catalogues.

:Authors: Martin Kilbinger, Elisa Russier

:Date: 2022
"""

import sys

from tqdm import tqdm

from optparse import OptionParser

import numpy as np

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

    TreeCorrConfig_arr = []
    if not physical or True:

        # Interpret scale_{min, max} as angular scales
        my_TreeCorrConfig = {
            'ra_units': coord_units,
            'dec_units': coord_units,
            'min_sep': scale_min,
            'max_sep': scale_max,
            'sep_units': sep_units,
            'nbins': n_theta,
            'num_threads': n_cpu,
        }
        TreeCorrConfig_arr.append(my_TreeCorrConfig)
    else:

        print('MKDEBUG physical')
        # Angular distance at object redshift
        d_ang = cosmo.angular_diameter_distance(z_arr)

        n_obj = len(z_arr)
        for idx in range(n_obj):

            # Interpret scale_{min, max} as physical scales (in Mpc)

            # Transfer physical to angular scales
            theta_min = rad_to_unit(scale_min / d_ang[idx], sep_units)
            theta_max = rad_to_unit(scale_max / d_ang[idx], sep_units)

            my_TreeCorrConfig = {
                'ra_units': coord_units,
                'dec_units': coord_units,
                'min_sep': theta_min,
                'max_sep': theta_max,
                'sep_units': sep_units,
                'nbins': n_theta,
                'num_threads': n_cpu,
            }
            TreeCorrConfig_arr.append(my_TreeCorrConfig)

    return TreeCorrConfig_arr


def ng_copy(ng):

    ng_c = treecorr.NGCorrelation(ng.config)

    ng_c.meanr = ng.meanr
    ng_c.meanlogr = ng.meanlogr
    ng_c.xi = ng.xi
    ng_c.xi_im = ng.xi_im
    ng_c.varxi = ng.varxi
    ng_c.weight = ng.weight
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
    else:
        cosmo = None

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
    TreeCorrConfig_arr = create_treecorr_config(
        coord_units,
        params['theta_min'],
        params['theta_max'],
        sep_units,
        params['n_theta'],
        params['n_cpu'],
        params['physical'],
        cosmo,
        data['fg'][params['key_z']],
    )

    # Compute correlation
    ng_step = []
    n_fg = len(cats['fg'])
    if params['verbose']:
        print(f'Correlating 1 bg with {n_fg} fg catalogues...')

    ng = treecorr.NGCorrelation(TreeCorrConfig_arr[0])
    if len(cats['fg']) > 1:

        # More than one foreground catalogue: run individual correlations
        for idx, cat_fg in tqdm(
            enumerate(cats['fg']),
            total=len(cats['fg']),
            disable=not params['verbose'],
        ):
            # Store previous correlation result
            if idx == 0:
                ng_prev = treecorr.NGCorrelation(TreeCorrConfig_arr[0])
            else:
                ng_prev = ng_copy(ng)

            # Perform correlation
            ng.process_cross(cat_fg, cats['bg'][0], num_threads=params['n_cpu'])

            # Grab last correlation: difference to previous run if not first
            # call
            if idx == 0:
                ng_step.append(ng_copy(ng))
            else:
                ng_step.append(ng_diff(ng, ng_prev))

        varg = treecorr.calculateVarG(cats['bg'])
        ng.finalize(varg)

    else:

        # One foreground catalogue: run single simultaneous correlation
        ng.process(cats['fg'][0], cats['bg'][0], num_threads=params['n_cpu'])

    if params['physical']:

        ng_comb = treecorr.NGCorrelation(TreeCorrConfig_arr[0])
        for idx in range(len(ng_step)):
            ng_comb.meanr += ng_step[idx].meanr * ng_step[idx].weight
            ng_comb.meanlogr += ng_step[idx].meanlogr * ng_step[idx].weight
            ng_comb.xi += ng_step[idx].xi * ng_step[idx].weight
            ng_comb.xi_im += ng_step[idx].xi_im * ng_step[idx].weight
            ng_comb.varxi += ng_step[idx].varxi * ng_step[idx].weight
            ng_comb.weight += ng_step[idx].weight
            ng_comb.npairs += ng_step[idx].npairs

        ng_comb.meanr /= ng_comb.weight
        ng_comb.meanlogr /= ng_comb.weight
        ng_comb.xi /= ng_comb.weight
        ng_comb.xi_im /= ng_comb.weight
        ng_comb.varxi /= ng_comb.weight
        
        # Without the following line the ng correlation internally
        # combined by treecorr will be used
        #ng = ng_comb

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
