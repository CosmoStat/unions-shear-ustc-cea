#!/usr/bin/env python

"""compute_ng_binned_samples.py

Compute GGL (ng-correlation) between two (binned) input catalogues.

:Authors: Martin Kilbinger, Elisa Russier

:Date: 2022
"""

import sys

from optparse import OptionParser

from astropy.io import fits

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
        'theta_min': 0.1,
        'theta_max': 200,
        'n_theta': 10,
        'physical' : False,
        'out_path' : './ggl_unions_sdss_matched.txt',
        'n_cpu': 1,
        'verbose': True,
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'sign_e1': 'int',
        'sign_e2': 'int',
        'physical': 'bool',
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
        'theta_min': 'minimum angular scale, default={}',
        'theta_max': 'maximum angular scale, default={}',
        'n_theta': 'number of angular scales, default={}',
        'physical' : '2D coordinates are physical [Mpc] if True, default={}',
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

    # Set treecorr config info for correlation
    sep_units = 'arcmin'
    TreeCorrConfig = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'min_sep': params['theta_min'],
        'max_sep': params['theta_max'],
        'sep_units': sep_units,
        'nbins': params['n_theta'],
        'num_threads': params['n_cpu'],
    }
    ng = treecorr.NGCorrelation(TreeCorrConfig)

    # Compute correlation
    n_fg = len(cats['fg'])
    if params['verbose']:
        print(f'Correlating 1 bg with {n_fg} fg catalogues...')
    if len(cats['fg']) > 1:
        for idx, cat_fg in enumerate(cats['fg']):
            ng.process_cross(cat_fg, cats['bg'][0]) #, num_threads=params['n_cpu'])
        varg = treecorr.calculateVarG(cats['bg'])
        ng.finalize(varg)
    else:
        ng.process(cats['fg'][0], cats['bg'][0]) #, num_threads=params['n_cpu'])

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
