#!/usr/bin/env python

"""ggl_compare_data_theory.py

Compare GGL from data to theory.
Use ccl to compute theoretical prediction.

:Authors: Martin Kilbinger, Elisa Russier

:Date: 2022

"""

import sys

import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser

import pyccl as ccl

from unions_wl import theory
from unions_wl import catalogue as cat
from unions_wl import defaults

from cs_util import logging
from cs_util import plots


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
        'corr_path': 'ggl_unions_sdss_matched.txt',
        'dndz_lens_path': 'dndz_lens.txt',
        'dndz_source_path': 'dndz_source.txt',
        'bias_1': None,
        'theta_min': 0.2,
        'theta_max': 200,
        'n_theta': 20,
        'physical' : False,
        'Delta_Sigma' : False,
        'out_base': 'gamma_tx',
        'out_cls_base': None,
        'verbose': True,
    }

    # Parameters which are not the default type ``str``
    types = {
        'bias_1': 'float',
        'theta_min': 'float',
        'theta_max': 'float',
        'n_theta': 'int',
        'physical': 'bool',
        'Delta_Sigma': 'bool',
    }

    # Parameters which can be specified as command line option
    help_strings = {
        'corr_path': 'path to treecorr output file, default={}',
        'dndz_lens_path': 'path to lens redshift histogram, default={}',
        'dndz_source_path': 'path to source redshift histogram, default={}',
        'bias_1': 'linear bias, default={}',
        'theta_min': 'minimum angular scale, default={}',
        'theta_max': 'minimum angular scale, default={}',
        'n_theta': 'number of angular scales, default={}',
        'physical' : '2D coordinates are physical [Mpc], default={}',
        'Delta_Sigma' : 'excess surface mass density instead of tangential'
            + ' shear  default={}',
        'out_base': 'output base path, default={}',
        'out_cls_base': (
            'output path for angular power spectrum, default no output'
        )
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


def check_options(options):                                                     
    """Check command line options.                                              
                                                                                
    Parameters                                                                  
    ----------                                                                  
    options: tuple                                                              
        Command line options                                                    
                                                                                
    Returns                                                                     
    -------                                                                     
    bool                                                                        
        Result of option check. False if invalid option value.                  
                                                                                
    """                                                                         
    if options['Delta_Sigma'] and not options['physical']:
        print('With Delta_Sigma=True physical needs to be True')
        return False

    return True


def main(argv=None):
    """Main

    Main program

    """

    params, short_options, types, help_strings = params_default()

    options = parse_options(params, short_options, types, help_strings)


    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    if check_options(params) is False:                                         
        return 1

    # Save calling command
    logging.log_command(argv)

    plt.rcParams['font.size'] = 18

    # Default cosmology
    cosmo = defaults.get_cosmo_default()

    # Read redshift distributions
    z_centers = {}
    nz = {}
    for sample in ('lens', 'source'):
        file_path = params[f'dndz_{sample}_path']
        z_centers[sample], nz[sample], _ = cat.read_dndz(file_path)

    # Set up scales

    n_theta = 2000
    if not params['physical']:
        # Angular scales on input are in arcmin. CCL asks for degree
        theta_min_deg = params['theta_min'] / 60
        theta_max_deg = params['theta_max'] / 60
        theta_arr_deg = np.geomspace(theta_min_deg, theta_max_deg, num=n_theta)
        theta_arr_amin = theta_arr_deg * 60
        x_data = theta_arr_amin
    else:
        r_min_Mpc = params['theta_min']
        r_max_Mpc = params['theta_max']
        r_arr_Mpc = np.geomspace(r_min_Mpc, r_max_Mpc, num=n_theta)
        x_data = r_arr_Mpc

    # Set up model for 3D galaxy-matter power spectrum
    pk_gm_info = {}
    if params['bias_1'] is not None:
        pk_gm_info['model_type'] = 'linear_bias'
        pk_gm_info['bias_1'] = params['bias_1']
    else:
        pk_gm_info['model_type'] = 'HOD'
        pk_gm_info['log10_Mmin'] = 11.5

    if not params['physical']:
        y_theo, ell, cls = theory.gamma_t_theo(
            theta_arr_deg,
            cosmo,
            (z_centers['lens'], nz['lens']),
            (z_centers['source'], nz['source']),
            pk_gm_info,
            integr_method='FFTlog',
        )
    else:
        y_theo = theory.gamma_t_theo_phys(
            r_arr_Mpc,
            cosmo,
            (z_centers['lens'], nz['lens']),
            (z_centers['source'], nz['source']),
            pk_gm_info,
            integr_method='FFTlog',
            Delta_Sigma=params['Delta_Sigma'],
        )

    # Write angular power spectrum (for testing)
    if params['out_cls_base']:
        cat.write_cls(f'{params["out_cls_base"]}.txt', ell, cls)
        plots.plot_data_1d(
            [ell],
            [cls],
            [np.nan],
            'Angular shear-galaxy cross-power spectrum',
            '$\ell$',
            '$C_\ell$',
            f'{params["out_cls_base"]}.pdf',
            xlog=True,
            ylog=True,
        )


    # Read treecorr ng correlation data
    ng = cat.get_ngcorr_data(
        params['corr_path'],
        theta_min=params['theta_min'],
        theta_max=params['theta_max'],
        n_theta=params['n_theta']
    )

    # Plot everything

    fac = 1.05

    y = [ng.xi, ng.xi_im, y_theo]
    dy = [np.sqrt(ng.varxi)] * 2 + [[]]
    title = 'GGL'
    if not params['physical']:
        x = [ng.meanr, ng.meanr * fac]
        xbase = r'\theta'
        xlabel = rf'${xbase}$ [arcmin]'
    else:
        x = [ng.rnom, ng.rnom * fac]
        xbase = r'r'
        xlabel = rf'${xbase}$ [Mpc]'
    x.append(x_data)
    ylabel = rf'$\gamma_{{\rm t}}({xbase})$'
    ls = ['-', '', '-']
    eb_linestyles = ['-', ':', ':']
    labels = [r'$\gamma_{\rm t}$', r'$\gamma_\times$', 'model']
    colors = ['g', 'r', 'g']

    xlim_fac = 1.5
    xlim = (params['theta_min'] / xlim_fac, params['theta_max'] * xlim_fac)
    ylim = (5e-6, 1e-2)

    for ymode, ystr in zip((False, True), ('lin', 'log')):
        out_path = f'{params["out_base"]}_{ystr}.pdf'
        plots.plot_data_1d(
            x,
            y,
            dy,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylog=ymode,
            labels=labels,
            colors=colors,
            linestyles=ls,
            eb_linestyles=eb_linestyles,
            xlim=xlim,
            ylim=ylim,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
