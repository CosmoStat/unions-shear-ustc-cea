#!/usr/bin/env python

"""fit_ggl_all.py

Fit GGL models to (previously computed) ng correlation.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import sys

import numpy as np
from joblib import Parallel, delayed

from uncertainties import ufloat
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import ascii

from optparse import OptionParser

import pyccl as ccl
from lmfit import minimize, Parameters, fit_report

from unions_wl import theory
from unions_wl import catalogue as cat

from cs_util import plots
from cs_util import logging

import treecorr


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
        'model_type': 'hod',
        'theta_min_fit_amin': 0.1,
        'theta_max_fit_amin': 300,
        'n_split_max': 2,
        'n_cpu': 1,
        'weight': 'w',
        'verbose': False,
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'theta_min_fit_amin': 'float',
        'theta_min_fit_amin': 'float',
        'n_split_max': 'int',
        'n_cpu': 'int',
    }

    # Parameters which can be specified as command line option
    help_strings = {
        'model_type': 'model type, \`linear\` or \`hod\`, default=\`{}\`',
        'theta_min_fit_amin': (
            'smallest angular scale for fit, in arcmin, default={}'
        ),
        'theta_max_fit_amin': (
            'largest angular scale for fit, in arcmin, default={}'
        ),
        'n_split_max': 'maximum number of black-hole mass bins, default={}',
        'n_cpu': 'number of CPUs for parallel processing, default={}',
        'weight': (
            'lens sample is weighted (\'w\') to create uniform redshift'
            + ' distribution, or unweighted (\'u\'), default={}'
        ),
    }

    # Options which have one-letter shortcuts (include dash, e.g. '-n')
    short_options = {
        'n_split_max': '-m',
        'n_cpu': '-n',
        'weight': '-w',
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


def read_mean_std_log_M_BH(n_split_arr, weight):
    """Read mean and std of log_M_BH.

    """
    mean_log10_M_BH = {}
    std_log10_M_BH = {}
    for n_split in n_split_arr:
        file_path = f'mean_logM_n_split_{n_split}_{weight}.txt'
        dat = ascii.read(file_path)
        mean_log10_M_BH[n_split] = {}
        std_log10_M_BH[n_split] = {}
        for idx in range(n_split):
            mean_log10_M_BH[n_split][idx] = dat['mean(logM)'][idx]
            std_log10_M_BH[n_split][idx] = dat['std(logM)'][idx]

    return mean_log10_M_BH, std_log10_M_BH


def read_z_data(n_split_arr, weight, shapes, blinds):

    # Set up dicts for sources and lenses
    z_centers = {}
    nz = {}
    for sample in ('source', 'lens'):
        z_centers[sample] = {}
        nz[sample] = {}

    # fg redshift distribution
    sample = 'lens'
    for n_split in n_split_arr:
        z_centers[sample][n_split] = {}
        nz[sample][n_split] = {}
        for idx in range(n_split):
            dndz_path = f'hist_z_{idx}_n_split_{n_split}_{weight}.txt'
            z_centers[sample][n_split][idx], nz[sample][n_split][idx], _ = (
                cat.read_dndz(dndz_path)
            )

    # bg redshift distribution
    sample = 'source'
    for sh in shapes:
        z_centers[sample][sh] = {}
        nz[sample][sh] = {}

        for blind in blinds:
            dndz_path = f'dndz_{sh}_{blind}.txt'
            z_centers[sample][sh][blind], nz[sample][sh][blind], _ = (
                cat.read_dndz(dndz_path)
            )

    return z_centers, nz


def read_correlation_data(n_split_arr, weight, shapes):
    # correlation data

    ng = {}

    for n_split in n_split_arr:
        ng[n_split] = {}
        for idx in range(n_split):
            ng[n_split][idx] = {}
            for sh in shapes:
                ng_path = (
                    f'{sh}/ggl_agn_{idx}_n_split_{n_split}_{weight}.fits'
                )
                ng[n_split][idx][sh] = cat.get_ngcorr_data(ng_path)

    return ng


def plot_data_only(ng, sep_units, n_split_arr, weight, shapes):

    # Plot data only
    fac = 1.05
    pow_idx = 0.8
    xlabel = rf'$\theta$ [{sep_units}]'
    ylabel = rf'$\theta^{pow_idx} \gamma_{{\rm t}}(\theta)$'
    colors = {
        'SP': 'r',
        'LF': 'b',
    }
    eb_linestyles = ['-', ':', '-.']
    for n_split in n_split_arr:
        x = []
        y = []
        dy = []
        labels = []
        ls = []
        eb_ls = []
        col = []
        title = f'{n_split} {weight}'
        my_fac = 1 / fac
        for sh in shapes:
            for idx in range(n_split):
                this_x = ng[n_split][idx][sh].meanr
                x.append(this_x * my_fac)
                my_fac *= fac
                y.append(ng[n_split][idx][sh].xi * this_x ** pow_idx)
                dy.append(
                    np.sqrt(ng[n_split][idx][sh].varxi) * this_x ** pow_idx
                )
                labels.append(f'{sh} {idx}')
                ls.append('')
                eb_ls.append(eb_linestyles[idx])
                col.append(colors[sh])

        out_path = f'gamma_t_n_split_{n_split}_{weight}_log.pdf'
        plots.plot_data_1d(
            x,
            y,
            dy,
            title,
            xlabel,
            ylabel,
            out_path=out_path,
            xlog=True,
            ylog=True,
            labels=labels,
            colors=col,
            linestyles=ls,
            eb_linestyles=eb_ls,
        )


def g_t_model(params, x_data, extra):
    """G_T_Model.

    Tangential shear model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.array
        x-values of the data (angular scales in deg)
    extra : dict
        additional parameters

    Returns
    -------
    numpy.array
        y-values of the model (tangential shear)

    """
    theta_arr_deg = x_data
    cosmo = extra['cosmo']

    z_centers = {}
    nz = {}
    for sample in ('source', 'lens'):
        z_centers[sample] = extra[f'z_centers_{sample}']
        nz[sample] = extra[f'nz_{sample}']

    # Set up model for 3D galaxy-matter power spectrum
    pk_gm_info = {}
    if 'bias_1' in params:
        pk_gm_info['model_type'] = 'linear_bias'
        pk_gm_info['bias_1'] = params['bias_1']
    else:
        pk_gm_info['model_type'] = 'HOD'
        pk_gm_info['log10_Mmin'] = params['log10_Mmin']

    y_model, _, _ = theory.gamma_t_theo(
        theta_arr_deg,
        cosmo,
        (z_centers['lens'], nz['lens']),
        (z_centers['source'], nz['source']),
        pk_gm_info,
        integr_method='FFTlog',
    )

    return y_model


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

    Returns
    -------
    numpy.array
        residuals

    """
    y_model = g_t_model(params, x_data, extra)

    residuals = (y_model - y_data) / err

    return residuals


def set_args_minimizer(
    ng,
    theta_min_fit_amin,
    theta_max_fit_amin,
    cosmo,
    z_centers,
    nz,
    n_split_arr,
    shapes,
    blinds
):
    # Set minimizer arguments

    # Set up all minimizing tasks
    args = []
    for n_split in n_split_arr:

        for idx in range(n_split):

            for sh in shapes:

                for blind in blinds:
                    extra = {
                        'cosmo': cosmo,
                        'z_centers_lens': z_centers['lens'][n_split][idx],
                        'nz_lens': nz['lens'][n_split][idx],
                        'z_centers_source': z_centers['source'][sh][blind],
                        'nz_source': nz['source'][sh][blind],
                    }

                    x = ng[n_split][idx][sh].meanr
                    y = ng[n_split][idx][sh].xi
                    err = np.sqrt(ng[n_split][idx][sh].varxi)
                    w = (
                        (x >= theta_min_fit_amin)
                        & (x <= theta_max_fit_amin)
                    )
                    theta_deg = x[w] / 60
                    gt = y[w]
                    dgt = err[w]

                    args.append((theta_deg, gt, dgt, extra))

    return args


def get_scales_pl(ng, n_split_arr, shapes):

    # Parameters to plot angular scales for theoretical prediction
    f_theta_pl = 1.1
    n_theta_pl = 2000

    theta_arr_amin = {}
    for n_split in n_split_arr:
        theta_arr_amin[n_split] = {}
        for idx in range(n_split):
            theta_arr_amin[n_split][idx] = {}
            for sh in shapes:
                theta_arr_amin[n_split][idx][sh] = (
                    np.geomspace(
                        ng[n_split][idx][sh].meanr[0] / f_theta_pl,
                        ng[n_split][idx][sh].meanr[-1] * f_theta_pl,
                        num=n_theta_pl
                    )
                )

    return theta_arr_amin

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


def retrieve_best_fit(res_arr, args, theta_arr_amin, n_split_arr, shapes, blinds, par_name):
    # Retrieve best-fit results and models

    par_bf = {}
    std_bf = {}
    g_t = {}

    jdx = 0
    for n_split in n_split_arr:
        par_bf[n_split] = {}
        std_bf[n_split] = {}
        g_t[n_split] = {}

        for idx in range(n_split):
            par_bf[n_split][idx] = {}
            std_bf[n_split][idx] = {}
            g_t[n_split][idx] = {}

            for sh in shapes:
                par_bf[n_split][idx][sh] = {}
                std_bf[n_split][idx][sh] = {}
                g_t[n_split][idx][sh] = {}

                for blind in blinds:

                    value = res_arr[jdx].params[par_name].value
                    err = res_arr[jdx].params[par_name].stderr
                    p_dp = ufloat(value, err)

                    par_bf[n_split][idx][sh][blind] = value
                    std_bf[n_split][idx][sh][blind] = err
                    print(
                        f'{sh} {blind} {idx+1}/{n_split}'
                        + f' {par_name} = {p_dp:.2ugP}'
                    )

                    theta_arr_deg = theta_arr_amin[n_split][idx][sh] / 60
                    extra = args[jdx][3]
                    g_t[n_split][idx][sh][blind] = g_t_model(
                        res_arr[jdx].params, theta_arr_deg, extra
                    )

                    jdx += 1

    return par_bf, std_bf, g_t


def plot_data_with_fits(
    ng,
    theta_min_fit_amin,
    theta_max_fit_amin,
    g_t,
    theta_arr_amin,
    par_bf,
    par_name_latex,
    sep_units,
    n_split_arr,
    weight,
    shapes,
    blinds,
):
    # Plot results

    fac = 1.05
    xlabel = rf'$\theta$ [{sep_units}]'
    ylabel = r'$\gamma_{\rm t, \times}(\theta)$'
    labels = [r'$\gamma_{\rm t}$', r'$\gamma_\times$', 'model']
    colors = ['g', 'g', 'g', 'b', 'b', 'b', 'r', 'r', 'r']
    eb_linestyles = ['-', ':', '', '-', ':', '', '-', ':', '']

    for n_split in n_split_arr:
        for sh in shapes:
            for blind in blinds:

                x = []
                y = []
                dy = []
                labels = []
                ls = []
                title = f'{n_split} {sh} {blind} {weight}'

                my_fac = 1 / fac
                for idx in range(n_split):

                    x.append(ng[n_split][idx][sh].meanr * my_fac)
                    my_fac *= fac
                    x.append(ng[n_split][idx][sh].meanr * my_fac)
                    my_fac *= fac
                    x.append(theta_arr_amin[n_split][idx][sh])

                    y.append(ng[n_split][idx][sh].xi)
                    y.append(ng[n_split][idx][sh].xi_im)
                    y.append(g_t[n_split][idx][sh][blind])

                    for i in (0, 1):
                        dy.append(
                            np.sqrt(ng[n_split][idx][sh].varxi)
                        )
                    dy.append([])

                    labels.append(fr'$\gamma_{{\rm t}}$  ($M_\ast$ bin {idx})')
                    labels.append(fr'$\gamma_\times$ ($M_\ast$ bin {idx})')
                    value = par_bf[n_split][idx][sh][blind]
                    labels.append(fr'${par_name_latex}={value:.2f}$')

                    ls.append('')
                    ls.append('')
                    ls.append('-')

                for ymode, ystr in zip((False, True), ('lin', 'log')):
                    out_path = (
                        f'{sh}/gtx_n_split_{n_split}'
                        + f'_{blind}_{weight}_{ystr}.pdf'
                    )
                    plots.figure(figsize=(15, 10))
                    plt.axvline(x=theta_min_fit_amin, linestyle='--')
                    plt.axvline(x=theta_max_fit_amin, linestyle='--')
                    plots.plot_data_1d(
                        x,
                        y,
                        dy,
                        title,
                        xlabel,
                        ylabel,
                        out_path=None,
                        xlog=True,
                        ylog=ymode,
                        labels=labels,
                        linestyles=ls,
                        colors=colors,
                        eb_linestyles=eb_linestyles,
                    )
                    plots.savefig(out_path)


def plot_M_BH_M_halo(
    mean_log10_M_BH,
    std_log10_M_BH,
    par_bf,
    std_bf,
    n_split_arr,
    weight,
    shapes,
    blinds,
):
    # Scatter plots of black-hole versus halo mass

    colors = {
        'SP': 'r',
        'LF': 'b',
    }
    markers = {
        'A': 'o',
        'B': 'd',
        'C': 's',
    }

    for n_split in n_split_arr:
        for sh in shapes:
            for blind in blinds:

                x = []
                dx = []
                y = []
                dy = []
                for idx in range(n_split):
                    x.append(mean_log10_M_BH[n_split][idx])
                    dx.append(std_log10_M_BH[n_split][idx])
                    y.append(par_bf[n_split][idx][sh][blind])
                    dy.append(std_bf[n_split][idx][sh][blind])
                plt.figure(figsize=(10, 10))
                plt.errorbar(
                    x,
                    y,
                    xerr=dx,
                    yerr=dy,
                    marker='o',
                    linestyle='',
                )

                plt.title(f'{sh} {blind} {weight}')
                plt.xlabel(r'$\log_{10} M_{\ast} / M_\odot$')
                plt.ylabel(r'$\log_{10} M_{\rm min} / M_\odot$')
                plt.ylim(10.8, 12.4)
                plt.tight_layout()
                plots.savefig(f'{sh}/logM_BH_log_Mmin_n_split_{n_split}_{blind}_{weight}.png')

    for n_split in n_split_arr:

        fx = 1.003
        ifx = -2
        plt.figure(figsize=(10, 10))
        for sh in shapes:
            for blind in blinds:

                x = []
                dx = []
                y = []
                dy = []
                for idx in range(n_split):
                    x.append(mean_log10_M_BH[n_split][idx] * fx ** float(ifx))
                    dx.append(std_log10_M_BH[n_split][idx])
                    y.append(par_bf[n_split][idx][sh][blind])
                    dy.append(std_bf[n_split][idx][sh][blind])
                if blind == 'A':
                    label = sh
                else:
                    label = ''
                plt.errorbar(
                    x,
                    y,
                    xerr=dx,
                    yerr=dy,
                    linestyle='',
                    color=colors[sh],
                    marker=markers[blind],
                    label=label,
                )

                ifx += 1

        plt.title(f'{weight}')
        plt.xlabel(r'$\log_{10} M_{\ast} / M_\odot$')
        plt.ylabel(r'$\log_{10} M_{\rm min} / M_\odot$')
        plt.ylim(10.6, 13.2)
        plt.legend(loc='best')
        plt.tight_layout()
        plots.savefig(f'logM_BH_log_Mmin_n_split_{n_split}_{weight}.png')


def set_up_cosmo():


    ccl.spline_params.ELL_MAX_CORR = 500_000
    ccl.spline_params.N_ELL_CORR = 5_000

    cosmo = ccl.Cosmology(
            Omega_c=0.27,
            Omega_b=0.045,
            h=0.67,
            sigma8=0.83,
            n_s=0.96,
    )

    return cosmo


def main(argv=None):
    """MAIN.

    Main program.

    """
    params, short_options, types, help_strings = params_default()

    options = parse_options(params, short_options, types, help_strings)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    # Save calling command
    logging.log_command(argv)

    cosmo = set_up_cosmo()

    plt.rcParams['font.size'] = 18

    sep_units = 'arcmin'

    weight = params['weight']
    shapes = ['SP', 'LF']
    blinds = ['A', 'B', 'C']

    n_split_arr = np.arange(1, params['n_split_max'] + 1)

    if params['verbose']:
        print('Reading input files...')

    # Binned black-hole masses
    mean_log10_M_BH, std_log10_M_BH = read_mean_std_log_M_BH(
        n_split_arr,
        weight
    )

    # Redshift distributions
    z_centers, nz = read_z_data(n_split_arr, weight, shapes, blinds)

    # Correlations (tangential shear)
    ng = read_correlation_data(n_split_arr, weight, shapes)

    args = set_args_minimizer(
        ng,
        params['theta_min_fit_amin'],
        params['theta_max_fit_amin'],
        cosmo,
        z_centers,
        nz,
        n_split_arr,
        shapes,
        blinds,
    )

    # Parameters to fit
    fit_params = Parameters()
    if params['model_type'] == 'linear_bias':
        par_name = 'bias_1'
        par_name_latex = 'b'
        fit_params.add(par_name, value=1.0)
    elif params['model_type'] == 'hod':
        par_name = 'log10_Mmin'
        par_name_latex = '\log_{{10}} M_{{\\rm min}} / M_\odot'
        fit_params.add(par_name, value=12.0, min=10.0, max=15.0)
    else:
        raise ValueError(f'Invalid model type {params["model_type"]}')

    # Perform fits
    res_arr = fit(args, fit_params, params['n_cpu'], params['verbose'])

    # Retrieve best-fit parameters, errors, and models
    theta_arr_amin = get_scales_pl(ng, n_split_arr, shapes)
    par_bf, std_bf, g_t = retrieve_best_fit(
        res_arr,
        args,
        theta_arr_amin,
        n_split_arr,
        shapes,
        blinds,
        par_name,
    )

    if params['verbose']:
        print('Create plots...')
    plot_data_only(ng, sep_units, n_split_arr, weight, shapes)
    plot_data_with_fits(
        ng,
        params['theta_min_fit_amin'],
        params['theta_max_fit_amin'],
        g_t,
        theta_arr_amin,
        par_bf,
        par_name_latex,
        sep_units,
        n_split_arr,
        weight,
        shapes,
        blinds,
    )
    if params['model_type'] == 'hod':
        plot_M_BH_M_halo(
            mean_log10_M_BH,
            std_log10_M_BH,
            par_bf,
            std_bf,
            n_split_arr,
            weight,
            shapes,
            blinds,
        )

if __name__ == "__main__":
    sys.exit(main(sys.argv))
