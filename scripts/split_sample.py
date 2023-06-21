#!/usr/bin/env python3

"""split_sample.py

Split input sample into equi-populated bins.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

:Date: 2022

"""

import sys
import os

import numpy as np

from optparse import OptionParser
import matplotlib.pylab as plt

from statsmodels.distributions.empirical_distribution import ECDF

from astropy.table import Table
from astropy.io import fits

from unions_wl import catalogue as cat_wl
from unions_wl import defaults

from cs_util import logging
from cs_util import calc
from cs_util import plots
from cs_util import cat as cat_csu
from cs_util import cosmo as cosmo_csu


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
        'input_path': 'agn.fits',
        'key_ra': 'ra',
        'key_dec': 'dec',
        'key_z': 'z',
        'key_logM': 'logM',
        'logM_min': None,
        'z_min': None,
        'z_max': None,
        'n_split': 2,
        'idx_ref': None,
        'n_bin_z_hist': 100,
        'dndz_source_path' : None,
        'output_dir': '.',
        'output_fname_base': 'agn',
    }

    # Parameters which are not the default, which is ``str``
    types = {
        'logM_min': 'float',
        'z_min': 'float',
        'z_max': 'float',
        'n_split': 'int',
        'idx_ref': 'int',
        'n_bin_z_hist': 'int',
    }

    # Parameters which can be specified as command line option
    help_strings = {
        'input_path': 'catalogue input path, default={}',
        'key_ra': 'right ascension column name, default={}',
        'key_dec': 'declination column name, default={}',
        'key_z': 'redshift column name, default={}',
        'key_logM': 'mass (in log) column name, default={}',
        'logM_min': 'minumum mass (log), default none',
        'z_min': 'minumum redshift, default none',
        'z_max': 'maximum redshift, default none',
        'n_split': 'number of equi-populated bins on output, default={}',
        'idx_ref': (
            'bin index for reference redshift histogram, default none'
            + ' (flat weighted histograms)'
        ),
        'n_bin_z_hist': 'number of bins for redshift histogram, default={}',
        'output_dir': 'output directory, default={}',
        'output_fname_base': 'output file base name, default={}',
    }

    # Options which have one-letter shortcuts
    short_options = {
        'input_path': '-i',
        'n_split': '-n',
        'output_dir': '-o',
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
    return True


def write_mean_std_logM(
        output_dir,
        key_logM,
        n_split,
        weigh_suffix,
        mask_list,
        means_logM,
        stds_logM,
):
    """Write Mean Std LogM

    Write mean and std of log(M) to file.

    """
    out_name = (
        f'{output_dir}/mean_{key_logM}_n_split_{n_split}{weigh_suffix}.txt'
    )
    with open(out_name, 'w') as f_out:
        print( f'# idx mean({key_logM}) std({key_logM})', file=f_out)
        for idx, _ in enumerate(mask_list):
            print(
                f'{idx} {means_logM[idx]:.3f} {stds_logM[idx]:.3f}',
                file=f_out,
            )


def plot_mass_histogram(mask_list, dat, params, labels, weighted=False):

    if weighted:
        suf = '_w'
        mod = 'weighted '
    else:
        suf = '_u'
        mod = ''

    xs = []
    ws = []
    means_logM = []
    stds_logM = []
    dat_mask = {}
    for idx, mask in enumerate(mask_list):
        for key in dat:
            dat_mask[key] = dat[key][mask]

        logM = dat_mask[params['key_logM']]
        xs.append(logM)
        if weighted:
            w = dat_mask[f'w_{idx}']
        else:
            w = np.ones_like(logM)
        ws.append(w)
        mean, std = calc.weighted_avg_and_std(logM, w)
        means_logM.append(mean)
        stds_logM.append(std)

    # Plot
    out_name = (
        f'{params["output_dir"]}'
        + f'/hist_{params["key_logM"]}_n_split_{params["n_split"]}{suf}.pdf'
    )
    plots.plot_histograms(
        xs,
        labels,
        'AGN SMBH {mod}mass distribution',
        r'$\log ( M_\ast / M_\odot )$',
        'frequency',
        [min(dat[params['key_logM']]), max(dat[params['key_logM']])],
        int(params['n_bin_z_hist'] / params['n_split']),
        out_name,
        weights=ws,
        density=True,
    )

    return means_logM, stds_logM


def write_z_hist(mask_list, z_hist_arr, z_edges_arr, params, suf):

    for idx, mask in enumerate(mask_list):
        out_name = (
            f'{params["output_dir"]}/hist_{params["key_z"]}'
            + f'_{idx}_n_split_{params["n_split"]}{suf}.txt'
        )
        if params['verbose']:
            if suf == '_w':
                mod = 'reweighted '
            else:
                mod = ''
            print(
                f'Writing {mod}redshift histogram #{idx+1}/{params["n_split"]}'
                f' to {out_name}'
            )
        z_hist_0 = np.append(z_hist_arr[idx], 0)
        np.savetxt(
            out_name,
            np.column_stack((z_edges_arr[idx], z_hist_0)),
            header='z dn_dz',
        )


def plot_reweighted_z_hist(                   
    mask_list,                                                              
    dat,                                                               
    params,                                                                 
    labels,                                                                 
    z_min,                                                                  
    z_max                                                                   
):

    # Prepare input
    xs = []
    ws = []
    dat_mask = {}
    for idx, mask in enumerate(mask_list):
        for key in dat:
            dat_mask[key] = dat[key][mask]
        xs.append(dat_mask[params['key_z']])
        ws.append(dat_mask[f'w_{idx}'])

    # Plot
    out_name = (
        f'{params["output_dir"]}'
        + f'/hist_{params["key_z"]}_n_split_{params["n_split"]}_w.pdf'
    )
    z_hist_rew_arr, z_edges_rew_arr = plots.plot_histograms(
        xs,
        labels,
        'AGN SMBH reweighted redshift distribution',
        '$z$',
        'frequency',
        [z_min, z_max],
        int(params['n_bin_z_hist'] / params['n_split']),
        out_name,
        weights=ws,
        density=True,
    )

    return z_hist_rew_arr, z_edges_rew_arr


def main(argv=None):

    params, short_options, types, help_strings  = params_default()

    options = parse_options(params, short_options, types, help_strings)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    # Check whether options are valid
    if not check_options(params):
        return 1

    # Save calling command
    logging.log_command(argv)

    plt.rcParams['font.size'] = 18

    # Open input catalogue and read into dictionary
    if params['verbose']:
        print(f'Reading catalogue {params["input_path"]}...')
    dat_fits = fits.getdata(params["input_path"])
    dat = {}
    for key in dat_fits.dtype.names:
        dat[key] = dat_fits[key]

    # To split into more equi-populated bins, compute cumulative
    # distribution function
    if params['verbose']:
        print(f'Computing cdf({params["key_logM"]})...')

    # Cut in mass if required
    dat = cat_wl.cut_data(
        dat,
        params['key_logM'],
        params['logM_min'],
        '>',
        verbose=params['verbose']
    )

    # Cuts in redshift if required
    dat = wl_cat.cut_data(
        dat,
        params['key_z'],
        params['z_min'],
        '>',
        verbose=params['verbose']
    )
    dat = cat_wl.cut_data(
        dat,
        params['key_z'],
        params['z_max'],
        '<',
        verbose=params['verbose']
    )

    # Get cumulative distribution function in log-mass
    cdf = ECDF(dat[params['key_logM']])

    # Split into two (check whether we get median from before)
    logM_bounds = cat_wl.y_equi(cdf, params['n_split'])

    # Add min and max to boundaries
    logM_bounds.insert(0, min(dat[params['key_logM']]))
    logM_bounds.append(max(dat[params['key_logM']]))

    # Create masks to select mass bins
    mask_list = []
    labels = []
    for idx in range(len(logM_bounds) - 1):
        label = f'{logM_bounds[idx]:g} <= logM < {logM_bounds[idx + 1]:g}'
        labels.append(label)
        if params['verbose']:
            print(
                f'Creating sample #{idx+1}/{params["n_split"]} with {label}'
            )
        mask = (
            (dat[params['key_logM']] >= logM_bounds[idx])
            & (dat[params['key_logM']] < logM_bounds[idx + 1])
        )
        mask_list.append(mask)

    if not os.path.exists(params['output_dir']):
        os.mkdir(params['output_dir'])

    # Plot (unweighted, original) mass histograms
    means_logM, stds_logM = plot_mass_histogram(
        mask_list,
        dat,
        params,
        labels,
        weighted=False,
    )

    # Add columns for weight for each sample
    for idx in range(len(mask_list)):
        dat[f'w_{idx}'] = np.ones_like(dat[params['key_z']])

    # Assign weights according to local density in redshift histogram.

    fac = 1.0001
    z_min = min(dat[params['key_z']]) / fac
    z_max = max(dat[params['key_z']]) * fac

    z_centres_arr = []
    z_hist_arr = []
    z_edges_arr = []
    for idx, mask in enumerate(mask_list):

        z_hist, z_edges = np.histogram(
            dat[params['key_z']][mask],
            bins=int(params['n_bin_z_hist'] / params['n_split']),
            density=True,
            range=(z_min, z_max),
        )
        z_centres = [
            (z_edges[i] + z_edges[i+1]) / 2 for i in range(len(z_edges) - 1)
        ]

        z_hist_arr.append(z_hist)
        z_centres_arr.append(z_centres)
        z_edges_arr.append(z_edges)

        # Plot histogram
        plt.step(z_centres, z_hist, where='mid', label=idx)

        weights = np.ones_like(dat[params['key_z']][mask])

        for idz, z in enumerate(dat[params['key_z']][mask]):
            w = np.where(z > z_edges)[0]
            if len(w) == 0:
                print('Error:', z)
            else:
                idh = w[-1]

            # Set weight to inverse redshift distribution density
            weights[idz] = 1 / z_hist[idh]

            # If required multiply weight by reference
            # redshift distribution density
            if params['idx_ref'] is not None:
                weights[idz] *= z_hist_arr[params['idx_ref']][idh]

        dat[f'w_{idx}'][mask] = weights

    # If required multiply weights by inverse square of effective
    # critical surface mass density
    if params['Delta_Sigma']:

        cosmo = defaults.get_cosmo_default()

        # Source redshift distribution and distances
        z_source, nz_source, _ = cat_csu.read_dndz(params['dndz_source_path'])
        a_source = 1 / (1 + z_source)
        d_ang_source = cosmo.angular_diameter_distance(a_source)

        # Create spline interpolation function
        nz_source_interp = interpolate.InterpolatedUnivariateSpline(z_source, nz_source)
        d_ang_source_interp = interpolate.InterpolatedUnivariateSpline(z_source, d_ang_source)

        # Rebin source to lower number to speed up Sigma_cr computation
        n_z_source_rebin = 25
        z_source_rebin = np.linspace(z_source[0], z_source[-1], n_z_source_rebin)
        nz_source_rebin = nz_source_interp(z_source_rebin)
        d_ang_source_rebin = d_ang_source_interp(z_source_rebin)

        # Loop over lens selections
        for idx, mask in enumerate(mask_list):

            n_z_lens = 25
            z_lens = np.linspace(z_min, z_max, n_z_lens)
            a_lens = 1 / (1 + z_lens)
            d_ang_lens = cosmo.angular_diameter_distance(a_lens)
            d_ang_lens_interp = interpolate.InterpolatedUnivariateSpline(z_lens, d_ang_lens)

            sig_cr_w = np.ones_like(dat[params['key_z']][mask])

            # Loop over lens objects
            for idz, z in tqdm(
                enumerate(dat[params['key_z']][mask]),
                total=len(dat[params['key_z']][mask]),
                disable=not params['verbose'],
                desc=f'split {idx}/{params["n_split"]}',
            ):
                #a_lens = 1 / (1 + z)
                #d_ang_lens = cosmo.angular_diameter_distance(a_lens)
                d_ang_lens_spline = d_ang_lens_interp(z)
                sig_crit_m1_eff = cosmo_csu.sigma_crit_m1_eff(
                    z,
                    z_source_rebin,
                    nz_source_rebin,
                    cosmo,
                    d_lens=d_ang_lens_spline,
                    d_source_arr=d_ang_source_rebin,
                )

                sig_cr_w[idz] = sig_crit_m1_eff.value ** 2

            # Apply weights
            dat[f'w_{idx}'][mask] = dat[f'w_{idx}'][mask] * sig_cr_w

    # Plot original redshift histograms
    out_name = (
        f'{params["output_dir"]}'
        + f'/hist_{params["key_z"]}_n_split_{params["n_split"]}_u.pdf'
    )
    plots.plot_data_1d(
        z_centres_arr,
        z_hist_arr,
        [np.nan] * len(z_centres_arr),
        'AGN SMBH redshift distribution',
        '$z$',
        'frequency',
        out_name,
    )

    ## Save to ASCII file
    write_z_hist(mask_list, z_hist_arr, z_edges_arr, params, '_u')


    # Reweighted redshift histograms
    # (if idx_ref is None, this should be flat)

    ## Plot
    z_hist_rew_arr, z_edges_rew_arr = plot_reweighted_z_hist(
        mask_list,
        dat,
        params,
        labels,
        z_min,
        z_max
    )

    ## Save to disk
    write_z_hist(
        mask_list,
        z_hist_rew_arr,
        z_edges_rew_arr,
        params,
        '_w',
    )

    # Plot reweighted mass histogram
    means_logM_w, stds_logM_w = plot_mass_histogram(
        mask_list,
        dat,
        params,
        labels,
        weighted=True,
    )

    # Write unweighted and weighted logM summaries to file
    for suf, mean, std in zip(
        ['_u', '_w'],
        [means_logM, means_logM_w],
        [stds_logM, stds_logM_w]
    ):
        write_mean_std_logM(
            params['output_dir'],
            params['key_logM'],
            params['n_split'],
            suf,
            mask_list,
            mean,
            std,
        )

    # Write catalogues
    dat_mask = {}
    for idx, mask in enumerate(mask_list):
        for key in dat:
            dat_mask[key] = dat[key][mask]
        t = Table(dat_mask)
        out_name = (
            f'{params["output_dir"]}/{params["output_fname_base"]}'
            + f'_{idx}_n_split_{params["n_split"]}.fits'
        )
        print(f'Writing catalogue {out_name}')

        cols = []
        for key in t.keys():
            cols.append(fits.Column(name=key, array=t[key], format='E'))
        cat_csu.write_fits_BinTable_file(cols, out_name)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
