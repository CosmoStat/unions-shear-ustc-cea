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

from unions_wl import catalogue as wl_cat

from cs_util import logging
from cs_util import calc
from cs_util import plots
from cs_util import cat as cs_cat


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
        'input_path': 'SDSS_SMBH_202206.fits',
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
        'output_dir': '.',
        'output_fname_base': 'SDSS_SMBH_202206',
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


def main(argv=None):

    params, short_options, types, help_strings  = params_default()

    options = parse_options(params, short_options, types, help_strings)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

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
    #dat = dat_fits

    # To split into more equi-populated bins, compute cumulative
    # distribution function
    if params['verbose']:
        print(f'Computing cdf({params["key_logM"]})...')

    # Cut in mass if required
    if params['logM_min']:
        if params['verbose']:
            print(f'Using minumum logM = {params["logM_min"]}')
        n_all = len(dat)
        w = dat[params['key_logM']] > params['logM_min']
        for key in dat:
            dat[key] = dat[key][w]
        n_cut = len(dat)
        if params['verbose']:
            print(
                f'Removed {n_all - n_cut}/{n_all} objects below minimum mass'
            )

    # Cut in redshift if required
    if params['z_min']:
        if params['verbose']:
            print(f'Using minumum z = {params["z_min"]}')
        n_all = len(dat)
        w = dat[params['key_z']] > params['z_min']
        for key in dat:
            dat[key] = dat[key][w]
        n_cut = len(dat)
        if params['verbose']:
            print(
                f'Removed {n_all - n_cut}/{n_all} objects below'
                + ' minimum redshift'
            )

    if params['z_max']:
        if params['verbose']:
            print(f'Using maximum z = {params["z_max"]}')
        n_all = len(dat)
        w = dat[params['key_z']] < params['z_max']
        for key in dat:
            dat[key] = dat[key][w]
        n_cut = len(dat)
        if params['verbose']:
            print(
                f'Removed {n_all - n_cut}/{n_all} objects above'
                + ' maximum redshift'
            )

    # Get cumulative distribution function in log-mass
    cdf = ECDF(dat[params['key_logM']])

    # Split into two (check whether we get median from before)
    logM_bounds = wl_cat.y_equi(cdf, params['n_split'])

    # Add min and max to boundaries
    logM_bounds.insert(0, min(dat[params['key_logM']]))
    logM_bounds.append(max(dat[params['key_logM']]))

    # Create masks to select mass bins
    mask_list = []
    labels = []
    for idx in range(len(logM_bounds) - 1):
        label = f'{logM_bounds[idx]:.1g} <= logM < {logM_bounds[idx + 1]:g}'
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

    # Plot mass histograms
    xs = []
    means_logM = []
    stds_logM = []
    n_bin = 100
    for mask in mask_list:
        xs.append(dat[params['key_logM']][mask])
        mean = np.mean(dat[params['key_logM']][mask])
        std = np.std(dat[params['key_logM']][mask])
        means_logM.append(mean)
        stds_logM.append(std)
    out_name = (
        f'{params["output_dir"]}'
        + f'/hist_{params["key_logM"]}_n_split_{params["n_split"]}_u.pdf'
    )
    plots.plot_histograms(
        xs,
        labels,
        'AGN SMBH mass distribution',
        r'$\log ( M_\ast / M_\odot )$',
        'frequency',
        [min(dat[params['key_logM']]), max(dat[params['key_logM']])],
        int(n_bin / params['n_split']),
        out_name,
    )

    # Print mean values
    out_name = (                                                                
        f'{params["output_dir"]}'                                               
        + f'/mean_{params["key_logM"]}_n_split_{params["n_split"]}_u.txt'
    ) 
    with open(out_name, 'w') as f_out:
        print(
            f'# idx mean({params["key_logM"]}) ({params["key_logM"]})',
            file=f_out,
        )
        for idx, _ in enumerate(mask_list):
            print(
                f'{idx} {means_logM[idx]:.3f} {stds_logM[idx]:.3f}',
                file=f_out,
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
            weights[idz] = 1 / z_hist[idh]

            if params['idx_ref'] is not None:
                weights[idz] *= z_hist_arr[params['idx_ref']][idh]

        dat[f'w_{idx}'][mask] = weights

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

    # Save original redshift histogram to ASCII file
    for idx, mask in enumerate(mask_list):
        out_name = (
            f'{params["output_dir"]}/hist_{params["key_z"]}'
            + f'_{idx}_n_split_{params["n_split"]}_u.txt'
        )
        if params['verbose']:
            print(
                f'Writing redshift histogram #{idx+1}/{params["n_split"]}'
                f' to {out_name}'
            )
        z_hist_0 = np.append(z_hist_arr[idx], 0)
        np.savetxt(
            out_name,
            np.column_stack((z_edges_arr[idx], z_hist_0)),
            header='z dn_dz',
        )

    # Test: plot reweighted redshift histograms, which should be flat
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

    # Save reweighted redshift histogram (= flat) to ASCII file
    for idx, mask in enumerate(mask_list):
        out_name = (
            f'{params["output_dir"]}/hist_{params["key_z"]}'
            + f'_{idx}_n_split_{params["n_split"]}_w.txt'
        )
        if params['verbose']:
            print(
                f'Writing reweighted redshift histogram'
                + f' #{idx+1}/{params["n_split"]}'
                f' to {out_name}'
            )
        z_hist_rew_0 = np.append(z_hist_rew_arr[idx], 0)
        np.savetxt(
            out_name,
            np.column_stack((z_edges_rew_arr[idx], z_hist_rew_0)),
            header='z dn_dz',
        )

    # Plot reweighted mass histogram
    # Prepare input
    xs = []
    means_logM_w = []
    stds_logM_w = []
    dat_mask = {}
    for idx, mask in enumerate(mask_list):
        for key in dat:
            dat_mask[key] = dat[key][mask]
        xs.append(dat_mask[params['key_logM']])
        w = dat_mask[f'w_{idx}']
        ws.append(w)
        mean, std = calc.weighted_avg_and_std(
            dat[params['key_logM']][mask], w
        )
        means_logM_w.append(mean)
        stds_logM_w.append(std)

    # Plot
    out_name = (
        f'{params["output_dir"]}'
        + f'/hist_{params["key_logM"]}_n_split_{params["n_split"]}_w.pdf'
    )
    plots.plot_histograms(
        xs,
        labels,
        'AGN SMBH reweighted mass distribution',
        r'$\log ( M_\ast / M_\odot )$',
        'frequency',
        [min(dat[params['key_logM']]), max(dat[params['key_logM']])],
        int(params['n_bin_z_hist'] / params['n_split']),
        out_name,
        weights=ws,
        density=True,
    )

    # Print mean values
    out_name = (                                                                
        f'{params["output_dir"]}'                                               
        + f'/mean_{params["key_logM"]}_n_split_{params["n_split"]}_w.txt'
    ) 
    with open(out_name, 'w') as f_out:
        print(
            f'# idx mean({params["key_logM"]}) std({params["key_logM"]})',
            file=f_out,
        )
        for idx, _ in enumerate(mask_list):
            print(
                f'{idx} {means_logM_w[idx]:.3f} {stds_logM_w[idx]:.3f}',
                file=f_out,
            )


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
        cs_cat.write_fits_BinTable_file(cols, out_name)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
