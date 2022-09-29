#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

import numpy as np
from uncertainties import ufloat
import matplotlib.pyplot as plt
from astropy.io import fits

from optparse import OptionParser

import pyccl as ccl
from lmfit import minimize, Parameters, fit_report

import lenspack
from unions_wl import theory
from unions_wl import catalogue as cat

from sp_validation import util
from sp_validation import plots

import treecorr


# In[ ]:


cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
)

sep_units = 'arcmin'


# In[ ]:


dndz_dir = '/home/mkilbing/astro/data/CFIS/v1.0/nz/'


# In[ ]:


weight = 'w'


# In[ ]:


z_centers = {}
nz = {}

for sample in ('source', 'lens'):
    z_centers[sample] = {}
    nz[sample] = {}


# In[ ]:


# fg redshift distribution
sample = 'lens'

for n_split in (1, 2):
    z_centers[sample][n_split] = {}
    nz[sample][n_split] = {}
    for idx in range(n_split):
        dndz_path = f'hist_z_{idx}_n_split_{n_split}_{weight}.txt'
        z_centers[sample][n_split][idx], nz[sample][n_split][idx], _ = (
            cat.read_dndz(dndz_path)
        )


# In[ ]:


# bg redshift distribution
sample = 'source'

for sh in ('SP', 'LF'):
    z_centers[sample][sh] = {}
    nz[sample][sh] = {}

    for blind in ('A', 'B', 'C'):
        dndz_path = f'{dndz_dir}/dndz_{sh}_{blind}.txt'
        z_centers[sample][sh][blind], nz[sample][sh][blind], _ = (
            cat.read_dndz(dndz_path)
        )


# In[ ]:


# correlation data
ng = {}

for n_split in (1, 2):
    ng[n_split] = {}
    for idx in range(n_split):
        ng[n_split][idx] = {}
        for sh in ('SP', 'LF'):
            ng_path = (
                f'{sh}/ggl_agn_{idx}_n_split_{n_split}_{weight}.fits'
            )
            ng[n_split][idx][sh] = cat.get_ngcorr_data(ng_path)


# In[ ]:


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


# In[ ]:


# THeoretical prediction for plot and bias fit
n_theta_pl = 2000
f_theta_pl = 1.1

# Smallest angular scale for fit
theta_min_fit_amin = 0.2
theta_max_fit_amin = 50

#model_type = 'linear_bias'
model_type = 'hod'

if model_type == 'linear_bias':
    par_name = 'bias_1'
elif model_type == 'hod':
    par_name = 'log10_Mmin'

g_t = {}
theta_arr_amin = {}
par_bf = {}

first = True

for n_split in (1, 2):
    g_t[n_split] = {}
    theta_arr_amin[n_split] = {}
    par_bf[n_split] = {}

    for idx in range(n_split):
        g_t[n_split][idx] = {}
        theta_arr_amin[n_split][idx] = {}
        par_bf[n_split][idx] = {}

        for sh in ('SP', 'LF'):
            g_t[n_split][idx][sh] = {}
            theta_arr_amin[n_split][idx][sh] = (
                np.geomspace(
                    ng[n_split][idx][sh].meanr[0] / f_theta_pl,
                    ng[n_split][idx][sh].meanr[-1] * f_theta_pl,
                    num=n_theta_pl
                )
            )
            par_bf[n_split][idx][sh] = {}

            for blind in ('A', 'B', 'C'):

                #if blind != 'A' or sh != 'SP' or n_split != 2:
                 #   continue

                theta_arr_deg = theta_arr_amin[n_split][idx][sh] / 60

                params = Parameters()
                if model_type == 'linear_bias':
                    params.add(par_name, value=1.0) #, min=0.0)
                elif model_type == 'hod':
                    params.add(par_name, value=12.0, min=10.0, max=15.0)

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
                res = minimize(
                    loss,
                    params,
                    args=(theta_deg, gt, dgt, extra)
                )

                if first:
                    print('all scales [deg] = ', theta_deg)
                    print('scales for fitting [arcmin] =', x[w])
                    first = False


                value = res.params[par_name].value
                p_dp = ufloat(value, res.params[par_name].stderr)

                par_bf[n_split][idx][sh][blind] = value
                print(
                    f'{sh} {blind} {idx+1}/{n_split}'
                    + f' {par_name} = {p_dp:.2ugP}'
                )

                g_t[n_split][idx][sh][blind] = g_t_model(
                    res.params, theta_arr_deg, extra
                )

                #if blind == 'A' and sh == 'SP' and n_split == 2:
                #    print(idx, y)
                #    print(res.params['bias_1'].value, g_t[n_split][idx][sh][blind])


# In[ ]:


fac = 1.05
xlabel = rf'$\theta$ [{sep_units}]'
ylabel = r'$\gamma_{\rm t}(\theta)$'
labels = [r'$\gamma_{\rm t}$', r'$\gamma_\times$', 'model']
#colors = ['g', 'r', 'g', 'b', 'orange', 'b']
colors = ['g', 'g', 'g', 'b', 'b', 'b']
eb_linestyles = ['-', ':', '', '-', ':', '']

for n_split in (1, 2):
    for sh in ('SP', 'LF'):
        for blind in ('A', 'B', 'C'):

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

                labels.append(fr'$\gamma_{{\rm t}}$ $M_\ast$ bin {idx}')
                labels.append(fr'$\gamma_\times$ $MM_\ast bin {idx}')
                value = par_bf[n_split][idx][sh][blind]
                labels.append(f'{par_name}={value:.2f}')

                ls.append('')
                ls.append('')
                ls.append('-')

                #if not (blind == 'A' and sh == 'SP' and n_split == 2):
                #    pass
                #continue
                #else:
                #    print(idx, ng[n_split][idx][sh].xi)
                #    print(b, g_t[n_split][idx][sh][blind])

            for ymode, ystr in zip((False, True), ('lin', 'log')):
                out_path = (
                    f'{sh}/gtx_n_split_{n_split}'
                    + f'_{blind}_{weight}_{ystr}.pdf'
                )
                #plt.axvline(x=theta_min_fit_amin)
                #plt.axvline(x=theta_max_fit_amin)
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
                    linestyles=ls,
                    colors=colors,
                    eb_linestyles=eb_linestyles,
                )


# In[ ]:




