# -*- coding: utf-8 -*-

"""THEORY MODULE.

:Description: This module provides theoretical predictions of
    weak-lensing observables.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>, Elisa Russier

"""

import numpy as np

from scipy.special import erf

import pyccl as ccl
from pyccl.core import Cosmology
import pyccl.nl_pt as pt
import pyccl.ccllib as lib


def pk_gm_theo(
    cosmo,
    bias_1,
    log10k_min=-4,
    log10k_max=2,
    nk_per_decade=20,
):
    """PK GM THEO.

    3D galaxy-matter power spectrum

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology
        Cosmological parameter
    bias_1 : float
        linear bias
    log10k_min : float, optional
        minimum 3D Fourier scale (log-10), default=-4
    log10k_max : float, optional
        maximum 3D Fourier scale (log-10), default=2
    nk_per_decade : int
        number of k-modes per log-10  interval in k, default=20

    Returns
    -------
     array_like
        3D power spectrum on a grid in (k, z)

    """

   # Tracers
    # Galaxies with constant linear bias
    ptt_g = pt.PTNumberCountsTracer(b1=bias_1)

    # Dark matter
    ptt_m = pt.PTMatterTracer()

    # Power spectrum pre-computation
    ptc = pt.PTCalculator(
        with_NC=True,
        with_IA=False,
        log10k_min=log10k_min,
        log10k_max=log10k_max,
        nk_per_decade=nk_per_decade,
    )

    # 3D galaxy - dark-matter cross power spectrum
    pk_gm = pt.get_pt_pk2d(cosmo, ptt_g, tracer2=ptt_m, ptc=ptc)

    return pk_gm


def pk_gm_theo_hod(
    cosmo,
    log10k_min=-4,
    log10k_max=2,
    nk_per_decade=20,
):

    # Mass definition
    mass_def = ccl.halos.MassDef200m()

    # c(M) relation
    c_of_M = ccl.halos.ConcentrationDuffy08(mass_def)

    # Mass function
    dlogn_dlogM = ccl.halos.MassFuncTinker10(cosmo, mass_def=mass_def)

    # Halo bias
    bh_of_M = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)

    # Halo model calculator
    hmc = ccl.halos.HMCalculator(cosmo, dlogn_dlogM, bh_of_M, mass_def)

    # Halo profile for galaxies
    # MKDEBUG: gamma_t with oscillations for default parameters
    # for HOD and k_arr (max log=1).
    # The following parameters are from user-defined HOD class
    # at https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=db75468d5f0230187bccf07770948818fecea074&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4c535354444553432f43434c582f646237353436386435663032333031383762636366303737373039343838313866656365613037342f48616c6f2d6d6f64656c2d506b2e6970796e62&logged_in=false&nwo=LSSTDESC%2FCCLX&path=Halo-model-Pk.ipynb&platform=android&repository_id=197742884&repository_type=Repository&version=101

    #prof_g = ccl.halos.HaloProfileHOD(
        #c_of_M,
        #lMmin_0=12.02,
        #lMmin_p=-1.34,
        #lM0_0=6.6,
        #lM0_p=-1.43,
        #lM1_0=13.27,
        #lM1_p=-0.323,
    #)
    prof_g = ccl.halos.HaloProfileHOD(
        c_of_M,
        lMmin_0=12.02,
        lMmin_p=0,
        lM0_0=6.6,
        lM0_p=0,
        lM1_0=13.27,
        lM1_p=0,
        fc_0=1.2,
        siglM_0=0.2,
    )

    # Halo profile for mass
    prof_m = ccl.halos.profiles.HaloProfileNFW(c_of_M)

    # Halo-model 3D power spectrum
    #k_arr = np.geomspace(1e-4, 1e1, 256)
    k_arr = np.logspace(
        log10k_min,
        log10k_max,
        (log10k_max - log10k_min) * nk_per_decade,
    )
    a_arr = np.linspace(0.1, 1, 32)
    pk_gm = ccl.Cosmology.halomod_Pk2D(
        cosmo,
        hmc,
        prof_g,
        prof2=prof_m,
        normprof1=True,
        normprof2=True,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
    )


    return pk_gm


def gamma_t_theo(
        theta_deg,
        cosmo,
        dndz_lens,
        dndz_source,
        pk_gm_info,
        ell=None,
        integr_method='FFTlog',
):
    """GAMMA T THEO.

    Theoretical prediction of the tangential shear of a source
    population around lenses using the ccl library.

    Parameters
    ----------
    theta_deg : array
        Angular scales in degrees
    cosmo : pyccl.core.Cosmology
        Cosmological parameters
    dndz_lens : tuple of arrays
        Lens redshift distribution (z, n(z))
    dndz_source : tuple of arrays
        Source redshift distribution (z, n(z))
    ell : array, optional
        2D Fourier mode, default is
        np.geomspace(2, 10000, 1000)
    pk_gm_info : dict
        information about 3D galaxy-matter power spectrum
    p_of_k : array_like, optional
        3D power spectrum on a grid in (k, z). If not given,
        the function ``pk_gm_theo`` is called
    integr_method : str, optional
        Method of integration over the Bessel function times
        the angular power spectrum, default is 'FFT_log'

    Returns
    -------
    array :
        Tangential shear at scales ``theta``

    """
    z_lens = dndz_lens[0]

    # 2D tracers

    # Galaxies (lenses)
    bias_g = np.ones_like(z_lens)

    # Multiply galaxy bias with linear bias according to model.
    # For HOD model galaxy bias is implemented in model, leave
    # value to unity here.
    if pk_gm_info['model_type'] == 'linear_bias':
        bias_g *= pk_gm_info['bias_1']

    print('MKDEBUG', bias_g[0])

    tracer_g = ccl.NumberCountsTracer(
            cosmo,
            False,
            dndz=dndz_lens,
            bias=(z_lens, bias_g),
    )

    # Weak lensing (sources)
    n_nz = len(dndz_source[0])
    tracer_l = ccl.WeakLensingTracer(
        cosmo,
        dndz=dndz_source,
        n_samples=n_nz,
    )

    # Bug when adding this (documented) option:
    n_samples=n_nz,

    # Angular cross-power spectrum
    if ell is None:
        ell_min = 2
        ell_max = 10000
        n_ell = 1000
        ell = np.geomspace(ell_min, ell_max, num=n_ell)

    # to check: bias twice?
    if pk_gm_info['model_type'] == 'linear_bias':
        print('linear bias')
        pk_gm = pk_gm_theo(cosmo, pk_gm_info['bias_1'])
    elif pk_gm_info['model_type'] == 'HOD':
        print('HOD')
        pk_gm = pk_gm_theo_hod(cosmo)
    else:
        raise ValueError(
            'Invalid power-spectrum model type '
            + pk_gm_info['model_type']
        )

    cls_gG = ccl.angular_cl(cosmo, tracer_g, tracer_l, ell, p_of_k_a=pk_gm)

    # Tangential shear
    gt = ccl.correlation(
        cosmo,
        ell,
        cls_gG,
        theta_deg,
        type='NG',
        method=integr_method,
    )

    return gt
