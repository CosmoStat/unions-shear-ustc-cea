# -*- coding: utf-8 -*-

"""THEORY MODULE.

:Description: This module provides theoretical predictions of
    weak-lensing observables.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>, Elisa Russier

"""

import numpy as np

import time

from scipy.special import erf

from astropy import units

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
    log10_Mmin,
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
        lMmin_0=log10_Mmin,
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
        Tangential shear at input scales
        ell
        cls

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

    # Angular cross-power spectrum
    if ell is None:
        ell_min = 2
        ell_max = 100_000
        ell = np.arange(ell_min, ell_max)

    # MKDEBUG to check: bias twice?
    if pk_gm_info['model_type'] == 'linear_bias':
        pk_gm = pk_gm_theo(cosmo, pk_gm_info['bias_1'])
    elif pk_gm_info['model_type'] == 'HOD':
        pk_gm = pk_gm_theo_hod(cosmo, pk_gm_info['log10_Mmin'])
    else:
        raise ValueError(
            'Invalid power-spectrum model type '
            + pk_gm_info['model_type']
        )

    cls_gG = ccl.angular_cl(
        cosmo,
        tracer_g,
        tracer_l,
        ell,
        p_of_k_a=pk_gm,
        limber_integration_method='qag_quad'
    )

    # Tangential shear
    gt = ccl.correlation(
        cosmo,
        ell,
        cls_gG,
        theta_deg,
        type='NG',
        method=integr_method,
    )

    return gt, ell, cls_gG


def gamma_t_theo_phys(
        r_Mpc,
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
    r_Mpc: array
        Angular physicak scales in Mpc
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
        Tangential shear at input scales

    """
    z_lens = dndz_lens[0]
    nz_lens = dndz_lens[1]
    bias_g = np.ones_like(z_lens)

    # Multiply galaxy bias with linear bias according to model.
    # For HOD model galaxy bias is implemented in model, leave
    # value to unity here.
    if pk_gm_info['model_type'] == 'linear_bias':
        bias_g *= pk_gm_info['bias_1']

    # 2D tracers

    # Weak lensing (sources)
    n_nz = len(dndz_source[0])
    tracer_l = ccl.WeakLensingTracer(
        cosmo,
        dndz=dndz_source,
        n_samples=n_nz,
    )

    # Angular cross-power spectrum
    if ell is None:
        ell_min = 2
        ell_max = 100_000
        ell = np.arange(ell_min, ell_max)

    # MKDEBUG to check: bias twice?
    if pk_gm_info['model_type'] == 'linear_bias':
        pk_gm = pk_gm_theo(cosmo, pk_gm_info['bias_1'])
    elif pk_gm_info['model_type'] == 'HOD':
        pk_gm = pk_gm_theo_hod(cosmo, pk_gm_info['log10_Mmin'])
    else:
        raise ValueError(
            'Invalid power-spectrum model type '
            + pk_gm_info['model_type']
        )

    # Galaxies (lenses)

    gt = []
    n_sub = 20
    if len(z_lens) % n_sub != 0:
        raise ValueError('n_sub is not divider of #nz_lens')
    z_lens_sub = np.split(z_lens, n_sub)
    nz_lens_sub = np.split(nz_lens, n_sub)
    bias_g_sub = np.split(bias_g, n_sub)

    nz_lens_mean_sub = []
    for idx in range(len(z_lens_sub)):
        tracer_g_sub = ccl.NumberCountsTracer(
            cosmo,
            False,
            dndz=(z_lens_sub[idx], nz_lens_sub[idx]),
            bias=(z_lens_sub[idx], bias_g_sub[idx]),
        )

        cls_gG_sub = ccl.angular_cl(
            cosmo,
            tracer_g_sub,
            tracer_l,
            ell,
            p_of_k_a=pk_gm,
            limber_integration_method='qag_quad'
        )

        a_lens = 1 / (1 + np.mean(z_lens_sub[idx]))
        d_ang = cosmo.angular_diameter_distance(a_lens)
        theta_rad = (r_Mpc / d_ang) * units.radian
        theta_deg = theta_rad.to('degree')

        # Tangential shear
        gt_sub = ccl.correlation(
            cosmo,
            ell,
            cls_gG_sub,
            theta_deg,
            type='NG',
            method=integr_method,
        )

        # Mean n(z) of sub-slice
        nz_lens_mean_sub.append(np.mean(nz_lens_sub[idx]))

        gt.append(gt_sub * nz_lens_mean_sub[idx])

    gt_tot = np.average(gt, axis=0, weights=nz_lens_mean_sub)

    return gt_tot


def pk_gm_theo_IA(cosmo, bias_1, dndz_lens, a_1_IA = 1.,log10k_min=-4, log10k_max=2, nk_per_decade=20):
    """PK GM THEO IA.

    3D galaxy-matter power spectrum for intrinsic alignments

    Parameters
    ----------
    cosmo : pyccl.core.Cosmology
        Cosmological parameter
    bias_1 : float
        linear bias
    a_1_IA : float

    dndz_lens : tuple of arrays
        Redshift distribution of galaxies
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
    # Redshift distribution of lens galaxies
    z = np.linspace(dndz_lens[0][0], dndz_lens[0][-1], 1000)
    # Normalization of IA tracers
    c_1, c_d, c_2 = pt.translate_IA_norm(cosmo, z, a1=a_1_IA, Om_m2_for_c2 = False)
    # Tracers
    # Galaxies with constant linear bias
    ptt_g = pt.PTNumberCountsTracer(b1=bias_1)

    # Dark matter
    ptt_IA_NLA = pt.PTIntrinsicAlignmentTracer(c1 = (z, c1))

    # Power spectrum pre-computation
    ptc = pt.PTCalculator(
        with_NC=True,
        with_IA=True,
        log10k_min=log10k_min,
        log10k_max=log10k_max,
        nk_per_decade=nk_per_decade,
    )

    # 3D galaxy - dark-matter cross power spectrum
    pk_gIA = pt.get_pt_pk2d(cosmo, ptt_g, tracer2=ptt_IA_NLA, ptc=ptc)

    return pk_gIA


def C_ell_pw(
    dndz_lens,
    rpar_min,
    r_transverse,
    bias_1,
    source_data_1_bin,
    sdss_data_cut,
    cosmo_ccl_1_bin,
    cosmo_ap_1_bin
):
    z_lens = dndz_lens[0] #redshift bins, named z_l_av_1_bin on previous code
    n_z_lens = dndz_lens[1] #number of galaxies in each redshift bins, named len(z_l_av_1_bin) on previous code
    r_s_1_bin = []
    z_s_1_bin = [] #has the redshift threshold value of the redshift bin
    t_g_1_bin = []
    t_l_1_bin = []
    C_ell = []
    g_t_1_bin_1 = []
    n_z_l_eff = []
    theta_1_bin = []
    ell_1_bin_1 = np.logspace(np.log(2),5,100000)
    count = 0
    for i in range (len(n_z_lens)):
        nz_s_1_bin = []
    #select from which range in redshift (that will be converted to r_par)
    #we want to cross correlate lens and source samples
        r_s_1_bin = cosmo_ap_1_bin.comoving_distance(z_lens[i]) + rpar_min*u.Mpc #Mpc #to convert z_l_i to r_li we need to have r_par (connu) = r_s_i (inconnu) - r_l_i (connu) > 0 (= min_rpar used in Romain's code)
        z_s_1_bin.append(astropy.cosmology.z_at_value(cosmo_ap_1_bin.comoving_distance, r_s_1_bin)) #to convert r_s_i to z_s_i
        if (z_s_1_bin[i] > 2):
            continue
        count += 1
        n_z_l_eff.append(n_z_lens[i])
        mask_z_s_1_bin = source_data_1_bin['Z'] >= z_s_1_bin[i]
    #print("z_l = ", z_l_av_1_bin[i], "z_s = ", z_s_1_bin[i])
    #print(len(np.where(mask_z_s==True)), len(source_data['Z']))
        nz_s_1_bin, z_s_f_1_bin = np.histogram(source_data_1_bin['Z'][mask_z_s_1_bin], bins = 20, density = True) #create a histogram from the redshift value in which we start our pair wise corr
    #subslice
        mask_z_l_1_bin = (sdss_data_cut['Z'] >= z_lens[i]) & (sdss_data_cut['Z'] < z_lens[i + 1])
        nz_l_1_bin_sub, z_l_f_1_bin = np.histogram(sdss_data_cut['Z'][mask_z_l_1_bin], bins = 5, density = True)
        z_l_f_av_1_bin = []
        for j in range(len(z_l_f_1_bin) - 1):
            slice_z_l_f_1_bin = z_l_f_1_bin[j: j+2]
            z_l_f_av_1_bin.append(np.mean(slice_z_l_f_1_bin))
        z_s_f_av_1_bin = []
        for k in range(len(z_s_f_1_bin) - 1):
            slice_z_s_f_1_bin = z_s_f_1_bin[k: k+2]
            z_s_f_av_1_bin.append(np.mean(slice_z_s_f_1_bin))
    #print("i = ", i, "z_s_f_av=", z_s_f_av_1_bin)#, "nz_s", nz_s_1_bin)
    # Lens
        t_g_1_bin = ccl.NumberCountsTracer(cosmo_ccl_1_bin, False, dndz=(z_l_f_av_1_bin, nz_l_1_bin_sub), bias=(z_l_f_av_1_bin, np.ones_like(z_l_f_av_1_bin)*bias_1))
    # Source
        t_l_1_bin = ccl.WeakLensingTracer(cosmo_ccl_1_bin, dndz=(z_s_f_av_1_bin, nz_s_1_bin))
        C_ell.append(ccl.angular_cl(cosmo_ccl_1_bin, t_g_1_bin, t_l_1_bin, ell_1_bin_1, p_of_k_a=pk_gm_1_bin))
        return C_ell

def gamma_t_theo_pw(
        theta_deg,
        cosmo,
        cosmo_ap,
        dndz_lens,
        bias_1,
        rpar_min,
        ell=None,
        p_of_k=None,
        integr_method='FFTlog',
):
    """GAMMA T THEO PAIRWISE.

    Theoretical prediction of the tangential shear of a source
    population around lenses within a rpar distance using the ccl library.

    Parameters
    ----------
    theta_deg : array
        Angular scales in degrees
    cosmo : pyccl.core.Cosmology
        Cosmological parameters
    dndz_lens : tuple of arrays
        Lens redshift distribution (z, n(z))
    bias_1 : float
        linear bias
    rpar_min : float
        minimum distance between source and lens galaxies in Mpc,
        default is 5 Mpc
    ell : array, optional
        2D Fourier mode, default is
        np.geomspace(2, 10000, 1000)
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
    bias_g = np.ones_like(z_lens) * bias_1
    tracer_g = ccl.NumberCountsTracer(
            cosmo,
            False,
            dndz=dndz_lens,
            bias=(z_lens, bias_g),
    )


    # Angular cross-power spectrum
    if ell is None:
        ell_min = 2
        ell_max = 10000
        n_ell = 1000
        ell = np.geomspace(ell_min, ell_max, num=n_ell)
        r_transverse = np.logspace(-2, 2, 10000) #needed for gamma_t(r_transverse) Mpc

    if not p_of_k:
        pk_gm = pk_gm_theo(cosmo, bias_1)
        r_transverse = np.logspace(-2, 2, 10000) #needed for gamma_t(r_transverse) Mpc
    else:
        pk_gm = p_of_k
        r_transverse = np.logspace(-2, 2, 10000) #needed for gamma_t(r_transverse) Mpc
    C_ell_1 = C_ell_pw(dndz_lens, rpar_min, r_transverse, bias_1, source_data_1_bin, sdss_data_cut, cosmo, cosmo_ap)

    theta_brut = r_transverse*u.Mpc/cosmo_ap_1_bin.comoving_distance(z_l_av_1_bin[i])
    theta_rad = theta_brut*u.radian
    theta_deg = theta_rad.to("degree")
    theta_1_bin.append(theta_deg.to_value())
    for l in range(len(C_ell_1)):
        g_t_1_bin_1.append(ccl.correlation(cosmo, ell_1_bin_1, C_ell_1[l], theta_1_bin[l], type='NG', method='FFTLog'))

    g_t_tot_1 = np.sum(g_t_1_bin_1, axis = 0)/count

    g_t_tot_weight_1 = np.average(g_t_1_bin_1, axis = 0, weights = n_z_l_eff)
    return g_t_tot_1

def gamma_t_ia_theo(
    theta_deg,
    cosmo,
    dndz_lens,
    dndz_source,
    bias_1,
    bias_1_IA,
    ell=None,
    p_of_k_IA=None,
    integr_method='FFTlog'):

    z_lens = dndz_lens[0]
    z_source = dndz_source[1]

    # 2D tracers

    # Galaxies (lenses)
    bias_g = np.ones_like(z_lens) * bias_1
    tracer_g = ccl.NumberCountsTracer(
            cosmo,
            False,
            dndz=dndz_lens,
            bias=(z_lens, bias_g),
    )

    # Weak lensing (sources)
    n_nz = len(dndz_source[0])
    bias_IA = np.ones_like(z_source) * bias_1_IA
    if not p_of_k_IA:#IA implementation within tracer
        tracer_l_IA = ccl.WeakLensingTracer(
            cosmo, has_shear = True,
            dndz=dndz_source, use_A_ia=True,
            n_samples=n_nz,
            ia_bias = (z_source, bias_IA),
        )
    else:#IA implementation within power spectrum
        tracer_l_IA = ccl.WeakLensingTracer(
            cosmo, has_shear = True,
            dndz=dndz_source, use_A_ia=False,
            n_samples=n_nz,
            ia_bias = (z_source, bias_IA),
        )

    # Angular cross-power spectrum
    if ell is None:
        ell_min = 2
        ell_max = 10000
        n_ell = 1000
        ell = np.geomspace(ell_min, ell_max, num=n_ell)

    if not p_of_k_IA: #IA implementation within tracer
        pk_gm_IA = None
    else: #IA implementation within power spectrum
        pk_gm_IA = pk_gm_theo_IA(cosmo, bias_1, dndz_lens)
    cls_gG = ccl.angular_cl(cosmo, tracer_g, tracer_l_IA, ell, p_of_k_a=pk_gm_IA)

    # Tangential shear
    gt_IA = ccl.correlation(
        cosmo,
        ell,
        cls_gG,
        theta_deg,
        type='NG',
        method=integr_method,
    )

    return gt_IA
