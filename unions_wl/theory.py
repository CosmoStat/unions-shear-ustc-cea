# -*- coding: utf-8 -*-                                                         
                                                                                
"""THEORY MODULE.                                                                 
                                                                                
This module provides theoretical predictions of weak-lensing observables.
                                                                                
""" 

import numpy as np

import pyccl as ccl                                                             
import pyccl.nl_pt as pt                                                        
import pyccl.ccllib as lib
from bin_sample import bin_edges2centers


def pk_gm_theo(cosmo, bias_1, log10k_min=-4, log10k_max=2, nk_per_decade=20):
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


def gamma_t_theo(                                                               
        theta_deg,                                                              
        cosmo,                                                                  
        dndz_lens,                                                              
        dndz_source,                                                            
        bias_1,                                                                 
        ell=None,                                                               
        p_of_k=None,                                                            
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
    bias_1 : float                                                              
        linear bias                                                             
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
                                                                                
    # Weak lensing (sources)                                                    
    n_nz = len(dndz_source[0])                                                  
    tracer_l = ccl.WeakLensingTracer(                                           
        cosmo,                                                                  
        dndz=dndz_source,                                                       
    )                                                                           
                                                                                
    # Angular cross-power spectrum                                              
    if ell is None:                                                             
        ell_min = 2                                                             
        ell_max = 10000                                                         
        n_ell = 1000                                                            
        ell = np.geomspace(ell_min, ell_max, num=n_ell)                         
                                                                                
    if not p_of_k:                                                              
        pk_gm = pk_gm_theo(cosmo, bias_1)                                       
    else:                                                                       
        pk_gm = p_of_k                                                          
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