#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pyccl as ccl
import pyccl.nl_pt as pt                                                        


# In[322]:


ccl.spline_params.ELL_MAX_CORR = 500_000                                        
ccl.spline_params.N_ELL_CORR = 5_000


# In[342]:


import numpy as np
import matplotlib.pylab as plt
import time


# In[300]:


from unions_wl import catalogue as cat


# In[330]:


bias_1 = 0.91


# In[302]:


z_lens = np.arange(0.05, 1.2, 0.01)
dndz_lens = np.ones_like(z_lens)


# In[303]:


z_source = np.arange(0.2, 1.5, 0.1)
z_0 = 0.5
beta = 1.5

dndz_source = (z_source / z_0)**2 * np.exp(- (z_source / z_0) ** beta)


# In[331]:


z_lens, dndz_lens, x = cat.read_dndz('/home/mkilbing/astro/Runs/UNIONS/ggl-agn/jobs/Shen22_and_Liu19_logM_min_7_nz_0_z_0.1_0.8/hist_z_0_n_split_1_w.txt')
z_source, dndz_source, x = cat.read_dndz('/home/mkilbing/astro/data/CFIS/v1.0/nz/dndz_SP_A.txt')


# In[332]:


plt.plot(z_lens, dndz_lens)
_ = plt.plot(z_source, dndz_source)


# In[333]:


cosmo = ccl.Cosmology(                                                  
        Omega_c=0.27,                                                           
        Omega_b=0.045,                                                          
        h=0.67,                                                                 
        sigma8=0.83,                                                            
        n_s=0.96,                                                               
)

bias_g = np.ones_like(z_lens) * bias_1


# In[334]:


tracer_g = ccl.NumberCountsTracer(                                          
    cosmo,                                                              
    False,                                                              
    dndz=(z_lens, dndz_lens),                                                     
    bias=(z_lens, bias_g),                                              
)


# In[335]:


tracer_l = ccl.WeakLensingTracer(                                           
    cosmo,                                                        
    dndz=(z_source, dndz_source),                                              
    n_samples=len(dndz_source),                                           
) 


# In[336]:


log10k_min = -4
log10k_max = 2
nk_per_decade = 20

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


# In[346]:


start = time.time()

ell_min = 2
ell_max = 100_000
ell = np.arange(ell_min, ell_max)

cls_gG = ccl.angular_cl(                                                    
    cosmo,                                                                  
    tracer_g,                                                               
    tracer_l,                                                               
    ell,                                                                    
    p_of_k_a=pk_gm,                                                         
    limber_integration_method='qag_quad'                                    
)

end = time.time()
print(f'cell: {end - start:.2f}s')


# In[347]:


start = time.time()

th_min_amin = 0.15
th_max_amin = 150
n_th = 1000

theta_amin = np.logspace(np.log10(th_min_amin), np.log10(th_max_amin), num=n_th)
theta_deg = theta_amin / 60

# Tangential shear                                                          
gt = ccl.correlation(                                                       
    cosmo,                                                                  
    ell,                                                                    
    cls_gG,                                                                 
    theta_deg,                                                              
    type='NG',                                                              
    method='FFTlog',          
)          

end = time.time()
print(f'gt: {end - start:.2f}s')


# In[339]:


plt.loglog(theta_amin, gt)


# In[ ]:





# In[ ]:





# In[ ]:




