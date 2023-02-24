#!/home/ustc/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
This script is for measuring lensing signal with TreeCorr module in angular scale. 

:Author: Qinxun Li <liqinxun@mail.ustc.edu.cn>
:Date: 2023
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import treecorr as tc
import pandas as pd
import multiprocessing as mp
import numpy.random as random
import time
h=cosmo.H0.value/100
pi=np.pi

Num_patch=[20,40,150]
catalogue_path = '/data/Qinxun/UNIONS_shape/Lensfit/lensfit_goldshape_2022v1.fits' # The path of shape catalog
lens_rou='/data/Qinxun/SDSSGroup/' # The path of lens catalog
lens_key=['low','high'] # 
lens_suffix='' # Suffix of lens catalog file name
sep_arc_min=0.1 # Smallest radius in phycical scale, unit: Mpc/h
sep_arc_max=2 # Largest radius in phycical scale, unit: Mpc/h
nbin=10 # Number of radial bins
Nsamp=500 #Times of boostrap sampling
Npro=60 # Number of processes
r=np.geomspace(sep_arc_min,sep_arc_max,nbin+1)
r_ph=(r[1:]+r[:-1])/2

def calculate_shear(lensdata,outputpath):
    # jobs = []
    # for gal_idx,gal_l in lensdata.iterrows():
    #     jobs.append(pool.apply_async(shear,[gal_idx,gal_l,len(lensdata['ra'])]))

    # res = []
    # for job in jobs:
    #     # if job.get() != 0:
    #     # print(job.get())
    #     if job.get() != 0:
    #         res.append(job.get())

        
    cattmp=tc.Catalog(ra=lensdata['ra'], dec=lensdata['dec'],ra_units='degrees', dec_units='degrees')
    NG = tc.NGCorrelation(min_sep=sep_arc_min, max_sep=sep_arc_max, nbins=nbin, sep_units='degrees',bin_type='Log',metric='Arc')
    print('Start calculating')
    NG.process(cattmp,cat)
    NG.write(outputpath)
    gamm1=NG.xi
    gamm2=NG.xi_im
    gerr=NG.varxi
    
    
    return [gamm1,gamm2,gerr]

def write_res(res,path):
    lensdata=pd.DataFrame([])
    lensdata['r']=r_ph
    lensdata['gt']=res[0]
    lensdata['gerr']=res[2]
    lensdata['gx']=res[1]
    # lensdata['gxerr']=res[3]
    lensdata.to_csv(path)
    return 0


# for i in range(2):
#     lens_name='Merge_UNIONS_%s%s'%(lens_key[i],lens_suffix)
#     lens_path='%s/%s'%(lens_rou,lens_name)
#     lensdata=np.loadtxt(lens_path,unpack=True)
#     index=['ra','dec','z','logM','logM_err','w']
#     lensdata=pd.DataFrame(lensdata.T,columns=index)

#     res=calculate_shear(pool,lensdata,Nsamp)
#     output_path='/data/Qinxun/KiDSdata/Shear_signal/shear_%s'%(lens_name)
#     write_res(res,output_path)
# for i in range(2):
sourcedata = fits.getdata(catalogue_path, 1)

# for i in range(3):
cat = tc.Catalog(ra=sourcedata['ra'], dec=sourcedata['dec'], g1=sourcedata['e1'], g2=sourcedata['e2'],w=sourcedata['w'], ra_units='degrees', dec_units='degrees')
print('Finish loading source catalog')
# Load source catalog
lens_name='modelC_group_Mh114.5+'
lens_path='%s/%s.csv'%(lens_rou,lens_name)
# lensdata=np.loadtxt(lens_path,unpack=True)
lensdata=pd.read_csv(lens_path)
# lensdata=np.loadtxt(lens_path,unpack=True)
# index=['ra','dec','z','logM','logM_err']
# lensdata=pd.DataFrame(lensdata.T,columns=index)
lensdata['w']=np.ones_like(lensdata['ra'])
print('Finish loading Lens catalog')


start_time=time.time()
output_path='/data/Qinxun/KiDSdata/Shear_signal/shear_%s_lensfit_0'%(lens_name)
res=calculate_shear(lensdata,output_path)

# write_res(res ,output_path)

end_time=time.time()
print('Running time:%.2f'%(end_time-start_time))
