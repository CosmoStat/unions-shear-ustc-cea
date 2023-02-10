
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from sklearn.neighbors import KDTree
import os
from numpy import logical_and as n_a
import pandas as pd
import multiprocessing as mp
import numpy.random as random
import time
h=cosmo.H0.value/100
pi=np.pi

catalogue_path = '/data/Qinxun/UNIONS_shape/Lensfit/lensfit_goldshape_2022v1.fits' # The path of shape catalog
lens_rou='/data/Qinxun/KiDSdata/SDSSBHlens' # The path of lens catalog
lens_key=['low','high'] # 
lens_suffix='' # Suffix of lens catalog file name

lens_name='testsample'
lens_path='%s/%s'%(lens_rou,lens_name)
output_path='/data/Qinxun/KiDSdata/Shear_signal/shear_%s_lensfit_KDTree_test'%(lens_name)

Sep_table=pd.read_csv('/data/Qinxun/AngularSeparation_Spherical2Euclidean.csv')

sep_ph_min=0.5 # Smallest radius in phycical scale, unit: Mpc/h
sep_ph_max=50 # Largest radius in phycical scale, unit: Mpc/h
# sep_ph_min=sep_ph_min_v*u.Mpc*h
# sep_ph_max=sep_ph_max_v*u.Mpc*h
Nbins=10 # Number of radial bins
Nsamp=500 #Times of boostrap sampling
Npro=80 # Number of processes
r=np.geomspace(sep_ph_min,sep_ph_max,Nbins+1)
r_ph=(r[1:]+r[:-1])/2
# Nbins=10
# r=r[0:8]/h
# r_ph=r_ph[0:7]/h

sourcedata = fits.getdata(catalogue_path, 1)
# cat = tc.Catalog(ra=sourcedata['ra'], dec=sourcedata['dec'], g1=sourcedata['e1'], g2=sourcedata['e2'],w=sourcedata['w'], ra_units='degrees', dec_units='degrees')
coor_source=np.array([sourcedata['ra'],sourcedata['dec']]).T
sourcenum=len(sourcedata['ra'])
tree=KDTree(coor_source, metric='euclidean')

FalseArray=[]
for i in range(sourcenum):
    FalseArray.append(False)
FalseArray=np.array(FalseArray)
print('Finish loading source catalog')
# Load source catalog

def reform(dat):
    dat['z']=dat['zl']
    dat['logM']=dat['logMBH']
    dat['ra']=dat['RA']
    dat['dec']=dat['DEC']
    return dat

def download_KDTree(cor_len,tree,Rmax):
    Rmax_E=np.interp(Rmax,Sep_table['d_A'],Sep_table['d_E'])# adjust the radius for source collecting. KDTree uses Euclidean metric, while our purpose is to select source galaxies with sperical (arc) metric.
    idx=tree.query_radius(cor_len,r=Rmax_E)[0]
    if len(idx) == 0:
        return 'lensing is not in our area'
    return idx.astype('int')

def shear(gal_idx,gal_l,rp,maxlens):
    wt_l=gal_l['w']
    d=cosmo.angular_diameter_distance(gal_l['z']).to(u.Mpc).value*h
    # sep_ph_min=rp[0]
    sep_ph_max=rp[-1]
    Nbins=len(rp)-1
    # sep_arc_min=(sep_ph_min/d*u.rad).to(u.deg).value
    sep_arc_max=(sep_ph_max/d*u.rad).to(u.deg).value
    sep_arc=(rp/d*u.rad).to(u.deg).value
    
    gals_s_idx = download_KDTree([[gal_l['ra'],gal_l['dec']]],tree,int(sep_arc_max)+0.2)
    idx=FalseArray.copy()
    idx[gals_s_idx]=True
    
    gals_s=sourcedata[idx]
    if type(gals_s) == str:
        return 0
    # col_s = ['RA','DEC','e1','e2','wt']
    # gals_s.columns = col_s
    # gals_s.reset_index(drop=True, inplace=True)

    xm0 = np.cos(pi/2 - gals_s['dec'] * pi / 180)
    xm1 = np.cos(pi/2 - gal_l['dec'] * pi / 180)
    xm2 = np.sin(pi/2 - gals_s['dec'] * pi / 180)
    xm3 = np.sin(pi/2 - gal_l['dec'] * pi / 180)
    xm4 = np.cos((gals_s['ra'] - gal_l['ra']) * pi / 180)
    the = np.arccos(xm0*xm1+xm2*xm3*xm4)
    
    tpsin = np.sin(((gals_s['ra']-gal_l['ra'])*np.cos(gal_l['dec']*pi/180))*pi/180)
    tpcin = np.sin((gals_s['dec']-gal_l['dec'])*pi/180)
    sph = (2*tpsin**2)/np.sin(the)**2-1
    cph = (2*tpsin*tpcin)/np.sin(the)**2
    
    gm1 = np.zeros(Nbins)
    gm2 = np.zeros(Nbins)
    wgt = np.zeros(Nbins)

    for i in range(Nbins):
        idx = n_a(the > sep_arc[i]*pi/180 , the < sep_arc[i+1]*pi/180)
        if idx.sum() == 0:
            gm1[i] = 0
            gm2[i] = 0
            wgt[i] = 0
        else:
            galb_s = gals_s[idx]
            wt =galb_s['w']
            ep=galb_s['e1']
            em=galb_s['e2']
        
            e45 = cph[idx] * ep + sph[idx] * em
            et = - sph[idx] * ep + cph[idx] * em
            gm1[i] = np.sum(et * wt)*wt_l
            gm2[i] = np.sum(e45 * wt)*wt_l
            wgt[i] = np.sum(wt)*wt_l
    print('%.2f'%((gal_idx/maxlens)*100))
    return [gm1,gm2,wgt]

    
def gamma(res,index = None,resp=1):
    nl = len(res)
    if type(index) != np.ndarray :
        index = range(nl)
    #result_d = dirname + 'result/'
    
    gamm1 = np.zeros(Nbins)
    gamm2 = np.zeros(Nbins)
    
    for i in range(Nbins):
        gm1_t = 0
        gm2_t = 0           
        wgt_t = 0
        gm1_t0 = 0
        gm2_t0 = 0           
        wgt_t0 = 0
        for k in range(nl):
            gm1_ki = res[k][0][i]
            gm2_ki = res[k][1][i]
            wgt_ki = res[k][2][i] 
            gm1_t0 += gm1_ki
            gm2_t0 += gm2_ki 
            wgt_t0 += wgt_ki           
        for k in index: 
            gm1_ki = res[index[k]][0][i]
            gm2_ki = res[index[k]][1][i]
            wgt_ki = res[index[k]][2][i]
            gm1_t += gm1_ki
            gm2_t += gm2_ki 
            wgt_t += wgt_ki
        if wgt_t==0:
            gamm1[i]=gm1_t0/wgt_t0/resp
            gamm2[i]=gm2_t0/wgt_t0/resp
        else:
            gamm1[i] = gm1_t/wgt_t/resp
            gamm2[i] = gm2_t/wgt_t/resp
    return [gamm1,gamm2]

def calculate_shear(pool,lensdata,Nsamp):
    jobs = []
    for gal_idx,gal_l in lensdata.iterrows():
        jobs.append(pool.apply_async(shear,[gal_idx,gal_l,r,len(lensdata['ra'])]))

    res = []
    for job in jobs:
        # if job.get() != 0:
        # print(job.get())
        if job.get() != 0:
            res.append(job.get())
    
    # res=np.array(res)
    
    nl = len(res)
    random.seed(2)
    idx = random.randint(nl,size = nl * Nsamp)
    
    print('Start bootstraping')
    boots = []
    for j in range(Nsamp):
        boots.append(pool.apply_async(gamma,[res,idx[j*nl:(j+1)*nl]]))

    g1  = np.zeros((Nsamp,Nbins))
    g2  = np.zeros((Nsamp,Nbins))

    j = 0
    for boot in boots:
        tmp = boot.get()
        g1[j] = tmp[0]
        g2[j] = tmp[1]
        j += 1

    g1err = np.zeros(Nbins)
    g2err = np.zeros(Nbins)

    for k in range(Nbins):
        g1err[k] = np.sqrt(np.var(g1[:,k]))
        g2err[k] = np.sqrt(np.var(g2[:,k]))

    
    gamm1, gamm2 = gamma(res)
    return [gamm1,gamm2,g1err,g2err]

def write_res(res,path):
    lensdata=pd.DataFrame([])
    lensdata['r']=r_ph
    lensdata['gt']=res[0]
    lensdata['gterr']=res[2]
    lensdata['gx']=res[1]
    lensdata['gxerr']=res[3]
    lensdata.to_csv(path)
    return 0

pool = mp.Pool(Npro)
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
# ssfix=['high','low']
# for i in range(2):

    # lensdata=np.loadtxt(lens_path,unpack=True)
lensdata=pd.read_csv(lens_path)
# lensdata=reform(lensdata)

lensdata['w']=np.ones_like(lensdata['ra'])

print('Finish loading Lens catalog')

print('Start calculating')
start_time=time.time()
res=calculate_shear(pool,lensdata,Nsamp)

write_res(res,output_path)

end_time=time.time()
print('Running time:%.2f'%(end_time-start_time))
pool.close()
