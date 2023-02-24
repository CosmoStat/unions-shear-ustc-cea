#!/home/ustc/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
This script is for split the shape catalog into several small patches. 

:Author: Ao Wang & Qinxun Li <liqinxun@mail.ustc.edu.cn>
:Date: 2022
"""
import os
import multiprocessing as mp
import math
from astropy.io import fits
import pandas as pd

data = fits.getdata('/data/Qinxun/UNIONS_shape/ShapePipe/unions_shapepipe_2022_v1.0.fits', 1)
f=open('/data/Qinxun/UNIONS_shape/ShapePipe/unions_shapepipe.dat','w+')
for i in range(len(data['RA'])):
    print(data['RA'][i],data['Dec'][i],data['e1'][i],data['e2'][i],data['w'][i],file=f)
f.close()

dirpath = '/data/Qinxun/UNIONS_shape/ShapePipe/'
filename = dirpath + 'unions_shapepipe.dat'
cata = 'UNIONS_shapepipe'

def process_wrapper(chunkStart,chunkSize):
    num = 0
    with open(filename,'r') as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        for line in lines:
            ra = float(line.split()[0])*10
            dec= float(line.split()[1])*10
            if dec < 0 :
                dec = 'm%d'%(abs(int(dec))+1)
            else:
                dec = 'p%d'%int(dec)
            ra = int(ra)
            path = dirpath + '%s/%d'%(cata,ra)
            try:
                os.mkdir(path)
            except OSError:
                pass
            ofa = open(dirpath + '%s/%d/%s.txt'%(cata,ra,dec),'a')
            ofa.writelines(line+'\n')
            ofa.close()
            num += 1
    return num

def chunkify(fname,size = 1024*1024):
	fileEnd = os.path.getsize(fname)
	with open(fname,'rb') as f:
		chunkEnd = f.tell()
		while True:
			chunkStart = chunkEnd
			f.seek(size,1)
			f.readline()
			chunkEnd = f.tell()
			yield chunkStart, chunkEnd - chunkStart
			if chunkEnd > fileEnd:
				break

if __name__ == '__main__':
	pool = mp.Pool(mp.cpu_count())
	
	jobs = []
	for chunkStart, chunkSize in chunkify(filename):
		jobs.append(pool.apply_async(process_wrapper,(chunkStart,chunkSize)))
	
	res = []
	for job in jobs:
		res.append(job.get())

	pool.close()
	print(sum(res))
