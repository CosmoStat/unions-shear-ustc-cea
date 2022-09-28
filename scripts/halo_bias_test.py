import sys

import numpy as np

import matplotlib.pylab as plt

import pyccl as ccl
from pyccl.halos import hbias

def bias_halo(cosmo, M, z):

    bh_obj = hbias.HaloBiasTinker10(cosmo)

    a = 1 / z - 1
    b = bh_obj.get_halo_bias(cosmo, M, a)

    return b


def main(argv=None):

    # Cosmology
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
    )

    z = 0.5
    M = np.geomspace(1e12, 1e14, 50)

    b = bias_halo(cosmo, M, z)

    bias_model = 'Tinker10'

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))                     
    axes.semilogx(M, b, '-', label=bias_model)
    axes.set_xlabel(r'mass [$M_\odot$]')
    axes.set_ylabel('b')
    plt.savefig(f'M_b_{bias_model}.png')


    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
