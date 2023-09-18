#!/usr/bin/env python3

"""z_to_dndz.py

Transform (SOM-based) redshift sample galaxy data (FITS format)
to (COSMOSIS-readable) redshift histogram ASCII file.

:Authors: Lisa Goh
          Martin Kilbinger <martin.kilbinger@cea.fr>
          Qinxun Li

:Date: 2023

"""

import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt

work_dir = '.'

nz_names = [
    'blind_nz_cfis_shapepipe_2022v1.fits',
    'blind_nz_cfis_lensfit_goldshape_2022v1.fits'
]
methods = ['SP', 'LF']

blinds = ['A', 'B', 'C']

for nz_name, method in zip(nz_names, methods):
    for blind in blinds:

        # Open FITS file with (SOM-derived) redshifts
        hdu = fits.open(f'{work_dir}/{nz_name}')
        z1 = hdu[1].data['Z_%s' % blind]
        wt_som = hdu[1].data['som_w']

        # Compute and plot histogram
        n, bins, _ = plt.hist(
            z1,
            bins=200,
            range=(0, 5.0),
            density=True,
            histtype='step',
            weights=wt_som,
            label=f'{method} blind_{blind}'
        )

        plt.xlabel('Redshifts')
        plt.ylabel('n(z)')
        plt.legend(fontsize=20)

        # Save plot
        output_base = f'dndz_{method}_{blind}'
        plt.savefig(f'{output_base}.png')

        print(f'{method} {blind} {min(z1)} {max(z1)}')

        # Add zero to right-most bin boundary
        n = np.append(n, 0.0)

        # Save histogram as ASCII file
        np.savetxt(
            f'{output_base}.txt',
            np.column_stack((bins, n)),
            header='z dn_dz',
        )
