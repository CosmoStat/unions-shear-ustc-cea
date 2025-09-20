# %%
# flexion_create_cats.py

# Reference for simulations: https://arxiv.org/pdf/2111.08745

# %%
from IPython import get_ipython

# %%
ipython = get_ipython()

# %%
if ipython:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("load_ext", "log_cell_time")
    ipython.run_line_magic("matplotlib", "inline")

# %%
import sys
import os
import numpy as np
import pandas as pd
from astropy import units
import healpy as hp
import matplotlib.pylab as plt
from cs_util import plots as cs_plots
from sp_validation import util as sp_util

import treecorr
from LE3_WL_Products import plot_style
from LE3_WL_Products import treecorr_aux as ta

# %%
from unions_wl import flexion

# %%
# Ray-tracing simulations
simul_dir = f"{os.environ['HOME']}/astro/Runs/flexion/simuls_MAB"
fname = f"{simul_dir}/master_catalogue_boxlen220_n2048_lcdmp18mnu0v1_00000_narrow_plane_sachs_0.000001_halos_withflexion.parquet"
# %%
df = pd.read_parquet(fname)

# %%
n_tot = len(df)
sp_util.print_millified("Number of all objects", n_tot)

# Area and number density  
area_deg2 = flexion.get_area(df)
print(f"Area = {area_deg2:.2f} deg^2")

n_gal_aminm2 = len(df["theta1"]) / area_deg2 / 60 ** 2
print(f"Number density for all = {n_gal_aminm2:.2f} arcmin^{-2}")

# Downsample df to desired (e.g. Euclid) number density,
# If set to -1: no downsamping
n_goal = 80
#n_goal = -1

if n_goal > 0:
    y = n_gal_aminm2 / n_goal
    print(f"Downsampling from {n_gal_aminm2} to {n_goal} galaxies per square arcmin")
else:
    print("Keeping all galaxies")
    y = 1

n_Euc = int(n_tot / y)
indices = np.random.choice(n_tot, n_Euc, replace=False)
df_Euc = df.iloc[indices].copy()
n_Euc = len(df_Euc)
sp_util.print_millified(f"Number of objects at desired number density", n_Euc)

# %%
# Fill in all lensing information
flexion.fill_lensing_quantities(df)
flexion.fill_lensing_quantities(df_Euc)

# %%
# Select density sample
sample = "mass_z"
#sample = "kappa"
if sample == "kappa":
    n = 500
    df_fg = df.nlargest(n, 'kappa')
    z_min_bg = 0.3

    print(f"Fg selection: {n} kappa peaks")

    df_bg = df_Euc[df_Euc["z0"] >= z_min_bg]

if sample == "mass_z":
    log_Mass_min = 13.0
    z_min_fg = 0.1
    z_max_fg = 0.5
    z_min_bg = 0.6
    
    print(f"Fg selection: {log_Mass_min} {z_min_fg} {z_max_fg}")

    df_fg = df[(df["log_Mass"] > log_Mass_min) & df["z0"].between(z_min_fg, z_max_fg)]
    df_bg = df_Euc[df_Euc["z0"] >= z_min_bg]

n_fg = len(df_fg)
n_bg = len(df_bg)

print(f"Sample = {sample}")
print(f"Number of selected fg halos = {n_fg}/{n_tot} = {n_fg / n_tot:.1e}")
print(f"Number of source bg gals    = {n_bg}/{n_Euc} = {n_bg / n_tot:.1%}")

# %%
# Write FITS catalogues
flexion.write_to_fits(df_fg, "fg.fits", shape=None)
for shape in ("gamma", "F", "G"):
    flexion.write_to_fits(df_bg, f"bg_{shape}.fits", shape=shape)

# %%
# Plot footprints

# Angle differences
figsize = 10
fix, axes = plt.subplots(
    nrows=1, ncols=1, figsize=(figsize, figsize)
)
# Lens (observed, deflected) angle
ra, dec = flexion.get_ra_dec(df_fg, angle="theta")
plt.scatter(ra, dec)
cs_plots.savefig("footprint_theta_fg.png")


# For testing: compute angular stacks
do_testing = True

c2_sign = +1
flip_dec = False

if do_testing:
    flexion.test(df_fg, df_bg, c2_sign=c2_sign, flip_dec=flip_dec)