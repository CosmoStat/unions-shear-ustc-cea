# %%
# flexion_create_cats.py

# Reference for simulations: https://arxiv.org/pdf/2111.08745

# %%
#from IPython import get_ipython

# %%
#python = get_ipython()

# %%

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
from sp_validation import io as sp_io

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

stats_file = sp_io.open_stats_file(".", "stats.log")

# %%
n_tot = len(df)
sp_util.print_millified("Number of all objects", n_tot)

# Area and number density  
area_deg2 = flexion.get_area(df)
sp_io.print_stats(f"Area = {area_deg2:.2f} deg^2", stats_file, verbose=True)

n_gal_aminm2 = len(df["theta1"]) / area_deg2 / 60 ** 2
sp_io.print_stats(f"Number density for all = {n_gal_aminm2:.2f} arcmin^{-2}", stats_file, verbose=True)

# Downsample df to desired (e.g. Euclid) number density,
# If set to -1: no downsamping
#n_goal = 60
n_goal = -1

if n_goal > 0:
    y = n_gal_aminm2 / n_goal
    sp_io.print_stats(
        f"Downsampling from {n_gal_aminm2} to {n_goal} galaxies per square arcmin",
        stats_file,
        verbose=True,
    )
else:
    sp_io.print_stats("Keeping all galaxies", stats_file, verbose=True)
    y = 1

n_Euc = int(n_tot / y)
indices = np.random.choice(n_tot, n_Euc, replace=False)
df_Euc = df.iloc[indices].copy()
n_Euc = len(df_Euc)
sp_util.print_millified(f"Number of objects at desired number density", n_Euc)

# %%
# Fill in all lensing information

# Correct results are obtained with:
# flip_dec  c1_sign c2_sign
# False     +1      +1
# True      -1      +1
flip_dec = False
c1_sign = +1
c2_sign = +1

sp_io.print_stats(
    f"flip_dec = {flip_dec}, c1/2 sign = {c1_sign:+d}/{c2_sign:+d}",
    stats_file,
    verbose=True
)
flexion.fill_lensing_quantities(df, flip_dec=flip_dec, c1_sign=c1_sign, c2_sign=c2_sign)
flexion.fill_lensing_quantities(df_Euc, flip_dec=flip_dec, c1_sign=c1_sign, c2_sign=c2_sign)

# %%
# Select density sample
#sample = "mass_z"
sample = "kappa"
if sample == "kappa":
    n = 500
    df_fg = df.nlargest(n, 'kappa')
    z_min_bg = 0.5

    sp_io.print_stats(f"Fg selection: {n} kappa peaks", stats_file, verbose=True)

    df_bg = df_Euc[df_Euc["z0"] >= z_min_bg]

if sample == "mass_z":
    log_Mass_min = 13.0
    z_min_fg = 0.3
    z_max_fg = 0.5
    z_min_bg = 0.6
    
    sp_io.print_stats(f"Fg selection: {log_Mass_min} {z_min_fg} {z_max_fg}", stats_file, verbose=True)

    df_fg = df[(df["log_Mass"] > log_Mass_min) & df["z0"].between(z_min_fg, z_max_fg)]
    df_bg = df_Euc[df_Euc["z0"] >= z_min_bg]

n_fg = len(df_fg)
n_bg = len(df_bg)

sp_io.print_stats(f"Sample = {sample}", stats_file, verbose=True)
sp_io.print_stats(f"Number of selected fg halos = {n_fg}/{n_tot} = {n_fg / n_tot:.1e}", stats_file, verbose=True)
sp_io.print_stats(f"Number of source bg gals    = {n_bg}/{n_Euc} = {n_bg / n_tot:.1%}", stats_file, verbose=True)

# %%
# Write FITS catalogues
flexion.write_to_fits(df_fg, "fg.fits", shape=None)
for shape in ("gamma", "F", "G"):
    flexion.write_to_fits(df_bg, f"bg_{shape}.fits", shape=shape)

# %%
# Plot footprints

figsize = 10
fix, axes = plt.subplots(
    nrows=1, ncols=1, figsize=(figsize, figsize)
)
# Lens (observed, deflected) angle
ra, dec = flexion.get_ra_dec(df_fg, angle="theta")
plt.scatter(df_fg["RA"], df_fg["Dec"])
cs_plots.savefig("footprint_theta_fg.png")


# For testing: compute angular stacks
do_testing = True


if do_testing:
    flexion.test(df_fg, df_bg)
