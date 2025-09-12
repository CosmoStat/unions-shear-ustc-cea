# %%
# flexion_stacks.py

# Reference: https://arxiv.org/pdf/2111.08745

# When running from terminal:
# export QT_QPA_PLATFORM=offscreen

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
import os
import numpy as np
import matplotlib.pylab as plt

from cs_util import plots
plt.rcParams['font.size'] = 20

from unions_wl import run

# ## Set input parameters

# Define paramer dictrionary
params_in = {}

# Input catalogue names
params_in["input_path_fg"] = "fg.fits"
params_in["input_path_bg"] = "bg_gamma.fits"

# Other paramters
params_in["key_ra_fg"] = "RA"
params_in["key_dec_fg"] = "Dec"
params_in["key_w_bg"] = "w"
params_in["verbose"] = True
params_in["npatch"] = 15

# %%
# Create compute_ng instance
obj = run.Compute_NG()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

# %%
# 1. Angular scales, treecorr automatic stack
obj._params["scales"] = "angular"
obj._params["stack"] = "auto"
obj._params["theta_min"] = 0.05
obj._params["theta_max"] = 10
obj._params["n_theta"] = 10
out_base = "shear_cl"
obj._params["out_path"] = f"{out_base}_angular.fits"

# %%
obj.run()
# %%
obj.plot_EB(out_path=f"{out_base}.png")

# %%
plt.close()

# %%
# 2. Physical coordinates, post-processing stack
obj._params["scales"] = "physical"
obj._params["stack"] = "post"
obj._params["theta_min"] = 0.1
obj._params["theta_max"] = 10
obj._params["out_path"] = f"{out_base}_physical_post.fits"
obj._params["npatch"] = 1

# %%
obj.run()

# %%
obj.plot_EB(out_path=f"{out_base}_physical_post.png")

# %%
plt.close()

# %%
# 3. Physical coordinates, cross stack
obj._params["scales"] = "physical"
obj._params["stack"] = "cross"
obj._params["theta_min"] = 0.1
obj._params["theta_max"] = 10
obj._params["out_path"] = f"{out_base}_physical_cross.fits"
obj._params["npatch"] = 1

# %%
obj.run()

# %%
obj.plot_EB(out_path=f"{out_base}_physical_cross.png")

# %%
plt.close()