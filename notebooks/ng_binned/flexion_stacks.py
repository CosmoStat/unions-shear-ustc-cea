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

from cs_util import canfar

# %%
from unions_wl import run
# %%

# ## Set input parameters

# Define paramer dictrionary
params_in = {}

# Input catalogue names
params_in["input_path_fg"] = "fg.fits"
params_in["input_path_bg"] = "bg_shear.fits"

# Output catalogue
params_in["out_path"] = "shear_cl.fits"

# Other paramters
params_in["key_ra_fg"] = "ra"
params_in["key_dec_fg"] = "dec"
params_in["key_w_bg"] = "w"
params_in["verbose"] = True

# %%
# Create compute_ng instance
obj = run.Compute_NG()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

# %%
# 1. Angular scales, treecorr automatic stack
obj._params["theta_min"] = 0.1
obj._params["theta_max"] = 200

# %%
obj.run()
# %%
obj.plot_EB(out_path="shear_cl.png")

# %%
# 2. Physical coordinates, automatic stack
obj._params["scales"] = "physical"
obj._params["stack"] = "post"
obj._params["theta_min"] = 0.1
obj._params["theta_max"] = 10

# %%
obj.run()

# %%
obj.plot_EB(out_path="shear_cl_physical_post.png")

# %%
# 3. Physical coordinates, cross stack
obj._params["scales"] = "physical"
obj._params["stack"] = "cross"
obj._params["theta_min"] = 0.1
obj._params["theta_max"] = 10

# %%
obj.run()

# %%
obj.plot_EB(out_path="shear_cl_physical_cross.png")
# %%
