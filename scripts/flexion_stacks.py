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


# %%
def set_params_in(shape):
    
    params_in = {}

    # Input catalogue names
    params_in["input_path_fg"] = "fg.fits"
    params_in["input_path_bg"] = f"bg_{shape}.fits"

    # Other paramters
    params_in["key_ra_fg"] = "RA"
    params_in["key_dec_fg"] = "Dec"

    if shape == "F":
        params_in["key_e1"] = "v1"
        params_in["key_e2"] = "v2"
    elif shape == "G":
        params_in["key_e1"] = "t1"
        params_in["key_e2"] = "t2"

    params_in["key_w_bg"] = "w"
    params_in["verbose"] = True
    params_in["npatch"] = 15

    return params_in

# %%
def run_and_plot(obj, params_in, shape, mode="angular", stack="auto"):
 
    out_base = f"{shape}_cl_{mode}_{stack}"

    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]

    obj._params["scales"] = mode
    obj._params["stack"] = stack

    obj._params["theta_min"] = 0.01
    obj._params["theta_max"] = 5
    obj._params["n_theta"] = 20
    obj._params["out_path"] = f"{out_base}.fits"
    
    if mode == "angular":
        obj._params["npatch"] = 15
    else:
        obj._params["npatch"] = 1
    
    obj.run()

    obj.plot_EB(out_base, shape=shape)


# %%
# %% Run different scale and stack cases
cases = [
    ("angular", "auto"),
    ("physical", "post"),
    ("physical", "cross"),
]

for shape in ("gamma", "F", "G"):
    params_in = set_params_in(shape)

    # Create compute_ng instance
    obj = run.Compute_NG()

    for mode, stack in cases:
        print("Running case:", shape, mode, stack)
        run_and_plot(obj, params_in, shape, mode=mode, stack=stack)

# %%
plt.close()
