# %%
# flexion_stacks.py

# Reference: https://arxiv.org/pdf/2111.08745

# When running from terminal:
# export QT_QPA_PLATFORM=offscreen

# %%
import os
import numpy as np
import matplotlib.pylab as plt

from cs_util import plots as cs_plots
plt.rcParams['font.size'] = 20

from unions_wl import run


# %%
def set_params_in(shape, input_dir=".", sig_shape=-1):
    
    params_in = {}

    params_in["shape"] = shape

    # Input catalogue names
    params_in["input_path_fg"] = f"{input_dir}/fg.fits"
    params_in["input_path_bg"] = f"{input_dir}/bg_{shape}.fits"

    # Other paramters
    params_in["key_ra_fg"] = "RA"
    params_in["key_dec_fg"] = "Dec"

    if shape == "F":
        params_in["key_e1"] = "v1"
        params_in["key_e2"] = "v2"
    elif shape == "G":
        params_in["key_e1"] = "t1"
        params_in["key_e2"] = "t2"
        
    params_in["sig_shape"] = sig_shape

    params_in["key_w_bg"] = "w"
    params_in["verbose"] = True
    params_in["npatch"] = 15

    return params_in

# %%
def run_and_plot(obj, params_in, ax=None, mode="angular", stack="auto"):
 
    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]

    out_base = f"{obj._params['shape']}_cl_{mode}_{stack}"

    obj._params["scales"] = mode
    obj._params["stack"] = stack

    obj._params["theta_min"] = 0.01
    obj._params["theta_max"] = 5
    obj._params["n_theta"] = 20
    obj._params["out_path"] = f"{out_base}.fits"
    
    if obj._params["stack"] == "post":
        obj._params["out_path_jk"] = f"{out_base}_jk.fits"
    
    if mode == "angular":
        obj._params["npatch"] = 15
    else:
        obj._params["npatch"] = 1
    
    if os.path.exists(obj._params["out_path"]):
        obj.read_correlation()
    else:
        obj.run()

    if ax is not None:
        # Mosaic plot: Don't save
        out_base = None
    obj.plot_EB(out_base=out_base, ax=ax)


# %%
# %% Run different scale and stack cases
cases = [
    ("angular", "auto"),
    ("physical", "post"),
    ("physical", "cross"),
]

mosaic_plot = True

sig_shape = {
    "gamma": 0.26,
    "F": 0.04,
    "G": 0.04,
}

input_dir = ".."

for mode, stack in cases:

    if mosaic_plot:
        # Create multiple axes
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

    for idx, shape in enumerate(("gamma", "F", "G")):
        params_in = set_params_in(shape, input_dir=input_dir, sig_shape=sig_shape[shape])

        # Create compute_ng instance
        obj = run.Compute_NG()

        if mosaic_plot is False:
            # No axes -> save single plot
            ax = None
        else:
            # Pick corresponding axis
            ax = axes[idx]

        print("Running case:", shape, mode, stack)
        run_and_plot(obj, params_in, ax=ax, mode=mode, stack=stack)

    if mosaic_plot:
        plt.suptitle(f"coordinates={mode}, stack={stack}")
        plt.tight_layout()
        cs_plots.savefig(f"cl_{mode}_{stack}.png")

# %%
if mosaic_plot is False:
    _ = plt.close()
