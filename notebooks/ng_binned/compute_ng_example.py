# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compute NG example
# unions_wl

# +
import os
import numpy as np
import matplotlib.pylab as plt

from cs_util import plots
plt.rcParams['font.size'] = 20

from cs_util import canfar

from unions_wl import run
# -

# ## Set input parameters

# Define paramer dictrionary
params_in = {}

# Input catalogue names
params_in["input_path_fg"] = "agn_0_n_split_1.fits"
params_in["input_path_bg"] = "cat_unions_SP.fits"

# Output catalogue
params_in["out_path"] = "ggl_agn_0_n_split_1_u.fits"

# Other paramters
params_in["key_ra_fg"] = "ra"
params_in["key_dec_fg"] = "dec"
params_in["key_w_bg"] = "w"
params_in["theta_min"] = 0.1
params_in["theta_max"] = 200
params_in["verbose"] = True

# ### Get input catalogues 

# +
vos_url = "vos:cfis/cosmostat/ggl/agn/coords_test/scales_angular_stack_auto"
canfar.download(f"{vos_url}/{params_in['input_path_fg']}", params_in["input_path_fg"], verbose=True)

source_path = "/home/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_2022_v1.0.fits"
if not os.path.exists(params_in["input_path_bg"]):
    os.symlink(source_path, params_in["input_path_bg"])
# -

# ## Set up ng correlation

# Create compute_ng instance
obj = run.Compute_NG()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

# ## Run
# (This can also be done with obj.run())

# Check parameter validity
obj.check_params()

# Read input catalogues
obj.read_data()

# Set up treecorr
obj.set_up_treecorr()

# Compute correlations and stack (if not auto)
obj.correlate()

# Write correlation outputs to disk
obj.write_correlations()

# +
# Plot

x = [obj._ng.meanr] * 2
y = [obj._ng.xi, obj._ng.xi_im]
dy = [np.sqrt(obj._ng.varxi)] * 2

title = "n-g correlation"
xlabel = rf'$\theta$ [{obj._sep_units}]'
ylabel = r'$\gamma_{\rm t}(\theta), \gamma_\times(\theta)$'
labels = ['$\gamma_{\rm t}$', '$\gamma_\times$']
          
plots.plot_data_1d(
    x,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    None,
    xlog=True,
)
# -


