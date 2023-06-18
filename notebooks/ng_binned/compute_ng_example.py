# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Compute NG example
# unions_wl

# +
import matplotlib.pylab as plt

from unions_wl import run

# -

# ## Set input parameters

# ### Input catalogues

# +
params_in = {}
# -

# ## Compute ng correlation

# Create compute_ng instance
obj = run.Compute_NG()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

# ## Run
# (This can also be done with obj.run())

# +
# Check parameter validity
obj.check_params()
# -

# +
# Read input catalogues
obj.read_data()
# -

# +
# Set up treecorr
obj.set_up_treecorr()
# -

# +
# Compute correlations and stack (if not auto)
obj.correlate()
# -

# +
# Write correlation outputs to disk
obj.write_correlations()
# -
