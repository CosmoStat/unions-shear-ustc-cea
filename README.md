# UNIONS shear analyses

| Usage | Development | Release |
| ----- | ----------- | ------- |
| [![docs](https://img.shields.io/badge/docs-Sphinx-blue)](https://martinkilbinger.github.io/unions_wl/) | [![build](https://github.com/martinkilbinger/unions_wl/workflows/CI/badge.svg)](https://github.com/martinkilbinger/unions_wl/actions?query=workflow%3ACI) | [![release](https://img.shields.io/github/v/release/martinkilbinger/unions_wl)](https://github.com/martinkilbinger/unions_wl/releases/latest) |
| [![license](https://img.shields.io/github/license/martinkilbinger/unions_wl)](https://github.com/martinkilbinger/unions_wl/blob/master/LICENCE.txt) | [![deploy](https://github.com/martinkilbinger/unions_wl/workflows/CD/badge.svg)](https://github.com/martinkilbinger/unions_wl/actions?query=workflow%3ACD) | [![pypi](https://img.shields.io/pypi/v/unions_wl)](https://pypi.org/project/unions_wl/) |
| [![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide) | [![codecov](https://codecov.io/gh/martinkilbinger/unions_wl/branch/master/graph/badge.svg?token=XHJIQXV7AX)](https://codecov.io/gh/martinkilbinger/unions_wl) | [![python](https://img.shields.io/pypi/pyversions/unions_wl)](https://www.python.org/downloads/source/) |
| [![contribute](https://img.shields.io/badge/contribute-read-lightgrey)](https://github.com/martinkilbinger/unions_wl/blob/master/CONTRIBUTING.md) | [![CodeFactor](https://www.codefactor.io/repository/github/martinkilbinger/unions_wl/badge)](https://www.codefactor.io/repository/github/martinkilbinger/unions_wl) | |
| [![coc](https://img.shields.io/badge/conduct-read-lightgrey)](https://github.com/martinkilbinger/unions_wl/blob/master/CODE_OF_CONDUCT.md) | [![Updates](https://pyup.io/repos/github/martinkilbinger/unions_wl/shield.svg)](https://pyup.io/repos/github/martinkilbinger/unions_wl/) | |

---
> Author: <a href="https://sfarrens.github.io/" target="_blank" style="text-decoration:none; color: #F08080">Martin Kilbinger</a>  
> Email: <a href="mailto:martin.kilbinger@cea.fr" style="text-decoration:none; color: #F08080">martin.kilbinger@cea.fr</a>  
> Year: 2022 
---

(Copied from the Pyralid template. See [pyraliddemo](https://github.com/sfarrens/pyraliddemo) for a demo package created with the Pyralid template.)

## Contents

1. [Requirements](#Requirements)
1. [AGN-GGL](#AGN-galaxy-galaxy-lensing)
1. [Management](#Management)
1. [Deployment](#Deployment)
1. [CosmoStat](#CosmoStat)

## Requirements

The python packages `treecorr`, `àstropy`, `statsmodel` and `cs_util` need to be installed.

## AGN galaxy-galaxy lensing

### Input data

First, you need the AGN catalogue, optically selected in the SDSS survey. This data file is part of this repository, `data/agn_ggl/SDSS_SMBH_202206.txt`.
Second, the weak-lensing catalogue used here is the (internally) release v1.0, produced by either the `ShapePipe` or `Lensfit` weak-lensing pipeline.
E.g. use the file `unions_shapepipe_2022_v1.0.fits`.

### Split input catalogues in black-hole mass bins

Run
```bash
scripts/split_sample.py -v -n 2
```
to create equi-populated mass sub-samples.

### Compute GGL

Ignoring weights, use
```bash
scripts/compute_ng_binned_samples.py --input_path_fg data_mass_sub/SDSS_SMBH_202206_0_n_split_2.fits --input_path_bg unions_shapepipe_2022_v1.0.fits --key_ra_fg ra --key_dec_fg dec -v --out_path data_mass_sub/ggl_agn_0.txt
```
to compute the weak-lensing tangential shear of UNIONS galaxies around the lower-mass bin (#0). Repeat for the high-mass bin (#1).












