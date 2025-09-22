import numpy as np
import matplotlib.pylab as plt

from astropy import units
from astropy.io import fits
from astropy.table import Table

import treecorr
import pandas as pd
import healpy as hp

from cs_util import plots as cs_plots


class Compute_ng(object):
    
    figsize = 10

    def __init__(self, min_sep=0.05, max_sep=10, nbins=10, npatch=15, sep_units="arcmin"):

        self._npatch = npatch
        if self._npatch > 1:
            var_method = "jackknife"
        else:
            var_method = "shot"

        if sep_units == "arcmin":
            self._config = {
                "min_sep": min_sep,
                "max_sep": max_sep,
                "nbins": nbins,
                "sep_units": sep_units,
                "var_method": var_method,
            }
            # treecorr instances
            self._nx = {}
            self._nx["gamma"] = treecorr.NGCorrelation(self._config, cross_patch_weight="match")
            self._nx["F"] = treecorr.NVCorrelation(self._config, cross_patch_weight="match")
            self._nx["G"] = treecorr.NTCorrelation(self._config, cross_patch_weight="match")

            # shape catalogues
            self._shapes = {}            

            plt.rcParams.update(
                {
                    "axes.titlesize": 18,
                    "axes.labelsize": 16,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "legend.fontsize": 14,
                    "font.size": 14,
                }
            )


    def set_cat(self, df, shape="gamma", angle="theta", from_fits=False):

        if shape is None:
            self._density = self.get_Catalogue(
                df,
                shape=None,
                angle=angle,
                from_fits=from_fits,
            )
        else:
            self._shapes[shape] = self.get_Catalogue(
                df,
                shape=shape,
                angle=angle,
                from_fits=from_fits,
            )

    def set_cat_from_fits(self, path, shape="gamma"):

        with fits.open(path) as hdu_list:
            data = Table(hdu_list[1].data)
        df = data.to_pandas()

        self.set_cat(df, shape=shape, from_fits=True)

    def extract_fields_for_cat(
        self,
        df,
        angle="theta",
        shape=None,
        from_fits=False,
        only_valid=True,
    ):

        g1 = g2 = v1 = v2 = t1 = t2 = None

        ra = df["RA"]
        dec = df["Dec"]

        if not from_fits:
            if shape == "gamma":
                g1 = np.real(df[shape])
                g2 = np.imag(df[shape])
            elif shape == "F":
                v1 = np.real(df["F_inv_asec"])
                v2 = np.imag(df["F_inv_asec"])
            elif shape == "G":
                t1 = np.real(df["G_inv_asec"])
                t2 = np.imag(df["G_inv_asec"])
            elif shape is not None:
                raise ValueError(f"shape {shape} not implemented")
        else:
            if shape == "gamma":
                g1 = df["e1"]
                g2 = df["e2"]
            elif shape == "F":
                v1 = df["v1"]
                v2 = df["v2"]
            elif shape == "G":
                t1 = df["t1"]
                t2 = df["t2"]
            elif shape is not None:
                raise ValueError(f"Invalid shape {shape}")

        if only_valid and shape is not None:
            # Get valid (non-NaN) indices
            if shape == "gamma":
                valid_indices = ~(np.isnan(g1) | np.isnan(g2))
                g1 = g1[valid_indices]
                g2 = g2[valid_indices]
            elif shape == "F":
                valid_indices = ~(np.isnan(v1) | np.isnan(v2))
                v1 = v1[valid_indices]
                v2 = v2[valid_indices]
            elif shape == "G":
                valid_indices = ~(np.isnan(t1) | np.isnan(t2))
                t1 = t1[valid_indices]
                t2 = t2[valid_indices]

            n_valid = np.sum(valid_indices)
            n_nan = np.sum(~valid_indices)
            n_total = len(ra)
            print(f"Removed {n_nan}/{n_total} = {n_nan/n_total:.1%} NaN values")

            ra = ra[valid_indices]
            dec = dec[valid_indices]

        return ra, dec, g1, g2, v1, v2, t1, t2       

    def get_Catalogue(
        self,
        df,
        angle="theta",
        shape="gamma",
        from_fits=False,
    ):

        ra, dec, g1, g2, v1, v2, t1, t2 = self.extract_fields_for_cat(
            df,
            angle=angle,
            shape=shape,
            from_fits=from_fits,
        )
        units = "deg"
        
        cat = treecorr.Catalog(
            ra=ra,
            dec=dec,
            g1=g1,
            g2=g2,
            v1=v1,
            v2=v2,
            t1=t1,
            t2=t2,
            ra_units=units,
            dec_units=units,
            npatch=self._npatch,
        )
        
        return cat
    
    def process(self, shape="gamma"):
        
        self._nx[shape].process(self._density, self._shapes[shape]) 
        
    def plot_EB(self, shape="shear", suf=""):

        fig, axes = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(self.figsize, self.figsize)
        )

        nx = self._nx[shape]
 
        # E
        dx = cs_plots.dx(0, nx=2, log=True)
        plt.semilogx(nx.rnom * dx, nx.xi, label="E", color="blue")
        plt.errorbar(
            nx.rnom * dx, nx.xi, yerr=np.sqrt(nx.varxi), marker="s", color="blue"
        )

        # B
        dx = cs_plots.dx(1, nx=2, log=True)
        plt.semilogx(nx.rnom * dx, nx.xi_im, label="B", color="orange")
        plt.errorbar(
            nx.rnom * dx, nx.xi_im, yerr=np.sqrt(nx.varxi), marker="o", color="orange"
        )
        
        plt.axhline(color="k")
        
        plt.xlabel(r"$\theta$ [arcmin]")
        if shape == "F":
            label = r"1-flexion $F$ [arcsec$^{-1}$]"
        elif shape == "G":
            label = r"3-flexion $G$ [arcsec$^{-1}$]"
        elif shape == "gamma":
            label = r"$\gamma$"
        else:
            label = shape
        plt.ylabel(rf"{label}")
        plt.legend()
        plt.savefig(f"{shape}{suf}_EB.png")

    def write_correlation(self, out_path, shape="gamma"):
        
        ng = self._nx[shape]

        ng.write(out_path, file_type=None, precision=None)


def get_kappa_gamma(*, a_11, a_12, a_21, a_22):

    # Note: a_ij = delta_ij - d^2 dphi_ij

    kappa = 1 - 0.5 * (a_11 + a_22)
    gamma = -0.5 * ((a_11 - a_22) + 1j * (a_12 + a_21))

    return kappa, gamma


def get_F_G(*, d_111, d_112, d_122, d_222, d_211=None, d_212=None):

    # Symmetrize
    if d_211 is not None:
        d_112_sym = 0.5 * (d_112 + d_211)
    else:
        d_112_sym = d_112
    if d_212 is not None:
        d_122_sym = 0.5 * (d_122 + d_212)
    else:
        d_122_sym = d_122

    F = 0.5 * ((d_111 + d_122_sym) + 1j * (d_112_sym + d_222))
    G = 0.5 * ((d_111 - 3 * d_122_sym) + 1j * (3 * d_112_sym - d_222))

    return F, G


def fill_lensing_quantities(df, angle="theta", flip_dec=True, c1_sign=+1, c2_sign=+1):

    ra, dec = get_ra_dec(df, angle=angle, flip_dec=flip_dec)
    df.loc[:, "RA"] = ra
    df.loc[:, "Dec"] = dec

    # Get lensing quantities
    kappa, gamma = get_kappa_gamma(
        a_11=df["a11"], a_12=df["a12"], a_21=df["a21"], a_22=df["a22"]
    )
    df.loc[:, "kappa"] = kappa
    df.loc[:, "gamma"] = gamma
    (
        df["F"],
        df["G"],
    ) = get_F_G(
        d_111=df["d111"],
        d_112=df["d112"],
        d_122=df["d122"],
        d_222=df["d222"],
        d_211=df["d211"],
        d_212=df["d212"],
    )

    if c1_sign * c2_sign not in (-1, +1):
        raise ValueError(f"Invalid c1/2 signs {c1_sign:+d}/{c2_sign:+d}")
    
    for shape in ("F", "G"):
        df[shape] = df[shape].apply(
        lambda x: complex(x.real * c1_sign, x.imag * c2_sign)
    )

    df.loc[:, "F_inv_asec"] = inv_rad_to_inv_asec(df["F"])
    df.loc[:, "G_inv_asec"] = inv_rad_to_inv_asec(df["G"])

    df["gamma_abs"] = np.abs(df["gamma"])
    df["F_inv_asec_abs"] = np.abs(df["F_inv_asec"])
    df["G_inv_asec_abs"] = np.abs(df["G_inv_asec"])


def rad_to_deg(x):

    if hasattr(x, "values"):
        # if x is pandas dataframe column
        x_val = x.values
    else:
        x_val = x

    return (x_val * units.rad).to("deg").value


def deg_to_rad(x):

    if hasattr(x, "values"):
        # if x is pandas dataframe column
        x_val = x.values
    else:
        x_val = x

    return (x_val * units.deg).to("rad").value


def inv_rad_to_inv_asec(x):

    if hasattr(x, "values"):
        x_val = x.values
    else:
        x_val = x
    y = x_val * 1 / units.rad

    return y.to("1/arcsec").value


# Footprint and area. TODO: move to cs_util
def get_binned_area(ra, dec, nside=512):
    
    # Pixel list of input data
    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)
    
    # Number of occupied pixels
    Nocc  = np.unique(ipix).size
    
    # Pixel area
    pix_area_deg2 = hp.nside2pixarea(nside, degrees=True)

    # Footprint area
    area_deg2 = Nocc * pix_area_deg2

    return area_deg2


def get_ra_dec(df, angle="theta", flip_dec=True):
    
    ra = rad_to_deg(df[f"{angle}1"])
    if flip_dec:
        # theta2 goes from 0 to 180 deg
        dec = 90 - rad_to_deg(df[f"{angle}2"])
    else:
        dec = rad_to_deg(df[f"{angle}2"])

    return ra, dec    


def get_area(df, nside= 8192, approx=False, angle="theta"):
    
    ra, dec = get_ra_dec(df, angle=angle)

    if approx:
        # Area of rectangle of largest extent
        ra_max = max(ra)
        ra_min = min(ra)
        dec_max = max(dec)
        dec_min = min(dec)
        dec_cen = np.mean(dec)
        area_deg2 = (
            (ra_max - ra_min)
            * (dec_max - dec_min)
            * np.cos(deg_to_rad(dec_cen))
        )
    else:
        area_deg2 = get_binned_area(ra, dec, nside=nside)
        
    return area_deg2


def plot_1d_lens_histograms(df, figsize=10):

    # 1D histograms of lensing quantities
    fix, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(figsize, figsize)
    )

    axes[0][0].set_xlabel("$|\gamma|$")
    axes[0][0].scatter(df["gamma_abs"], df["kappa"], s=0.5)
    axes[0][0].set_ylabel(r"$\kappa$")
    axes[0][0].set_ylim(-0.2, 1)

    axes[0][1].scatter(df["F_inv_asec_abs"], df["G_inv_asec_abs"], s=0.5)
    axes[0][1].set_xlabel(r"$|\cal F| \;\; [\rm{arcsec}^{-1}]$")
    axes[0][1].set_ylabel(r"$|\cal G| \;\; [\rm{arcsec}^{-1}]$")

    axes[1][0].scatter(df["gamma_abs"], df["F_inv_asec_abs"], s=0.5)
    axes[1][0].set_xlabel("$|\gamma|$")
    axes[1][0].set_ylabel(r"$|\cal F| \;\; [\rm{arcsec}^{-1}]$")

    axes[1][1].scatter(df["gamma_abs"], df["G_inv_asec_abs"], s=0.5)
    axes[1][1].set_xlabel("$|\gamma|$")
    axes[1][1].set_ylabel(r"$|\cal G| \;\; [\rm{arcsec}^{-1}]$")

    plt.tight_layout()
    cs_plots.savefig("histograms.png")
    
    
def plot_2d_lens_scatter(df, figsize=10):
    
    # 2D scatter plots of lensing quantities
    fix, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(figsize, figsize)
    )

    hist_kappa = axes[0][0].hist(df["kappa"], bins=100, range=(-0.3, 1.0))
    axes[0][0].set_yscale("log")
    axes[0][0].set_xlabel("$\kappa$")

    hist_gamma_1 = axes[1][0].hist(
        df["gamma"].values.real, bins=100, range=(-0.5, 0.5)
    )
    axes[1][0].set_yscale("log")
    axes[1][0].set_xlabel("$\gamma_1$")

    hist_gamma_2 = axes[1][1].hist(
        df["gamma"].values.imag, bins=100, range=(-0.5, 0.5)
    )
    axes[1][1].set_yscale("log")
    axes[1][1].set_xlabel("$\gamma_2$")

    hist_F_1 = axes[2][0].hist(
        df["F_inv_asec"].values.real, bins=100, range=(-1, 1)
    )
    axes[2][0].set_yscale("log")
    axes[2][0].set_xlabel(r"${\cal F}_1 \;\; [\rm{arcsec}^{-1}]$")

    hist_F_2 = axes[2][1].hist(
        df["F_inv_asec"].values.imag, bins=100, range=(-1, 1)
    )
    axes[2][1].set_yscale("log")
    axes[2][1].set_xlabel(r"${\cal F}_2 \;\; [\rm{arcsec}^{-1}]$")

    hist_G_1 = axes[3][0].hist(
        df["G_inv_asec"].values.real, bins=100, range=(-2, 2)
    )
    axes[3][0].set_yscale("log")
    axes[3][0].set_xlabel(r"${\cal G}_1 \;\; [\rm{arcsec}^{-1}]$")

    hist_G_2 = axes[3][1].hist(
        df["G_inv_asec"].values.imag, bins=100, range=(-2, 2)
    )
    axes[3][1].set_yscale("log")
    axes[3][1].set_xlabel(r"${\cal G}_2 \;\; [\rm{arcsec}^{-1}]$")

    for row in (0, 3):
        for col in (0, 1):
            axes[row][col].set_ylabel("frequency")
    plt.tight_layout()
    cs_plots.savefig("scatters.png")


def add_two_comp(cols, base, df, key):
    cols.append(
        fits.Column(name=f"{base}1", array=df[key].values.real, format="E")
    )
    cols.append(
        fits.Column(name=f"{base}2", array=df[key].values.imag, format="E")
    )

def write_to_fits(df, output_path, shape="gamma"):
    """Write To Fits.
        Write position, shear, flexion F or G flexion catalogue
        to a FITS file.
    
    """
    cols = []
        
    cols.append(fits.Column(name="RA", array=df["RA"], format="E"))
    cols.append(fits.Column(name="Dec", array=df["Dec"], format="E"))

    if shape is None:
        # foreground position + redshift catalogue
        cols.append(fits.Column(name="z", array=df["z0"], format="E"))
    else:
        if shape == "gamma":
            add_two_comp(cols, "e", df, "gamma")
        elif shape == "F":
            add_two_comp(cols, "v", df, "F_inv_asec")
        elif shape == "G":
            add_two_comp(cols, "t", df, "G_inv_asec")
        else:
            raise ValueError(f"Invalid shape {shape}")

    cols.append(fits.Column(name="w", array=np.ones_like(df["RA"]), format="E"))
        
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(output_path, overwrite=True)


def test(df_fg, df_bg):
    """Test.
    
    Test of shear and flexion correlations. Is being called from
    flexion_create_cats.py
    
    Parameters
    -----------
    df_fg: pandas.df
        foreground (cluster) catalogue
    df_bg: pandas.df
        background shear and flexion catalogue
    
    """
    # Use live df or read from file
    for from_fits in (False, True):

        cng = Compute_ng(npatch=15, min_sep=0.05, max_sep=10, nbins=20)

        if not from_fits:
            # Position fg catalogue
            cng.set_cat(df_fg, shape=None)

            # Lensing bg catalogues
            for shape in ("gamma", "F", "G"):
                cng.set_cat(df_bg, shape=shape)
        else:
            cng.set_cat_from_fits("fg.fits", shape=None)
            for shape in ("gamma", "F", "G"):
                cng.set_cat_from_fits(f"bg_{shape}.fits", shape=shape)

        for shape in ("gamma", "F", "G"):
            cng.process(shape=shape)
            cng.write_correlation(f"{shape}_EB_{from_fits}.fits", shape=shape)

        for shape in "gamma", "F", "G":
            cng.plot_EB(shape=shape, suf=f"_{from_fits}_test")