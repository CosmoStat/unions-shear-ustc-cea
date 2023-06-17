"""RUN.

This module sets up runs of the unions_wl computations.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import numpy as np

from astropy.io import fits
from astropy import units                                                       

from tqdm import tqdm
import treecorr

from cs_util import logging

from unions_wl import defaults
from unions_wl.stack_ng import ng_essentials, ng_stack



from optparse import OptionParser
# MKDEBUG TODO to cs_utils (see sp_validation)
def parse_options(p_def, short_options, types, help_strings):
    """Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    help_strings : dict
        help strings for options

    Returns
    -------
    options: tuple
        Command line options
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    for key in p_def:
        if key in help_strings:

            if key in short_options:
                short = short_options[key]
            else:
                short = ''

            if key in types:
                typ = types[key]
            else:
                typ = 'string'

            if typ == 'bool':
                parser.add_option(
                    f'{short}',
                    f'--{key}',
                    dest=key,
                    default=False,
                    action='store_true',
                    help=help_strings[key].format(p_def[key]),
                )
            else:
                parser.add_option(
                    short,
                    f'--{key}',
                    dest=key,
                    type=typ,
                    default=p_def[key],
                    help=help_strings[key].format(p_def[key]),
                )

    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help=f'verbose output'
    )

    options, args = parser.parse_args()

    return options

class Compute_NG(object):
    """Compute NG.

    This class computes number-shear correlations between a foreground and
    background sample.

    """

    def __init__(self):
        # Set default parameters
        self.params_default()

    def set_params_from_command_line(self, args):
        """Set Params From Command line.

        Only use when calling using python from command line.
        Does not work from ipython or jupyter.

        """
        # Read command line options
        options = parse_options(
            self._params,
            self._short_options,
            self._types,
            self._help_strings,
        )

        # Update parameter values from options
        for key in vars(options):
            self._params[key] = getattr(options, key)

        # del options ?

        # Save calling command
        logging.log_command(args)

    def params_default(self):
        """Params Default.

        Return default parameter values and additional information
        about type and command line options.

        Returns
        -------
        list :
            parameter dict
            types if not default (``str``)
            help string dict for command line option
            short option letter dict

        """
        # Specify all parameter names and default values
        self._params = {
            'input_path_fg': 'unions_sdss_matched_lens.fits',
            'input_path_bg': 'unions_sdss_matched_source.fits',
            'key_ra_fg': 'RA',
            'key_dec_fg': 'DEC',
            'key_ra_bg': 'RA',
            'key_dec_bg': 'DEC',
            'key_w_fg': None,
            'key_w_bg': None,
            'key_e1': 'e1',
            'key_e2': 'e2',
            'sign_e1': +1,
            'sign_e2': +1,
            'key_z': 'z',
            'theta_min': 0.1,
            'theta_max': 200,
            'n_theta': 10,
            'scales' : 'angular',
            'stack': 'auto',
            'out_path' : './ggl_unions_sdss_matched.txt',
            'out_path_jk' : None,
            'n_cpu': 1,
            'verbose': False,
        }

        # Parameters which are not the default, which is ``str``
        self._types = {
            'sign_e1': 'int',
            'sign_e2': 'int',
            'n_theta': 'int',
            'n_cpu': 'int',
        }

        # Parameters which can be specified as command line option
        self._help_strings = {
            'input_path_fg': 'background catalogue input path, default={}',
            'input_path_bg': 'foreground catalogue input path, default={}',
            'key_ra_fg': 'foreground right ascension column name, default={}',
            'key_dec_fg': 'foreground declination column name, default={}',
            'key_ra_bg': 'background right ascension column name, default={}',
            'key_dec_bg': 'background declination column name, default={}',
            'key_w_fg': 'foreground weight column name, default={}',
            'key_w_bg': 'background weight column name, default={}',
            'key_e1': 'first ellipticity component column name, default={}',
            'key_e2': 'second ellipticity component column name, default={}',
            'sign_e1': 'first ellipticity multiplier (sign), default={}',
            'sign_e2': 'first ellipticity multiplier (sign), default={}',
            'key_z': (
                'foreground redshift column name (if scales=physical),'
                + ' default={}'
            ),
            'theta_min': 'minimum angular scale, default={}',
            'theta_max': 'maximum angular scale, default={}',
            'n_theta': 'number of angular scales, default={}',
            'scales' : (
                '2D coordinates (scales) are angular (arcmin) or physical'
                + ' [Mpc], default={}'
            ),
            'stack' : 'allowed are auto, cross, post, default={}',
            'out_path' : 'output path, default={}',
            'out_path_jk' : 'output path, default=<out_path>_jk.<ext>',
            'n_cpu' : 'number of CPUs for parallel processing, default={}',
        }

        # Options which have one-letter shortcuts
        self._short_options = {
        }

    def check_params(self):
        """Check Params.

        Check whether parameter values are valid.

        Raises
        ------
        ValueError
            if a parameter value is not valid

        """
        if self._params['scales'] not in ('angular', 'physical'):
            raise ValueError(
                'Scales (option -s) need to be angular or physical'
            )

        # Set verbose to False if not given on input
        if "verbose" not in self._params:
            self._params["verbose"] = False

    def read_data(self):
        """Read Data.

        Read input catalogues.

        """
        self._data = {}
        for sample in ('fg', 'bg'):
            input_path = self._params[f'input_path_{sample}']
            if self._params['verbose']:
                print(f'Reading catalogue {input_path}')
            self._data[sample] = fits.getdata(input_path)

        # Test run with only 0.4M source galaxies:
        #data['bg'] = data['bg'][400_000:600_000]

    def set_up_treecorr(self):
        """Set Up Trecorr.

        Set up configuration and catalogues for treecorr call(s).

        """
        self._coord_units = 'degrees'
        self._sep_units = 'arcmin'

        self.set_up_treecorr_cats()

        self.set_up_treecorr_config()

    def set_up_treecorr_cats(self):
        """Set Up Treecorr Cats.

        Set up catalogues for treecorr call(s).

        """
        params = self._params

        self._cats = {}

        # Check ellipticity signs
        if (
            params['verbose']
            and params['sign_e1'] != +1
            and params['sign_e2'] != +1
        ):
            print(
                'Non-standard signs for ellipticity components ='
                + f' ({params["sign_e1"]:+d}, {params["sign_e2"]:+d})'
            )

        # Set fg and bg sample data columns
        # Shear components g1, g2: Set `None` for foreground
        g1 = {
            'fg': None,
            'bg': self._data['bg'][params['key_e1']] * params['sign_e1']
        }
        g2 = {
            'fg': None,
            'bg': self._data['bg'][params['key_e2']] * params['sign_e2']
        }
        w = {}
        for sample in ['fg', 'bg']:
            n = len(self._data[sample][params[f'key_ra_{sample}']])

            # Set weight
            if params[f'key_w_{sample}'] is None:
                w[sample] = [1] * n
                if params['verbose']:
                    print(f'Not using weights for {sample} sample')
            else:
                w[sample] = self._data[sample][params[f'key_w_{sample}']]
                if params['verbose']:
                    print(f'Using catalog weights for {sample} sample')

        # Create treecorr catalogues
        for sample in ('fg', 'bg'):

            # Split cat into single objects if fg and physical
            if (
                sample == 'fg' and
                (params['scales'] == 'physical' or params['stack'] != 'auto')
            ):
                split = True
            else:
                split = False

            self._cats[sample] = create_treecorr_catalogs(
                self._data,
                sample,
                params[f'key_ra_{sample}'],
                params[f'key_dec_{sample}'],
                g1,
                g2,
                w,
                self._coord_units,
                split,
            )

            if params['verbose']:
                print(
                    f"Correlating 1 bg with {len(self._cats['fg'])} fg"
                    + f" catalogues..."
                )

    def set_up_treecorr_config(self):
        """Set Up Trecorr Config.

        Set up configuration for treecorr call(s).

        """
        if self._params['scales'] == 'physical':
            self._cosmo = defaults.get_cosmo_default()

            # Angular distances to all objects
            a_arr = 1 / (1 + self._data['fg'][self._params['key_z']])
            self._d_ang_arr = self._cosmo.angular_diameter_distance(a_arr)

            theta_min, theta_max = get_theta_min_max(
                self._params['theta_min'],
                self._params['theta_max'],
                self._d_ang_arr,
                self._sep_units,
                verbose=self._params['verbose'],
            )

        else:
            self._cosmo = None
            self._d_ang_arr = None
            theta_min = self._params['theta_min']
            theta_max = self._params['theta_max']

        self._TreeCorrConfig = create_treecorr_config(
            self._coord_units,
            theta_min,
            theta_max,
            self._sep_units,
            self._params['n_theta'],
            self._params['n_cpu'],
        )

    def correlate(self):
        """Correlate.

        Main function to compute correlations.

        """
        self._ng = treecorr.NGCorrelation(self._TreeCorrConfig)
        if len(self._cats['fg']) > 1:
            self.correlate_n_fg()
            self.stack()
        else:
            self.correlate_1()

    def correlate_n_fg(self):
        """Correlate N FG.

        Carry out n_fg correlations.

        """
        params = self._params

        n_corr = 0
        ng_prev = ng_essentials(self._params["n_theta"])
        self._all_ng = []

        # More than one foreground catalogue: run individual correlations
        for idx, cat_fg in tqdm(
            enumerate(self._cats['fg']),
            total=len(self._cats['fg']),
            disable=not params['verbose'],
        ):

            # Save previous cumulative correlations (empty if first)
            ng_prev.copy_from(self._ng)

            # Perform correlation
            self._ng.process_cross(
                cat_fg,
                self._cats['bg'][0],
                num_threads=params['n_cpu']
            )

            # Last correlation = difference betwen two cumulative results
            ng_diff = ng_essentials(self._params["n_theta"])
            ng_diff.difference(self._ng, ng_prev)

            # Count (and add) correlations
            n_corr += 1

            self._all_ng.append(ng_diff)
            # Update previous cumulative correlations
            ng_prev.copy_from(self._ng)

        if params['stack'] == 'cross':
            if params['verbose']:
                print('Cross (treecorr process_cross) stacking of fg objects')
            varg = treecorr.calculateVarG(self._cats['bg'])
            self._ng.finalize(varg)

        if n_corr == 0:
            raise ValueError('No correlations computed')
        print(f'Computed {n_corr} correlations')

    def correlate_1(self):
        """Correlate One.

        Carry one one correlation with entire foreground catalogue.

        """
        # One foreground catalogue: run single simultaneous correlation
        if self._params['verbose']:
            print('Automatic (treecorr process) stacking of fg objects')
            self._ng.process(
                self._cats['fg'][0],
                self._cats['bg'][0],
                num_threads=self._params['n_cpu'],
            )

    def stack(self):
        """Stack.

        Stack correlations.

        """
        params = self._params

        if params['scales'] == 'physical':                                          
                                                                                
            # Create new config for correlations stacked on physical scales.        
            # Use (command-line) input scales but interpret in Mpc                  
            TreeCorrConfig_for_stack = create_treecorr_config(                                
                self._coord_units,                                                  
                params['theta_min'],                                                
                params['theta_max'],                                                
                self._sep_units,                                                    
                params['n_theta'],                                                  
                1,                                                                  
            )                                                                       

        else:

            # Re-use previous config for stacking on angular scales
            TreeCorrConfig_for_stack = self._TreeCorrConfig
                                                                                
        if len(self._cats['fg']) > 1 and not params['stack'] == 'cross':                  
            # Stack now (in post-processing) if more than one fg catalogue,         
            # and not cross stacking done                                           
            if params['verbose']:                                                   
                print('Post-process (this script) stacking of fg objects')          

            # MKDEBUG TODO: distinguish from previous ng (not used anymore here)
            self._ng, self._ng_jk = ng_stack(                                                   
                TreeCorrConfig_for_stack,                                                     
                self._all_ng,                                                             
                self._d_ang_arr,                                                              
            )                                                                       
        else:                                                                       
            self._ng_jk = None 

    def write_corr(self):
        """Write Corr.

        Write correlation outputs to disk.

        """
        if self._params['verbose']:                                                       
            print(f"Writing output file {self._params['out_path']}")                                
        self._ng.write(                                                                   
            self._params["out_path"],                                                               
            rg=None,                                                                
            file_type=None,                                                         
            precision=None,                                                         
        )                                                                           
                                                                                
        # Write stack with jackknife resamples summaries to file                    
        if self._ng_jk:                                                                   
            if not self._params['out_path_jk']:                                           
                base, ext = os.path.splitext(self._params['out_path'])                    
                out_path_jk = f'{base}_jk{ext}'                                     
            else:                                                                   
                out_path_jk = self._params['out_path_jk']                                 
            if self._params['verbose']:                                                   
                print(f"Writing output file {self._params['out_path_jk']}")                         
            ng_jk.write(                                                            
                out_path_jk,                                                        
                rg=None,                                                            
                file_type=None,                                                     
                precision=None,                                                     
            )

    def run(self):
        """Run.

        Main processing of scale-dependent leakage.

        """
        # Check parameter validity
        self.check_params()

        # Read input catalogues
        self.read_data()

        # Set up treecorr
        self.set_up_treecorr()

        # Compute correlations and stack (if not auto)
        self.correlate()

        # Write correlation outputs to disk
        self.write_corr()


def create_treecorr_catalogs(
    positions,
    sample,
    key_ra,
    key_dec,
    g1,
    g2,
    w,
    coord_units,
    split,
):
    """Create Treecorr Catalogs.

    Return treecorr catalog(s).

    Parameters
    ----------
    positions : dict
        input positions
    sample : str
        sample string
    key_ra : str
        data key for right ascension
    key_dec : str
        data key for declination
    g1 : dict
        first shear component
    g2 : dict
        second shear component
    w : dict
        weight
    coord_units : str
        coordinate unit string
    split : bool
        if True split foreground sample into individual objects

    Returns
    -------
    list
        treecorr Cataloge objects

    """
    cat = []
    if not split:
        my_cat = treecorr.Catalog(
            ra=positions[sample][key_ra],
            dec=positions[sample][key_dec],
            g1=g1[sample],
            g2=g2[sample],
            w=w[sample],
            ra_units=coord_units,
            dec_units=coord_units,
        )
        cat = [my_cat]
    else:
        n_obj = len(positions[sample][key_ra])
        for idx in range(n_obj):
            if not g1[sample]:
                my_g1 = None
                my_g2 = None
            else:
                my_g1 = g1[sample][idx:idx+1]
                my_g2 = g2[sample][idx:idx+1]

            my_cat = treecorr.Catalog(
                ra=positions[sample][key_ra][idx:idx+1],
                dec=positions[sample][key_dec][idx:idx+1],
                g1=my_g1,
                g2=my_g2,
                w=w[sample][idx:idx+1],
                ra_units=coord_units,
                dec_units=coord_units,
            )
            cat.append(my_cat)

    return cat


def create_treecorr_config(                                                     
    coord_units,                                                                
    scale_min,                                                                  
    scale_max,                                                                  
    sep_units,                                                                  
    n_theta,                                                                    
    n_cpu,                                                                      
):                                                                              
                                                                                
    TreeCorrConfig = {                                                          
        'ra_units': coord_units,                                                
        'dec_units': coord_units,                                               
        'min_sep': scale_min,                                                   
        'max_sep': scale_max,                                                   
        'sep_units': sep_units,                                                 
        'nbins': n_theta,                                                       
        'num_threads': n_cpu,                                                   
    }                                                                           
                                                                                
    return TreeCorrConfig


def rad_to_unit(value, unit):

    return (value * units.rad).to(unit).value


def unit_to_rad(value, unit):

    return value * units.Unit(units).to('rad')


def get_theta_min_max(r_min, r_max, d_ang_arr, sep_units, verbose=False):
    """Get Theta Min MaX.

    Return scale ranges.

    Parameters
    ----------
    r_min : float
        minium physical scale [Mpc]
    r_max : float
        maxium physical scale [Mpc]
    d_ang_arr : list
        angular diameter distances [Mpc]
    sep_units : str
        unit for angular scales
    verbose : bool, optional
        verbose output if True, default is False

    Returns
    -------
    float
        minimum angular scale
    float
        maximum angular scale

    """
    # Min and max angular distance at object redshift
    d_ang_min = min(d_ang_arr)
    d_ang_max = max(d_ang_arr)
    d_ang_mean = np.mean(d_ang_arr)

    # Transfer physical to angular scales

    theta_min = 1e30
    theta_max = -1
    for d_ang in d_ang_arr:
        th_min = float(r_min) / d_ang
        if th_min < theta_min:
            theta_min = th_min
        th_max = float(r_max) / d_ang
        if th_max > theta_max:
            theta_max = th_max

    if verbose:
        print(f'physical to angular scales, r = {r_min}  ... {r_max:} Mpc')
        print(
            f'physical to angular scales, d_ang = {min(d_ang_arr):.2f}  ... '
            + f'{max(d_ang_arr):.2f} (mean {d_ang_mean:.2f}) Mpc'
        )
        print(
            f'physical to angular scales, theta = {theta_min:.2g}  ... '
            + f'{theta_max:.2g} rad'
        )

    theta_min = rad_to_unit(theta_min, sep_units)
    theta_max = rad_to_unit(theta_max, sep_units)

    print(
        f'physical to angular scales, theta = {theta_min:.2g}  ... '
        + f'{theta_max:.2f} arcmin'
    )

    return theta_min, theta_max
