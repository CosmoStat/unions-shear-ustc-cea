#!/usr/bin/env python

"""compute_ng_binned_samples.py

Command-line script to run computation of
GGL (ng-correlation) between two (redshift-binned) input catalogues.

:Authors: Martin Kilbinger, Elisa Russier

"""

from unions_wl.run import run_compute_ng_binned_samples

def main(argv=None):
    """Main

    Main program

    """
    run_compute_ng_binned_samples(*argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
