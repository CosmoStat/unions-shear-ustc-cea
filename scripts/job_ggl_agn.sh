#!/usr/bin/env bash

# Name: job_ggl_agn.sh
# Description: Galaxy-galaxy lensing with AGNs job script
# Autor: Martin Kilbinger <martin.kilbinger@cea.fr>

# Variables
repo_path=$HOME/astro/repositories/github/unions-shear-ustc-cea

bin_path=${repo_path}/scripts
agn_data_path=${repo_path}/data/agn_ggl
unions_data_path=/home/mkilbing/astro/data/CFIS/v1.0

# Command line arguments
## Default variables
methods=(SP LF)
blinds=(A B C)
weights=("u" "w")
n_split_arr=(1 2 3)
stack=angular
Delta_Sigma=0
AGN=Liu19
logM_min=0
z_min=0
z_max=10
theta_min=0.1
theta_max=200
footprint_mask=""
n_cpu=1
idx_ref=

## Help string
usage="Usage: $(basename "$0") [OPTIONS]
\n\nOptions:\n
   -h\tthis message\n
   -a, --agn_cat AGN\n
   \tAGN catalogue, '$AGN' (default) or 'Shen22_and_Liu19'\n
   -f, --footprint_mask FNAME\n
   \tfootprint mask file FNAME\n
   \tno mask is applied if not given (default)\n
   -n, --n_cpu N_CPU\n
   \tNumber of CPUs, default is ${n_cpu}\n
   --logM_min LOGM_MIN\n
   \tUse only AGN with mass > LOGM_MIN, default is ${logM_min} (no cut)\n
   --z_min Z_MIN\n
   \tminimum redshift, default is ${z_min} (no cut)\n
   --z_max Z_MAX\n
   \tmaximum redshift, default is ${z_max} (no cut)\n
  --idx_ref IDX_REF\n
   \tbin index IDX_REF for reference redshift histogram\n
   \tdefault none (flat weighted histograms)\n
   -s, --stack STACK\n
   \tstack using angular or physical coordinates, default=${stack}\n
   --Delta_Sigma\n
   \testimator is excess surface mass density instead of tangential shear\n
   --theta_min THETA_MIN\n
   \tminimum angular scale, default=${theta_min}\n
   --theta_max THETA_MAX\n
   \tmaximum angular scale, default=${theta_max}\n

"

## Save command line to log file
echo  $(basename "$0") $@ > log_job_ggl_agn.sh

## Parse command line
while [ $# -gt 0 ]; do
  case "$1" in
    -h)
      echo -ne $usage
      exit 0
      ;;\
    -a|--agn_cat)
      AGN="$2"
      shift
      ;;
    -f|--footprint_mask)
      footprint_mask="$2"
      shift
      ;;
    -n|--n_cpu)
      n_cpu="$2"
      shift
      ;;
    --logM_min)
      logM_min="$2"
      shift
      ;;
    --z_min)
      z_min="$2"
      shift
      ;;
    --z_max)
      z_max="$2"
      shift
      ;;
    --theta_min)
      theta_min="$2"
      shift
      ;;
    --theta_max)
      theta_max="$2"
      shift
      ;;
    --idx_ref)
      idx_ref="$2"
      shift
      ;;
    -s|--stack)
      stack="$2"
      shift
      ;;
    --Delta_Sigma)
      Delta_Sigma=1
      ;;
  esac
  shift
done


# Check options
if [ "${Delta_Sigma}" == 1 ] && [ "$stack" == "angular" ]; then
  echo "With Delta_Sigma=1 stack cannot be angular"
  exit 5
fi

function create_one_link() {
  file=$1
  dir=$2

  if [ ! -e $file ]; then
    ln -s $dir/$file
 fi
}

function create_links() {
  AGN=$1

  # AGN catalogues
  if [ "$AGN" == "Liu19" ]; then
    input_cat=SDSS_SMBH_202206.txt
  elif [ "$AGN" == "Shen22_and_Liu19" ]; then
    input_cat=SDSS_SMBH_202210.txt
  else
    echo "Invalid AGN catalogue '$AGN'"
    exit 2
  fi
  create_one_link $input_cat $agn_data_path

  # Lensing catalogues
  files=(
    "unions_shapepipe_2022_v1.0.fits"
    "mask_all.fits"
  )
  dirs=(
    "${unions_data_path}/ShapePipe"
    "${unions_data_path}/ShapePipe/masks/healpix/nside_1024"
  )
  for idx in ${!files[@]}; do
    create_one_link ${files[$idx]} ${dirs[$idx]}
  done

  files=(
    "lensfit_goldshape_2022v1.fits"
    "CFIS3500_THELI_mask_hp_4096.fits"
  )
  dirs=(
    "${unions_data_path}/Lensfit"
    "${unions_data_path}/Lensfit/masks"
  )
  for idx in ${!files[@]}; do
    create_one_link ${files[$idx]} ${dirs[$idx]}
  done

  # Redshift distributions
  for file in ${unions_data_path}/nz/*.txt; do
    ln -s $file
  done

  # Rename UNIONS WL catalogues (links)
  mv unions_shapepipe_2022_v1.0.fits cat_unions_SP.fits
  mv lensfit_goldshape_2022v1.fits cat_unions_LF.fits

  for sh in ${methods[@]}; do
    mkdir -p $sh
  done
}


# Transform AGN text files to FITS format
function agn_txt2fits() {

  echo "*** Transform AGN sample to FITS file..."
  if [ "$AGN" == "Liu19" ]; then
    ascii2fits.py -i SDSS_SMBH_202206.txt -o agn.fits -H "RA Dec z logM"
  elif [ "$AGN" == "Shen22_and_Liu19" ]; then
    ascii2fits.py -i SDSS_SMBH_202210.txt -o agn.fits -H "RA Dec z logM err_logM"
  else
    echo "Invalid AGN catalogue '$AGN'"
    exit 2
  fi

}

function footprint() {

  if [ "$footprint_mask" != "" ]; then
    echo "*** Select AGNs in UNIONS footprint with mask file $footprint_mask..."
    ${bin_path}/check_footprint.py -v -m ${footprint_mask} -p -g 0
  else
    echo "*** Ignore UNIONS footprint"
    ln -sf agn.fits agn_in_footprint.fits
  fi

}


# Wait for parallel jobs to finish
function wait_parallel() {

  # If n_cpu jobs already running: wait for one to finish before starting
  # new job
  if [[ $(jobs -r -p | wc -l) -ge ${n_cpu} ]]; then
    wait
  fi

}

# Split AGN sample into redshift bins
function split_sample() {

  echo "*** Split sample..."
  if [ "${idx_ref}" != "" ]; then
    nz_opt="--idx_ref=${idx_ref}"
  else
    nz_opt=""
  fi

  parallel -j ${n_cpu} ${bin_path}/split_sample.py -v -n {1} --z_min=${z_min} --z_max=${z_max} --logM_min=${logM_min} ${nz_opt} \>\> log_job.sh ::: ${n_split_arr[@]}

}


# Compute GGL correlation functions
function compute_ng() {

  echo "*** Compute correlations..."

  # Create file pattern string from sample IDs
  c_n_str_arr=()
  c_arr=()
  for n_split in ${n_split_arr[@]}; do
    for (( c=0; c<$n_split; c++ )); do
      c_n_str_arr+=("${c}_n_split_${n_split}")
      c_arr+=($c)
    done
  done

  # Set stacking argument(s)
  if [ "$stack" == "angular" ]; then
    arg_s="--stack=auto"
  elif [ "$stack" == "physical" ]; then
    arg_s="--physical --key_z z"
  else
    echo "Invalid stack value"
  fi

  if [ "${Delta_Sigma}" == "1" ]; then

    # Weighted fg
    parallel -j ${n_cpu} ${bin_path}/compute_ng_binned_samples.py -v --input_path_fg agn_{1}.fits --input_path_bg cat_unions_{3}.fits --key_ra_fg ra --key_dec_fg dec --out_path {3}/ggl_agn_{1}_w.fits --key_w_bg=w --key_w_fg=w_{2} $arg_s --theta_min ${theta_min} --theta_max=${theta_max} --Delta_Sigma --dndz_source dndz_{3}_{4}.txt \>\> log_job.sh ::: ${c_n_str_arr[@]} :::+ ${c_arr[@]} ::: ${methods[@]} ::: ${blinds[@]}

    # Unweighted fg
    parallel -j ${n_cpu} ${bin_path}/compute_ng_binned_samples.py -v --input_path_fg agn_{1}.fits --input_path_bg cat_unions_{3}.fits --key_ra_fg ra --key_dec_fg dec --out_path {3}/ggl_agn_{1}_u.fits --key_w_bg=w $arg_s --theta_min ${theta_min} --theta_max=${theta_max} --Delta_Sigma --dndz_source dndz_{3}_{4}.txt \>\> log_job.sh ::: ${c_n_str_arr[@]} :::+ ${c_arr[@]} ::: ${methods[@]} ::: ${blinds[@]}

  else

    # Weighted fg
    parallel -j ${n_cpu} ${bin_path}/compute_ng_binned_samples.py -v --input_path_fg agn_{1}.fits --input_path_bg cat_unions_{3}.fits --key_ra_fg ra --key_dec_fg dec --out_path {3}/ggl_agn_{1}_w.fits --key_w_bg=w --key_w_fg=w_{2} $arg_s --theta_min ${theta_min} --theta_max=${theta_max} \>\> log_job.sh ::: ${c_n_str_arr[@]} :::+ ${c_arr[@]} ::: ${methods[@]}

    # Unweighted fg
    parallel -j ${n_cpu} ${bin_path}/compute_ng_binned_samples.py -v --input_path_fg agn_{1}.fits --input_path_bg cat_unions_{3}.fits --key_ra_fg ra --key_dec_fg dec --out_path {3}/ggl_agn_{1}_u.fits --key_w_bg=w $arg_s --theta_min ${theta_min} --theta_max=${theta_max} \>\> log_job.sh ::: ${c_n_str_arr[@]} :::+ ${c_arr[@]} ::: ${methods[@]}

  fi
}


# Compare to theory
function compare_to_theory() {

  for n_split in ${n_split_arr[@]}; do
    for (( c=0; c<$n_split; c++ )); do
      for sh in ${methods[@]}; do
        for blind in ${blinds[@]}; do
          for weight in ${weights[@]}; do

              ${bin_path}/ggl_compare_data_theory.py --corr_path $sh/ggl_agn_${c}_n_split_${n_split}_${weight}.fits --dndz_lens_path hist_z_${c}_n_split_${n_split}_${weight}.txt --dndz_source_path dndz_${sh}_${blind}.txt -v --theta_min 0.1 --theta_max 200 --n_theta 10 --bias_1 1.0 --out_base ${sh}/gamma_tx_${c}_n_split_${n_split}_${blind}_${weight} | tee -a log_job.sh

          done
        done
      done
    done
  done

}


### Main program ###

# Remove existing job log file
rm -f log_job.sh

# Set links to data files
create_links $AGN

# Transform AGN files
agn_txt2fits

# Select AGNs in UNIONS footprint
footprint

# Split AGNs into redshift bins
split_sample

# Compute correlations
compute_ng

# TODO: use HOD instead of linear bias model
#echo "*** Compare to theory..."
#compare_to_theory
