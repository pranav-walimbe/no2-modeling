#!/bin/bash
#SBATCH --job-name=nox-change-eda
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio4_m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/global/home/users/pranavwalimbe/no2-modeling}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/no2-eda-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

srun python -u -m eda.plot_hourly_nox_changes
