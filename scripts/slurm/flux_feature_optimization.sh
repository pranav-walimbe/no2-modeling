#!/bin/bash
#SBATCH --job-name=flux-feature-opt
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
output_dir="/global/home/users/pranavwalimbe/vis/flux-feature-opt-${SLURM_JOB_ID}"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"

srun python -u -m eda.flux_feature_optimization \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${output_dir}" \
    "$@"
