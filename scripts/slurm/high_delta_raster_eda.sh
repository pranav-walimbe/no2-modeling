#!/bin/bash
#SBATCH --job-name=coal-dominance-raster-eda
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

output_dir="/global/home/users/pranavwalimbe/vis/coal-dominance-raster-eda-${SLURM_JOB_ID}"
output_image="${output_dir}/coal_dominance_raster_comparison.png"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.high_delta_raster_eda \
    --coal-comparison-only \
    --coal-samples 20 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${output_dir}" \
    "$@"

test -s "${output_image}"
printf '%s\n' 'The coal-dominance raster comparison is attached.' \
    | mailx -s 'Coal-dominance NO2 raster EDA' -a "${output_image}" \
        pranav.walimbe@berkeley.edu
