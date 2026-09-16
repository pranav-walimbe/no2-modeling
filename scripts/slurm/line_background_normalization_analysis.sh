#!/bin/bash
#SBATCH --job-name=line-bg-analysis
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
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

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

output_prefix="/global/home/users/pranavwalimbe/vis/line-background-analysis-${SLURM_JOB_ID}"

srun python -u -m eda.line_background_normalization_eda \
    --analysis \
    --sample-count 500 \
    --bootstrap-replicates 1000 \
    --output-prefix "${output_prefix}"

for suffix in \
    record-metrics.csv \
    bootstrap-summary.csv \
    stratified-summary.csv \
    class-separation.csv \
    agreement.png \
    paired-effects.png; do
    test -s "${output_prefix}-${suffix}"
done
