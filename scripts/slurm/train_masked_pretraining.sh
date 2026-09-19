#!/bin/bash
#SBATCH --job-name=train-masked-no2
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src/pretraining/masked-model:/global/home/users/pranavwalimbe/no2-modeling/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

training_output="${SLURM_TMPDIR:-/tmp}/train-masked-no2-${SLURM_JOB_ID}.log"
recipient="pranav.walimbe@berkeley.edu"

srun python -u -m modeling.train \
    --device cuda \
    --batch-size 128 \
    --epochs 300 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --prefetch-factor 2 \
    --seed 42 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --gradient-clip-norm 5.0 \
    --scheduler-patience 10 \
    --scheduler-factor 0.50 \
    --early-stop-patience 25 \
    | tee "${training_output}"
echo "Training command completed; locating result artifacts"

run_dir="$(sed -n 's/^Training .*; outputs: //p' "${training_output}" | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
    echo "Could not determine the masked-model run directory from the training output." >&2
    exit 1
fi

bash scripts/slurm/email_masked_model_results.sh \
    "${run_dir}" \
    "${SLURM_JOB_ID}" \
    "${recipient}"
