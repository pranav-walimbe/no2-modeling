#!/bin/bash
#SBATCH --job-name=train-no2-regression
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src:/global/home/users/pranavwalimbe/no2-modeling/src/delta-model"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/global/home/users/pranavwalimbe/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

mkdir -p "/global/home/users/pranavwalimbe/.cache"
training_output=$(mktemp "/global/home/users/pranavwalimbe/.cache/train-no2-${SLURM_JOB_ID}.XXXXXX.log")
trap 'rm -f -- "${training_output}"' EXIT
recipient="pranav.walimbe@berkeley.edu"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m modeling.train \
    --device cuda \
    --batch-size 128 \
    --epochs 100 \
    --tabular-epochs 75 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --prefetch-factor 2 \
    --seed 42 \
    --head-dim 128 \
    --dropout 0.30 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --gradient-clip-norm 5.0 \
    --scheduler-patience 10 \
    --scheduler-factor 0.50 \
    --early-stop-patience 12 \
    --loss-weighting lds_sqrt_inverse \
    --lds-bins 101 \
    --lds-sigma 2.0 \
    --maximum-loss-weight 5.0 \
    --huber-delta 0.1 \
    | tee "${training_output}"
echo "Training command completed; locating result artifacts"

run_dir="$(sed -n 's/^Training .*; outputs: //p' "${training_output}" | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
    echo "Could not determine the model run directory from the training output." >&2
    exit 1
fi

echo "Emailing the two-model prediction scatterplot and training plots"
bash scripts/slurm/email_model_results.sh \
    "${run_dir}" \
    "${SLURM_JOB_ID}" \
    "${recipient}"
