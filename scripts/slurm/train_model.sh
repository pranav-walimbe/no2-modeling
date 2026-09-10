#!/bin/bash
#SBATCH --job-name=train-no2
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --time=02:00:00
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

training_output="${SLURM_TMPDIR:-/tmp}/train-no2-${SLURM_JOB_ID}.log"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m modeling.train \
    --device cuda \
    --inputs full \
    --batch-size 128 \
    --epochs 300 \
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
    --early-stop-patience 25 \
    | tee "${training_output}"

run_dir="$(sed -n 's/^Training .*; outputs: //p' "${training_output}" | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
    echo "Could not determine the model run directory from the training output." >&2
    exit 1
fi

comparison_plot="${run_dir}/model_comparison.png"
loss_plot="${run_dir}/loss_curve.png"
for artifact in "${comparison_plot}" "${loss_plot}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected result artifact is missing or empty: ${artifact}" >&2
        exit 1
    fi
done

mail_log_offset=$(ssh -o BatchMode=yes "${mail_host}" stat -c %s "${mail_log}")
printf 'NO2 model training completed successfully.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${SLURM_JOB_ID}" \
    | ssh -o BatchMode=yes "${mail_host}" \
        "mailx -s 'NO2 modeling results (${SLURM_JOB_ID})' \
            -a '${comparison_plot}' -a '${loss_plot}' '${recipient}'"

delivery_confirmed=false
for _ in {1..30}; do
    if ssh -o BatchMode=yes "${mail_host}" \
        tail -c "+$((mail_log_offset + 1))" "${mail_log}" \
        | grep -F "to=<${recipient}>" \
        | grep -Fq 'status=sent'; then
        delivery_confirmed=true
        break
    fi
    sleep 1
done
if [[ "${delivery_confirmed}" != true ]]; then
    echo "Could not confirm successful SMTP delivery in ${mail_log}" >&2
    exit 1
fi
echo "Confirmed SMTP delivery to ${recipient}"
