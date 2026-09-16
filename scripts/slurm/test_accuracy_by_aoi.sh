#!/bin/bash
#SBATCH --job-name=test-accuracy-aoi
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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

run_dir="/global/home/users/pranavwalimbe/model_runs/delta_nox_classification_20260912_060114"
output_prefix="/global/home/users/pranavwalimbe/vis/test-accuracy-by-aoi-${SLURM_JOB_ID}"
figure_path="${output_prefix}.png"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.test_accuracy_by_aoi \
    --run-dir "${run_dir}" \
    --output-prefix "${output_prefix}"

if [[ ! -s "${figure_path}" ]]; then
    echo "Expected result artifact is missing or empty: ${figure_path}" >&2
    exit 1
fi

mail_log_offset=$(ssh -o BatchMode=yes "${mail_host}" stat -c %s "${mail_log}")
printf 'Test accuracy by AOI analysis completed successfully.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${SLURM_JOB_ID}" \
    | ssh -o BatchMode=yes "${mail_host}" \
        "mailx -s 'Test accuracy by AOI (${SLURM_JOB_ID})' -a '${figure_path}' '${recipient}'"

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
