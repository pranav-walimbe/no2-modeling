#!/bin/bash
#SBATCH --job-name=line-bg-normalization
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
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

output_prefix="/global/home/users/pranavwalimbe/vis/line-background-normalization-${SLURM_JOB_ID}"
output_image="${output_prefix}.png"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

srun python -u -m eda.line_background_normalization_eda \
    --sample-count 20 \
    --output-prefix "${output_prefix}"

test -s "${output_image}"
mail_log_offset=$(ssh -o BatchMode=yes "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' 'The upwind-median and cross-sectional line-background comparison is attached.' \
    | ssh -o BatchMode=yes "${mail_host}" \
        "mailx -s 'NO2 line-background normalization (${SLURM_JOB_ID})' \
            -a '${output_image}' '${recipient}'"

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
