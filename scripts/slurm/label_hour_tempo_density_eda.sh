#!/bin/bash
#SBATCH --job-name=label-hour-tempo-density
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

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
output="/global/home/users/pranavwalimbe/vis/label-hour-tempo-density-${SLURM_JOB_ID}.png"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" logs

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.label_hour_tempo_density --output "${output}"

if [[ ! -s "${output}" ]]; then
    echo "Expected density plot was not created: ${output}" >&2
    exit 1
fi

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Attached is the density of label-ending TEMPO scan timing across the matched CAMPD hour.' \
    "Figure: ${output}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'TEMPO timing within CAMPD label hour (${SLURM_JOB_ID})' -a '${output}' '${recipient}'"

delivery_confirmed=false
for _ in {1..30}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
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
