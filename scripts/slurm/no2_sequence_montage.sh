#!/bin/bash
#SBATCH --job-name=no2-sequence-montage
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
mail_tmp_dir="/global/home/users/pranavwalimbe/.cache/mailx"
artifact="/global/home/users/pranavwalimbe/vis/no2-sequence-montage-${SLURM_JOB_ID}.png"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.no2_sequence_montage \
    --split train \
    --samples 20 \
    --seed 20260917 \
    --output "${artifact}"

if [[ ! -s "${artifact}" ]]; then
    echo "Expected montage is missing or empty: ${artifact}" >&2
    exit 1
fi

mkdir -p "${mail_tmp_dir}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf 'The 20-sample NO2 sequence montage is attached.\nJob: %s\nArtifact: %s\n' \
    "${SLURM_JOB_ID}" \
    "${artifact}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "TMPDIR='${mail_tmp_dir}' TMP='${mail_tmp_dir}' TEMP='${mail_tmp_dir}' \
            mailx -s 'NO2 sequence montage (${SLURM_JOB_ID})' -a '${artifact}' '${recipient}'"

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
echo "Confirmed SMTP delivery of ${artifact} to ${recipient}"
