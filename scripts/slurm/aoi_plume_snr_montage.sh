#!/bin/bash
#SBATCH --job-name=aoi-label-snr
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio4_m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
output="/global/home/users/pranavwalimbe/vis/aoi-label-aligned-snr-montage-${SLURM_JOB_ID}.png"
manifest="/global/home/users/pranavwalimbe/vis/aoi-label-aligned-snr-montage-${SLURM_JOB_ID}.csv"
aoi_scores="/global/home/users/pranavwalimbe/vis/aoi-label-aligned-snr-scores-${SLURM_JOB_ID}.csv"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.aoi_plume_snr_montage \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output "${output}" \
    --manifest-output "${manifest}" \
    --aoi-score-output "${aoi_scores}" \
    "$@"

for artifact in "${output}" "${manifest}" "${aoi_scores}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected artifact was not created: ${artifact}" >&2
        exit 1
    fi
done

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Attached: 20 standardized prior-dataset sequences comparing bottom- and top-quartile label-aligned plume-SNR AOIs.' \
    'Each stratum contains five current-EMA decrease and five current-EMA increase records from distinct AOIs.' \
    'Each record combines matched-filter SNR with soft agreement between current EMA label direction and EMA-aligned plume-amplitude change.' \
    'AOI scores use a geometric balance of increase and decrease compatibility plus their direction-correct fractions.' \
    'Matched filters use high-pass NO2 residuals, crosswind controls, localization, source connectivity, and broad-signal penalties.' \
    'AOI histories use up to 32 deterministic records. Rasters use the fixed 10,000-bundle robust global normalization.' \
    'Cyan + marks the source hotspot and cyan arrows show local downwind direction.' \
    "Sample manifest: ${manifest}" \
    "AOI score table: ${aoi_scores}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'AOI label-aligned plume-SNR montage (${SLURM_JOB_ID})' -a '${output}' '${recipient}'"

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
