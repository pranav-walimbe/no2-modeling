#!/bin/bash
#SBATCH --job-name=aoi-nox-montage
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
output="/global/home/users/pranavwalimbe/vis/aoi-nox-score-standardized-montage-${SLURM_JOB_ID}.png"
manifest="/global/home/users/pranavwalimbe/vis/aoi-nox-score-standardized-montage-${SLURM_JOB_ID}.csv"
normalization="/global/home/users/pranavwalimbe/vis/aoi-nox-score-robust-normalization-${SLURM_JOB_ID}.json"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.aoi_nox_score_montage \
    --output "${output}" \
    --manifest-output "${manifest}" \
    --normalization-output "${normalization}" \
    --normalization-sample-size 10000 \
    "$@"

if [[ ! -s "${output}" ]]; then
    echo "Expected montage was not created: ${output}" >&2
    exit 1
fi
if [[ ! -s "${manifest}" ]]; then
    echo "Expected sample manifest was not created: ${manifest}" >&2
    exit 1
fi
if [[ ! -s "${normalization}" ]]; then
    echo "Expected robust normalization was not created: ${normalization}" >&2
    exit 1
fi

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Attached: 60 prior-dataset NO2 sequences, with 10 samples in every current EMA class and total-NOx AOI-score half.' \
    'AOI scores average hourly total NOx after retaining hours at or above each AOI median mean unit operating time.' \
    'Classes were recomputed with the current overlap-weighted four-timestep EMA method; t4 is post-label.' \
    'NO2 normalization uses the pooled valid pixels from a deterministic 10,000-bundle sample (50,000 timestep rasters).' \
    'The robust center is the sample median and the scale is sample IQR / 1.349. Cyan + marks the stored source hotspot.' \
    "Sample manifest: ${manifest}" \
    "Normalization statistics: ${normalization}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'AOI total-NOx score raster montage (${SLURM_JOB_ID})' -a '${output}' '${recipient}'"

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
