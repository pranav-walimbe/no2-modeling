#!/bin/bash
#SBATCH --job-name=stratify-plants
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
strat_dir="/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
diagnostic="/global/home/users/pranavwalimbe/vis/stratification-ema-balance-${SLURM_JOB_ID}.png"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m preprocessing.stratify_plants \
    --diagnostic-output "${diagnostic}" \
    "$@"

if [[ ! -s "${diagnostic}" ]]; then
    echo "Expected stratification diagnostic was not created: ${diagnostic}" >&2
    exit 1
fi

train_records=$(($(wc -l < "${strat_dir}/train_records.csv") - 1))
val_records=$(($(wc -l < "${strat_dir}/val_records.csv") - 1))
test_records=$(($(wc -l < "${strat_dir}/test_records.csv") - 1))
total_records=$((train_records + val_records + test_records))

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Causal EMA stratification completed successfully.' \
    'Selected the highest coal-NOx-ranked half of coal-containing AOIs.' \
    'Raw EMA-change threshold: +/-100' \
    'Four rasters retained; label interval ends at t2; EMA history is four hours.' \
    'Filtered AOI clusters were assigned by class to approximately 70/15/15 splits.' \
    'Every split is independently balanced across decrease, steady, and increase.' \
    "Train: ${train_records} records" \
    "Validation: ${val_records} records" \
    "Test: ${test_records} records" \
    "Total: ${total_records} records" \
    "Diagnostic: ${diagnostic}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 stratification diagnostics (${SLURM_JOB_ID})' -a '${diagnostic}' '${recipient}'"

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
