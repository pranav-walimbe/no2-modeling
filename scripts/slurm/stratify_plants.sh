#!/bin/bash
#SBATCH --job-name=stratify-plants
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio4_m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

# The m512 constraint allocates 8 GB per core, so 16 cores provide 128 GB.
# The extra capacity leaves room for streaming emissions aggregation. Polars
# also uses the requested cores for sorting.

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
strat_dir="/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
histogram="/global/home/users/pranavwalimbe/vis/stratification-scaled-label-histograms-${SLURM_JOB_ID}.png"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m preprocessing.stratify_plants --histogram-output "${histogram}" "$@"

if [[ ! -s "${histogram}" ]]; then
    echo "Expected histogram was not created: ${histogram}" >&2
    exit 1
fi

train_records=$(($(wc -l < "${strat_dir}/train_records.csv") - 1))
val_records=$(($(wc -l < "${strat_dir}/val_records.csv") - 1))
test_records=$(($(wc -l < "${strat_dir}/test_records.csv") - 1))
total_records=$((train_records + val_records + test_records))

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Causal EMA stratification completed successfully.' \
    "Train: ${train_records} records" \
    "Validation: ${val_records} records" \
    "Test: ${test_records} records" \
    "Total: ${total_records} records" \
    "Histogram: ${histogram}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 stratification scaled labels (${SLURM_JOB_ID})' -a '${histogram}' '${recipient}'"

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
