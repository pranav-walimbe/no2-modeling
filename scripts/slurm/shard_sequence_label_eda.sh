#!/bin/bash
#SBATCH --job-name=shard-sequence-eda
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
output_dir="/global/home/users/pranavwalimbe/vis/shard-sequence-label-eda-${SLURM_JOB_ID}"
output_image="${output_dir}/shard_sequence_label_eda.png"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

export PYTHONPATH="${repo_dir}/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}" "${output_dir}"

srun python -u -m eda.shard_sequence_label_eda \
    --shard-dir /global/scratch/projects/fc_nitrates/ddp/nox/dataset/shards/train/000002 \
    --sample-count 10 \
    --output "${output_image}"

test -s "${output_image}"
test -s "${output_image%.png}.csv"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Attached: ten balanced examples from completed training shard 2.' \
    'Each row shows the full five-timestep NO2 sequence followed by its stored emissions-change label.' \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'Five-timestep NO2 sequence EDA (${SLURM_JOB_ID})' -a '${output_image}' '${recipient}'"

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
