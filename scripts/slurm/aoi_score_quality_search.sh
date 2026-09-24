#!/bin/bash
#SBATCH --job-name=aoi-score-search
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
output_dir="/global/home/users/pranavwalimbe/vis/aoi-score-quality-search-${SLURM_JOB_ID}"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.aoi_score_quality_search \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${output_dir}" \
    "$@"

for artifact in \
    "${output_dir}/summary.json" \
    "${output_dir}/heuristic_sweep.csv" \
    "${output_dir}/feature_threshold_sweep.csv" \
    "${output_dir}/aoi_score_quality_search.png"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected analysis artifact was not created: ${artifact}" >&2
        exit 1
    fi
done

echo "AOI score quality search completed: ${output_dir}"
