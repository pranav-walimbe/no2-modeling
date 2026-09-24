#!/bin/bash
#SBATCH --job-name=aoi-score-refine
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio4_m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

repo_dir="/global/home/users/pranavwalimbe/no2-modeling"
output_dir="/global/home/users/pranavwalimbe/vis/aoi-score-refinement-${SLURM_JOB_ID}"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m eda.aoi_score_refinement \
    --output-dir "${output_dir}" \
    "$@"

if [[ ! -s "${output_dir}/summary.json" ]]; then
    echo "Expected refinement summary was not created: ${output_dir}/summary.json" >&2
    exit 1
fi

echo "AOI score refinement completed: ${output_dir}"
