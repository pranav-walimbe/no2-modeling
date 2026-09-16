#!/bin/bash

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
mkdir -p logs

index_job_id="$(sbatch --parsable scripts/slurm/build_tempo_mapping_index.sh "$@")"
observation_job_id="$(sbatch --parsable --dependency="afterok:${index_job_id}" scripts/slurm/build_tempo_mapping_observations.sh "$@")"

echo "TEMPO granule-index job: ${index_job_id}"
echo "TEMPO AOI-mapping array: ${observation_job_id}"
