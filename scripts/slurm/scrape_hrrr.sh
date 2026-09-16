#!/bin/bash
#SBATCH --job-name=scrape_hrrr
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%A_%a.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%A_%a.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src"

START_DATES=(2023-08-01 2024-05-09 2025-02-15 2025-11-24)
END_DATES=(2024-05-08 2025-02-14 2025-11-23 2026-09-10)

start_date="${START_DATES[$SLURM_ARRAY_TASK_ID]}"
end_date="${END_DATES[$SLURM_ARRAY_TASK_ID]}"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m collection.scrape_hrrr \
  --start-date "$start_date" \
  --end-date "$end_date" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --overwrite
