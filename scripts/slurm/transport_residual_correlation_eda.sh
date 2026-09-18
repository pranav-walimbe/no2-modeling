#!/bin/bash
#SBATCH --job-name=transport-residual-correlation
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src:/global/home/users/pranavwalimbe/no2-modeling/src/delta-model"
export MPLCONFIGDIR="/global/home/users/pranavwalimbe/.cache/matplotlib"
mkdir -p logs

srun python -u -m eda.transport_residual_correlation_eda \
  --shard-dir /global/scratch/projects/fc_nitrates/ddp/nox/dataset/shards/val/000002 \
  --source-records /global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data/val_records.csv
