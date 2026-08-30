#!/bin/bash
#SBATCH --job-name=power_prices
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-6%3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%A_%a.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%A_%a.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src"

ISOS=(CAISO ERCOT ISONE MISO NYISO PJM SPP)
ISO="${ISOS[$SLURM_ARRAY_TASK_ID]}"

srun python -u -m collection.scrape_power_prices --iso "$ISO"
