#!/bin/bash                                                                                                                     
#SBATCH --job-name=<JOB NAME>
#SBATCH --account=fc_nitrates                                                                                                   
#SBATCH --partition=savio2                                                                                                  
#SBATCH --nodes=<NUMBER OF NODES (default=1)>                                                                                                             
#SBATCH --ntasks=<NUMBER OF TASKS (default=1)>                                                                                                      
#SBATCH --cpus-per-task=<CPUs PER TASK (default=1)>
#SBATCH --time=<TIME LIMIT (format=HH:MM:SS)>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "<PATH TO REPOSITORY>" # example /global/home/users/<USERNAME>/no2-modeling
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

srun python -u -m "<MODULE>" # example preprocessing.generate_dataset
