#!/bin/bash                                                                                                                     
#SBATCH --job-name=<JOB NAME>
#SBATCH --account=fc_nitrates                                                                                                   
#SBATCH --partition=savio2                                                                                                  
#SBATCH --nodes=<NUMBER OF NODES (default=1)>                                                                                                             
#SBATCH --ntasks=<NUMBER OF TASKS (default=1)>                                                                                                      
#SBATCH --cpus-per-task=<CPUs PER TASK (default=1)>
#SBATCH --time=<TIME LIMIT (format=HH:MM:SS)>
#SBATCH --output=/global/home/users/<YOUR USERNAME>/job_logs/<JOB NAME>_%j.out
#SBATCH --error=/global/home/users/<YOUR USERNAME>/job_logs/<JOB_NAME>_%j.err

cd "<PATH TO REPOSITORY>" # example /global/home/users/<USERNAME>/no2-modeling
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate

python -u -m "<MODULE>" # example no2_modeling.preprocessing.generate_dataset
