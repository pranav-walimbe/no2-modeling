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

source "<PATH TO VENV>/bin/activate" # example /global/home/users/pranavwalimbe/no2_emissions/.venv/bin/activate

cd "<PATH TO SCRIPT DIR>" # example /global/home/users/pranavwalimbe/no2_emissions/src/powerplants/
python -u "<SCRIPT NAME>"
