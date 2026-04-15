#!/bin/bash                                                                                                                 
#SBATCH --job-name=<JOB NAME>
#SBATCH --account=fc_nitrates                                                                                               
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --gres=gpu:A40:1                                                                                    
#SBATCH --nodes=1
#SBATCH --ntasks=1                                                                                                          
#SBATCH --cpus-per-task=8                                                                                                  
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR EMAIL>
#SBATCH --output=/global/home/users/<USERNAME>/job_logs/<JOB NAME>_%j.out                                                    
#SBATCH --error=/global/home/users/<USERNAME>/job_logs/<JOB NAME>_%j.err 

source /global/home/users/<USERNAME>/venv/bin/activate # assuming you have venv setup at this directory

cd /global/home/users/<USERNAME>/conus_co2/src/<PATH TO MODEL DIRECTORY> # example: /nox/model/
python -u <TRAIN SCRIPT> # example: train.py