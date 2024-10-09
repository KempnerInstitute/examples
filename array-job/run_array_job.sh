#! /bin/bash
#SBATCH --job-name=job-array
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_dev
#SBATCH --nodes=1           
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=15:00
#SBATCH --array=1-4
#SBATCH --output=%A_%a.out

module load python

python hyperparameter_tuning.py --task_id $SLURM_ARRAY_TASK_ID