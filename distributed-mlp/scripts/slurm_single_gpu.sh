#! /bin/bash
#SBATCH --job-name=mlp-single-gpu
#SBATCH --output=mlp.out
#SBATCH --error=mlp.err
#SBATCH --time=00:10:00
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=kempner_dev  # Add your own account here
#SBATCH --gres=gpu:1

module load python
conda activate dist_computing

python mlp_single_gpu.py