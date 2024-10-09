#! /bin/bash
#SBATCH --job-name=mlp_single_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

#SBATCH --time=00:10:00
#SBATCH --mem=64G

#SBATCH --partition=kempner    # A100
#SBATCH --account=kempner_dev  # Add your own account here

module load python cuda cudnn
conda activate dist_computing

python mlp_single_gpu.py