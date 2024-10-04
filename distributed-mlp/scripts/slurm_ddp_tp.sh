#! /bin/bash
#SBATCH --job-name=mlp_multi_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

#SBATCH --time=00:10:00
#SBATCH --mem=64G
#SBATCH --partition=kempner    # A100
#SBATCH --account=kempner_dev  # Add your own account here

# Check if the first argument ($1) is provided
if [ -z "$1" ]; then
    echo "Pass in the mlp_ddp.py or mlp_tensor_parallel.py to run"
else
    echo "Running: $1"
fi

module load python cuda cudnn
conda activate dist_computing

export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    python -u "$1"
