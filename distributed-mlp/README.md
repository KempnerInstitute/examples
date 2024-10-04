In this example, we take a simple MLP network and illustrates the insight of different distributed training approaches.

Following is the mlp network diagram and its corrsponding parameters (weight and bias matrices).

| 2-layer MLP | Corresponding Matrices |
|:-----:|:-----:|
| <img src="figures/mlp_network.png" style="width: 60%;"/>| <img src="figures/mlp_matrices.png"/> |

# 1. Environment Setup On HPC Cluster
## 1.1. Create conda environment
Creating the conda envireonment named `dist_computing` (one can use their own customized name).
```{code} bash
conda create -n dist_computing python=3.10
```
## 1.2. Installing PyTorch
```{code} bash
# Activating the conda environment and install PyTorch:
conda activate dist_computing
pip3 install torch
```

# 2. Run On HPC Cluster
[scripts](scripts/) directory contains mlp scripts to run single GPU as well as different multi-GPU distributed approaches such as Distributed Data Parallelism (DDP), Model Parallelism (MP), Tensor Parallelism (TP) and Fully Sharded Data Parallelism (FSDP). It also contains slurm script skeletons that you can use to run the exmaples on the HPC cluster after creating the codna environment as described above.

While in [scripts](scripts/) directory, follow below instructions on how run each script of multi-gpu scenarios:
## 2.1
It runs the mlp example on the (sigle-node) single-GPU setup using one A100 GPU.
```{code}bash
sbatch slurm_single_gpu.sh 
```
## 2.2. Run DDP
It runs the mlp example on the multi-node multi-GPU distributed data parallel setup using two A100 GPUs from two different nodes. 
```{code}bash
sbatch slurm_ddp_tp_fsdp.sh mlp_ddp.py
```
## 2.3. Run MP
It runs the mlp example on the single-node multi-GPU model parallel setup using two A100 GPUs on the same node. 
```{code}bash
sbatch slurm_single_node_mp.sh 
```
## 2.4. Run TP
It runs the mlp example on the multi-node multi-GPU tensor parallel setup using two A100 GPUs from two different nodes. 
```{code}bash
sbatch slurm_ddp_tp_fsdp.sh mlp_tensor_parallel.py 
```
## 2.5. Run FSDP
It runs the mlp example on the multi-node multi-GPU fully sharded data parallel setup using two A100 GPUs from two different nodes. 
```{code}bash
sbatch slurm_ddp_tp_fsdp.sh mlp_fsdp.py 
```