# SLURM Job Array Example for Hyperparameter Tuning

This repository contains a simple example of running hyperparameter tuning using SLURM job arrays. Each task in the job array processes a different set of hyperparameters from a CSV file, writes the results to an output file, and simulates a workload by sleeping for 2 minutes.

## Files in this Repository

- **run_array_job.sh**: The SLURM submission script that configures and runs the job array.
- **hyperparameters.csv**: A CSV file containing different hyperparameter sets to be used by the SLURM array jobs.
- **hyperparameter_tuning.py**: The Python script that reads hyperparameters for each task based on its array task ID, writes the parameters to an output file, and simulates some workload.


## How to Use

### Prerequisites

- Access to a SLURM cluster
- Python 3 installed on the cluster


### Steps to Clone and Run the Job

1. Clone this repository to your SLURM cluster:

```bash
git clone https://github.com/KempnerInstitute/examples.git
cd examples/array-job
```

2. Submit the SLURM job array:

```bash
sbatch run_array_job.sh
```

This will submit a SLURM array job with 12 tasks, where only two tasks run at the same time `(%2 in #SBATCH --array=1-12%2)`.

3. Monitor the job status:

```bash
squeue -u <username>
```

or simply use

```bash
squeue -u $USER
```

4. Once the job is complete, check the output files in the `output_<task_id>.txt` files:

```bash
cat output_1.txt
```
Which should contain the hyperparameters for task 1.

```bash
learning_rate: 0.01
batch_size: 32
num_epochs: 10
```

5. You can define different kinds of restrictions for the job array. For example,

- **Range of Task IDs**: `#SBATCH --array=1-12` This will run tasks 1 to 12.
- **Comma-separated Task IDs**: `#SBATCH --array=1,3,5,7,9,11` This will run tasks 1, 3, 5, 7, 9, and 11.
- **Range with Step Size**: `#SBATCH --array=1-12:2` (only even task IDs) This will run tasks 1, 3, 5, 7, 9, and 11.
- **Multiple Ranges**: `#SBATCH --array=1-6,8-12` This will run tasks 1 to 6 and 8 to 12.
- **Limit Number of Tasks Running at Once**: `#SBATCH --array=1-12%2` This will run only two tasks at a time.

6. Requeue failed array jobs

If a few jobs within the array job submission fail, we can requeue them using the following instructions.

- Simulate failed jobs by uncommenting lines 29-30 in the `hyperparameter_tuning.py` script.

```python
    if args.task_id %3 == 0:
       raise ValueError()
```

- Run the array job following Step 2, and wait for the jobs to finish. 

- Requeue the failed jobs using the following bash command.

```bash
job_id=<ENTER JOB ID>
for i in $(sacct -j $job_id | grep FAILED | grep -v batch | awk '{ print $1 }'); do scontrol requeue $i; done
```

or use

```bash
job_id=<ENTER JOB ID>
for i in $(squeue -j $job_id --array --states=FAILED | awk 'NR > 1 { print $1 }'); do scontrol requeue $i; done
```

In the above commands, we first find the array job ID of the failed jobs and requeue them using `scontrol requeue <failed_array_job_id>`.

Done!