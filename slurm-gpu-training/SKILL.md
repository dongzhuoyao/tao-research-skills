---
name: slurm-gpu-training
description: Use when running ML training on HPC clusters with Slurm, including job submission, environment setup, monitoring, and failure triage. Applies to any GPU training workload on Slurm-managed clusters.
---

# Slurm GPU Training

## When to Use

- Submitting training jobs to a Slurm cluster
- Setting up conda/venv environments for non-interactive Slurm shells
- Debugging failed Slurm jobs (OOM, timeout, module issues)
- Planning walltime and resource requests for GPU training

## Core Principles

### Offline-First

HPC nodes often lack internet access. Default to offline mode for all package managers and model hubs:

```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Pre-cache everything (models, datasets, tokenizers) on the login node before submitting jobs. W&B can run online if the cluster allows outbound HTTPS — but always have an offline fallback.

### Preflight Before Submit

Run a preflight check script before `sbatch` to verify:
- All dataset shards/files exist in cache
- Model weights are downloaded
- Environment variables are set (API keys, paths)
- GPU is detectable (for interactive debug sessions)

```python
# scripts/preflight_training_offline.py
def check_dataset_cache(data_dir):
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"Dataset not cached: {data_dir}")
    shard_count = len(list(Path(data_dir).glob("*.tar")))
    if shard_count == 0:
        raise FileNotFoundError(f"No shards in {data_dir}")
    print(f"OK: {shard_count} shards in {data_dir}")
```

### Conda Init for Non-Interactive Shells

Slurm jobs run in non-interactive shells where `conda activate` doesn't work by default. Always source conda's init script first:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

The `LD_LIBRARY_PATH` export is critical — without it, CUDA libraries from conda may not be found.

## Patterns

### Sbatch Template

```bash
#!/bin/bash
#SBATCH --job-name=train-experiment
#SBATCH --account=$ACCOUNT
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --output=%j.log
#SBATCH --error=%j.log

set -euo pipefail

# Environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Offline defaults
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Secrets from .env
set -a; source .env; set +a

# Run
python train.py mode=train training=fullrun
```

### Log Naming

Use Slurm job ID only — no date stamps, no experiment names in the filename:

```bash
#SBATCH --output=%j.log    # Good: 12345678.log
#SBATCH --output=train_%j_%x.log  # Avoid: redundant, hard to parse
```

### Run Naming with Job ID

Append Slurm job ID to W&B run names for traceability:

```python
run_name = f"{experiment_name}_{os.environ.get('SLURM_JOB_ID', 'local')}"
```

### Walltime Planning

| GPU | Typical Training | Suggested Walltime |
|-----|------------------|--------------------|
| A100/H100 | 100k iters, bs=16 | 5 days (`5-00:00:00`) |
| A100/H100 | 1200 iters (fastrun) | 1 hour (`01:00:00`) |
| Any | Smoke test / dryrun | 30 min (`00:30:00`) |

### Monitoring

```bash
squeue -u "$USER"                                    # Job status
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS # Post-mortem
tail -f <jobid>.log                                  # Live output
```

### Background Monitor Pattern

For long runs, launch a detached monitor that tails output:

```bash
nohup bash -c "while ! [ -f outputs/${SLURM_JOB_ID}.log ]; do sleep 5; done; tail -f outputs/${SLURM_JOB_ID}.log" > outputs/${SLURM_JOB_ID}_monitor.log 2>&1 &
```

## Anti-Patterns

- **Hardcoding hyperparameters in sbatch scripts**: Sbatch sets environment and calls `python train.py` with config overrides. Hyperparameters live in config files.
- **Running GPU-heavy work on login nodes**: Always use `srun --pty bash` for interactive GPU work, or submit via `sbatch`.
- **Skipping `LD_LIBRARY_PATH`**: Conda environments need this for CUDA/cuDNN to resolve correctly inside Slurm jobs.
- **Date-stamped log files**: Use `%j.log` (job ID only). Date stamps create clutter and the job ID is already unique and traceable via `sacct`.
- **Assuming internet access**: Never `pip install` or `huggingface-cli download` inside a Slurm job. Cache everything beforehand.
- **Ignoring exit codes**: Always use `set -euo pipefail` in sbatch scripts. Silent failures waste GPU hours.
