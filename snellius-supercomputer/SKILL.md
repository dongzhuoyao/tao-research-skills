---
name: snellius-supercomputer
description: Use when running workloads on SURF Snellius supercomputer, including GPU job submission (NVIDIA A100/H100), conda/venv setup, Slurm configuration, and Snellius-specific infrastructure.
---

# Snellius Supercomputer

## When to Use

- Submitting GPU jobs to Snellius (partitions, accounting, resource requests)
- Setting up Python/PyTorch environments on Snellius (conda, venv, modules)
- Debugging Snellius-specific failures (filesystem, modules, GPU access)
- Planning single-node or multi-node GPU training on Snellius
- Understanding storage paths and filesystem layout

## Quick Reference

| Fact | Value |
|------|-------|
| GPU (thin) | NVIDIA A100 40 GB, 4 per node |
| GPU (fat) | NVIDIA H100 SXM 80 GB, 4 per node |
| CPU | AMD EPYC 7H12 (Rome, 2x 64 cores = 128 cores/node) |
| CPU memory | 256 GB (thin), 512 GB (fat GPU nodes) |
| Network | InfiniBand HDR-100 (100 Gbps) |
| Software stack | CUDA, Environment Modules (`module load`), conda/venv |
| SSH | `snellius.surf.nl` |
| Docs | `https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius` |
| Quota check | `myquota` |
| Allocation check | `accinfo` or `budget-overview` |
| Support | `helpdesk@surf.nl` |

## Key Patterns

### 1. Environment Setup (Conda)

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <env_name>
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Always source conda in sbatch scripts -- the shell is non-interactive and doesn't load `.bashrc`.

### 2. Environment Setup (Modules + venv)

```bash
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
python -m venv .venv
source .venv/bin/activate
pip install -e ".[extras]"
```

Use `module spider <name>` to discover available versions. Modules are version-pinned yearly (2022, 2023, etc.).

### 3. Partition Selection

| Partition | Type | Max Walltime | GPUs | Use Case |
|-----------|------|-------------|------|----------|
| `gpu` | A100 nodes | 5 days | 4x A100 40GB/node | Standard GPU training |
| `gpu_h100` | H100 nodes | 5 days | 4x H100 80GB/node | Large models, fast training |
| `gpu_mig` | A100 MIG | 5 days | MIG slices | Small/interactive GPU jobs |
| `thin` | CPU-only | 5 days | None | Preprocessing, CPU workloads |
| `fat` | High-mem CPU | 5 days | None | Memory-intensive CPU jobs |
| `short` | Debug | 1 hour | Varies | Quick testing |

### 4. GPU Job Template (Single Node)

```bash
#!/bin/bash
#SBATCH --account=<account>
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%j.log
#SBATCH --error=%j.log

set -euo pipefail

# Conda init (adjust for your env)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python train.py
```

### 5. Filesystem Layout

| Area | Path | Quota | Purpose |
|------|------|-------|---------|
| Home | `/home/$USER` | 200 GB | Code, configs, small files |
| Scratch | `/scratch-shared/$USER` | Large (project-dependent) | Training data, checkpoints |
| Project | `/projects/<project>` | Varies | Shared project data |

**Warning**: Scratch is NOT backed up and may be purged. Home has daily snapshots but limited space.

### 6. Interactive GPU Session

```bash
srun --account=<account> --partition=gpu --time=01:00:00 \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --mem=32G \
     --pty bash
```

### 7. Monitoring

```bash
squeue -u "$USER"                                              # Your jobs
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode # Job details
scancel <jobid>                                                # Cancel job
accinfo                                                        # Budget/SBU usage
myquota                                                        # Disk quotas
nvidia-smi                                                     # GPU status (on compute node)
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `--account=` | Every job needs an account. Check with `accinfo`. |
| Not sourcing conda in sbatch | Add `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>` |
| Missing `LD_LIBRARY_PATH` | Export `$CONDA_PREFIX/lib` to avoid shared library errors |
| Using `gpu` partition for H100 | Use `gpu_h100` partition for H100 nodes |
| Writing large data to `$HOME` | Use `/scratch-shared/` for training data and checkpoints |
| `module load` without year | Specify the module year: `module load 2023` first |
| No `set -euo pipefail` | Always add to sbatch scripts to catch errors early |

## Failure Triage

- **`ModuleNotFoundError`**: Activate your env and reinstall. Check `module list` for loaded modules.
- **CUDA not available**: Verify `--gpus=` in sbatch, check `nvidia-smi` on the node, verify CUDA module is loaded.
- **OOM kill**: Reduce batch size, increase `--mem`, or switch to H100 (80 GB).
- **Timeout**: Increase `--time` or checkpoint more frequently.
- **`Permission denied` on scratch**: Check `myquota` -- you may have hit inode limits.
- **Conda not found**: Add `source "$(conda info --base)/etc/profile.d/conda.sh"` before `conda activate`.

## Detailed References

- [references/snellius-subpages.md](references/snellius-subpages.md) -- Full index of all Snellius documentation subpages

## When This Skill Doesn't Have the Answer

If a Snellius-specific question isn't covered above or in the reference files, **scrape the official documentation**:

```
WebFetch url="https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius" prompt="<your question>"
```

Key documentation sections:
- Hardware: `https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660208/Snellius+hardware`
- Partitions: `https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting`
- Filesystems: `https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems`
- Job scripts: `https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660220/Writing+a+job+script`
- Machine Learning: `https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/74227856/Machine+Learning`
- Support: `helpdesk@surf.nl`

After finding useful new information, consider updating this skill's reference files so future sessions benefit.
