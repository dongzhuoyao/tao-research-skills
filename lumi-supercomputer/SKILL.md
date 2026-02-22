---
name: lumi-supercomputer
description: Use when running workloads on LUMI supercomputer, including GPU job submission, PyTorch with ROCm/AMD MI250X, container workflows, and LUMI-specific Slurm configuration.
---

# LUMI Supercomputer

## When to Use

- Submitting GPU jobs to LUMI (partitions, billing, resource requests)
- Setting up PyTorch on AMD MI250X GPUs (ROCm, not CUDA)
- Debugging LUMI-specific failures (MIOpen cache, Slingshot network, container issues)
- Planning multi-node distributed training on LUMI-G
- Configuring storage paths and understanding Lustre constraints

## Quick Reference

| Fact | Value |
|------|-------|
| GPU | AMD Instinct MI250X (2 GCDs each = 8 logical GPUs/node) |
| GPU memory | 64 GB HBM2e per GCD, 512 GB total/node |
| CPU | AMD EPYC 7A53 "Trento", 64 cores (**56 usable**) |
| CPU memory | 512 GB DDR4 per node |
| Network | HPE Slingshot-11, 200 Gbps/NIC, 4 NICs/node |
| Software stack | ROCm (not CUDA), Singularity containers, `module load PyTorch/...` |
| SSH | `lumi.csc.fi` |
| Outbound IP | `193.167.209.128/26` (for firewall allowlists) |

## Key Differences from NVIDIA Clusters

These are the critical gotchas when moving from NVIDIA (Snellius, etc.) to LUMI:

1. **ROCm, not CUDA**: All GPU code runs through ROCm/HIP. PyTorch works transparently but CUDA-specific extensions (cuDNN calls, custom CUDA kernels) won't compile. Use `ROCR_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES`.

2. **GCDs, not GPUs**: Each MI250X has 2 Graphics Compute Dies. Slurm sees 8 "GPUs" per node, but they're 4 physical modules with 2 GCDs each. In-package bandwidth (same MI250X) is 4x faster than cross-module.

3. **56 cores, not 64**: Low-noise mode disables 1 core per L3 region (8 regions) = 8 disabled. Only 56 are schedulable. Use `--cpus-per-task=7` per GCD (not 8).

4. **Container-first Python**: Never `pip install` directly on Lustre. Use the provided Singularity containers via `module load PyTorch/...`. Extend with `pip install` inside the container shell, then `make-squashfs` to consolidate.

5. **MIOpen cache on `/tmp`**: MIOpen (AMD's cuDNN equivalent) uses a file-locked database. On Lustre this causes hangs. Always redirect: `MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache`.

6. **Slingshot RCCL vars**: Multi-GPU/multi-node communication requires explicit network config:
   ```bash
   export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
   export NCCL_NET_GDR_LEVEL=3
   ```

7. **No email notifications**: Slurm on LUMI does not support `--mail-type`.

8. **Auto-requeue**: Enabled by default. Always use `--no-requeue` and `--open-mode=append` to avoid duplicated output.

9. **Account is mandatory**: Every job needs `--account=project_<id>`. Check allocation with `lumi-allocations`.

## GPU Job Template (Single Node)

```bash
#!/bin/bash
#SBATCH --account=project_<id>
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=%j.log
#SBATCH --error=%j.log

module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

srun singularity exec $SIFPYTORCH conda-python-simple -u train.py
```

## Multi-Node Distributed Training

See [references/pytorch-gpu-jobs.md](references/pytorch-gpu-jobs.md) for full multi-node templates with CPU binding masks and `torch.distributed.run`.

## Partitions

| Partition | Type | Max Walltime | Max Nodes | Billing |
|-----------|------|-------------|-----------|---------|
| `standard-g` | Full-node GPU | 2 days | 1,024 | `nodes * 4 * hours` GPU-h |
| `small-g` | Shared GPU | 3 days | 4 | `max(ceil(cores/8), ceil(mem/64GB), GCDs) * hours * 0.5` |
| `dev-g` | Debug GPU | 30 min--2 hrs | 8--32 | Same as `small-g` |
| `standard` | Full-node CPU | 2 days | 512 | `nodes * 128 * hours` core-h |
| `small` | Shared CPU | 3 days | 4 | `max(cores, ceil(mem/2GB)) * hours` |
| `debug` | Debug CPU | 30 min | 4 | Same as `small` |

## Storage

| Area | Path | Default Quota | Purpose |
|------|------|---------------|---------|
| Home | `/users/<username>` | 20 GB / 100k files | Config, scripts |
| Project | `/project/<project>` | 50 GB (expandable to 500 GB) | Shared code, small data |
| Scratch | `/scratch/<project>` | 50 TB (expandable to 500 TB) | Training data, checkpoints |
| Flash | `/flash/<project>` | 2 TB (expandable to 100 TB) | Hot data (3x billing) |
| Object | LUMI-O (S3) | 150 TB | Cold storage (0.25x billing) |

**Warning**: No backups on any storage. `/tmp` on compute nodes is memory-backed (no local disk). Never install Python packages directly on Lustre — use containers + SquashFS.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using `CUDA_VISIBLE_DEVICES` | Use `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID` |
| `--cpus-per-task=8` per GCD | Use 7 (only 56 cores available, not 64) |
| `pip install` on Lustre | Use container shell + `make-squashfs` |
| Missing MIOpen redirect | Set `MIOPEN_USER_DB_PATH=/tmp/...` |
| Missing NCCL vars | Set `NCCL_SOCKET_IFNAME` and `NCCL_NET_GDR_LEVEL` |
| Expecting `conda activate` to work | Use `module load PyTorch/...` + Singularity |
| No `--no-requeue` | Jobs auto-requeue on preemption, duplicating output |
| `--mail-type` in sbatch | Not supported on LUMI |

## Detailed References

- [references/hardware.md](references/hardware.md) — Full hardware specs, GCD architecture, CPU-GPU affinity, network topology
- [references/pytorch-gpu-jobs.md](references/pytorch-gpu-jobs.md) — Container workflow, multi-node templates, environment variables, venv extension

## When This Skill Doesn't Have the Answer

If a LUMI-specific question isn't covered above or in the reference files, **scrape the official documentation**:

```
WebFetch url="https://docs.lumi-supercomputer.eu/" prompt="<your question>"
```

Key documentation sections:
- Hardware: `https://docs.lumi-supercomputer.eu/hardware/`
- Running jobs: `https://docs.lumi-supercomputer.eu/runjobs/`
- Software: `https://docs.lumi-supercomputer.eu/software/`
- PyTorch: `https://docs.lumi-supercomputer.eu/software/packages/pytorch/`
- Storage: `https://docs.lumi-supercomputer.eu/storage/`

After finding useful new information, consider updating this skill's reference files so future sessions benefit.
