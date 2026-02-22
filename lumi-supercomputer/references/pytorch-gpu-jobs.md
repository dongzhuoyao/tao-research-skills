# PyTorch GPU Jobs on LUMI

## Container Module

LUMI provides pre-built PyTorch containers as modules. Always use these instead of installing PyTorch manually.

```bash
module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527
```

### Available Versions

PyTorch 2.4.1 through 2.7.1, paired with ROCm 5.x through 6.2.4. Latest:
```
PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827
```

### What the Module Sets

| Variable | Purpose |
|----------|---------|
| `$SIF` / `$SIFPYTORCH` | Path to the Singularity `.sif` container image |
| `$SINGULARITY_BIND` | Pre-configured system bind mounts |
| `$RUNSCRIPTS` | Helper scripts directory |
| `$CONTAINERROOT` | User container installation space |

### Included Packages

PyTorch, torchvision, torchdata, torchtext, torchaudio, DeepSpeed, flash-attention, transformers, xformers, vllm (in recent versions).

### Wrapper Scripts

The module provides wrapper scripts that handle container execution transparently:

| Script | Use |
|--------|-----|
| `conda-python-simple` | Run Python inside container (single process) |
| `conda-python-distributed` | Run Python with distributed setup |
| `start-shell` | Interactive shell inside container |
| `pip` | Install packages into user venv |
| `python` | Python inside container |
| `torchrun` | Distributed launch (PyTorch 2.6+) |
| `accelerate` | HuggingFace Accelerate launcher |
| `huggingface-cli` | HF CLI tools |

## Critical Environment Variables

These must be set in every GPU job script:

| Variable | Value | Why |
|----------|-------|-----|
| `ROCR_VISIBLE_DEVICES` | `$SLURM_LOCALID` | Maps each rank to its GCD. AMD equivalent of `CUDA_VISIBLE_DEVICES`. |
| `MIOPEN_USER_DB_PATH` | `/tmp/${USER}-miopen-cache-$SLURM_NODEID` | MIOpen kernel cache. Must be on `/tmp` (RAM), not Lustre — file locking causes hangs. |
| `MIOPEN_CUSTOM_CACHE_DIR` | `$MIOPEN_USER_DB_PATH` | Additional MIOpen cache path. |
| `MIOPEN_SYSTEM_DB_PATH` | (empty) | Clear system DB to avoid stale entries. |
| `NCCL_SOCKET_IFNAME` | `hsn0,hsn1,hsn2,hsn3` | Slingshot network interfaces for RCCL (AMD's NCCL). |
| `NCCL_NET_GDR_LEVEL` | `3` | GPU Direct RDMA level for Slingshot. |
| `MPICH_GPU_SUPPORT_ENABLED` | `1` | Enable GPU-aware MPI (needed for multi-node). |

### Distributed Training Variables

| Variable | Value | When |
|----------|-------|------|
| `MASTER_ADDR` | `$(scontrol show hostnames $SLURM_JOB_NODELIST \| head -n1)` | Multi-node |
| `MASTER_PORT` | `29500` | Multi-node (safe with exclusive nodes) |
| `WORLD_SIZE` | `$SLURM_NPROCS` | Multi-node |
| `RANK` | `$SLURM_PROCID` | Multi-node |

## Single-Node Training (8 GCDs)

For training on one full LUMI-G node with all 8 GCDs:

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

set -euo pipefail

module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

# GPU environment
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export MIOPEN_SYSTEM_DB_PATH=

srun singularity exec $SIFPYTORCH conda-python-simple -u train.py
```

For single-GPU work (e.g., 1 GCD), adjust:
```bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
```

## Multi-Node Distributed Training

### Method 1: torchrun (PyTorch 2.6+)

```bash
#!/bin/bash -e
#SBATCH --account=project_<id>
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=%j.log
#SBATCH --error=%j.log

set -euo pipefail

module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

# GPU environment
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-$SLURM_NODEID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export MIOPEN_SYSTEM_DB_PATH=
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# CPU binding: 7 cores per GCD, respecting NUMA affinity
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

srun --cpu-bind=${CPU_BIND} singularity exec $SIFPYTORCH \
    python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py
```

### Method 2: Explicit Singularity exec with GPU selection script

For finer control over GPU assignment and CPU masks:

```bash
#!/bin/bash -e
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --account=project_<id>
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=%j.log
#SBATCH --error=%j.log

set -euo pipefail

module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export MPICH_GPU_SUPPORT_ENABLED=1

# Per-rank GPU selection script
cat << 'EOF' > /tmp/select_gpu.sh
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*
EOF
chmod +x /tmp/select_gpu.sh

# CPU affinity masks: 7 cores per GCD (0xfe = 11111110, skip core 0 per L3 group)
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

srun --cpu-bind=mask_cpu:$MYMASKS /tmp/select_gpu.sh \
    singularity exec $SIFPYTORCH \
    conda-python-distributed -u train.py
```

## Extending the Container with pip

### Install packages

```bash
# Load the module first
module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527

# Enter container shell
singularity shell $SIF

# Install packages (writes to $CONTAINERROOT/user-software/venv/pytorch)
pip install wandb hydra-core omegaconf mir_eval

# For packages needing compilation on LUMI
CXX=g++-12 pip install torch-scatter
```

### Consolidate with SquashFS

Many small files on Lustre cause severe metadata performance issues. After installing packages, consolidate into a single SquashFS image:

```bash
# Consolidate venv into SquashFS (run after pip installs are done)
make-squashfs

# This replaces $CONTAINERROOT/user-software with a .sqsh file
# The module auto-mounts it on next load

# To edit later (unpacks the SquashFS back to files)
unmake-squashfs
pip install <new-package>
make-squashfs
```

### Manual Container Binding

If not using the module's pre-configured binds, you need these mounts:

```bash
singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray/ \
    -B /usr/lib64/libcxi.so.1 \
    -B /pfs,/scratch,/projappl,/project,/flash,/appl \
    $SIF python train.py
```

## Debugging Tips

### Check GPU visibility

```bash
srun --account=project_<id> --partition=dev-g --nodes=1 --gpus=1 --time=00:10:00 --pty bash
module load LUMI PyTorch/2.7.0-rocm-6.2.4-python-3.12-singularity-20250527
singularity exec $SIF python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Note: PyTorch uses `torch.cuda.*` API even on AMD GPUs (ROCm maps to the CUDA API surface).

### Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: No HIP GPUs` | Missing `ROCR_VISIBLE_DEVICES` or wrong partition | Set `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`, use `-g` partition |
| Hang during `init_process_group` | Missing NCCL vars or firewall | Set `NCCL_SOCKET_IFNAME` and `NCCL_NET_GDR_LEVEL` |
| MIOpen hang / `flock` timeout | MIOpen cache on Lustre | Redirect to `/tmp` via `MIOPEN_USER_DB_PATH` |
| `ImportError` for pip package | Package not in container, not in SquashFS | `singularity shell $SIF && pip install ...` then `make-squashfs` |
| Slow I/O, metadata stalls | Too many small files on Lustre | Use SquashFS, WebDataset/tar, or `/flash` |
| `CUDA extension` build failure | ROCm doesn't have CUDA headers | Use ROCm-compatible versions or HIP-ported extensions |
