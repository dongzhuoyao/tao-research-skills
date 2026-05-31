---
name: experiment-preflight
description: Use when preparing to launch any remote or multi-GPU ML experiment. Triggers include "preflight", "sanity check", "launch validation", "before training", "GPU check", "dataset check", "environment check"
---

# Experiment Preflight

## When to Use

- Before submitting a remote Slurm, RunPod, or cloud GPU job
- Before starting multi-GPU distributed training
- After environment changes (new container, new code, new dataset)
- When resuming a long-run from checkpoint

## When NOT to Use

- Local CPU-only debugging with no remote or GPU dependency
- Repeated preflights without any changes (results are still valid)

## Core Workflow

Run all 8 checks in order. Fail fast on any check.

| # | Check | What to Verify | Fail If |
|---|-------|---------------|---------|
| 1 | Remote access | SSH/config, jump nodes, VPN | Cannot reach target host |
| 2 | Storage | Quota, paths, permissions, write access | Insufficient space or no write access |
| 3 | Python / environment | Container, modules, `uv sync`, imports | Import errors, version mismatches |
| 4 | GPU availability | `nvidia-smi`, ROCm, device count | No GPUs, wrong GPU type, driver issues |
| 5 | Tracking | W&B auth, project exists, can log | Auth fails, project missing |
| 6 | Dataset preparation | Shards exist, format correct, cache warm | Missing files, zero shards, wrong format |
| 7 | Single-job sanity | Dryrun or fastrun completes without crash | Crash, NaN, import error, config mismatch |
| 8 | Parallel launch | Multi-node/GPU mock or small-scale test | Communication error, rank mismatch |

## Implementation

```python
def preflight(cfg):
    """Fail fast, fail cheap. Run before any GPU work."""
    # 1. Remote
    if cfg.remote.host:
        assert ssh_reachable(cfg.remote.host), f"Cannot reach {cfg.remote.host}"
    
    # 2. Storage
    assert shutil.disk_usage(cfg.output_dir).free > cfg.min_free_gb * 1e9
    assert os.access(cfg.output_dir, os.W_OK)
    
    # 3. Environment
    assert importlib.import_module("torch") is not None
    assert torch.__version__.startswith(cfg.required_torch)
    
    # 4. GPU
    assert torch.cuda.is_available() or torch.backends.mps.is_available()
    assert torch.cuda.device_count() >= cfg.min_gpus
    
    # 5. Tracking
    if cfg.wandb.enabled:
        import wandb
        wandb.login()
    
    # 6. Dataset
    shards = list(Path(cfg.data.dir).glob("*.tar"))
    assert len(shards) >= cfg.min_shards, f"Only {len(shards)} shards found"
    
    # 7. Single-job sanity (dryrun)
    # Run minimal steps, verify no crash and loss is finite
    ...
    
    # 8. Parallel launch (if multi-GPU)
    # torchrun --nnodes=1 --nproc_per_node=2 for local multi-GPU smoke test
    ...
    
    print("Preflight OK")
```

## Key Principles

### Fail Cheap, Not Expensive
Order matters: validate config and data before allocating GPUs or launching remote jobs.

### Environment Isolation
Preflight in the **exact** environment the job will run in:
- Same container / module stack
- Same Python environment (not local laptop if job runs on HPC)
- Same network constraints (offline mode if cluster is air-gapped)

### No Silent Skips
If a check is not applicable, explicitly mark it `N/A` with justification. Do not omit checks.

## Anti-Patterns

- **Local-only preflight for remote jobs**: Passing preflight on your laptop means nothing for HPC
- **Skipping sanity run**: "It worked last week" is not evidence
- **Ignoring warnings**: Warnings in preflight become errors at scale
- **Partial preflight**: Running only checks 1-4 because 5-8 "take too long"

## See Also

- `fail-fast-ml-engineering` — General fail-fast patterns
- `slurm-gpu-training` — Slurm-specific launch workflows
- `lumi-supercomputer` — LUMI-specific preflight and probe requirements
- `wandb-experiment-tracking` — W&B auth and project setup
