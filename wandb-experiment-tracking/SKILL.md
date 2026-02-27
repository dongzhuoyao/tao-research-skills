---
name: wandb-experiment-tracking
description: >-
  Use when integrating W&B experiment tracking into ML training pipelines,
  including logging strategy, run configuration, and online/offline mode
  management. Triggers: "W&B", "wandb", "weights and biases", "experiment
  logging", "wandb.log", "wandb.init", "training dashboard"
---

# W&B Experiment Tracking

## When to Use

- Setting up W&B logging for a new training pipeline
- Deciding what to log and at what granularity
- Configuring online vs offline mode for HPC environments
- Debugging missing or misleading W&B metrics

## Core Principles

### Log Everything an LLM Needs to Debug

Users paste W&B logs to LLMs for diagnosis. Every piece of context needed to understand a training run must be in W&B — not hidden in console output only. If you `print()` something diagnostic, also log it to W&B.

### Console Log Parity

All logs that appear in the Slurm/terminal log must also appear in W&B's Logs tab. The user should be able to debug entirely from W&B without SSH access to the compute node. W&B's default console capture can be lossy — especially for tqdm `\r`-based progress bars. Ensure reliable capture with `settings=wandb.Settings(console="wrap_pty")` or by using Python `logging` (full `\n`-terminated lines) for important messages rather than tqdm-style overwrites.

### Runtime Config at Init

Log the full resolved config at initialization, not just hyperparameters:

```python
wandb_run = wandb.init(project="my-project", name=run_name, config={})
wandb_run.config.update(OmegaConf.to_container(cfg, resolve=True))

# Also log derived values
wandb_run.config.update({
    "model/total_params_M": total_params / 1e6,
    "model/trainable_params_M": trainable_params / 1e6,
    "data/num_shards": num_shards,
    "runtime/gpu_name": torch.cuda.get_device_name(0),
})
```

### Param Counts in Millions

Always log parameter counts in millions (M) for readability:

```python
wandb.config["model/total_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
```

### Two-Tier Logging

Log at two granularities:

- **Batch-level** (`batch/*`): loss, learning rate, throughput — logged every N steps
- **Epoch-level** (`epoch/*`): evaluation metrics, aggregated stats — logged at epoch boundaries or eval intervals

```python
# Batch-level
self._log_wandb({"batch/loss": loss.item(), "batch/lr": lr}, step=global_step)

# Epoch-level
self._log_wandb({"epoch/accuracy": acc, "epoch/eval_loss": eval_loss}, step=global_step)
```

## Patterns

### Online/Offline Mode

```bash
# Default: online (requires WANDB_API_KEY)
export WANDB_MODE=online

# HPC without internet: offline, sync later
export WANDB_MODE=offline
# After job completes, on login node:
wandb sync outputs/wandb/offline-run-*
```

### API Key Management

Store in `.env` at repo root, never commit:

```bash
# .env
WANDB_API_KEY=your_key_here
```

Load in sbatch:
```bash
set -a; source .env; set +a
```

### Run Naming Convention

```python
# For Slurm jobs: experiment name + job ID
run_name = f"fullrun-tiny-{os.environ.get('SLURM_JOB_ID', 'local')}"

# For local runs: experiment name + timestamp
run_name = f"debug-{datetime.now().strftime('%H%M')}"
```

### GPU Memory Logging

Use `memory_reserved()` to match what `nvidia-smi` reports:

```python
wandb.log({
    "system/gpu_mem_used_GB": torch.cuda.memory_reserved() / 1e9,
    "system/gpu_mem_allocated_GB": torch.cuda.memory_allocated() / 1e9,
})
```

`memory_allocated()` shows only active tensors; `memory_reserved()` shows the full caching allocator pool and matches `nvidia-smi`.

### Logging Helper

Centralize W&B logging to avoid scattered `wandb.log()` calls:

```python
def _log_wandb(self, metrics: dict, step: int):
    if self.wandb_run is not None:
        self.wandb_run.log(metrics, step=step)
```

### Version-Prefixed Run Names

Prefix W&B run names with a config version for easy filtering across experiments:

```python
version = cfg.get("version", "0.0")
slurm_id = os.environ.get("SLURM_JOB_ID", "local")
run_name = f"v{version}-{experiment_name}_{slurm_id}"
# Result: "v0.1-snellius-fullrun_19812345"
```

This lets you filter W&B dashboards by version (e.g., `v0.1-*`) to compare runs from the same codebase version.

### Comparing Runs via W&B API

When comparing two training jobs, check the git commit via W&B metadata first — W&B auto-captures git state:

```python
import wandb
api = wandb.Api()
runs = api.runs("my-project", filters={"display_name": "fullrun_19812345"})
run = runs[0]
print(run.metadata["git"]["commit"])   # Which code version ran
print(run.config["training"]["lr"])     # What config was used
```

Fallback when W&B metadata is unavailable:
```bash
# Cross-reference job submit time with git log
sacct -j 19812345 --format=Submit
git log --oneline --since="2024-01-15 10:00" --until="2024-01-15 11:00"
```

### Step Discipline

Always pass explicit `step=` in every `wandb.log()` call. Never rely on W&B's implicit internal step counter:

```python
# Good: explicit step
wandb.log({"batch/loss": loss.item()}, step=global_step)

# Bad: implicit step (W&B auto-increments, causing step misalignment)
wandb.log({"batch/loss": loss.item()})
```

This is **critical** when:
- Using `reinit=True` (ablation variant loops) — implicit counter resets per `wandb.init()`
- Logging from multiple code paths (train loop + eval callback + recognize callback)
- Logging at irregular intervals (every N steps, not every step)

### Include `iter_num` in Batch Logs

Always log the total target iterations alongside current step so W&B charts show progress toward completion:

```python
wandb.log({
    "batch/loss": loss.item(),
    "batch/iter_num": total_iterations,  # e.g., 100000
    "batch/step": global_step,
}, step=global_step)
```

This lets anyone viewing the dashboard immediately see "step 5000 of 100000" without checking the config.

### W&B Group Pattern for Ablations

Group related runs (e.g., ablation variants) for side-by-side comparison in the W&B UI:

```python
job_id = os.environ.get("SLURM_JOB_ID", "local")
wandb_group = f"ablate-{experiment}_{job_id}"

for variant in variants:
    wandb.init(
        project="my-ablations",
        name=f"ablate-{variant}_{job_id}",
        group=wandb_group,
        reinit=True,
    )
    # ... train variant ...

    # Write final metrics to summary for the runs table
    for key, value in final_results.items():
        wandb.run.summary[f"final/{key}"] = value
    wandb.finish()
```

The `group` parameter creates a collapsible group in the W&B runs table. Use `final/<metric>` in `wandb.run.summary` so key results appear as columns without scrolling through charts.

### Separate W&B Projects for Exploration

Keep ablation/synthetic runs in a different W&B project from production training:

```python
# Ablation script
wandb.init(project="my-ablations", ...)

# Production training
wandb.init(project="my-training", ...)
```

Hundreds of short exploratory runs drown out production fullruns in the dashboard. Separate projects keep the main project clean.

### W&B API Access (Not WebFetch)

W&B pages are JavaScript-rendered SPAs. `WebFetch` or `curl` on a W&B URL returns empty HTML with no data. Always use the Python API:

```python
import wandb
api = wandb.Api()

# Query runs
runs = api.runs("entity/project", filters={"display_name": "my-run"})
run = runs[0]

# Access data
print(run.config)           # training config
print(run.summary)          # final metrics
print(run.metadata["git"])  # git commit info
print(run.history())        # time-series metrics as DataFrame
```

## Anti-Patterns

- **Console-only diagnostics**: If it's worth printing, it's worth logging to W&B. Users debug from W&B dashboards, not terminal scrollback.
- **Logging raw param counts**: `432891904` is unreadable. Log `432.9M` or store as float in millions.
- **Forgetting config logging**: `wandb.init()` without `config=` means you can't filter or compare runs by hyperparameters.
- **Using `memory_allocated()` for GPU monitoring**: This shows ~5 GB when `nvidia-smi` shows ~58 GB. Use `memory_reserved()` for the number that matches system tools.
- **Hardcoded `WANDB_MODE`**: Make it configurable. Default to online, let users override to offline via env var or config.
- **Logging every step**: Batch logging every step floods W&B. Log every N steps (e.g., every 10 or 100).

## See Also

- `slurm-gpu-training` — Slurm job ID in W&B run names, monitoring patterns
- `ml-ablation-design` — Grouping ablation variant runs in W&B
- `hydra-experiment-config` — Logging resolved Hydra config at init
