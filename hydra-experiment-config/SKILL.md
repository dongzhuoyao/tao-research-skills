---
name: hydra-experiment-config
description: Use when structuring ML experiment configs with Hydra, adding new config groups, or debugging config resolution. Applies to any project using Hydra for hyperparameter management. Triggers: "Hydra", "config", "yaml config", "OmegaConf", "config groups", "defaults list", "config override"
---

# Hydra Experiment Config

## When to Use

- Setting up Hydra config structure for a new ML project
- Adding new experiment variants (model sizes, training schedules, datasets)
- Debugging config override resolution or composition errors
- Reviewing whether runtime constants are properly externalized to config

## Core Principles

### Config is King

Every runtime constant lives in YAML config, never hardcoded in Python. Checkpoint metadata is diagnostic only — the config file is the single source of truth for reproducing a run.

### Hierarchical Groups with Flat Aliases

Organize configs into semantic groups, but provide flat backward-compatible aliases so users don't need to memorize the full path:

```yaml
# conf/training/default.yaml
run:
  iter_num: 100000
  seed: 42
optim:
  lr: 1e-4
  weight_decay: 0.01
loader:
  batch_size: 16
  num_workers: 4

# Flat aliases (resolved at runtime via OmegaConf interpolation or code)
iter_num: ${run.iter_num}
lr: ${optim.lr}
batch_size: ${loader.batch_size}
```

Both of these work:
```bash
python train.py training.loader.batch_size=8   # hierarchical
python train.py training.batch_size=8           # flat alias
```

### One Entry Point, Mode Switch

Use a single entry point with a `mode` field to dispatch:

```python
@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "evaluate":
        evaluate(cfg)
    elif cfg.mode == "recognize":
        recognize(cfg)
```

## Patterns

### Config Group Layout

```
conf/
  config.yaml               # Top-level: defaults list + mode
  model/
    base.yaml                # Default model
    tiny.yaml                # Small model for fast iteration
    dryrun.yaml              # Minimal model for smoke tests
  training/
    default.yaml             # Standard training config
    dryrun.yaml              # Smoke test (few iterations)
    fastrun.yaml             # Quick validation (frequent eval)
    fullrun.yaml             # Production run
  data/
    dataset_a.yaml           # Dataset-specific loader config
    dataset_b.yaml
```

### Defaults List

```yaml
# conf/config.yaml
defaults:
  - model: base
  - training: default
  - data: dataset_a
  - _self_

mode: train
```

### Derived Values

Compute derived values from config, never from magic numbers:

```python
# Good: derived from config
fps = cfg.model.encoder.sample_rate / cfg.model.encoder.hop_samples

# Bad: magic number
fps = 25
```

### Eliminating Module-Level Constants

Module-level constants that duplicate config values must be replaced with config-driven parameters:

```python
# Bad: hardcoded constant duplicates config
TARGET_SAMPLE_RATE = 24000  # lives in model.encoder.sample_rate

def resample(waveform, orig_sr):
    if orig_sr != TARGET_SAMPLE_RATE:
        ...

# Good: accept from config, use function default only as safety net
def resample(waveform, orig_sr, target_sr: int):
    if orig_sr != target_sr:
        ...
```

Function parameter defaults like `target_sr: int = 24000` are acceptable as fallbacks, but the actual value must always be threaded from config at call sites. Use `functools.partial` to bind config values into pipeline callbacks that can't accept extra arguments directly.

### CLI Override Patterns

```bash
# Single override
python train.py training.optim.lr=3e-4

# Switch config group
python train.py model=tiny training=fastrun

# Multi-run sweep
python train.py --multirun training.optim.lr=1e-4,3e-4,1e-3
```

## Anti-Patterns

- **Hardcoded constants in Python**: If a value might change between experiments, it belongs in config.
- **Sbatch scripts with inline hyperparameters**: Hyperparameters live in `conf/` YAML; sbatch scripts only set environment variables and call `python train.py` with config group overrides.
- **Checkpoint as config source**: Checkpoints may store config snapshots for diagnostics, but the YAML files are the authority for reproduction. Never load training config from a checkpoint to resume — use the original config file.
- **Deeply nested overrides without aliases**: If users frequently override `training.optim.scheduler.warmup_steps`, provide a flat alias `training.warmup_steps`.
- **Mixing config and argparse**: Choose one. Hydra replaces argparse entirely.

## See Also

- `fail-fast-ml-engineering` — Config as single source of truth, preflight validation
- `wandb-experiment-tracking` — Logging resolved Hydra config to W&B at init
