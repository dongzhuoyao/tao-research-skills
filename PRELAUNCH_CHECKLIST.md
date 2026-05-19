# Prelaunch Checklist

Use this checklist before launching any remote or multi-GPU experiment. The goal is to make the pipeline runnable without a human in the loop after project-specific settings are provided once.

## Rule

Separate settings into two categories:

- Generic / discoverable
  - detect automatically
  - fail loudly if invalid
- Project-specific
  - ask once if missing
  - save to project config

Examples of project-specific settings:

- GPU quota policy
- one-job-per-GPU policy
- W&B usage
- W&B entity
- W&B project
- dataset choice
- result directory root

Examples of generic / discoverable settings:

- current user context
- writable workspace
- available GPUs
- GPU occupancy
- Python / venv availability
- dependency installability
- remote reachability
- dataset race safety

## Required phases

1. `preflight`
2. `prepare`
3. `sanity run`
4. `parallel launch`
5. `collect results`
6. `retry failed jobs only`

Do not skip directly from login to parallel launch.

## Preflight checks

### Remote access

- can reach the target machine / instance
- correct org / account is active
- correct user context is active
- shell commands run non-interactively

### Storage

- result root exists or can be created
- workspace root is writable
- enough free disk exists for:
  - repo clone
  - venv
  - dataset
  - checkpoints / logs / artifacts

### Python runtime

- `python3` exists
- `pip` or venv tooling exists
- a project venv can be created
- dependency install step succeeds

### GPUs

- `nvidia-smi` works
- visible GPUs are counted
- current occupancy is measured
- selected GPU ids exist
- requested GPU quota does not exceed:
  - visible GPUs
  - currently free GPUs

### Tracking

- if W&B is enabled:
  - API key exists
  - entity is valid
  - project is writable
  - a test `wandb init/log/finish` succeeds

### Dataset

- dataset root exists or is downloadable
- dataset-preparation lock path is writable
- shared download/extraction path is serialized

## Prepare phase

Run these once before job fanout:

- create workspace root
- clone / sync repo
- create venv
- install dependencies
- prepare dataset
- warm up model / encoder downloads if useful

This phase should be serialized.

## Sanity run

Launch exactly one tiny job first.

Verify:

- expected command starts
- expected GPU is used
- logs are written
- result JSON is written
- W&B run appears

Only after this passes should parallel jobs start.

## Parallel launch rules

- obey configured GPU quota
- one process per GPU unless explicitly overridden
- choose GPUs from an explicit list
- use independent output files per job
- use per-job log files

## Collection rules

- collect JSONs
- collect log tails
- summarize:
  - finished
  - failed
  - running
- retry only failed jobs

## Minimal project config

Example:

```json
{
  "execution": {
    "gpu_quota": 2,
    "one_job_per_gpu": true,
    "gpu_ids": [4, 5]
  },
  "tracking": {
    "use_wandb": true,
    "wandb_entity": "team-name",
    "wandb_project": "project-name"
  },
  "paths": {
    "workspace_root": "/data/user/project",
    "results_root": "/data/user/results",
    "venv_root": "/data/user/venvs/project"
  }
}
```

## Failure policy

The launcher should stop immediately if:

- remote is unreachable
- storage is not writable
- quota exceeds free GPUs
- W&B entity/project is invalid
- dataset cannot be prepared safely

Do not improvise fallback behavior silently.
