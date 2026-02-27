---
name: hf-dataset-management
description: >-
  Use when curating, uploading, or managing HuggingFace datasets for ML
  training, including offline caching, preflight verification, and data
  directory conventions. Triggers: "HuggingFace", "datasets", "push_to_hub",
  "load_dataset", "HF Hub", "dataset cache"
---

# HuggingFace Dataset Management

## When to Use

- Uploading a new dataset to HuggingFace Hub
- Setting up offline dataset caching for HPC clusters
- Verifying dataset integrity before training
- Organizing local data directories

## Core Principles

### Offline-First Caching

HPC training nodes typically lack internet. Pre-cache all datasets on the login node and verify before submitting jobs:

```bash
# On login node (has internet)
python -c "from datasets import load_dataset; load_dataset('user/dataset', cache_dir='data/dataset')"

# In training script
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

### Preflight Verification

Never let a training job discover missing data 30 minutes in. Run preflight checks before `sbatch`:

```python
def preflight_dataset(data_dir: str, expected_format: str = "parquet"):
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset cache missing: {data_dir}")

    files = list(path.rglob(f"*.{expected_format}"))
    if not files:
        raise FileNotFoundError(f"No {expected_format} files in {data_dir}")

    print(f"OK: {len(files)} {expected_format} files in {data_dir}")
    return len(files)
```

### Upload with Coverage Verification

After uploading, verify the dataset is complete and loadable:

```python
from datasets import load_dataset

# Upload
ds = load_dataset("audiofolder", data_dir="data/my_dataset")
ds.push_to_hub("user/my-dataset")

# Verify round-trip
ds_check = load_dataset("user/my-dataset")
assert len(ds_check["train"]) == expected_count, f"Expected {expected_count}, got {len(ds_check['train'])}"
```

## Patterns

### Data Directory Convention

```
data/
  dataset_a/          # HF cache or raw files
  dataset_b/
  dataset_c/
outputs/              # Training artifacts (checkpoints, logs)
```

Keep data under `data/` and outputs under `outputs/`. Add `data/` and `outputs/` to `.gitignore`.

### Dataset Registry Table

Maintain a table in your `CLAUDE.md` or `README.md`:

```markdown
| Dataset | HF Repo | Cache Dir | Use |
|---------|---------|-----------|-----|
| DatasetA | `user/dataset-a` | `data/dataset-a` | Training |
| DatasetB | `user/dataset-b` | `data/dataset-b` | Evaluation |
```

### WebDataset for Large-Scale Training

For datasets too large for HF's default format, convert to WebDataset (tar shards):

```python
# Target ~100 shards for good parallelism
# Keep estimated_size in sync with actual shard count in config
data:
  webdataset:
    path: "data/my_dataset_wds"
    estimated_size: 10000
    num_workers: 4
```

### Parallel Chord Layers Gotcha

HF datasets with music annotations may have parallel annotation layers (e.g., multiple chord transcriptions). Always select a single canonical layer:

```python
# Bad: concatenating layers
chords = sample["chord_layer_1"] + sample["chord_layer_2"]

# Good: use one basic layer as ground truth
chords = sample["chord_layer_basic"]
```

## Anti-Patterns

- **Downloading inside Slurm jobs**: Network access is unreliable or unavailable on compute nodes. Always pre-cache.
- **No preflight check**: A training job that crashes on missing data after 30 minutes of setup wastes GPU hours.
- **Scattered data locations**: Keep all datasets under `data/`. Don't put some in `~/.cache/huggingface` and others in random paths.
- **Uploading without verification**: Always round-trip test: upload, then download and check counts.
- **Committing data to git**: Add `data/` to `.gitignore`. Use HF Hub or shared filesystem for data distribution.
- **Ignoring `num_workers` for WebDataset**: `num_workers: 0` causes data-starved GPU (GPU util flashing 0-100%). Use at least 4 workers.

## See Also

- `webdataset-streaming` — For large-scale datasets that exceed HF's default format
- `slurm-gpu-training` — Offline-first caching strategy for HPC clusters
- `fail-fast-ml-engineering` — Preflight verification before training
