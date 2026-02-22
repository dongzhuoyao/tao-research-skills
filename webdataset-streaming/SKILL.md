---
name: webdataset-streaming
description: Use when streaming large datasets from tar shards with WebDataset, replacing file-based DataLoaders, or precomputing encoder latents into shards.
---

# WebDataset Streaming

## When to Use

- Training on datasets too large for file-based random access
- Streaming precomputed encoder latents from tar shards
- Replacing HuggingFace or file-based DataLoaders with streaming pipelines
- Precomputing expensive encoder outputs into reusable shard archives

## Core Concept

WebDataset stores samples as consecutive files in tar archives. Each sample is a group of files sharing the same key (basename):

```
shard_000000.tar
  ├── sample_001.flac    # audio
  ├── sample_001.json    # metadata
  ├── sample_002.flac
  ├── sample_002.json
  └── ...
```

The key advantage: sequential reads from tar files are much faster than random file I/O, especially on network filesystems (NFS, GPFS, Lustre).

## Two-Stage Shard Creation

### Stage 1: Preprocess

Convert raw data into WebDataset-compatible pairs:

```python
import webdataset as wds

with wds.ShardWriter("shards/train_%06d.tar", maxcount=50) as sink:
    for item in dataset:
        key = f"{item['id']:06d}"
        sink.write({
            "__key__": key,
            "flac": encode_flac(item["audio"]),
            "json": json.dumps(item["metadata"]).encode(),
        })
```

**`maxcount=50`**: Keep shards small enough for good shuffle granularity but large enough to amortize tar overhead. Adjust based on sample size.

### Stage 2: Create sizes.json

Track shard metadata for progress bars and epoch estimation:

```python
import json, tarfile, glob

sizes = {}
for tar_path in sorted(glob.glob("shards/train_*.tar")):
    with tarfile.open(tar_path) as tf:
        # Count unique keys (each sample has multiple files)
        keys = set(m.name.rsplit(".", 1)[0] for m in tf.getmembers() if not m.isdir())
        sizes[os.path.basename(tar_path)] = len(keys)

with open("shards/sizes.json", "w") as f:
    json.dump(sizes, f)
```

## DataLoader Construction

### Glob Resolution Before WebDataset

`wds.WebDataset` supports brace expansion (`{000..010}`) but NOT shell globs (`*.tar`). Always resolve globs first:

```python
import glob as globmod
import webdataset as wds

if globmod.has_magic(tar_pattern):
    urls = sorted(globmod.glob(tar_pattern))
    if not urls:
        raise FileNotFoundError(f"No tar files match: {tar_pattern}")
else:
    urls = tar_pattern

dataset = (
    wds.WebDataset(urls, shardshuffle=False)
    .shuffle(100)  # sample-level shuffle buffer
    .to_tuple("flac", "json", "__key__")
    .map(decode_sample)
)
```

### Shard Shuffle vs Sample Shuffle

For small datasets (< 100 shards): use `shardshuffle=False` + sample-level `.shuffle(N)`. Shard shuffling with few shards gives poor randomization.

For large datasets (1000+ shards): use `shardshuffle=True` for shard-level randomization, plus `.shuffle(N)` for within-buffer mixing.

### Conditional DataLoader kwargs

`persistent_workers` and `prefetch_factor` raise errors when `num_workers=0`:

```python
dl_kwargs: dict = {}
if num_workers > 0:
    dl_kwargs["persistent_workers"] = persistent_workers
    if prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
    **dl_kwargs,
)
```

## Latent Shard Variant

For expensive encoders (e.g., frozen audio encoders), precompute encoder outputs and store them as `.pth` + `.json` pairs in tar shards:

```
latent_shard_000000.tar
  ├── sample_001.pth     # torch.save({"encoder_hidden": tensor, "encoder_length": int})
  ├── sample_001.json    # {"chord_indices": [...], "code": "...", "title": "..."}
  └── ...
```

This eliminates encoder forward passes during training, dramatically reducing GPU memory and compute:

```python
dataset = (
    wds.WebDataset(urls, shardshuffle=False)
    .shuffle(100)
    .to_tuple("pth", "json", "__key__")
    .map(decode_latent_sample)
)
```

### Decoding `.pth` Payloads

```python
def decode_latent_sample(sample: dict) -> dict:
    payload = sample["pth"]
    if isinstance(payload, bytes):
        payload = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

    hidden = payload["encoder_hidden"]  # (T, D) tensor
    # ... validate shape, extract metadata from sample["json"] ...
    return {"encoder_hidden": hidden, "targets": targets, ...}
```

## Placeholder Dataset for Progress Bars

WebDataset's `IterableDataset` has no `__len__`. For tqdm progress bars and epoch tracking, wrap with an estimated size:

```python
class SizedIterableWrapper:
    """Wraps an IterableDataset with an estimated __len__ for progress bars."""

    def __init__(self, dataset, estimated_size: int):
        self.dataset = dataset
        self.estimated_size = estimated_size

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.estimated_size
```

Load `estimated_size` from `sizes.json` or compute as `total_samples // batch_size`.

## Gotchas

### Accelerate + IterableDataset

**Never** pass a WebDataset-backed DataLoader through `accelerator.prepare()`. Accelerate's `concatenate` fails on non-tensor batch values (strings, lists of tuples):

```python
# Bad: crashes with TypeError in concatenate
dataloader = accelerator.prepare(wds_dataloader)

# Good: only prepare model + optimizer
model, optimizer = accelerator.prepare(model, optimizer)
# Use wds_dataloader directly
```

### Separate num_workers Settings

WebDataset DataLoader and file-based DataLoader often need different worker counts. Keep them as independent config keys:

```yaml
# data config
webdataset:
  num_workers: 4      # for tar shard streaming
  shuffle_buffer: 100

# training config
loader:
  num_workers: 2      # for file-based HF DataLoader
```

Confusing these causes silent performance issues — the wrong DataLoader gets too many or too few workers.

### GPU Utilization Flashing 0-100%

`num_workers: 0` (the default) causes data starvation on GPU. The GPU idles while the main process reads and decodes the next batch:

```yaml
# Bad: GPU starved
webdataset:
  num_workers: 0

# Good: 4+ workers keep GPU fed
webdataset:
  num_workers: 4
```

Monitor with `watch nvidia-smi` — stable 90%+ utilization means workers are keeping up.

### __key__ Naming Conventions

WebDataset groups files by key (the part before the last `.`). If your filenames have extra dots, samples get split incorrectly:

```
# Bad: "track.v2" is the key, ".flac" is the extension
track.v2.flac
track.v2.json

# Good: use underscores in keys
track_v2.flac
track_v2.json
```

## Anti-Patterns

- **Passing WebDataset loaders to `accelerator.prepare()`**: Guaranteed crash. Only prepare model + optimizer.
- **`num_workers: 0` in production**: Starves the GPU. Always use 4+ workers for real training.
- **Shell globs in `wds.WebDataset()`**: WebDataset doesn't expand `*.tar`. Resolve with `glob.glob()` first.
- **Shard shuffling with few shards**: With < 10 shards, `shardshuffle=True` gives poor randomization. Use sample-level `.shuffle(N)` instead.
- **Forgetting `weights_only=False` for `.pth` loading**: `torch.load` defaults to `weights_only=True` in newer PyTorch, which rejects dict payloads. Explicitly set `weights_only=False` for latent shards.
