---
name: fail-fast-ml-engineering
description: >-
  Use when designing ML training pipelines, data loaders, or inference systems.
  Enforces engineering discipline — no silent fallbacks, explicit errors on
  critical paths, config as single source of truth. Triggers: "silent failure",
  "fallback", "preflight", "assertion", "error handling", "fail fast", "config
  truth"
---

# Fail-Fast ML Engineering

## When to Use

- Designing error handling for training pipelines
- Reviewing code for silent failure modes
- Setting up preflight validation before expensive operations
- Deciding where to put runtime constants (config vs code vs checkpoint)

## Core Principles

### No Silent Fallbacks

Critical paths must not silently fall back to defaults. If a required resource is missing, raise an explicit error immediately:

```python
# Bad: silent fallback
def load_encoder(path=None):
    if path and Path(path).exists():
        return torch.load(path)
    return RandomEncoder()  # Silently trains garbage

# Good: explicit failure
def load_encoder(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Encoder weights not found: {path}")
    return torch.load(path)
```

### Config as Single Source of Truth

Runtime behavior is determined by config files, not checkpoint metadata, environment guessing, or hardcoded defaults:

```python
# Bad: trusting checkpoint metadata
config = checkpoint["config"]  # May be stale, incomplete, or from different code version

# Good: config file is authoritative
config = OmegaConf.load("conf/training/fullrun.yaml")
# Checkpoint metadata is diagnostic only
checkpoint_meta = checkpoint.get("config", {})
```

### Preflight Pattern

Validate everything that can fail before starting expensive computation:

```python
def preflight(cfg):
    """Run before any GPU work. Fail fast, fail cheap."""
    # Data exists?
    assert Path(cfg.data.path).exists(), f"Data missing: {cfg.data.path}"

    # Model weights exist?
    if cfg.model.pretrained:
        assert Path(cfg.model.pretrained).exists(), f"Weights missing: {cfg.model.pretrained}"

    # GPU available?
    assert torch.cuda.is_available(), "No GPU detected"

    # Config sanity
    assert cfg.training.batch_size > 0
    assert cfg.training.lr > 0

    print("Preflight OK")
```

### Fail Cheap, Not Expensive

Order operations so failures happen before GPU hours are consumed:

1. Config validation
2. Data existence checks
3. Model weight loading
4. GPU allocation
5. Training loop

## Patterns

### Explicit Error Messages

Include the failing value and what was expected:

```python
# Bad
raise ValueError("Invalid batch size")

# Good
raise ValueError(f"batch_size must be > 0, got {cfg.training.batch_size}")
```

### Guard Clauses Over Nested Ifs

```python
# Bad: deeply nested
def process(sample):
    if sample is not None:
        if "audio" in sample:
            if sample["audio"].shape[0] > 0:
                return transform(sample["audio"])
    return None  # Silent failure

# Good: guard clauses with explicit errors
def process(sample):
    if sample is None:
        raise ValueError("Sample is None")
    if "audio" not in sample:
        raise KeyError(f"Sample missing 'audio' key. Keys: {list(sample.keys())}")
    if sample["audio"].shape[0] == 0:
        raise ValueError("Audio tensor is empty")
    return transform(sample["audio"])
```

### Assertion-Heavy Data Loading

Data pipelines are the #1 source of silent bugs in ML. Assert aggressively:

```python
def collate_fn(batch):
    audio = torch.stack([s["audio"] for s in batch])
    labels = [s["labels"] for s in batch]

    assert audio.dim() == 3, f"Expected 3D audio tensor, got {audio.dim()}D"
    assert audio.shape[0] == len(labels), f"Batch size mismatch: {audio.shape[0]} vs {len(labels)}"
    assert all(len(l) > 0 for l in labels), "Empty label sequence in batch"

    return {"audio": audio, "labels": labels}
```

### Framework Compatibility Guards

When combining frameworks (e.g., Accelerate + WebDataset), verify compatibility upfront:

```python
# Accelerate's prepare() may fail on custom dataloaders
if isinstance(dataloader.dataset, IterableDataset):
    # Skip accelerator.prepare() for external dataloaders
    # Only prepare model + optimizer
    model, optimizer = accelerator.prepare(model, optimizer)
else:
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

## Anti-Patterns

- **`try: ... except: pass`**: Never swallow exceptions in training code. Every exception is information.
- **Default arguments that hide failures**: `def load(path=None)` where `None` triggers a fallback is a bug waiting to happen.
- **Trusting checkpoint config**: Checkpoints store config snapshots for diagnostics. The YAML config files are the authority.
- **Late validation**: Checking data format inside the training loop wastes GPU time. Validate in preflight.
- **Magic numbers**: `fps = 25` hides the derivation. Use `fps = sample_rate / hop_samples` so the dependency is explicit.
- **Optimistic error handling**: Don't catch errors just to log and continue. If the error matters, stop. If it doesn't, don't catch it.

## See Also

- `hydra-experiment-config` — Config as single source of truth
- `hf-dataset-management` — Preflight verification for datasets
- `slurm-gpu-training` — Preflight before Slurm job submission
