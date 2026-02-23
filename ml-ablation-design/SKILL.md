---
name: ml-ablation-design
description: Use when designing ablation studies to compare model components, loss functions, or architectural choices. Covers synthetic data experiments, variant loops, production metrics, and W&B grouping. Triggers: "ablation", "ablation study", "variant comparison", "controlled experiment", "synthetic data experiment"
---

# ML Ablation Design

## When to Use

- Comparing model components (e.g., CTC vs AR heads, different loss functions)
- Designing controlled experiments with synthetic data before committing GPU hours
- Building self-contained ablation scripts that run end-to-end without external dependencies
- Setting up multi-variant experiments with proper W&B tracking

## Workflow

- [ ] **Hypothesis**: Define what you're testing and expected outcome
- [ ] **Synthetic first**: Build toy data, run all variants (~10 min)
- [ ] **Analyze synthetic**: Check if signal separates variants
- [ ] **Real pipeline**: Run on real data only if synthetic results are promising
- [ ] **Compare**: W&B grouped runs, dual-table console output
- [ ] **Decide**: Pick winner based on production metrics, not proxies

## Core Pattern: Self-Contained Ablation Script

One script, zero external dependencies (no dataset downloads, no pretrained models), production metrics, W&B grouping. Runs in minutes on a single GPU.

```
scripts/ablate_<experiment_name>.py   # Self-contained script
slurm_scripts/ablate_<name>.sbatch    # Slurm launcher (passes "$@")
```

## Two-Tier Strategy

1. **Synthetic first** (~10 min): Build a toy dataset that captures the essential structure of your real data. Train all variants. If a component shows no benefit on synthetic data, it won't help on real data either. This catches design flaws cheaply.
2. **Real pipeline second** (hours/days): Only after synthetic results look promising, run the full experiment on real data with the production training loop.

## Synthetic Data Design

### Orthogonal Embeddings via QR Decomposition

Create maximally separable class embeddings so the learning signal is clean:

```python
def make_embeddings(num_classes: int, feature_dim: int, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    raw = torch.randn(num_classes, feature_dim, generator=rng)
    q, _ = torch.linalg.qr(raw.T)
    return q.T[:num_classes]  # (num_classes, feature_dim)
```

### Single Noise Knob

Control task difficulty with one parameter (`noise_sigma`). Higher noise = harder classification. This lets you sweep difficulty to find where each variant breaks:

```python
noise = torch.randn(T, feature_dim, generator=rng) * noise_sigma
features = clean_embedding.expand(T, -1) + noise
```

### Shared Embeddings, Different Seeds

Train and eval datasets must use the **same** class embeddings (same underlying signal) but **different** random seeds (different noise realizations):

```python
chord_embs = make_embeddings(num_classes, feature_dim)  # shared
train_ds = SyntheticDataset(seed=42, embeddings=chord_embs)
eval_ds  = SyntheticDataset(seed=1042, embeddings=chord_embs)  # different seed
```

## Variant Loop

### Fresh Model Per Variant

Never share weights between ablated components. Each variant gets a fresh model from the **same** initialization seed:

```python
for variant in args.variants:
    torch.manual_seed(args.seed)  # same init every time
    model = MyModel(...).to(device)

    # Surgical freezing — not new model classes
    if variant == "ctc_only":
        for p in model.ar_params:
            p.requires_grad_(False)
    elif variant == "ar_only":
        for p in model.ctc_params:
            p.requires_grad_(False)
```

### Selective Reruns via CLI

Allow partial reruns without re-training all variants:

```python
p.add_argument("--variants", nargs="+", default=["ctc_only", "ar_only", "combined"])
```

```bash
# Re-run just one variant after a fix
python scripts/ablate_experiment.py --variants combined
```

## Evaluation: Use Production Metrics

Never use proxy metrics that diverge from your real evaluation pipeline. If your production pipeline uses `mir_eval.chord`, your ablation must too:

```python
# Good: same metrics as production
scores = mir_eval.chord.evaluate(ref_intervals, ref_labels, est_intervals, est_labels)

# Bad: accuracy on synthetic token IDs (doesn't catch Harte conversion bugs)
acc = (pred_tokens == gt_tokens).float().mean()
```

### Robust Per-Sample Evaluation

Wrap each sample in `try/except` so one bad sample doesn't crash the whole eval:

```python
for i, sample in enumerate(eval_samples):
    try:
        scores = evaluate_sample(sample)
        for key in METRIC_KEYS:
            all_scores[key].append(scores[key])
    except Exception:
        pass  # skip malformed samples, don't crash
```

## W&B Integration

### One Run Per Variant, Grouped for Comparison

```python
job_id = os.environ.get("SLURM_JOB_ID", "local")
wandb_group = args.wandb_group or f"ablate-{experiment}_{job_id}"

for variant in args.variants:
    wb_run = wandb.init(
        project="my-ablations",     # separate from production project
        name=f"ablate-{variant}_{job_id}",
        group=wandb_group,          # groups all variants together
        config=vars(args) | {"variant": variant},
        reinit=True,                # multiple wandb.init() in one process
    )
    # ... train ...
    wandb.finish()
```

### `final/` Summary Metrics

Write final results to `wandb.run.summary` so they appear in the W&B runs table without scrolling through charts:

```python
for key, value in final_results.items():
    if isinstance(value, float):
        wandb.run.summary[f"final/{key}"] = value
wandb.finish()
```

### Separate W&B Project

Keep ablation/synthetic runs in a different W&B project from production training. This prevents cluttering the main dashboard with hundreds of short exploratory runs.

## Slurm Launcher

Pass `"$@"` through so all CLI args work from the sbatch command line:

```bash
#!/bin/bash
#SBATCH --job-name=ablate-experiment
#SBATCH --time=00:30:00
#SBATCH --gpus=1

# ... conda init, env vars ...

python scripts/ablate_experiment.py --wandb "$@"
```

```bash
# Override from command line
sbatch slurm_scripts/ablate.sbatch --lr 1e-4 --noise_sigma 0.5 --wandb_group "sweep-lr"
```

## Console Results: Dual-Table Output

For multi-objective ablations, print separate tables for each objective class so results are scannable:

```
  Temporal alignment (frame-level):
  Variant      | Frame Acc  |      Seg |  OverSeg | UnderSeg
  ctc_only     |     92.3%  |   85.1%  |   91.2%  |   88.4%
  combined     |     93.1%  |   86.0%  |   91.8%  |   89.1%

  Chord label quality:
  Variant      |  Head |     Root |   Thirds |    MIREX |   Time
  ctc_only     |   CTC |   95.2%  |   91.0%  |   88.3%  |    12s
  ar_only      |    AR |   93.8%  |   89.5%  |   86.1%  |    15s
  combined     |   CTC |   95.5%  |   91.3%  |   88.7%  |    18s
               |    AR |   94.1%  |   90.0%  |   86.8%  |
```

## Anti-Patterns

- **Shared weights between ablated components**: If CTC encoder and AR encoder share parameters, you can't attribute improvement to either head. Use fully decoupled architectures for clean ablations.
- **Inconsistent init seeds**: Different random seeds across variants mean you're comparing initialization luck, not architecture. Always `torch.manual_seed(args.seed)` before each variant.
- **Proxy metrics that diverge from production**: Token-level accuracy on synthetic IDs won't catch bugs in your Harte conversion, segmentation, or interval computation. Use the real evaluation pipeline.
- **Polluting the main W&B project**: Hundreds of short ablation runs drown out production fullruns. Use a separate W&B project for exploration.
- **Giant ablation scripts**: The script should be self-contained but focused. If it exceeds ~500 lines, you're probably reimplementing your training loop instead of testing a specific hypothesis.

## See Also

- `wandb-experiment-tracking` — Grouping ablation runs in W&B, `final/` summary metrics
- `hydra-experiment-config` — Variant configs using Hydra config groups
- `slurm-gpu-training` — Slurm launcher pattern for ablation scripts
