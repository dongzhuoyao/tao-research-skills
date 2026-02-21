---
name: weak-supervision-alignment
description: Use when training models with weakly labeled sequential data — ordered labels without per-element timestamps. Covers CTC alignment, hybrid CTC-AR architectures, label expansion for repeat structures, and GT ambiguity resolution.
---

# Weak Supervision Alignment

## When to Use

- Training sequence models where only ordered labels are available (no per-frame timestamps)
- Designing hybrid CTC-AR architectures
- Handling label ambiguity (missing segments, uncertain boundaries)
- Working with repeat structures in sequential data (e.g., music, speech with repeated phrases)

## Core Principles

### CTC Enables Timestamp-Free Training

Connectionist Temporal Classification (CTC) aligns ordered label sequences to frame-level features without requiring per-frame ground truth. The key requirements:

- Monotonic alignment: labels appear in order but timing is unknown
- Blank token: a special `<blank>` token allows the model to emit "nothing" at frames between labels
- Label sequence can be much shorter than the feature sequence

```
Features:  [f1] [f2] [f3] [f4] [f5] [f6] [f7] [f8]
CTC path:  [A]  [_]  [_]  [B]  [B]  [_]  [C]  [C]
Output:     A              B              C
```

### Hybrid CTC-AR: Best of Both Worlds

Use dual heads — CTC for alignment and autoregressive (AR) for sequence modeling:

- **CTC head**: provides frame-level alignment, trained with CTC loss
- **AR head**: models label dependencies, trained with cross-entropy on aligned targets
- **Annealing**: gradually shift loss weight from CTC to AR as alignment improves

```python
# CTC annealing: linear from 1.0 to 0.2
ctc_weight = max(0.2, 1.0 - (step / anneal_steps) * 0.8)
total_loss = ctc_weight * ctc_loss + (1 - ctc_weight) * ar_loss
```

### Label Expansion for Repeats

When data contains repeated structures (e.g., verse-chorus-verse in music), expand the label sequence to match. CTC tolerates over-expansion but fails on under-expansion:

```python
# Song structure: Verse Chorus Verse Chorus
# Chord sequence (one pass): Am F C G
# Expanded for CTC:          Am F C G Am F C G

# Over-expansion: CTC handles it (extra blanks absorb surplus)
# Under-expansion: CTC fails (forced alignment breaks)
```

### GT Ambiguity Resolution Hierarchy

When ground truth labels don't fully cover the input, apply this priority:

1. **Trim to covered region**: If labels cover only part of the input, trim the input to the labeled region
2. **Model uncovered as blank**: If trimming isn't possible, treat uncovered regions as `<blank>`
3. **Downweight sample**: If ambiguity is too severe, reduce the sample's loss weight rather than excluding it

## Patterns

### CTC Target Preparation

```python
def prepare_ctc_targets(labels: list[int], blank_id: int = 0):
    """Insert blanks between labels for CTC."""
    # CTC requires: blank L1 blank L2 blank ... Ln blank
    targets = [blank_id]
    for label in labels:
        targets.append(label)
        targets.append(blank_id)
    return targets
```

### AR Target from CTC Alignment

Use CTC's Viterbi alignment to generate frame-level targets for the AR head:

```python
def ctc_to_ar_targets(ctc_output: torch.Tensor, blank_id: int = 0):
    """Convert CTC frame predictions to AR training targets."""
    # Viterbi decode
    best_path = ctc_output.argmax(dim=-1)  # [T]

    # Remove blanks and dedup for AR targets
    ar_targets = []
    prev = blank_id
    for token in best_path:
        if token != blank_id and token != prev:
            ar_targets.append(token)
        prev = token
    return ar_targets
```

### Boundary-Free Evaluation

Since CTC provides soft boundaries, evaluate at the sequence level or use standardized time grids:

```python
# Don't evaluate boundary precision for weakly supervised models
# Instead, evaluate:
# 1. Sequence accuracy (ordered labels match)
# 2. Frame-level accuracy on a fixed grid (e.g., 100ms frames)
# 3. Standard metrics that tolerate boundary jitter (e.g., mir_eval chord overlap)
```

### Conditional Head Training

Enable/disable heads via config flags to isolate training signals:

```yaml
training:
  heads:
    train:
      ctc: true
      ar: true
      capo: true
    # Set ar: false to train CTC-only first, then enable AR
```

```python
if cfg.training.heads.train.ctc:
    ctc_loss = compute_ctc_loss(ctc_logits, targets)
    total_loss += ctc_weight * ctc_loss

if cfg.training.heads.train.ar:
    ar_loss = compute_ar_loss(ar_logits, aligned_targets)
    total_loss += (1 - ctc_weight) * ar_loss
```

## Anti-Patterns

- **Requiring timestamps for CTC training**: The whole point of CTC is that you don't need them. If you have timestamps, use frame-level cross-entropy instead.
- **Under-expanding repeated labels**: CTC can absorb extra labels via blanks, but it cannot hallucinate missing ones. Always err on the side of over-expansion.
- **Single-head architecture**: CTC alone gives noisy frame predictions. AR alone needs pre-aligned targets. The hybrid is strictly better for weakly supervised settings.
- **Evaluating boundary precision**: Weakly supervised models have uncertain boundaries by design. Evaluate sequence correctness and overlap metrics, not boundary F1.
- **Concatenating parallel annotation layers**: If multiple annotators provide different label sequences, pick one canonical layer. Concatenation creates impossible CTC targets.
- **Ignoring blank regions**: Uncovered input regions should be explicitly handled (trim, blank, or downweight), not silently left for the model to figure out.
