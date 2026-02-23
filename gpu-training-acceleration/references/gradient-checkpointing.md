# Gradient Checkpointing

Trade ~10% speed for ~50% memory. Wrap repeated blocks (transformer layers, SSM blocks) so activations are recomputed in backward instead of stored:

```python
# Per-block checkpointing (preferred — fine-grained control)
for block in self.blocks:
    if self.use_checkpoint:
        hidden, residual = torch.utils.checkpoint.checkpoint(
            block, hidden, residual, cond, use_reentrant=False
        )
    else:
        hidden, residual = block(hidden, residual=residual, c=cond)
```

Config-gate it:
```yaml
runtime:
  gradient_checkpointing: true  # ~50% memory reduction, ~10% slower
```
