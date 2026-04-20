# Memory Optimization

## Memory Layout & In-Place Operations

CUDA kernels require contiguous memory. Enforce it after permutations, and prefer in-place ops to reduce allocations:

```python
# Enforce contiguous after index/permute ops (required by custom CUDA kernels)
if x.stride(-1) != 1:
    x = x.contiguous()

# In-place ops reduce peak memory
x = x.mul_(0.18215)          # Not x = x * 0.18215
ema_param.mul_(decay).add_(param.data, alpha=1 - decay)  # EMA update
```

## Strategic `empty_cache()`

Call before memory-intensive phases (FID sampling, evaluation) to reclaim the caching allocator pool:

```python
# Before evaluation/sampling that needs a large contiguous allocation
torch.cuda.empty_cache()
fid_samples = generate_samples(model, n=50000)
torch.cuda.empty_cache()  # Reclaim after
```

Don't sprinkle `empty_cache()` in training loops -- it forces CUDA to re-allocate and hurts throughput.

## NVCC Build Optimization

When compiling custom CUDA extensions, use aggressive flags:

```python
# setup.py for custom CUDA kernels
nvcc_flags = [
    "-O3",                         # Maximum optimization
    "--use_fast_math",             # Fast reciprocal/sqrt on GPU
    "--threads", "4",              # Parallel compilation
    "-U__CUDA_NO_HALF_OPERATORS__",      # Enable fp16 ops
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",  # Enable bf16 ops
]
```

`--use_fast_math` trades strict IEEE compliance for speed -- acceptable for training, verify for inference.
