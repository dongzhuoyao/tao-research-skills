# LUMI Hardware Reference

## LUMI-G GPU Nodes (2,978 nodes, 379.70 PFlop/s aggregate)

### GPU: AMD Instinct MI250X

Each node has **4 MI250X modules**, each containing **2 Graphics Compute Dies (GCDs)**. Slurm sees each GCD as a separate GPU, so **8 logical GPUs per node**.

| Spec | Per GCD | Per MI250X | Per Node |
|------|---------|------------|----------|
| HBM2e memory | 64 GB | 128 GB | 512 GB |
| Compute Units | 110 (112 physical, 2 disabled) | 220 | 880 |
| FP64 peak | ~26.5 TFlop/s | ~53 TFlop/s | ~212 TFlop/s |
| L2 cache | 8 MB | 16 MB | 64 MB |
| L1 cache/CU | 16 KB | -- | -- |
| LDS/CU | 64 KB | -- | -- |
| VGPRs/CU | 512 (64-wide, 4-byte) | -- | -- |
| Matrix Cores/CU | 4 | -- | -- |
| Wavefront size | 64 threads | -- | -- |

Architecture: 2nd Gen CDNA (MI250X).

### GPU-GPU Interconnect

| Link Type | Bandwidth (bidirectional) |
|-----------|--------------------------|
| Same MI250X (between 2 GCDs) | 400 GB/s |
| Different MI250X, single link | 100 GB/s |
| Different MI250X, double link | 200 GB/s |

Each MI250X has 5 inter-GPU Infinity Fabric links, 2 CPU-GPU links, and 1 PCIe link.

### CPU: AMD EPYC 7A53 "Trento" (Zen 3)

| Spec | Value |
|------|-------|
| Physical cores | 64 |
| **Usable cores** | **56** (low-noise mode: 1 core disabled per L3 region x 8 regions) |
| Memory | 512 GB DDR4 |
| NUMA nodes | 4 (NPS4 mode, 128 GB each) |
| L1 cache | 32 KiB data + 32 KiB instruction per core |
| L2 cache | 512 KiB per core |
| L3 cache | 256 MiB total (32 MiB per 8-core group, 8 groups) |
| Vector | AVX2 (256-bit) |

### CPU-GPU Affinity Map

Each MI250X module is associated with a specific NUMA node and L3 cache group. For optimal performance, bind CPU cores to their affiliated GPU:

**MPI-only binding (1 rank per GCD, 8 ranks/node):**
```bash
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
```

**Hybrid MPI+OpenMP binding (7 cores per GCD):**
```bash
c=fe   # Binary: 11111110 (7 cores, skip first in each L3 group)
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"
srun --cpu-bind=mask_cpu:$MYMASKS ...
```

The mask `0xfe` = `11111110` skips core 0 of each 8-core L3 group (reserved by low-noise mode). The ordering maps GCD 0-7 to their nearest CPU cores.

### Network: HPE Slingshot-11

| Spec | Value |
|------|-------|
| Technology | HPE Cray Slingshot-11 |
| Link speed | 200 Gbps per NIC (50 GB/s bidirectional) |
| NICs per LUMI-G node | 4 (one per MI250X module) |
| Topology | Dragonfly (3 hierarchical ranks) |
| LUMI-G groups | 24 electrical groups, 32 switches/group |
| Aggregate G-to-G | 276 TB/s |
| Per MI250X to NIC | 25+25 GB/s peak |

### Local Storage

**None.** LUMI-G compute nodes have no local disk. `/tmp` is memory-backed (uses RAM). All I/O goes to the Lustre parallel filesystem.

## LUMI-C CPU Nodes (2,048 nodes)

| Spec | Value |
|------|-------|
| CPU | 2x AMD EPYC 7763 (Zen 3 "Milan"), 128 cores total, 256 threads |
| Memory | 256 GiB (1,888 nodes), 512 GiB (128 nodes), 1,024 GiB (32 nodes) |
| Network | 1x Slingshot-11 NIC (200 Gbps) |

## LUMI-D Data Analytics Nodes (16 nodes)

- **8 without GPU**: 2x AMD EPYC 7742 (Rome), 128 cores, 4 TiB RAM, 25 TB SSD
- **8 with GPU**: 2x AMD EPYC 7742, 128 cores, 2 TiB RAM, 8x NVIDIA A40 (48 GB each), 14 TB SSD

Note: LUMI-D has **NVIDIA** A40 GPUs (not AMD). These are for visualization/analytics, not ML training.

## Storage Systems

| Area | Path | Default Quota | Max Files | Expandable To | Billing Rate |
|------|------|---------------|-----------|---------------|-------------|
| Home | `/users/<username>` | 20 GB | 100k | No | Free |
| Project | `/project/<project>` | 50 GB | 100k | 500 GB | 1x |
| Scratch | `/scratch/<project>` | 50 TB | 2M | 500 TB | 1x |
| Flash | `/flash/<project>` | 2 TB | 1M | 100 TB | 3x |
| Object (LUMI-O) | S3-compatible | 150 TB | 500M | 2.1 PB | 0.25x |

**Policies:**
- No backups on any filesystem
- 90-day retention after project deactivation
- Lustre warning: many small files (e.g., pip packages, conda envs) cause severe metadata performance issues. Use containers and SquashFS.

## Billing

### GPU partitions

- **`standard-g`** (full node, exclusive): `nodes * 4 * hours` GPU-hours
- **`small-g`** / **`dev-g`** (shared): `max(ceil(cores/8), ceil(memory/64GB), GCDs) * hours * 0.5` GPU-hours

### CPU partitions

- **`standard`** (full node, exclusive): `nodes * 128 * hours` CPU-core-hours
- **`small`** (shared): `max(CPU-cores, ceil(memory/2GB)) * hours` CPU-core-hours

Check balance: `lumi-allocations`

## Software Stack

| Component | Details |
|-----------|---------|
| OS module system | Lmod (`module load ...`) |
| Software stacks | CrayEnv, LUMI/24.03, Spack, Local-CSC |
| Recommended toolchain | cpeGNU (for ROCm 6.2.x) |
| Container runtime | Singularity CE (Apptainer-compatible, no module needed) |
| ROCm versions | 6.2.2, 6.2.4, 6.4.4 (current GPU driver) |
| Container build tools | cotainr, PRoot, or local build + transfer |

**Note:** cpeCray is NOT compatible with ROCm 6.2.x. Use cpeGNU for GPU code.
