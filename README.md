# tao-research-skills

Reusable Claude Code skills for ML research workflows. Distilled from real training pipelines.

## Skills

| Skill | Description |
|-------|-------------|
| [hydra-experiment-config](hydra-experiment-config/) | Structuring ML experiment configs with Hydra: hierarchical groups, flat aliases, config-is-king |
| [slurm-gpu-training](slurm-gpu-training/) | Running GPU training on HPC/Slurm: offline-first, preflight checks, conda init, job monitoring |
| [wandb-experiment-tracking](wandb-experiment-tracking/) | W&B integration: online/offline modes, run naming, param logging, runtime config |
| [hf-dataset-management](hf-dataset-management/) | HuggingFace dataset curation: upload verification, offline caching, preflight checks |
| [gpu-training-acceleration](gpu-training-acceleration/) | PyTorch GPU optimization: CUDA flags, torch.compile strategy, fused optimizers, mixed precision |
| [fail-fast-ml-engineering](fail-fast-ml-engineering/) | Engineering discipline: no silent fallbacks, explicit errors, config as truth, preflight patterns |
| [lumi-supercomputer](lumi-supercomputer/) | LUMI supercomputer: AMD MI250X/ROCm GPU jobs, PyTorch containers, Slingshot network, Slurm on LUMI |
| [snellius-supercomputer](snellius-supercomputer/) | SURF Snellius supercomputer: NVIDIA A100/H100 GPU jobs, conda/venv setup, Slurm on Snellius |

## Installation

### As Claude Code skills (symlink)

```bash
# From your project root
ln -s /path/to/tao-research-skills/hydra-experiment-config skills/hydra-experiment-config
```

Then reference in your project's `CLAUDE.md`:

```markdown
## Skills
- `hydra-experiment-config`: See `skills/hydra-experiment-config/SKILL.md`
```

### As git submodule

```bash
git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared
```

## Format

Each skill follows the Claude Code skill format:

```
skill-name/
  SKILL.md    # Frontmatter + when to use + principles + patterns + anti-patterns
```
