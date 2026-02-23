# tao-research-skills

Reusable Claude Code skills for ML research workflows. Distilled from real training pipelines.

## Installation

### One-prompt install (all skills)

Add this to your project's `CLAUDE.md` to install all skills as a git submodule in one shot:

```
Add tao-research-skills as a git submodule and register all skills:

git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared

Then append the following to CLAUDE.md under a "### Shared skills" section:

- `hydra-experiment-config`: Hydra config patterns, hierarchical groups, flat aliases.
- `slurm-gpu-training`: HPC/Slurm job submission, offline-first, conda init.
- `wandb-experiment-tracking`: W&B logging strategy, online/offline modes.
- `hf-dataset-management`: HF dataset caching, preflight, upload verification.
- `gpu-training-acceleration`: PyTorch GPU optimization, torch.compile, gradient checkpointing, Triton fusion, latent-space training.
- `fail-fast-ml-engineering`: No silent fallbacks, config as truth, preflight.
- `ml-ablation-design`: Synthetic ablation design, variant loops, production metrics.
- `webdataset-streaming`: WebDataset tar-shard streaming, Accelerate compatibility, DataLoader gotchas.
- `lumi-supercomputer`: LUMI supercomputer, AMD MI250X/ROCm GPU jobs, PyTorch containers.
- `snellius-supercomputer`: SURF Snellius supercomputer, NVIDIA A100/H100 GPU jobs, conda/venv setup.
- `tao-tmux-zsh`: Personal tmux + zsh dotfiles, Gruvbox theme, vim-style bindings, Oh My Zsh.
```

Or run it directly:

```bash
# From your project root — adds submodule + commits
git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared
git commit -m "Add tao-research-skills shared skills submodule"
```

### As individual symlinks

```bash
# From your project root — symlink only the skills you need
ln -s /path/to/tao-research-skills/hydra-experiment-config skills/hydra-experiment-config
```

Then reference in your project's `CLAUDE.md`:

```markdown
## Skills
- `hydra-experiment-config`: See `skills/hydra-experiment-config/SKILL.md`
```

### As git submodule (all skills)

```bash
git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared
```

## Skills

| Skill | Description |
|-------|-------------|
| [hydra-experiment-config](hydra-experiment-config/) | Structuring ML experiment configs with Hydra: hierarchical groups, flat aliases, config-is-king |
| [slurm-gpu-training](slurm-gpu-training/) | Running GPU training on HPC/Slurm: offline-first, preflight checks, conda init, job monitoring |
| [wandb-experiment-tracking](wandb-experiment-tracking/) | W&B integration: online/offline modes, run naming, param logging, runtime config |
| [hf-dataset-management](hf-dataset-management/) | HuggingFace dataset curation: upload verification, offline caching, preflight checks |
| [gpu-training-acceleration](gpu-training-acceleration/) | PyTorch GPU optimization: CUDA flags, torch.compile, fused optimizers, mixed precision, gradient checkpointing, Triton kernel fusion, latent-space training |
| [fail-fast-ml-engineering](fail-fast-ml-engineering/) | Engineering discipline: no silent fallbacks, explicit errors, config as truth, preflight patterns |
| [lumi-supercomputer](lumi-supercomputer/) | LUMI supercomputer: AMD MI250X/ROCm GPU jobs, PyTorch containers, Slingshot network, Slurm on LUMI |
| [ml-ablation-design](ml-ablation-design/) | Designing ablation studies: synthetic data, variant loops, production metrics, W&B grouping |
| [webdataset-streaming](webdataset-streaming/) | WebDataset tar-shard streaming: shard creation, DataLoader gotchas, Accelerate compatibility |
| [snellius-supercomputer](snellius-supercomputer/) | SURF Snellius supercomputer: NVIDIA A100/H100 GPU jobs, conda/venv setup, Slurm on Snellius |
| [tao-tmux-zsh](tao-tmux-zsh/) | Personal tmux + zsh dotfiles: Gruvbox theme, vim-style bindings, Oh My Zsh, consistent terminal setup |

## Format

Each skill follows the Claude Code skill format:

```
skill-name/
  SKILL.md    # Frontmatter + when to use + principles + patterns + anti-patterns
```
