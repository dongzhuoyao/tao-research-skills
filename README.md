<div align="center">

# tao-research-skills

**Battle-tested Claude Code skills for ML research workflows.**

*Lessons learned from training diffusion models and vision transformers on A100/H100 clusters — at [UvA](https://ivi.fnwi.uva.nl/vislab/) and [CompVis (LMU)](https://ommer-lab.com/).*

[![Skills](https://img.shields.io/badge/skills-13-blue)]() [![Claude Code](https://img.shields.io/badge/Claude%20Code-compatible-blueviolet)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

</div>

---

## Quick Start

### One-prompt install (recommended)

Copy this into your project's `CLAUDE.md` and Claude will set everything up:

```
Add tao-research-skills as a git submodule and register all skills:

git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared

Then append the following to CLAUDE.md under a "### Shared skills" section:

- `hydra-experiment-config`: Hydra config patterns, hierarchical groups, flat aliases.
- `slurm-gpu-training`: HPC/Slurm job submission, offline-first, conda init.
- `wandb-experiment-tracking`: W&B logging strategy, online/offline modes.
- `hf-dataset-management`: HF dataset caching, preflight, upload verification.
- `gpu-training-acceleration`: PyTorch GPU optimization, torch.compile, gradient checkpointing, Triton fusion, latent-space training.
- `genai-evaluation-metrics`: GenAI evaluation metrics (FID, IS, KID, sFID, FDD, FVD, PRDC, AuthPct, Vendi), feature extractors, online/offline eval.
- `fail-fast-ml-engineering`: No silent fallbacks, config as truth, preflight.
- `ml-ablation-design`: Synthetic ablation design, variant loops, production metrics.
- `webdataset-streaming`: WebDataset tar-shard streaming, Accelerate compatibility, DataLoader gotchas.
- `lumi-supercomputer`: LUMI supercomputer, AMD MI250X/ROCm GPU jobs, PyTorch containers.
- `snellius-supercomputer`: SURF Snellius supercomputer, NVIDIA A100/H100 GPU jobs, conda/venv setup.
- `tmux`: Tmux dotfiles, Gruvbox theme, vim-style bindings, copy mode.
- `zsh`: ZSH + Oh My Zsh dotfiles, conda init, PATH config.

Also check for upstream updates daily by running:
cd skills/shared && git pull origin main && cd ../..
```

### Manual install

```bash
# As git submodule (all skills)
git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared
git commit -m "Add tao-research-skills shared skills submodule"

# Or symlink individual skills
ln -s /path/to/tao-research-skills/gpu-training-acceleration skills/gpu-training-acceleration
```

Then reference in your `CLAUDE.md`:

```markdown
## Skills
- `gpu-training-acceleration`: See `skills/gpu-training-acceleration/SKILL.md`
```

---

## How It Works

Each skill is a self-contained `SKILL.md` with YAML frontmatter. Claude Code automatically loads the right skill based on **trigger keywords** in the description — no manual invocation needed.

```
skill-name/
  SKILL.md              # Frontmatter + when to use + patterns + anti-patterns
  references/           # (optional) Detailed docs for progressive disclosure
```

**Trigger keywords** — Skills include specific terms (e.g., `"sbatch"`, `"FID"`, `"torch.compile"`) so Claude matches them to your task automatically.

**Progressive disclosure** — Large skills keep a concise overview in `SKILL.md` and link to `references/` for deep dives, so Claude only loads what it needs.

**Cross-references** — Each skill has a `See Also` section linking related skills for easy navigation.

---

## Available Skills

### Training & Optimization

| Skill | Description |
|-------|-------------|
| [gpu-training-acceleration](gpu-training-acceleration/) | PyTorch GPU optimization: CUDA flags, torch.compile, fused optimizers, mixed precision, gradient checkpointing, Triton kernel fusion, latent-space training |
| [genai-evaluation-metrics](genai-evaluation-metrics/) | GenAI evaluation: FID, IS, KID, sFID, FDD, FVD, PRDC, LPIPS, SSIM, AuthPct, Vendi — feature extractors, online/offline eval, distributed computation |
| [ml-ablation-design](ml-ablation-design/) | Designing ablation studies: synthetic data, variant loops, production metrics, W&B grouping |

### Experiment Management

| Skill | Description |
|-------|-------------|
| [hydra-experiment-config](hydra-experiment-config/) | Structuring ML experiment configs with Hydra: hierarchical groups, flat aliases, config-is-king |
| [wandb-experiment-tracking](wandb-experiment-tracking/) | W&B integration: online/offline modes, run naming, param logging, runtime config |
| [hf-dataset-management](hf-dataset-management/) | HuggingFace dataset curation: upload verification, offline caching, preflight checks |
| [webdataset-streaming](webdataset-streaming/) | WebDataset tar-shard streaming: shard creation, DataLoader gotchas, Accelerate compatibility |

### Engineering Discipline

| Skill | Description |
|-------|-------------|
| [fail-fast-ml-engineering](fail-fast-ml-engineering/) | No silent fallbacks, explicit errors, config as single source of truth, preflight patterns |

### HPC & Supercomputers

| Skill | Description |
|-------|-------------|
| [slurm-gpu-training](slurm-gpu-training/) | Running GPU training on HPC/Slurm: offline-first, preflight checks, conda init, job monitoring |
| [lumi-supercomputer](lumi-supercomputer/) | LUMI supercomputer: AMD MI250X/ROCm GPU jobs, PyTorch containers, Slingshot network |
| [snellius-supercomputer](snellius-supercomputer/) | SURF Snellius supercomputer: NVIDIA A100/H100 GPU jobs, conda/venv setup |

### Dev Environment

| Skill | Description |
|-------|-------------|
| [tmux](tmux/) | Tmux dotfiles: Gruvbox theme, vim-style bindings, copy mode, mouse toggle |
| [zsh](zsh/) | ZSH + Oh My Zsh: robbyrussell theme, conda init, PATH config |

---

## Contributing

1. Fork this repo
2. Create a new skill directory with `SKILL.md`
3. Include YAML frontmatter with `name` and `description` (with trigger keywords)
4. Add patterns, anti-patterns, and a `See Also` section
5. Submit a PR
