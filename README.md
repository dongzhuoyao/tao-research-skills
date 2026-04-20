<div align="center">

# tao-research-skills

**Battle-tested agent skills for ML research & iOS development workflows.**

*Lessons learned from training diffusion models and vision transformers on A100/H100 clusters — at [UvA](https://ivi.fnwi.uva.nl/vislab/) and [CompVis (LMU)](https://ommer-lab.com/).*

[![Skills](https://img.shields.io/badge/skills-28-blue)]() [![Open Agent Skills](https://img.shields.io/badge/Open%20Agent%20Skills-compatible-blueviolet)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

</div>

---

## Quick Start

### One-prompt install (recommended)

Copy the prompt below into your `CLAUDE.md` (for Claude Code) or `AGENTS.md` (for Codex), and the agent will set everything up:

```
Add tao-research-skills as a git submodule and register all skills:

git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared

Then append the following to CLAUDE.md (or AGENTS.md) under a "### Shared skills" section:

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
- `github-cli`: GitHub CLI (`gh`) patterns, `gh api`, PRs, issues, releases.
- `agents-md-writing`: Writing effective CLAUDE.md/AGENTS.md files, project memory patterns, workflow rules.
- `memory-sync`: One canonical Markdown memory file for Codex + Claude Code, generated AGENTS.md/CLAUDE.md wrappers, drift checks.
- `mem-init`: Bootstrap project memory plus a starter README.md for new repos.
- `vercel-cost-optimization`: Vercel cost optimization, ISR-breaking patterns, function constraints, caching.
- `ios-swiftui-app`: iOS SwiftUI app patterns, UIKit bridging, SSH terminal, voice input, xcodegen + SPM.
- `claude-code-config`: Claude Code setup — permissions, statusline, plugins, settings.json.
- `academic-deep-research`: Paper evaluation and topic surveys — venue, citations, reproducibility scoring.
- `arxiv-latex-reader`: Progressive paper reading — index all sections then deep-read on demand, never truncates.
- `pdf-reader`: PDF → markdown + figures + tables workspace, then delegates to arxiv-latex-reader for progressive reading.
- `blog-reader`: Faithful, figure-aware summaries of long technical blog posts — section chunking, multimodal figure Read, coverage test.
- `gemini-generate-img`: Gemini text-to-image (nano-banana) — models, aspect ratios, prompt patterns, retry logic.
- `gemini-edit-img`: Gemini image editing / multi-image composition — input encoding, preservation tricks, iterative refinement.
- `login-cdp`: Re-authenticate CDP browser sessions — detects expired platforms, guides interactive re-login via MCP browser tools.
- `meta-init`: Install tao-research-skills globally across Claude Code and Codex with auto-update.

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

Each skill is a self-contained `SKILL.md` with YAML frontmatter — the [open agent skills format](https://github.com/anthropics/skills) supported by both **Claude Code** and **Codex**.

```
skill-name/
  SKILL.md              # Frontmatter + when to use + patterns + anti-patterns
  references/           # (optional) Detailed docs for progressive disclosure
```

**Compatible agents** — Skills use the [Open Agent Skills](https://agentskills.io) standard (`name` + `description` frontmatter), supported by 30+ agents:

| Agent | Skills Path | Instruction File |
|-------|-------------|-----------------|
| [Claude Code](https://claude.ai/code) | `.claude/skills/` | `CLAUDE.md` |
| [Codex](https://developers.openai.com/codex) | `.agents/skills/` | `AGENTS.md` |
| [Cursor](https://cursor.com) | `.cursor/skills/` | `.cursorrules` |
| [Gemini CLI](https://geminicli.com) | `.gemini/skills/` | `GEMINI.md` |
| [VS Code / Copilot](https://code.visualstudio.com) | `.github/skills/` | `.github/copilot-instructions.md` |
| [TRAE](https://trae.ai) | `.trae/skills/` | `TRAE.md` |
| [Roo Code](https://roocode.com) | `.roo/skills/` | `.roo/rules` |

And many more — see [agentskills.io](https://agentskills.io) for the full list.

**Trigger keywords** — Descriptions include specific terms (e.g., `"sbatch"`, `"FID"`, `"torch.compile"`) so the agent matches them to your task automatically.

**Progressive disclosure** — Large skills keep a concise overview in `SKILL.md` and link to `references/` for deep dives, so the agent only loads what it needs.

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
| [agents-md-writing](agents-md-writing/) | Writing effective CLAUDE.md / AGENTS.md: section structure, memory patterns, workflow rules, anti-patterns |
| [memory-sync](memory-sync/) | One canonical Markdown memory file for Codex + Claude Code, generated AGENTS.md and CLAUDE.md, drift checks |
| [mem-init](mem-init/) | Bootstrap shared project memory and a starter README.md for a new repository |
| [meta-init](meta-init/) | Install tao-research-skills globally across Claude Code and Codex with auto-update hooks and memory bootstrap |

### Research

| Skill | Description |
|-------|-------------|
| [academic-deep-research](academic-deep-research/) | Paper evaluation and topic surveys: venue, citations, GitHub stats, social buzz, reproducibility, author signals, scored markdown reports |
| [arxiv-latex-reader](arxiv-latex-reader/) | Progressive paper reading: index all sections (~2k tokens), deep-read on demand, chunk-summarize long sections, never truncates |
| [pdf-reader](pdf-reader/) | PDF → markdown + figures + tables workspace (marker / pymupdf4llm / poppler fallback chain), then delegates to arxiv-latex-reader for progressive two-layer reading |
| [blog-reader](blog-reader/) | Faithful, figure-aware digests of long technical blog posts: section-based chunking, parallel-subagent summarization, multimodal figure reading, coverage + claim-traceability test |

### Generative AI

| Skill | Description |
|-------|-------------|
| [gemini-generate-img](gemini-generate-img/) | Gemini text-to-image (nano-banana): model choice, aspect ratios, prompt patterns, multi-candidate sampling, OpenRouter routing, retry logic |
| [gemini-edit-img](gemini-edit-img/) | Gemini image editing and multi-image composition: input encoding, outfit swap / subject transfer / style transfer, preservation tricks, iterative refinement |

### HPC & Supercomputers

| Skill | Description |
|-------|-------------|
| [slurm-gpu-training](slurm-gpu-training/) | Running GPU training on HPC/Slurm: offline-first, preflight checks, conda init, job monitoring |
| [lumi-supercomputer](lumi-supercomputer/) | LUMI supercomputer: AMD MI250X/ROCm GPU jobs, PyTorch containers, Slingshot network |
| [snellius-supercomputer](snellius-supercomputer/) | SURF Snellius supercomputer: NVIDIA A100/H100 GPU jobs, conda/venv setup |

### Deployment & Cost

| Skill | Description |
|-------|-------------|
| [vercel-cost-optimization](vercel-cost-optimization/) | Vercel cost optimization: ISR-breaking patterns, function constraints, Cache-Control headers, Fluid Compute, build minutes |

### iOS & Mobile

| Skill | Description |
|-------|-------------|
| [ios-swiftui-app](ios-swiftui-app/) | iOS SwiftUI app development: UIKit bridging, SSH terminal (Citadel/SwiftTerm), voice input STT pipeline, @MainActor services, xcodegen + SPM, Keychain storage |

### Dev Environment

| Skill | Description |
|-------|-------------|
| [tmux](tmux/) | Tmux dotfiles: Gruvbox theme, vim-style bindings, copy mode, mouse toggle |
| [zsh](zsh/) | ZSH + Oh My Zsh: robbyrussell theme, conda init, PATH config |
| [github-cli](github-cli/) | GitHub CLI (`gh`): `gh api`, PRs, issues, releases, replacing WebFetch for GitHub URLs |
| [claude-code-config](claude-code-config/) | Claude Code setup: permissions, statusline script, plugins (superpowers, code-simplifier, ralph-loop), settings.json |
| [login-cdp](login-cdp/) | Re-authenticate CDP browser sessions: auto-detect expired platforms, guide interactive re-login via MCP browser tools |

---

## Contributing

1. Fork this repo
2. Create a new skill directory with `SKILL.md`
3. Include YAML frontmatter with `name` and `description` (with trigger keywords)
4. Add patterns, anti-patterns, and a `See Also` section
5. Submit a PR
