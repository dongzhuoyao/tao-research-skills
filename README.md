<div align="center">

# tao-research-skills

**Battle-tested agent skills for ML research, HPC, and AI engineering workflows.**

*Distilled from training diffusion models and vision transformers on A100/H100 clusters at [UvA](https://ivi.fnwi.uva.nl/vislab/) and [CompVis (LMU)](https://ommer-lab.com/).*

[![Skills](https://img.shields.io/badge/skills-28-blue)]() [![Open Agent Skills](https://img.shields.io/badge/Open%20Agent%20Skills-compatible-blueviolet)]() [![Claude Code](https://img.shields.io/badge/Claude%20Code-ready-brightgreen)]() [![Codex](https://img.shields.io/badge/Codex-ready-brightgreen)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

</div>

---

A plug-and-play collection of **28 self-contained agent skills** — HPC job submission, W&B logging, PyTorch GPU optimization, PDF/arXiv reading, and more — that your coding agent auto-loads the moment a trigger keyword appears in your prompt.

- 🧠 **Battle-tested** — every skill is distilled from real research projects, not hypothetical best practices.
- 🔌 **Multi-agent** — works in Claude Code, Codex, Cursor, Gemini CLI, and any other [Open Agent Skills](https://agentskills.io)–compatible agent.
- ♻️ **Auto-update** — a single `SessionStart` hook pulls new skills into every project every day, automatically.
- 🎯 **Trigger-based** — skills activate from keywords like `"sbatch"`, `"FID"`, `"torch.compile"` — no manual loading.
- 📦 **Progressive disclosure** — large skills keep a tight overview in `SKILL.md` and link to `references/` for deep dives, so the agent only pulls what it needs.

---

## Install

**Recommended — global install across every project on the machine** (via the built-in `meta-init` skill):

```bash
git clone https://github.com/dongzhuoyao/tao-research-skills.git ~/lab/tao-research-skills
python ~/lab/tao-research-skills/skills/infra/meta-init/scripts/meta_init.py
```

This symlinks every skill into `~/.claude/skills/` and `~/.agents/skills/`, injects a non-blocking `SessionStart` git-pull hook so skills stay current, and bootstraps the global memory file. Re-run any time — it's idempotent. See [`skills/infra/meta-init/SKILL.md`](skills/infra/meta-init/SKILL.md) for flags (`--dry-run`, `--platforms`, `--repo`).

**Per-project — git submodule** (for teams who want to pin a version):

```bash
git submodule add https://github.com/dongzhuoyao/tao-research-skills.git skills/shared
python skills/shared/skills/infra/meta-init/scripts/meta_init.py --repo skills/shared
```

The second command symlinks each skill into the project's agent skill paths so the agent can auto-discover them.

---

## Skill Catalog

### 🏋️ Training & Optimization

| Skill | Description |
|-------|-------------|
| [gpu-training-acceleration](skills/training/gpu-training-acceleration/) | PyTorch GPU optimization: CUDA flags, `torch.compile`, fused optimizers, mixed precision, gradient checkpointing, Triton kernel fusion, latent-space training |
| [ml-ablation-design](skills/training/ml-ablation-design/) | Designing ablation studies: synthetic data, variant loops, production metrics, W&B grouping |
| [genai-evaluation-metrics](skills/training/genai-evaluation-metrics/) | GenAI evaluation: FID, IS, KID, sFID, FDD, FVD, PRDC, LPIPS, SSIM, AuthPct, Vendi — feature extractors, online/offline eval, distributed computation |

### 📊 Experiment & Data Management

| Skill | Description |
|-------|-------------|
| [hydra-experiment-config](skills/experiments/hydra-experiment-config/) | Structuring ML experiment configs with Hydra: hierarchical groups, flat aliases, config-is-king |
| [wandb-experiment-tracking](skills/experiments/wandb-experiment-tracking/) | W&B integration: online/offline modes, run naming, param logging, runtime config |
| [hf-dataset-management](skills/experiments/hf-dataset-management/) | HuggingFace dataset curation: upload verification, offline caching, preflight checks |
| [webdataset-streaming](skills/experiments/webdataset-streaming/) | WebDataset tar-shard streaming: shard creation, DataLoader gotchas, Accelerate compatibility |

### 🖥️ HPC & Supercomputers

| Skill | Description |
|-------|-------------|
| [slurm-gpu-training](skills/hpc/slurm-gpu-training/) | Running GPU training on HPC/Slurm: offline-first, preflight checks, conda init, job monitoring |
| [lumi-supercomputer](skills/hpc/lumi-supercomputer/) | LUMI supercomputer: AMD MI250X/ROCm GPU jobs, PyTorch containers, Slingshot network |
| [snellius-supercomputer](skills/hpc/snellius-supercomputer/) | SURF Snellius supercomputer: NVIDIA A100/H100 GPU jobs, conda/venv setup |

### 📚 Research & Reading

| Skill | Description |
|-------|-------------|
| [academic-deep-research](skills/research/academic-deep-research/) | Paper evaluation and topic surveys: venue, citations, GitHub stats, social buzz, reproducibility, author signals, scored markdown reports |
| [arxiv-latex-reader](skills/research/arxiv-latex-reader/) | Progressive paper reading: index all sections (~2k tokens), deep-read on demand, chunk-summarize long sections, never truncates |
| [pdf-reader](skills/research/pdf-reader/) | PDF → markdown + figures + tables workspace (marker / pymupdf4llm / poppler fallback chain), then delegates to arxiv-latex-reader for progressive two-layer reading |
| [blog-reader](skills/research/blog-reader/) | Faithful, figure-aware digests of long technical blog posts: section-based chunking, parallel-subagent summarization, multimodal figure reading, coverage + claim-traceability tests |

### 🎨 Generative AI

| Skill | Description |
|-------|-------------|
| [gemini-generate-img](skills/genai/gemini-generate-img/) | Gemini text-to-image (nano-banana): model choice, aspect ratios, prompt patterns, multi-candidate sampling, OpenRouter routing, retry logic |
| [gemini-edit-img](skills/genai/gemini-edit-img/) | Gemini image editing and multi-image composition: input encoding, outfit swap / subject transfer / style transfer, preservation tricks, iterative refinement |

### 🧠 Agent Tooling & Memory

| Skill | Description |
|-------|-------------|
| [fail-fast-ml-engineering](skills/infra/fail-fast-ml-engineering/) | No silent fallbacks, explicit errors, config as single source of truth, preflight patterns |
| [agents-md-writing](skills/infra/agents-md-writing/) | Writing effective `CLAUDE.md` / `AGENTS.md`: section structure, memory patterns, workflow rules, anti-patterns |
| [memory-sync](skills/infra/memory-sync/) | One canonical Markdown memory file for Codex + Claude Code, generated `AGENTS.md`/`CLAUDE.md` wrappers, drift checks |
| [mem-init](skills/infra/mem-init/) | Bootstrap shared project memory and a starter `README.md` for a new repository |
| [meta-init](skills/infra/meta-init/) | Install tao-research-skills globally across Claude Code and Codex with auto-update hooks and memory bootstrap |

### 🛠️ Dev Environment

| Skill | Description |
|-------|-------------|
| [tmux](skills/devenv/tmux/) | Tmux dotfiles: Gruvbox theme, vim-style bindings, copy mode, mouse toggle |
| [zsh](skills/devenv/zsh/) | ZSH + Oh My Zsh: robbyrussell theme, conda init, PATH config |
| [github-cli](skills/devenv/github-cli/) | GitHub CLI (`gh`): `gh api`, PRs, issues, releases, replacing `WebFetch` for GitHub URLs |
| [claude-code-config](skills/devenv/claude-code-config/) | Claude Code setup: permissions, statusline script, plugins (superpowers, code-simplifier, ralph-loop), `settings.json` |
| [login-cdp](skills/devenv/login-cdp/) | Re-authenticate CDP browser sessions: auto-detect expired platforms, guide interactive re-login via MCP browser tools |
| [vercel-cost-optimization](skills/devenv/vercel-cost-optimization/) | Vercel cost optimization: ISR-breaking patterns, function constraints, `Cache-Control` headers, Fluid Compute, build minutes |

### 📱 Application Development

| Skill | Description |
|-------|-------------|
| [ios-swiftui-app](skills/apps/ios-swiftui-app/) | iOS SwiftUI app development: UIKit bridging, SSH terminal (Citadel/SwiftTerm), voice input STT pipeline, `@MainActor` services, xcodegen + SPM, Keychain storage |

---

## How It Works

Each skill is a self-contained directory with a single `SKILL.md` at its root, grouped into `skills/<category>/<name>/`:

```
skills/
└── <category>/         # training, experiments, hpc, research, genai, infra, devenv, apps
    └── <skill-name>/
        ├── SKILL.md    # YAML frontmatter + when-to-use + patterns + anti-patterns
        └── references/ # (optional) deep-dive docs loaded on demand
```

The YAML frontmatter follows the [Open Agent Skills](https://agentskills.io) standard:

```yaml
---
name: gpu-training-acceleration
description: "Use when optimizing PyTorch training speed or memory on CUDA GPUs. Triggers: \"torch.compile\", \"mixed precision\", \"gradient checkpointing\", \"FSDP\"..."
---
```

When a user's prompt hits a trigger keyword, the agent loads the corresponding `SKILL.md` and follows its patterns. Large skills link to `references/` for deep dives, so context stays lean.

---

## Compatible Agents

| Agent | Skills Path | Instruction File |
|-------|-------------|-----------------|
| [Claude Code](https://claude.ai/code) | `.claude/skills/` | `CLAUDE.md` |
| [Codex](https://developers.openai.com/codex) | `.agents/skills/` | `AGENTS.md` |
| [Cursor](https://cursor.com) | `.cursor/skills/` | `.cursorrules` |
| [Gemini CLI](https://geminicli.com) | `.gemini/skills/` | `GEMINI.md` |
| [VS Code / Copilot](https://code.visualstudio.com) | `.github/skills/` | `.github/copilot-instructions.md` |
| [TRAE](https://trae.ai) | `.trae/skills/` | `TRAE.md` |
| [Roo Code](https://roocode.com) | `.roo/skills/` | `.roo/rules` |

30+ more listed at [agentskills.io](https://agentskills.io).

---

## Contributing

PRs welcome. Add a new skill in three steps:

1. Create `skills/<category>/<skill-name>/SKILL.md` with the required YAML frontmatter (`name`, `description` ending in a `Triggers:` list).
2. Add `## When to Use`, patterns, `## Anti-Patterns`, and `## See Also` sections.
3. Update the badge count, install block, and catalog in this `README.md`.

See [`AGENTS.md`](AGENTS.md) for the full contributor guide — frontmatter conventions, verification checklist, and commit format. Open an issue if you're missing a skill you'd love to see.

---

<div align="center">
<sub>MIT licensed · Built by <a href="https://github.com/dongzhuoyao">@dongzhuoyao</a> · Inspired by <a href="https://agentskills.io">Open Agent Skills</a></sub>
</div>
