---
name: agents-md-writing
description: Use when writing or improving CLAUDE.md, AGENTS.md, GEMINI.md, or any agent instruction file. Covers section structure, memory patterns, workflow rules, and anti-patterns. Triggers: "CLAUDE.md", "AGENTS.md", "agent instructions", "project memory", "MEMORY.md", "instruction file"
---

# Writing Effective Agent Instruction Files

Battle-tested patterns for writing CLAUDE.md, AGENTS.md, and project memory files that make AI coding agents dramatically more effective.

## When to Use

- Creating a CLAUDE.md / AGENTS.md for a new project
- Improving an existing instruction file that feels incomplete
- Setting up persistent memory (MEMORY.md) for cross-session context
- Reviewing whether your instruction file covers the right things

## Philosophy

Agent instruction files are **the highest-leverage artifact in AI-assisted development**. A well-written CLAUDE.md saves more time than any other optimization because:

1. It prevents the agent from asking questions it should already know the answer to
2. It encodes hard-won lessons so they're never forgotten across sessions
3. It sets behavioral expectations (coding style, workflow, safety rules) once

The goal: **an agent that behaves like a team member who read all the docs on day one**.

## CLAUDE.md / AGENTS.md Structure

A complete instruction file has these sections, roughly in this order. Not all projects need every section — start with Commands + Architecture + Gotchas and grow from there.

### 1. Project Summary (1-2 lines)

What the project is. Helps the agent understand context for ambiguous instructions.

```markdown
# MyProject

Real-time anomaly detection pipeline for IoT sensor data.
```

### 2. Commands

Copy-pasteable commands for every common operation. The agent will use these verbatim.

```markdown
## Commands

\`\`\`bash
# Install
pip install -e ".[dev]"

# Train
python train.py --config configs/default.yaml

# Test
pytest                    # all tests
pytest tests/test_model.py  # specific module

# Lint
ruff check . --fix

# Deploy
docker compose up -d
\`\`\`
```

**Why this matters**: Without explicit commands, the agent guesses (`python -m pytest`? `make test`? `npm test`?). Wrong guesses waste time and break trust.

### 3. Architecture

A tree or diagram of the key files. Annotate each with a brief purpose.

```markdown
## Architecture

\`\`\`
src/
  pipeline/
    ingest.py       # Kafka consumer, raw → cleaned events
    features.py     # Feature extraction (rolling stats, FFT)
    model.py        # LightGBM anomaly scorer
    alert.py        # Threshold logic + PagerDuty integration
configs/
  default.yaml      # Production config
  dev.yaml          # Local dev (smaller windows, no alerting)
tests/
scripts/
  backfill.py       # One-off: reprocess historical data
\`\`\`
```

**Why this matters**: The agent doesn't know your project layout. Without this, it `grep`s and `find`s for minutes before starting real work.

### 4. Config System

How configuration works — what tool, key overrides, precedence rules.

```markdown
## Config

- Tool: Hydra (hierarchical YAML in `configs/`)
- Override via CLI: `python train.py model.lr=3e-4`
- Key settings: `model.hidden_dim` (256), `data.window_size` (1024)
- Secrets in `.env` (never commit): API_KEY, DB_URL
```

### 5. Environment

How to set up the dev environment. Include runtime, package manager, and platform-specific notes.

```markdown
## Environment

- Python 3.11, managed via `uv` (or conda/venv)
- GPU: NVIDIA A100, CUDA 12.1
- Required env vars: `WANDB_API_KEY` (in `.env`)
```

### 6. Gotchas

**The most valuable section.** Encode every hard-won lesson, non-obvious behavior, and "don't do this" rule. Organized by category.

```markdown
## Gotchas

### Data
- Timestamps are UTC, never local time. Conversion happens only at display layer.
- The `events` table has duplicate rows — always `DISTINCT` on `event_id`.

### Training
- Batch size > 32 causes OOM on A100 with this model. Use gradient accumulation instead.
- Learning rate warmup is required — without it, loss explodes in first 100 steps.

### Engineering
- **Config is king**: Never hardcode runtime constants. Read from config.
- **Fail fast**: Critical paths must raise explicit errors, never silently fall back.
- **Offline-first**: HF/transformers stay offline in production. Datasets must be pre-cached.
```

**Anti-pattern**: Gotchas that are too vague ("be careful with data") or too specific to a single session ("I fixed a bug in line 42 yesterday").

### 7. Style

Coding conventions the agent should follow. Keep it short — only rules that aren't obvious from the existing code.

```markdown
## Style

- Type hints on all public functions
- Docstrings: Google style, only on non-obvious functions
- Imports: stdlib → third-party → local, separated by blank lines
- No print() in library code — use `logging.getLogger(__name__)`
```

### 8. Skills / Tools

Register any skills or tools the agent should know about.

```markdown
## Skills

- `gpu-training-acceleration`: See `skills/shared/gpu-training-acceleration/SKILL.md`
- `wandb-experiment-tracking`: See `skills/shared/wandb-experiment-tracking/SKILL.md`
```

### 9. Datasets / Resources (if applicable)

Tables work well for structured references.

```markdown
## Datasets

| Dataset | Location | Use |
|---------|----------|-----|
| TrainSet | `s3://bucket/train/` | Training (10k samples) |
| EvalSet | `data/eval/` | Local evaluation (500 samples) |
```

## Memory File (MEMORY.md) Structure

Memory files persist across conversations. They complement the CLAUDE.md (which is checked into git) with session-learned knowledge.

### Recommended Sections

```markdown
# Project Memory

## User Preferences
- Workflow preferences, tool choices, communication style
- Auth/deployment details (SSH keys, accounts, deploy targets)

## Workflow Orchestration
- How the agent should approach tasks (plan first? TDD? subagents?)
- When to ask vs. when to act autonomously

## Core Principles
- Engineering values that apply to all code changes
- E.g., "simplicity first", "TDD", "no silent fallbacks"

## Resolved Bugs
- Bugs that were hard to diagnose — symptoms, root cause, fix
- Prevents the agent from reintroducing or re-investigating

## Shorthand Commands
- Abbreviations the user types that map to complex workflows
- E.g., `"deploy staging"` → specific sequence of commands

## Key Architecture Notes
- Domain-specific knowledge the agent needs across sessions
- Token formats, protocol details, data schemas
```

### What Belongs in Memory vs. CLAUDE.md

| Put in CLAUDE.md | Put in MEMORY.md |
|------------------|------------------|
| Stable project structure | User-specific preferences |
| Commands and config | Resolved bugs (evolving) |
| Permanent gotchas | Session-learned patterns |
| Coding style rules | Shorthand commands |
| Checked into git | Machine-local, not in git |

## Workflow Rules (for Memory)

These are proven patterns for steering agent behavior. Copy what fits your workflow.

### Plan Before Build

```markdown
## Workflow Orchestration

### Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan — don't keep pushing
- Write detailed specs upfront to reduce ambiguity
```

### Verification Before Done

```markdown
### Verification Before Done
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Ask: "Would a senior engineer approve this?"
```

### TDD for AI-Assisted Development

```markdown
### TDD for Vibe Coding
- Red: Write a failing test that defines the desired result
- Green: Write minimal code to pass the test
- Refactor: Clean up while tests guarantee nothing breaks
- Tests give the AI clear verification criteria — without them, it guesses
```

### Autonomous Bug Fixing

```markdown
### Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding.
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
```

### Self-Improvement Loop

```markdown
### Self-Improvement Loop
- After ANY correction from the user: update memory with the pattern
- Write rules that prevent the same mistake
- Iterate on lessons until the mistake rate drops
```

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Write a 500-line CLAUDE.md on day one | Start small (Commands + Architecture + Gotchas), grow organically |
| Duplicate info already in code comments | Reference the file: "See `src/model.py` header for architecture" |
| Include session-specific context in CLAUDE.md | Put ephemeral knowledge in MEMORY.md |
| Write vague gotchas ("be careful") | Be specific: symptom, cause, fix |
| List every file in Architecture | Only list the key entry points and non-obvious structure |
| Hardcode values that belong in config | Reference config keys: "`model.lr` (default 1e-4)" |
| Write rules the agent already follows | Only encode surprising or non-obvious behaviors |

## Checklist: Is Your Instruction File Good Enough?

- [ ] Can the agent install and run the project without asking questions?
- [ ] Does Architecture cover the 5-10 most important files?
- [ ] Are the top 3 gotchas documented (things that burned you before)?
- [ ] Is there a testing command? (agents need fast feedback loops)
- [ ] Are secrets/env vars documented (without exposing values)?
- [ ] Does the style section prevent the agent's most common formatting mistakes?

## See Also

- `claude-code-config` — Settings.json, permissions, statusline, plugins
- `fail-fast-ml-engineering` — Engineering discipline rules for CLAUDE.md Gotchas section
- `hydra-experiment-config` — Config patterns to document in the Config section
