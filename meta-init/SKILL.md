---
name: meta-init
description: "Use when installing tao-research-skills into a machine's global AI agent configuration. Creates symlinks, sets up auto-update hooks, and bootstraps memory directories across Claude Code and Codex. Triggers: \"install skills\", \"setup skills globally\", \"meta-init\", \"cross-platform skills install\""
---

# Meta Init

Install this skills repo into the global configuration of multiple AI agent platforms so that all skills are available in every project on the machine.

## When to Use

- Setting up tao-research-skills on a new machine
- Adding support for a new AI agent platform
- Re-running after a platform was added or reinstalled
- Verifying the installation is intact (`--dry-run`)

## What It Does

| Step | Claude Code | Codex |
|------|------------|-------|
| Symlinks | Per-skill: `~/.claude/skills/<name>` each | Per-skill: `~/.agents/skills/<name>` each |
| Auto-update | `SessionStart` hook in `settings.json` | Instruction in `~/.agents/AGENTS.md` |
| Memory bootstrap | Auto-memory dir with `MEMORY.md` + `README.md` | N/A |
| Workflow rules | Appended to `~/.claude/CLAUDE.md` | Appended to `~/.agents/AGENTS.md` |

All operations are **idempotent** — safe to run multiple times.

## Script Usage

```bash
# Full install (all platforms)
python meta-init/scripts/meta_init.py

# Preview changes without applying
python meta-init/scripts/meta_init.py --dry-run

# Single platform
python meta-init/scripts/meta_init.py --platforms claude-code

# Skip specific steps
python meta-init/scripts/meta_init.py --skip-hooks --skip-memory
```

Output is JSON showing what was done per platform.

## Workflow Rules Injected

The script appends pre-commit/push checks to the global instruction file of each platform:

1. **README freshness** — verify badge count matches actual skill count, all skills in three README sections
2. **Project memory freshness** — check if memory files (skill inventory, known issues) need updating

These rules are scoped to the tao-research-skills repo and apply to all AI models on the machine.

## Post-Install Agent Guidance

After the script runs, the agent should:

1. **Scan the repo** — read `README.md` and a sample of `SKILL.md` files to understand the skills collection
2. **Populate memory** — create memory files in the auto-memory directory with project overview, skill inventory, and known issues
3. **Audit CLAUDE.md** — check if the project's `CLAUDE.md` is up to date with current repo state

## Adding a New Platform

Add an entry to the `PLATFORMS` dict in `meta_init.py`:

```python
PLATFORMS = {
    "claude-code": Path.home() / ".claude" / "skills",
    "codex": Path.home() / ".agents" / "skills",
    "cursor": Path.home() / ".cursor" / "skills",  # new
}
```

Then add platform-specific hook injection if the platform supports it.

## Anti-Patterns

- Running with `--skip-hooks` and forgetting to update manually — skills will go stale
- Editing the symlink target instead of `git pull` in the repo — symlinks point to the repo, not a copy
- Running on a repo that is not tao-research-skills — the script auto-detects its own repo root

## See Also

- `mem-init` — Bootstrap project memory files in a target repo (different purpose: target repo memory, not global install)
- `memory-sync` — Keep AGENTS.md and CLAUDE.md in sync from one canonical source
- `claude-code-config` — Claude Code permissions, statusline, plugins setup
