---
name: mem-init
description: Initialize shared project memory for Codex and Claude Code in a repository. Use when a repo does not yet have canonical memory, AGENTS.md, or CLAUDE.md, and you want to bootstrap the standard memory layout quickly.
---

# Mem Init

Bootstrap shared memory for a repo with one canonical Markdown source and generated host files.

Default layout:

- `memory/project.md` as the canonical shared memory
- `AGENTS.md` generated from `memory/project.md`
- `CLAUDE.md` generated as a thin `@AGENTS.md` wrapper
- `memory/claude.md` for optional Claude-only notes
- `README.md` as a starter project introduction

## Use This Skill When

- A repo does not yet have project memory
- The user wants a quick memory bootstrap, not a full sync/debug workflow
- You want the standard Codex + Claude Code memory layout in one step

## Command

From the skill directory:

```bash
python scripts/mem_init.py --repo /path/to/repo
```

## Result

After initialization, the target repo should contain:

- `memory/project.md`
- `memory/claude.md`
- `AGENTS.md`
- `CLAUDE.md`
- `README.md`

## Notes

- This skill is intentionally narrow. For later updates, drift repair, or sync checks, use `memory-sync`.
