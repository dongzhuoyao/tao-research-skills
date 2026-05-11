---
name: project-mem-init
description: "Use when a repo does not yet have canonical memory, AGENTS.md, or CLAUDE.md. Bootstraps the standard Codex + Claude Code project memory layout in one step. Triggers: \"project-mem-init\", \"bootstrap memory\", \"initialize project memory\""
---

# Project Mem Init

Bootstrap shared project memory for a repo with one canonical Markdown source and generated host files.

Default layout:

- `memory/project.md` as the canonical shared memory
- `AGENTS.md` generated from `memory/project.md`
- `CLAUDE.md` generated as a thin `@AGENTS.md` wrapper
- `memory/claude.md` for optional Claude-only notes
- `README.md` as a starter project introduction

## When to Use

- A repo does not yet have project memory
- The user wants a quick memory bootstrap, not a full sync/debug workflow
- You want the standard Codex + Claude Code memory layout in one step

## Command

From the skill directory:

```bash
python scripts/project_mem_init.py --repo /path/to/repo
```

## Migration

If the target repo already has a `CLAUDE.md` or `AGENTS.md` with real content (not a generated wrapper), that content is automatically migrated into `memory/project.md` before initialization. This preserves existing project instructions while adopting the canonical memory layout.

## Result

After initialization, the target repo should contain:

- `memory/project.md` — canonical shared memory (migrated from existing files or template)
- `memory/claude.md` — Claude-specific overlay
- `AGENTS.md` — generated from `memory/project.md`
- `CLAUDE.md` — generated as `@AGENTS.md` wrapper
- `README.md` — starter template (skipped if already exists)

## Notes

- This skill is intentionally narrow. For later updates, drift repair, or sync checks, use `memory-sync`.

## Anti-Patterns

| Anti-Pattern | What goes wrong | Fix |
|--------------|-----------------|-----|
| Running `project-mem-init` on a repo that already has `memory/project.md` | Overwrites canonical memory and loses history | Use `memory-sync` instead — `project-mem-init` is bootstrap only |
| Hand-writing `CLAUDE.md` right after init | Drift on the next `memory-sync sync` | Edit `memory/project.md`; `CLAUDE.md` is generated |
| Skipping the auto-migration step on an existing repo | Pre-existing instructions in old `CLAUDE.md`/`AGENTS.md` get lost | Let `project-mem-init` migrate; review the resulting `memory/project.md` |
| Treating `project-mem-init` as the ongoing sync workflow | No drift detection, no idempotence guarantee | Use `memory-sync` for updates after bootstrap |
| Initializing without an existing git repo | Generated files land in an un-tracked state and may be lost | `git init` (or run inside an existing repo) before `project-mem-init` |

## See Also

- `memory-sync` — Ongoing sync, drift check, and repair (use after `project-mem-init`)
- `meta-init` — Install tao-research-skills globally (different "init" — installs skills, not memory)
- `agents-md-writing` — Patterns for writing effective `AGENTS.md`/`CLAUDE.md` content
