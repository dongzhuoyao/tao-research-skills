---
name: memory-sync
description: Set up and maintain shared project memory across Codex and Claude Code using one canonical Markdown source. Use when a repo should keep a single memory file in sync with generated AGENTS.md and a thin CLAUDE.md wrapper, or when memory drift needs to be checked or repaired.
---

# Memory Sync

Use a single canonical Markdown memory file and generate host-facing files from it.

Default pattern:

- Canonical source: `memory/project.md`
- Codex file: `AGENTS.md` contains the shared memory content
- Claude Code file: `CLAUDE.md` is a thin wrapper that imports `@AGENTS.md`
- Optional Claude-only overlay: `memory/claude.md`

This avoids maintaining two different memory bodies by hand.

## Use This Skill When

- The user wants one shared project memory for Codex and Claude Code
- `AGENTS.md` and `CLAUDE.md` have drifted
- A repo currently uses manual dual-write memory and should move to one source
- The user wants a reusable memory-sync workflow or helper script

## Workflow

1. Pick or confirm the target repo.
2. Initialize the memory layout if it does not exist.
3. Sync the generated files.
4. Run the drift check.
5. Report the canonical file and the generated outputs.

## Commands

From the skill directory:

```bash
python scripts/memory_sync.py init --repo /path/to/repo
python scripts/memory_sync.py sync --repo /path/to/repo
python scripts/memory_sync.py check --repo /path/to/repo
```

## Output Pattern

`AGENTS.md`:

- generated from `memory/project.md`
- contains the shared memory directly

`CLAUDE.md`:

- generated wrapper
- imports `@AGENTS.md`
- appends `memory/claude.md` if present

## Rules

- Keep the canonical memory in Markdown, not YAML.
- Do not hand-edit generated `AGENTS.md` or `CLAUDE.md`; edit `memory/project.md` instead.
- Keep `CLAUDE.md` thin unless there is a real Claude-specific need.
- If the repo already has important `CLAUDE.md` content, move only the shared part into `memory/project.md` and preserve the Claude-only remainder as `memory/claude.md`.

## Validation

After any change, run:

```bash
python scripts/memory_sync.py check --repo /path/to/repo
```

Success means the generated files match the canonical source.
