---
name: meta-init
description: "Use when installing tao-research-skills into a machine's global AI agent configuration. Symlinks every skill into Claude Code and Codex, injects a SessionStart git-pull hook for auto-updates, bootstraps memory, and appends workflow rules. Triggers: \"install skills\", \"install to computer level\", \"setup skills globally\", \"meta-init\", \"cross-platform skills install\", \"bootstrap skills\""
---

# Meta Init

Install this skills repo into the global configuration of supported AI agent platforms so every skill is available in every project on the machine.

## When to Use

- Setting up tao-research-skills on a new machine
- Adding support for a new AI agent platform
- Re-running after a platform was added or reinstalled
- Verifying the installation is intact (dry-run mode)

## Workflow (when invoked)

Follow these steps exactly, in order.

### 1. Confirm repo location

Run `pwd`. If cwd is not the tao-research-skills root, ask the user to confirm the repo path, or pass `--repo <path>` to the script.

### 2. Dry-run preview

```bash
python meta-init/scripts/meta_init.py --dry-run
```

Parse the JSON report. Summarize to the user, per platform:

- **symlinks** — how many `would_create` vs. `already_linked` vs. `skipped:*`
- **hook** — `would_create` or `already_exists` (Claude Code only)
- **memory** — which files would be created
- **workflow_rules** — `would_create` or `already_exists`

If any `skipped:different_target` appears, **stop and ask the user** — an existing symlink points elsewhere; overwriting would clobber it.

### 3. Apply

```bash
python meta-init/scripts/meta_init.py
```

The script is idempotent — safe to re-run.

### 4. Verify symlinks

```bash
ls -la ~/.claude/skills/ | grep tao-research-skills
ls -la ~/.agents/skills/ 2>/dev/null | head -5
```

Every skill directory from the repo should appear as a symlink pointing back into the repo (one symlink per skill, not a single repo-level link).

### 5. Verify the SessionStart hook

```bash
jq '.hooks.SessionStart' ~/.claude/settings.json
```

Expected output contains a command referencing `tao-research-skills`, with `"async": true`. This is the auto-update hook — `git pull` runs non-blocking on every session start.

### 6. Report

Print the JSON report from step 3, plus a one-line confirmation per step.

## What Gets Installed

| Step | Claude Code | Codex |
|------|-------------|-------|
| Symlink per skill | `~/.claude/skills/<skill>` → repo skill dir | `~/.agents/skills/<skill>` → repo skill dir |
| Auto-update | `SessionStart` hook in `~/.claude/settings.json` | Instruction in `~/.agents/AGENTS.md` |
| Memory bootstrap | `MEMORY.md` + `README.md` in auto-memory dir | — |
| Workflow rules | Appended to `~/.claude/CLAUDE.md` | Appended to `~/.agents/AGENTS.md` |

All operations are **idempotent** — safe to run multiple times.

## The SessionStart Hook

The script writes this into `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "command": "git -C <repo-path> pull --quiet origin main 2>/dev/null || true",
        "type": "command",
        "async": true,
        "statusMessage": "Updating tao-research-skills..."
      }]
    }]
  }
}
```

- `async: true` — doesn't block session startup
- `|| true` — network failures are silently swallowed
- Detected via `tao-research-skills` marker in the command; re-running the script won't duplicate
- If the user already added a different `tao-research-skills` pull hook manually, the script detects it via the marker and skips (`already_exists`)

## Script Flags

```bash
python meta-init/scripts/meta_init.py [options]

  --dry-run              Preview without applying
  --platforms LIST       Comma-separated subset (default: claude-code,codex)
  --skip-hooks           Skip hook + workflow-rule injection
  --skip-memory          Skip memory bootstrap
  --repo PATH            Override repo root (default: auto-detect)
```

## Adding a New Platform

Add to the `PLATFORMS` dict in `meta_init.py`:

```python
PLATFORMS = {
    "claude-code": Path.home() / ".claude" / "skills",
    "codex": Path.home() / ".agents" / "skills",
    "cursor": Path.home() / ".cursor" / "skills",  # new
}
```

Then add platform-specific hook injection if the platform supports it.

## Post-Install Guidance

After the script finishes:

1. **Restart Claude Code** (or open `/hooks`) — the SessionStart hook only takes effect after the settings watcher reloads
2. **Scan the repo** — read `README.md` and a sample of `SKILL.md` files to understand the skills collection
3. **Populate memory** — create files in the auto-memory directory with project overview, skill inventory, and known issues
4. **Audit CLAUDE.md** — check if the project's `CLAUDE.md` is up to date with current repo state

## Anti-Patterns

- Running `--skip-hooks` and forgetting manual updates — skills silently go stale
- Editing the symlink target instead of `git pull` in the repo — symlinks point to the repo, not a copy
- Running from outside the tao-research-skills repo without `--repo` — script auto-detects via script location, but ambiguity is safer avoided
- Skipping the dry-run — destructive target conflicts (`skipped:different_target`) are reported but not blocked
- Running on a repo that isn't tao-research-skills — the script assumes this repo's layout; pointing it elsewhere will create bogus symlinks

## See Also

- `project-mem-init` — Bootstrap project memory files in a target repo (different purpose: target repo memory, not global install)
- `memory-sync` — Keep AGENTS.md and CLAUDE.md in sync from one canonical source
- `claude-code-config` — Claude Code permissions, statusline, plugins setup
