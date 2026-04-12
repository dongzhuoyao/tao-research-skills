# meta-init Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a cross-platform skill that installs tao-research-skills into a machine's global AI agent configuration (Claude Code + Codex), with auto-update.

**Architecture:** A Python script handles mechanical operations (symlink creation, settings.json hook injection, platform detection). A SKILL.md guides agents on post-install judgment calls (memory initialization, quality audit). The script is idempotent and merge-safe.

**Tech Stack:** Pure Python 3.9+ (pathlib, json, argparse), no third-party dependencies.

---

### Task 1: Create `meta_init.py` — Platform Detection & Symlink

**Files:**
- Create: `meta-init/scripts/meta_init.py`

**Step 1: Write the script with platform registry and symlink logic**

The script defines a `PLATFORMS` dict mapping platform names to their global skills directory. It:
- Auto-detects the skills repo root from its own `__file__` path
- For each platform, creates `~/<platform-skills-dir>/tao-research-skills` → repo root
- Skips if symlink already exists and points to the right place
- Reports what it did as JSON to stdout

```python
PLATFORMS = {
    "claude-code": Path.home() / ".claude" / "skills",
    "codex": Path.home() / ".agents" / "skills",
}
```

Key behaviors:
- If target is an empty directory, remove it and replace with symlink
- If target is a symlink pointing elsewhere, warn and skip (don't overwrite)
- If target is a symlink pointing to the right place, skip silently
- `--platforms` flag to select specific platforms (default: all detected)
- `--dry-run` flag to preview changes

**Step 2: Run script to verify symlinks**

```bash
python meta-init/scripts/meta_init.py --dry-run
```

Expected: JSON output listing planned symlinks without creating them.

**Step 3: Commit**

```bash
git add meta-init/scripts/meta_init.py
git commit -m "feat(meta-init): add cross-platform symlink installer"
```

---

### Task 2: Add Auto-Update Hook Injection

**Files:**
- Modify: `meta-init/scripts/meta_init.py`

**Step 1: Add Claude Code hook injection**

Add a function that:
- Reads `~/.claude/settings.json` (or creates minimal `{}` if missing)
- Checks if a SessionStart hook with `tao-research-skills` already exists
- If not, appends a new entry to the `SessionStart` hooks array:
  ```json
  {
    "hooks": [{
      "command": "git -C <repo-path> pull --quiet origin main 2>/dev/null || true",
      "type": "command",
      "async": true,
      "statusMessage": "Updating tao-research-skills..."
    }]
  }
  ```
- Merges carefully: preserves all existing hooks and settings
- Writes back with `json.dump(indent=2)`

**Step 2: Add Codex auto-update guidance**

Codex has no hook mechanism. Instead:
- Check if `~/.agents/AGENTS.md` exists
- If it exists, check if it already mentions tao-research-skills update
- If not, append a section:
  ```markdown
  ## Auto-Update Skills
  Before starting work, run: `git -C <repo-path> pull --quiet origin main`
  ```
- If file doesn't exist, create it with just this section

**Step 3: Verify hook injection is idempotent**

```bash
python meta-init/scripts/meta_init.py && python meta-init/scripts/meta_init.py
```

Expected: Second run reports "already configured" for both platforms, no duplicate hooks.

**Step 4: Commit**

```bash
git add meta-init/scripts/meta_init.py
git commit -m "feat(meta-init): add auto-update hook for Claude Code and Codex"
```

---

### Task 3: Add Memory Initialization

**Files:**
- Modify: `meta-init/scripts/meta_init.py`

**Step 1: Add Claude Code auto-memory bootstrap**

Add a function that:
- Computes the Claude auto-memory path: `~/.claude/projects/-<sanitized-cwd>/memory/`
  - Sanitized: replace `/` with `-`, strip leading `-`
- Creates the directory if it doesn't exist
- Creates `MEMORY.md` with a minimal index template if it doesn't exist
- Creates `README.md` with a project introduction template if it doesn't exist
- Skips if files already exist (never overwrite)

**Step 2: Verify memory init**

```bash
python meta-init/scripts/meta_init.py --dry-run
```

Expected: Shows memory directory path and files to create.

**Step 3: Commit**

```bash
git add meta-init/scripts/meta_init.py
git commit -m "feat(meta-init): add memory directory bootstrap"
```

---

### Task 4: Add CLI Interface & JSON Report

**Files:**
- Modify: `meta-init/scripts/meta_init.py`

**Step 1: Add argparse CLI and structured output**

```
python meta_init.py [--platforms claude-code,codex] [--dry-run] [--skip-hooks] [--skip-memory]
```

Final JSON report format:
```json
{
  "repo": "/path/to/tao-research-skills",
  "platforms": {
    "claude-code": {"symlink": "created", "hook": "created", "memory": "created"},
    "codex": {"symlink": "created", "agents_md": "updated"}
  }
}
```

**Step 2: Full end-to-end test**

```bash
python meta-init/scripts/meta_init.py
```

Verify:
- `ls -la ~/.claude/skills/tao-research-skills` → symlink
- `ls -la ~/.agents/skills/tao-research-skills` → symlink
- `jq '.hooks.SessionStart' ~/.claude/settings.json` → contains git pull hook
- No duplicates on second run

**Step 3: Commit**

```bash
git add meta-init/scripts/meta_init.py
git commit -m "feat(meta-init): add CLI interface and JSON report"
```

---

### Task 5: Write SKILL.md

**Files:**
- Create: `meta-init/SKILL.md`

**Step 1: Write the skill document**

Follow repo conventions:
- YAML frontmatter: `name: meta-init`, `description: Use when ...`
- `## When to Use` — first section
- Script usage section
- Post-install agent guidance (memory content generation, CLAUDE.md audit)
- Platform-specific notes (what each platform gets)
- `## Anti-Patterns`
- `## See Also` — backtick style, link to `mem-init`, `memory-sync`, `claude-code-config`

**Step 2: Commit**

```bash
git add meta-init/SKILL.md
git commit -m "feat: add meta-init skill for cross-platform installation"
```

---

### Task 6: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Update three places**

1. Badge: `skills-22-blue` (21 → 22, adding meta-init)
2. Install block: add `- \`meta-init\`: ...` line
3. Available Skills table: add to "Project Memory" category alongside mem-init and memory-sync

Also fix: `academic-deep-research` is still missing — add it to all three places too. Badge becomes `skills-23-blue` if adding both.

Wait — check current state first. academic-deep-research may have been added in the latest pull.

**Step 2: Verify README is correct**

```bash
grep -c 'academic-deep-research' README.md  # should be > 0 in table
grep 'skills-' README.md                     # badge count matches
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add meta-init to README, fix skill count"
```

---

### Task 7: Update CLAUDE.md Known Issues

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Remove fixed issues, update count**

After README is updated, remove the `academic-deep-research` and badge count issues from Known Issues.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md known issues after meta-init"
```
