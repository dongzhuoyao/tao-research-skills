#!/usr/bin/env python3
"""Install tao-research-skills into global AI agent configurations.

Creates symlinks from each platform's global skills directory to this repo,
sets up auto-update hooks, and bootstraps memory directories.

Supported platforms: Claude Code, Codex (extensible via PLATFORMS dict).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform registry
# ---------------------------------------------------------------------------

PLATFORMS: dict[str, Path] = {
    "claude-code": Path.home() / ".claude" / "skills",
    "codex": Path.home() / ".agents" / "skills",
}

SKILL_REPO_NAME = "tao-research-skills"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the root of this skills repo (two levels up from this script)."""
    return Path(__file__).resolve().parents[2]


def _sanitize_cwd_for_claude(repo: Path) -> str:
    """Sanitize a path the way Claude Code does for auto-memory directories."""
    return str(repo).replace("/", "-").lstrip("-")


# ---------------------------------------------------------------------------
# Symlink creation
# ---------------------------------------------------------------------------

def _find_skill_dirs(repo: Path) -> list[Path]:
    """Find all skill directories (those containing SKILL.md) in the repo."""
    return sorted(d.parent for d in repo.glob("*/SKILL.md"))


def install_symlinks(
    repo: Path,
    platforms: list[str],
    dry_run: bool = False,
) -> dict[str, dict[str, str]]:
    """Create per-skill symlinks from each platform's skills dir.

    Claude Code and Codex discover skills at ~/.claude/skills/<name>/SKILL.md
    (one level deep), so each skill needs its own symlink — NOT a single
    symlink to the whole repo.

    Returns a dict of platform -> {skill_name: action}.
    """
    skill_dirs = _find_skill_dirs(repo)
    results: dict[str, dict[str, str]] = {}

    for platform in platforms:
        skills_dir = PLATFORMS[platform]
        skills_dir.mkdir(parents=True, exist_ok=True)
        platform_results: dict[str, str] = {}

        # Remove stale repo-level symlink if it exists
        old_repo_link = skills_dir / SKILL_REPO_NAME
        if old_repo_link.is_symlink():
            if not dry_run:
                old_repo_link.unlink()
            platform_results["_old_repo_symlink"] = "removed"

        for skill_path in skill_dirs:
            name = skill_path.name
            target = skills_dir / name

            if target.is_symlink():
                if target.resolve() == skill_path.resolve():
                    platform_results[name] = "already_linked"
                    continue
                # Points elsewhere — skip
                platform_results[name] = "skipped:different_target"
                continue

            if target.is_dir():
                try:
                    target.rmdir()
                except OSError:
                    platform_results[name] = "skipped:directory_not_empty"
                    continue

            if dry_run:
                platform_results[name] = "would_create"
                continue

            target.symlink_to(skill_path)
            platform_results[name] = "created"

        results[platform] = platform_results

    return results


# ---------------------------------------------------------------------------
# Auto-update: Claude Code SessionStart hook
# ---------------------------------------------------------------------------

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
HOOK_MARKER = "tao-research-skills"


def _build_hook_command(repo: Path) -> str:
    return f"git -C {repo} pull --quiet origin main 2>/dev/null || true"


def inject_claude_hook(repo: Path, dry_run: bool = False) -> str:
    """Add a SessionStart git-pull hook to ~/.claude/settings.json.

    Returns action taken: created | already_exists | would_create.
    """
    if not CLAUDE_SETTINGS.exists():
        settings: dict = {}
    else:
        settings = json.loads(CLAUDE_SETTINGS.read_text(encoding="utf-8"))

    hooks = settings.setdefault("hooks", {})
    session_start = hooks.setdefault("SessionStart", [])

    # Check for existing hook
    for entry in session_start:
        for hook in entry.get("hooks", []):
            cmd = hook.get("command", "")
            if HOOK_MARKER in cmd:
                return "already_exists"

    if dry_run:
        return "would_create"

    session_start.append({
        "hooks": [
            {
                "command": _build_hook_command(repo),
                "type": "command",
                "async": True,
                "statusMessage": "Updating tao-research-skills...",
            }
        ]
    })

    CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_SETTINGS.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return "created"


# ---------------------------------------------------------------------------
# Auto-update: Codex AGENTS.md instruction
# ---------------------------------------------------------------------------

CODEX_AGENTS_MD = Path.home() / ".agents" / "AGENTS.md"
CODEX_UPDATE_SECTION = "## Auto-Update Skills"


def inject_codex_update(repo: Path, dry_run: bool = False) -> str:
    """Append an update instruction to ~/.agents/AGENTS.md.

    Returns action taken: created | already_exists | would_create.
    """
    if CODEX_AGENTS_MD.exists():
        content = CODEX_AGENTS_MD.read_text(encoding="utf-8")
        if HOOK_MARKER in content:
            return "already_exists"
    else:
        content = ""

    if dry_run:
        return "would_create"

    section = (
        f"\n{CODEX_UPDATE_SECTION}\n\n"
        f"Before starting work, update shared skills:\n\n"
        f"```bash\n"
        f"git -C {repo} pull --quiet origin main\n"
        f"```\n"
    )

    CODEX_AGENTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(CODEX_AGENTS_MD, "a", encoding="utf-8") as f:
        f.write(section)
    return "created"


# ---------------------------------------------------------------------------
# Memory bootstrap (Claude Code only)
# ---------------------------------------------------------------------------

MEMORY_INDEX_TEMPLATE = """# {repo_name} Memory

## Project
<!-- Add project memory file links here -->

## User
<!-- Add user memory file links here -->

## Feedback
<!-- Add feedback memory file links here -->
"""

MEMORY_README_TEMPLATE = """# {repo_name} -- Project Memory

This directory stores persistent memory for the `{repo_name}` project,
used by Claude Code across conversations.

## Memory structure

- `MEMORY.md` -- Index of all memory files (loaded into context automatically)
- `user_*.md` -- User profile and preferences
- `feedback_*.md` -- Guidance on approach and conventions
- `project_*.md` -- Ongoing project state and known issues
- `reference_*.md` -- Pointers to external resources
"""


def bootstrap_memory(repo: Path, dry_run: bool = False) -> dict[str, str]:
    """Create Claude Code auto-memory directory with templates.

    Returns dict of file -> action taken.
    """
    sanitized = _sanitize_cwd_for_claude(repo)
    memory_dir = Path.home() / ".claude" / "projects" / sanitized / "memory"
    repo_name = repo.name

    results: dict[str, str] = {}
    files = {
        "MEMORY.md": MEMORY_INDEX_TEMPLATE.format(repo_name=repo_name).lstrip(),
        "README.md": MEMORY_README_TEMPLATE.format(repo_name=repo_name).lstrip(),
    }

    for filename, content in files.items():
        target = memory_dir / filename
        if target.exists():
            results[filename] = "already_exists"
            continue
        if dry_run:
            results[filename] = "would_create"
            continue
        memory_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        results[filename] = "created"

    return results


# ---------------------------------------------------------------------------
# Workflow rules: inject pre-commit checks into global instruction files
# ---------------------------------------------------------------------------

WORKFLOW_RULES_MARKER = "## tao-research-skills Workflow Rules"

WORKFLOW_RULES = """## tao-research-skills Workflow Rules

**Before commit/push in tao-research-skills repo**, always verify:

1. **README freshness** — Run `ls */SKILL.md | wc -l` and compare with the badge count in `README.md`. Check that every skill directory appears in all three README sections (badge count, one-prompt install block, Available Skills table).
2. **Project memory freshness** — Check whether any memory files (skill inventory, known issues, etc.) need updating to reflect changes made in this session. For Claude Code, update files in the auto-memory directory. For Codex, update project-level `AGENTS.md` if applicable.
3. **Never truncate SKILL.md** — When reading or loading skills, always read the full file. If a skill is too large for the context window, use an LLM to summarize it instead of truncating. Never use `head`, `[:N]`, or `limit` on SKILL.md files.
4. **No silent fallbacks** — Never use fallback values or default behaviors to mask errors. If something fails, raise it explicitly. Silent fallbacks hide bugs and make debugging impossible.
"""


def inject_workflow_rules_claude(dry_run: bool = False) -> str:
    """Append workflow rules to ~/.claude/CLAUDE.md.

    Returns action taken: created | already_exists | would_create.
    """
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8")
        if WORKFLOW_RULES_MARKER in content:
            return "already_exists"
    else:
        content = ""

    if dry_run:
        return "would_create"

    claude_md.parent.mkdir(parents=True, exist_ok=True)
    with open(claude_md, "a", encoding="utf-8") as f:
        f.write("\n" + WORKFLOW_RULES)
    return "created"


def inject_workflow_rules_codex(dry_run: bool = False) -> str:
    """Append workflow rules to ~/.agents/AGENTS.md.

    Returns action taken: created | already_exists | would_create.
    """
    if CODEX_AGENTS_MD.exists():
        content = CODEX_AGENTS_MD.read_text(encoding="utf-8")
        if WORKFLOW_RULES_MARKER in content:
            return "already_exists"
    else:
        content = ""

    if dry_run:
        return "would_create"

    CODEX_AGENTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(CODEX_AGENTS_MD, "a", encoding="utf-8") as f:
        f.write("\n" + WORKFLOW_RULES)
    return "created"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install tao-research-skills into global AI agent configs.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Path to tao-research-skills repo (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--platforms",
        default=",".join(PLATFORMS.keys()),
        help=f"Comma-separated platforms (default: {','.join(PLATFORMS.keys())})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them",
    )
    parser.add_argument(
        "--skip-hooks",
        action="store_true",
        help="Skip auto-update hook injection",
    )
    parser.add_argument(
        "--skip-memory",
        action="store_true",
        help="Skip memory directory bootstrap",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    repo = Path(args.repo).resolve() if args.repo else _repo_root()
    platforms = [p.strip() for p in args.platforms.split(",")]

    for p in platforms:
        if p not in PLATFORMS:
            print(f"Unknown platform: {p}. Available: {', '.join(PLATFORMS.keys())}")
            return 1

    report: dict = {"repo": str(repo), "platforms": {}}

    # 1. Symlinks
    symlink_results = install_symlinks(repo, platforms, dry_run=args.dry_run)

    # 2. Hooks
    hook_results: dict[str, str] = {}
    if not args.skip_hooks:
        if "claude-code" in platforms:
            hook_results["claude-code"] = inject_claude_hook(repo, dry_run=args.dry_run)
        if "codex" in platforms:
            hook_results["codex"] = inject_codex_update(repo, dry_run=args.dry_run)

    # 3. Memory
    memory_results: dict[str, str] = {}
    if not args.skip_memory and "claude-code" in platforms:
        memory_results = bootstrap_memory(repo, dry_run=args.dry_run)

    # 4. Workflow rules
    rules_results: dict[str, str] = {}
    if not args.skip_hooks:
        if "claude-code" in platforms:
            rules_results["claude-code"] = inject_workflow_rules_claude(dry_run=args.dry_run)
        if "codex" in platforms:
            rules_results["codex"] = inject_workflow_rules_codex(dry_run=args.dry_run)

    # Build report
    for p in platforms:
        p_symlinks = symlink_results.get(p, {})
        # Summarize symlinks: count by action
        actions = {}
        for action in p_symlinks.values():
            actions[action] = actions.get(action, 0) + 1
        entry: dict = {"symlinks": actions, "symlink_count": len(p_symlinks)}
        if p in hook_results:
            key = "hook" if p == "claude-code" else "agents_md"
            entry[key] = hook_results[p]
        if memory_results and p == "claude-code":
            entry["memory"] = memory_results
        if p in rules_results:
            entry["workflow_rules"] = rules_results[p]
        report["platforms"][p] = entry

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
