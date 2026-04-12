#!/usr/bin/env python3
"""Initialize shared memory files for a repo using the memory-sync layout."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


README_TEMPLATE = """# Project Name

Short introduction to the project.

## Purpose

Explain what this project is for and what problem it solves.

## Main Components

- Describe the main modules or directories.

## Workflow

- Explain the normal way to work in this repo.

## Memory

- Shared project memory lives in `memory/project.md`.
- `AGENTS.md` is generated for Codex.
- `CLAUDE.md` is generated for Claude Code.
"""


def _load_memory_sync_module():
    skill_root = Path(__file__).resolve().parents[2]
    target = skill_root / "memory-sync" / "scripts" / "memory_sync.py"
    spec = importlib.util.spec_from_file_location("memory_sync_skill", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize shared memory for Codex and Claude Code.")
    parser.add_argument("--repo", default=".", help="Target repository path")
    parser.add_argument("--force", action="store_true", help="Overwrite starter templates if present")
    return parser


def init_repo(repo: Path, force: bool = False) -> None:
    repo = repo.resolve()
    module = _load_memory_sync_module()
    module.init_repo(repo, force=force)
    readme = repo / "README.md"
    if force or not readme.exists():
        readme.write_text(README_TEMPLATE.strip() + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo).resolve()
    init_repo(repo, force=args.force)
    print(f"Initialized shared memory in {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
