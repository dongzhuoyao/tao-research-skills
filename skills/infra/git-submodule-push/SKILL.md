---
name: git-submodule-push
description: Use when pushing changes in a repository that contains git submodules, especially when the submodule has its own commits that must not be left stranded. Triggers include "push", "git push", "submodule", "commit submodule", "workspace push"
---

# Git Submodule Push

## When to Use

- The repo contains git submodules with uncommitted or unpushed changes
- Pushing the outer repo would leave submodule work stranded locally
- A submodule pin needs updating in the outer repo
- Mirroring or syncing a workspace submodule to a standalone clone

## When NOT to Use

- No submodules in the repo (use standard `git push`)
- Submodule has no new commits since last outer-repo push

## Core Workflow

Always push submodule first, then outer repo, then mirror.

1. **Submodule first**
   ```bash
   git -C <submodule-path> add -A
   git -C <submodule-path> commit -m "<message>"
   git -C <submodule-path> push
   ```

2. **Outer repo** (bumps the submodule pin)
   ```bash
   git add <submodule-path>
   git commit -m "<message>"
   git push
   ```

3. **Mirror clone** (if configured)
   ```bash
   git -C <mirror-path> pull --ff-only
   ```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Pushing outer repo before submodule | Submodule commits are stranded; reverse the order |
| Forgetting `git add <submodule-path>` | Pin doesn't update; outer repo still points to old commit |
| Using `git push --recurse-submodules=on-demand` without checking | Fails silently if submodule remote rejects; explicit push is safer |
| Skipping the mirror sync | Standalone clone drifts out of date |

## Anti-Patterns

- **Bare outer push**: Pushing only the outer repo when submodule is dirty
- **Manual pin editing**: Editing `.gitmodules` or the gitlink directly instead of `git add <submodule-path>`
- **Force push in submodule**: Risks breaking collaborators' checkouts
