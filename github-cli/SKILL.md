---
name: github-cli
description: Use when interacting with GitHub repos, PRs, issues, releases, or API data. Covers gh CLI usage patterns, authentication, and common queries. Triggers: "gh", "github", "pull request", "PR", "issue", "gh api", "gh pr", "gh issue", "github release"
---

# GitHub CLI (`gh`)

Patterns for using the GitHub CLI to interact with repos, PRs, issues, and the GitHub API — without leaving the terminal.

## When to Use

- Fetching GitHub data (PRs, issues, commits, releases, repo info)
- Creating or reviewing pull requests from the terminal
- Querying the GitHub API for structured data
- Replacing WebFetch for GitHub URLs (which fails on JS-rendered pages)

## Quick Reference

| Fact | Value |
|------|-------|
| Tool | `gh` (GitHub CLI) |
| Auth | `gh auth login` or `GITHUB_TOKEN` env var |
| Docs | `gh help <command>` or `man gh-<command>` |
| API base | `gh api` wraps `https://api.github.com/` with auth |

## Why `gh` Over WebFetch

GitHub pages are JS-rendered SPAs — `WebFetch` only gets empty HTML/CSS scaffolding. The `gh` CLI uses authenticated API calls and returns structured JSON, making it the reliable choice for all GitHub interactions.

## Common Patterns

### Pull Requests

```bash
# List open PRs
gh pr list

# View a specific PR (by number or URL)
gh pr view 123
gh pr view https://github.com/owner/repo/pull/123

# Create a PR
gh pr create --title "Title" --body "Description"

# Check PR status (CI checks, reviews)
gh pr checks 123

# View PR diff
gh pr diff 123

# List PR comments
gh api repos/{owner}/{repo}/pulls/123/comments
```

### Issues

```bash
# List open issues
gh issue list

# View an issue
gh issue view 42

# Create an issue
gh issue create --title "Bug: ..." --body "Steps to reproduce..."

# Search issues
gh issue list --search "label:bug sort:updated-desc"
```

### Repository Info

```bash
# View repo details
gh repo view owner/repo

# List releases
gh release list

# View latest release
gh release view --repo owner/repo

# Clone a repo
gh repo clone owner/repo
```

### Raw API Queries

`gh api` is the most powerful pattern — it wraps any GitHub REST or GraphQL endpoint with automatic authentication and pagination.

```bash
# REST: Get repo info
gh api repos/owner/repo

# REST: List commits on a branch
gh api repos/owner/repo/commits?sha=main&per_page=5

# REST: Get a specific workflow run
gh api repos/owner/repo/actions/runs/12345

# REST: Paginated results (auto-paginate)
gh api --paginate repos/owner/repo/issues

# GraphQL: Custom query
gh api graphql -f query='{ repository(owner:"owner", name:"repo") { stargazerCount } }'

# Output formatting with jq
gh api repos/owner/repo/pulls --jq '.[].title'
```

### Authentication

```bash
# Interactive login
gh auth login

# Check auth status
gh auth status

# Use a specific SSH key for git operations (separate from gh auth)
GIT_SSH_COMMAND="ssh -i ~/.ssh/my_key -o IdentitiesOnly=yes" git push
```

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| `WebFetch` on GitHub URLs | `gh api` or `gh pr view` |
| `curl` with manual auth headers | `gh api` (handles auth automatically) |
| Parse GitHub HTML | Use `gh api` for structured JSON |
| Hardcode GitHub tokens in scripts | Use `gh auth` or `GITHUB_TOKEN` env var |

## Tips

- **`--jq` flag**: Filter JSON output inline without piping to `jq`. E.g., `gh api repos/o/r/pulls --jq '.[].title'`.
- **`--paginate`**: Automatically follows pagination links for large result sets.
- **`-X POST`**: Use with `gh api` for write operations. E.g., `gh api repos/o/r/issues -f title="Bug" -f body="..."`.
- **`--json` flag**: On `gh pr list`, `gh issue list`, etc., select specific fields: `gh pr list --json number,title,author`.

## See Also

- `fail-fast-ml-engineering` — Preflight checks that may query GitHub for repo state
- `wandb-experiment-tracking` — W&B run metadata includes git commit info
