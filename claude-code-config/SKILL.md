---
name: claude-code-config
description: Use when setting up Claude Code on a new machine, configuring permissions, statusline, or plugins. Contains the standard settings.json and statusline script.
---

# Tao's Claude Code Configuration

Standard Claude Code settings for research workflows: bypass all tool permissions, custom statusline, plan-mode default, and plugin setup.

## When to Use

- Setting up Claude Code on a new machine or project
- Restoring permissions/statusline after a config reset
- Checking what plugins or permissions are enabled
- Configuring a new project's `.claude/settings.json`

## Quick Reference

| Setting | Value |
|---------|-------|
| Default mode | `plan` (forces planning before implementation) |
| Permissions | All tools allowed (full bypass) |
| Statusline | Custom bash script with git, cost, tokens, context % |
| Plugins | superpowers, code-simplifier, ralph-loop, claude-md-management, claude-code-setup |

## Global Settings (`~/.claude/settings.json`)

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Write(*)",
      "Edit(*)",
      "Glob(*)",
      "Grep(*)",
      "WebFetch(*)",
      "WebSearch(*)",
      "NotebookEdit(*)",
      "Task(*)"
    ],
    "deny": [],
    "defaultMode": "plan"
  },
  "statusLine": {
    "type": "command",
    "command": "bash /home/tlong01/.claude/statusline-command.sh"
  },
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true,
    "ralph-loop@claude-plugins-official": true,
    "claude-md-management@claude-plugins-official": true,
    "claude-code-setup@claude-plugins-official": true
  }
}
```

## Project-Level Settings (`<project>/.claude/settings.json`)

Add IDE MCP tools on top of global permissions when using VS Code / Cursor:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "Read(*)",
      "Write(*)",
      "Edit(*)",
      "Glob(*)",
      "Grep(*)",
      "WebFetch(*)",
      "WebSearch(*)",
      "NotebookEdit(*)",
      "Task(*)",
      "mcp__ide__getDiagnostics(*)",
      "mcp__ide__executeCode(*)"
    ],
    "deny": []
  }
}
```

## Statusline Script (`~/.claude/statusline-command.sh`)

Custom status bar showing: `directory (branch) model $cost tokens ctx% [vim]`

```bash
#!/usr/bin/env bash
# Claude Code status line script
# Displays: dir | git branch | model | cost | tokens | context % | vim mode

input=$(cat)

cwd=$(echo "$input" | jq -r '.cwd // empty')
model=$(echo "$input" | jq -r '.model.display_name // empty')
used_pct=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
vim_mode=$(echo "$input" | jq -r '.vim.mode // empty')
cost_usd=$(echo "$input" | jq -r '.cost.total_cost_usd // empty')
in_tokens=$(echo "$input" | jq -r '.context_window.total_input_tokens // empty')
out_tokens=$(echo "$input" | jq -r '.context_window.total_output_tokens // empty')

# ANSI escape helper
ESC=$'\033'
RESET="${ESC}[0m"
CYAN="${ESC}[0;36m"
YELLOW="${ESC}[0;33m"
GREEN="${ESC}[0;32m"
RED="${ESC}[0;31m"
BLUE="${ESC}[0;34m"
MAGENTA="${ESC}[0;35m"

# Directory: basename of cwd
if [ -n "$cwd" ]; then
  dir=$(basename "$cwd")
else
  dir=$(basename "$(pwd)")
fi

# Git branch
git_branch=""
if git -C "${cwd:-.}" --no-optional-locks rev-parse --git-dir > /dev/null 2>&1; then
  branch=$(git -C "${cwd:-.}" --no-optional-locks symbolic-ref --short HEAD 2>/dev/null)
  if [ -n "$branch" ]; then
    git_branch=" ${YELLOW}($branch)${RESET}"
  fi
fi

# Context usage (color-coded: green < 50%, yellow 50-80%, red > 80%)
ctx_str=""
if [ -n "$used_pct" ]; then
  used_int=${used_pct%.*}
  if [ "$used_int" -ge 80 ] 2>/dev/null; then
    ctx_color="$RED"
  elif [ "$used_int" -ge 50 ] 2>/dev/null; then
    ctx_color="$YELLOW"
  else
    ctx_color="$GREEN"
  fi
  ctx_str=" ${ctx_color}ctx:${used_pct}%${RESET}"
fi

# Vim mode indicator
vim_str=""
if [ -n "$vim_mode" ]; then
  if [ "$vim_mode" = "NORMAL" ]; then
    vim_str=" ${BLUE}[N]${RESET}"
  else
    vim_str=" ${CYAN}[I]${RESET}"
  fi
fi

# Model (shortened)
model_str=""
if [ -n "$model" ]; then
  model_str=" ${MAGENTA}${model}${RESET}"
fi

# Session cost
cost_str=""
if [ -n "$cost_usd" ]; then
  cost_str=" ${YELLOW}\$${cost_usd}${RESET}"
fi

# Token usage (compact: e.g. 12.3k/4.1k)
token_str=""
if [ -n "$in_tokens" ] && [ -n "$out_tokens" ]; then
  format_tokens() {
    local t=$1
    if [ "$t" -ge 1000000 ] 2>/dev/null; then
      printf "%.1fM" "$(echo "scale=1; $t / 1000000" | bc)"
    elif [ "$t" -ge 1000 ] 2>/dev/null; then
      printf "%.1fk" "$(echo "scale=1; $t / 1000" | bc)"
    else
      printf "%d" "$t"
    fi
  }
  in_fmt=$(format_tokens "$in_tokens")
  out_fmt=$(format_tokens "$out_tokens")
  token_str=" ${CYAN}${in_fmt}in/${out_fmt}out${RESET}"
fi

printf "%s%s%s%s%s%s%s" \
  "${CYAN}${dir}${RESET}" \
  "$git_branch" \
  "$model_str" \
  "$cost_str" \
  "$token_str" \
  "$ctx_str" \
  "$vim_str"
```

**Requires**: `jq` and `bc` installed on the system.

## Installation on a New Machine

```bash
# 1. Create Claude config directory
mkdir -p ~/.claude

# 2. Write global settings
cat > ~/.claude/settings.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
      "Glob(*)", "Grep(*)", "WebFetch(*)", "WebSearch(*)",
      "NotebookEdit(*)", "Task(*)"
    ],
    "deny": [],
    "defaultMode": "plan"
  },
  "statusLine": {
    "type": "command",
    "command": "bash ~/.claude/statusline-command.sh"
  },
  "enabledPlugins": {
    "superpowers@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true,
    "ralph-loop@claude-plugins-official": true,
    "claude-md-management@claude-plugins-official": true,
    "claude-code-setup@claude-plugins-official": true
  }
}
EOF

# 3. Copy statusline script (from tao-research-skills or existing machine)
# The script contents are in this SKILL.md above

# 4. For each project, add .claude/settings.json with IDE tools if needed
```

## Permissions Explained

| Permission | Purpose |
|------------|---------|
| `Bash(*)` | Run any shell command (git, conda, sbatch, etc.) |
| `Read(*)` | Read any file |
| `Write(*)` | Create new files |
| `Edit(*)` | Modify existing files |
| `Glob(*)` | Find files by pattern |
| `Grep(*)` | Search file contents |
| `WebFetch(*)` | Fetch URLs |
| `WebSearch(*)` | Web search |
| `NotebookEdit(*)` | Edit Jupyter notebooks |
| `Task(*)` | Launch subagents |
| `mcp__ide__*` | IDE integration (VS Code / Cursor only) |

The `(*)` wildcard means all arguments are allowed. The `deny: []` list is empty, meaning nothing is blocked.

## Plugins

| Plugin | Purpose |
|--------|---------|
| `superpowers` | Brainstorming, debugging, TDD, code review, planning workflows |
| `code-simplifier` | Simplify and refine code |
| `ralph-loop` | Autonomous development loop |
| `claude-md-management` | Audit and improve CLAUDE.md files |
| `claude-code-setup` | Recommend Claude Code automations for a project |

## Notes

- `defaultMode: "plan"` forces Claude to plan before implementing. Override per-session with `/chat` for quick questions.
- The statusline script reads JSON from stdin (provided by Claude Code) and outputs ANSI-colored text.
- Context usage is color-coded: green (< 50%), yellow (50-80%), red (> 80%) — helps you know when to start a new conversation.
