---
name: tmux
description: "Use when setting up tmux on a new machine, looking up key bindings, debugging terminal colors, or restoring tmux dotfiles. Triggers: \"tmux\", \"terminal multiplexer\", \"tmux.conf\", \"pane\", \"window split\", \"copy mode\", \"prefix key\""
---

# Tmux Configuration

Personal tmux dotfiles for a consistent terminal environment across machines (Snellius, LUMI, personal).

## When to Use

- Setting up tmux on a new machine or server
- Restoring tmux configuration after a reinstall
- Looking up tmux key bindings or copy-mode commands
- Debugging terminal color or clipboard issues

## Quick Reference

| Fact | Value |
|------|-------|
| Repo | `git@github.com:dongzhuoyao/tao-tmux-zsh.git` |
| Prefix | `C-b` (default) + `C-a` (secondary, Screen-compatible) |
| Theme | Gruvbox dark (self-contained, no TPM needed) |
| Config size | ~200 lines of pure tmux config |
| Scrollback | 50,000 lines |
| True color | Yes (24-bit) |
| Mouse | Toggleable with `prefix + m` |
| tmux required | 2.6+ |

## Installation

```bash
# Clone
git clone git@github.com:dongzhuoyao/tao-tmux-zsh.git ~/tao-tmux-zsh

# Backup existing config
[ -f ~/.tmux.conf ] && mv ~/.tmux.conf ~/.tmux.conf.bak

# Symlink
ln -sf ~/tao-tmux-zsh/tmux.conf ~/.tmux.conf

# Reload (if tmux is already running)
tmux source-file ~/.tmux.conf
```

## Key Bindings

All bindings use the prefix key (`C-b` or `C-a`) unless noted.

### Navigation

| Binding | Action |
|---------|--------|
| `h` / `j` / `k` / `l` | Pane navigation (vim-style) |
| `Tab` | Switch to last active window |
| `C-h` / `C-l` | Previous / next window |

### Panes and Windows

| Binding | Action |
|---------|--------|
| `-` | Split horizontal (retains current path) |
| `_` | Split vertical (retains current path) |
| `H` / `J` / `K` / `L` | Resize pane |
| `c` | New window (retains current path) |
| `m` | Toggle mouse on/off |
| `r` | Reload tmux config |

### Copy Mode

| Binding | Action |
|---------|--------|
| `Enter` | Enter copy mode |
| `v` | Begin selection |
| `C-v` | Toggle rectangle selection |
| `y` | Yank (copy) selection |

macOS clipboard integration available via `prefix + y`.

## Features

- **Gruvbox dark theme**: Self-contained color scheme, no plugin manager (TPM) needed
- **Mouse support**: Toggle with `prefix + m` — useful for scrolling and pane selection
- **Focused pane highlighting**: Active pane visually distinct from inactive ones
- **True color (24-bit)**: Proper color rendering for modern terminals
- **50K scrollback**: Large history buffer for log inspection
- **Status bar**: Session name (left), prefix indicator + time + date + user@host (right)

## Common Tasks

### Reload config after edits

```bash
# Inside tmux: prefix + r
# Or from shell:
tmux source-file ~/.tmux.conf
```

### Update dotfiles

```bash
cd ~/tao-tmux-zsh && git pull
tmux source-file ~/.tmux.conf
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Colors look wrong | Ensure terminal supports true color; check `echo $TERM` shows `xterm-256color` or similar |
| Clipboard not working (macOS) | Install `reattach-to-user-namespace`: `brew install reattach-to-user-namespace` |
| Tmux config not loading | Verify symlink: `ls -la ~/.tmux.conf` should point to `~/tao-tmux-zsh/tmux.conf` |

## Anti-Patterns

| Anti-Pattern | What goes wrong | Fix |
|--------------|-----------------|-----|
| Installing TPM and adding many plugins | Slow startup, fails on remote servers without internet | This config is self-contained — no TPM needed |
| Leaving `TERM=xterm` on a remote server | Colors mangled, key bindings break, copy-mode glitches | Set `TERM=xterm-256color` (in zshrc or `~/.ssh/config` via `SetEnv`) |
| Editing `~/.tmux.conf` directly after symlinking | Edits get lost on next `git pull` of the dotfiles repo | Edit `~/tao-tmux-zsh/tmux.conf` and re-source |
| Patching `pbcopy` into Linux server config | Errors — `pbcopy` is macOS only | Let the config branch by OS; do not patch in `pbcopy` on Linux |
| Starting multiple tmux servers per user | Sessions diverge across servers; confusing reattach | Use `tmux a` to attach to the existing server |
| Forgetting to reload after editing config | Changes don't take effect until restart | `prefix + r` or `tmux source-file ~/.tmux.conf` |

## See Also

- `zsh` — Companion zsh configuration from the same dotfiles repo
