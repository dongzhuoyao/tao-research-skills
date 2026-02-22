---
name: tao-tmux-zsh
description: Use when setting up terminal environment on a new machine, restoring tmux/zsh dotfiles, or looking up tmux key bindings and zsh configuration.
---

# Tao's Tmux + ZSH Configuration

Personal dotfiles for a consistent terminal environment across machines (Snellius, LUMI, personal).

## When to Use

- Setting up tmux + zsh on a new machine or server
- Restoring terminal configuration after a reinstall
- Looking up tmux key bindings or copy-mode commands
- Debugging terminal color or clipboard issues
- Configuring Oh My Zsh on a remote server

## Quick Reference

| Fact | Value |
|------|-------|
| Repo | `git@github.com:dongzhuoyao/tao-tmux-zsh.git` |
| Tmux prefix | `C-b` (default) + `C-a` (secondary, Screen-compatible) |
| Theme | Gruvbox dark (self-contained, no TPM needed) |
| Config size | ~200 lines of pure tmux config |
| ZSH framework | Oh My Zsh with `robbyrussell` theme |
| Scrollback | 50,000 lines |
| True color | Yes (24-bit) |
| Mouse | Toggleable with `prefix + m` |
| tmux required | 2.6+ |

## Installation

```bash
# 1. Prerequisites
# Install Oh My Zsh (if not already present)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 2. Clone
git clone git@github.com:dongzhuoyao/tao-tmux-zsh.git ~/tao-tmux-zsh

# 3. Backup existing configs
[ -f ~/.zshrc ] && mv ~/.zshrc ~/.zshrc.bak
[ -f ~/.tmux.conf ] && mv ~/.tmux.conf ~/.tmux.conf.bak

# 4. Symlink
ln -sf ~/tao-tmux-zsh/zshrc ~/.zshrc
ln -sf ~/tao-tmux-zsh/tmux.conf ~/.tmux.conf

# 5. Reload (if tmux is already running)
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

## ZSH Configuration

| Setting | Value |
|---------|-------|
| Framework | Oh My Zsh |
| Theme | `robbyrussell` |
| Plugins | `git` |
| PATH additions | `~/.local/bin`, `~/.opencode/bin` |
| Conda | Miniconda3 initialization (if installed) |
| Cursor | Blinking cursor enabled (for Termius) |

## Tmux Features

- **Gruvbox dark theme**: Self-contained color scheme, no plugin manager (TPM) needed
- **Mouse support**: Toggle with `prefix + m` — useful for scrolling and pane selection
- **Focused pane highlighting**: Active pane visually distinct from inactive ones
- **True color (24-bit)**: Proper color rendering for modern terminals
- **50K scrollback**: Large history buffer for log inspection
- **Status bar**: Session name (left), prefix indicator + time + date + user@host (right)

## Common Tasks

### Set up a new remote server

```bash
# SSH in, then:
sudo apt install tmux zsh  # or: yum install tmux zsh
chsh -s $(which zsh)       # set zsh as default shell (re-login required)

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Clone and symlink
git clone git@github.com:dongzhuoyao/tao-tmux-zsh.git ~/tao-tmux-zsh
ln -sf ~/tao-tmux-zsh/zshrc ~/.zshrc
ln -sf ~/tao-tmux-zsh/tmux.conf ~/.tmux.conf
```

### Reload config after edits

```bash
# Inside tmux: prefix + r
# Or from shell:
tmux source-file ~/.tmux.conf
source ~/.zshrc
```

### Update dotfiles

```bash
cd ~/tao-tmux-zsh && git pull
tmux source-file ~/.tmux.conf
source ~/.zshrc
```

## Dependencies

| Dependency | Required | Install |
|------------|----------|---------|
| tmux 2.6+ | Yes | `apt install tmux` / `yum install tmux` |
| Oh My Zsh | Yes | `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"` |
| Miniconda3 | Optional | [miniconda docs](https://docs.conda.io/en/latest/miniconda.html) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Colors look wrong | Ensure terminal supports true color; check `echo $TERM` shows `xterm-256color` or similar |
| Conda not found after setup | Re-source: `source ~/.zshrc` or check Miniconda3 is installed at `~/miniconda3` |
| Clipboard not working (macOS) | Install `reattach-to-user-namespace`: `brew install reattach-to-user-namespace` |
| Tmux config not loading | Verify symlink: `ls -la ~/.tmux.conf` should point to `~/tao-tmux-zsh/tmux.conf` |
| Oh My Zsh not found | Re-run the Oh My Zsh installer; check `~/.oh-my-zsh/` exists |
