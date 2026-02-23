---
name: zsh
description: Use when setting up zsh and Oh My Zsh on a new machine, configuring shell plugins, PATH, or conda initialization. Triggers: "zsh", "zshrc", "Oh My Zsh", "shell config", "dotfiles", "conda init", "PATH"
---

# ZSH Configuration

Personal zsh dotfiles with Oh My Zsh for a consistent shell environment across machines.

## When to Use

- Setting up zsh + Oh My Zsh on a new machine or server
- Configuring PATH, conda, or shell plugins
- Restoring shell configuration after a reinstall
- Debugging shell startup or conda issues

## Quick Reference

| Setting | Value |
|---------|-------|
| Repo | `git@github.com:dongzhuoyao/tao-tmux-zsh.git` |
| Framework | Oh My Zsh |
| Theme | `robbyrussell` |
| Plugins | `git` |
| PATH additions | `~/.local/bin`, `~/.opencode/bin` |
| Conda | Miniconda3 initialization (if installed) |
| Cursor | Blinking cursor enabled (for Termius) |

## Installation

```bash
# 1. Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 2. Clone
git clone git@github.com:dongzhuoyao/tao-tmux-zsh.git ~/tao-tmux-zsh

# 3. Backup existing config
[ -f ~/.zshrc ] && mv ~/.zshrc ~/.zshrc.bak

# 4. Symlink
ln -sf ~/tao-tmux-zsh/zshrc ~/.zshrc

# 5. Reload
source ~/.zshrc
```

## Common Tasks

### Set up a new remote server

```bash
# SSH in, then:
sudo apt install zsh       # or: yum install zsh
chsh -s $(which zsh)       # set zsh as default shell (re-login required)

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Clone and symlink
git clone git@github.com:dongzhuoyao/tao-tmux-zsh.git ~/tao-tmux-zsh
ln -sf ~/tao-tmux-zsh/zshrc ~/.zshrc
```

### Reload after edits

```bash
source ~/.zshrc
```

### Update dotfiles

```bash
cd ~/tao-tmux-zsh && git pull
source ~/.zshrc
```

## Dependencies

| Dependency | Required | Install |
|------------|----------|---------|
| Oh My Zsh | Yes | `sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"` |
| Miniconda3 | Optional | [miniconda docs](https://docs.conda.io/en/latest/miniconda.html) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Conda not found after setup | Re-source: `source ~/.zshrc` or check Miniconda3 is installed at `~/miniconda3` |
| Oh My Zsh not found | Re-run the Oh My Zsh installer; check `~/.oh-my-zsh/` exists |

## See Also

- `tmux` — Companion tmux configuration from the same dotfiles repo
