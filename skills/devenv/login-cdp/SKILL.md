---
name: login-cdp
description: Use when user says "login", "登录", "fix expired sessions", "refresh login", or needs to re-authenticate CDP browser sessions for any platform. Auto-detects expired platforms and guides interactive re-login via MCP browser tools.
---

# CDP Platform Login

Auto-detect expired CDP browser sessions and re-authenticate interactively via MCP browser tools. Supports multiple CDP ports for multi-account setups.

## Prerequisites

- Chrome running with `--remote-debugging-port=<port>`
- MCP browser tools available (Playwright MCP or Chrome MCP)

## CDP Port Configuration

Two CDP ports, two independent Chrome instances, each with its own login sessions:

| Port | Account | Session Directory | Purpose |
|------|---------|-------------------|---------|
| 9222 | Account 1 (default) | `$HOME/chrome-debug-v2` | General browsing, Twitter scraping, posting |
| 9223 | Account 2 | `$HOME/chrome-debug2` | RedNote publishing (secondary account) |

Startup commands:
```bash
# Account 1 — CDP 9222 — Session: ~/chrome-debug-v2
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/chrome-debug-v2"

# Account 2 — CDP 9223 — Session: ~/.chrome-cdp/rednote-account2
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9223 \
  --user-data-dir="$HOME/chrome-debug2"
```

Quick check:
```bash
curl -s --noproxy '*' http://127.0.0.1:9222/json/version  # Account 1
curl -s --noproxy '*' http://127.0.0.1:9223/json/version  # Account 2

# Check which Chrome instances are running and their session directories
ps aux | grep "[G]oogle Chrome" | grep -v "Helper\|GPU\|Renderer" | grep -o 'user-data-dir=[^ ]*'
```

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--port` | `9222` | CDP port to check/fix |
| platform names | `rednote, twitter` | Comma-separated or space-separated platform names |

Examples:
- `/login-cdp` — check rednote + twitter on port 9222
- `/login-cdp twitter` — check only twitter
- `/login-cdp --port 9223 rednote` — check rednote on port 9223

## Flow

### Step 0: Verify CDP is Running

```bash
curl -s --noproxy '*' http://127.0.0.1:<port>/json/version
```

If connection refused → **print the exact Chrome startup command as a fenced bash block for the user to paste into their own terminal**, then STOP. Do NOT try to run it yourself — Chrome must be launched from the user's shell so it stays attached to their session.

Format the output like this (fill in the port/profile that matches the request):

```
CDP <port> is not running. Paste this into your terminal:

​```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=<port> \
  --user-data-dir="$HOME/<profile-dir>"
​```

Then re-run /login-cdp.
```

Port → profile mapping: 9222 → `chrome-debug-v2`, 9223 → `chrome-debug2`.

If CDP is reachable, also check which `--user-data-dir` is being used:
```bash
ps aux | grep "[G]oogle Chrome" | grep -v "Helper\|GPU\|Renderer\|Utility\|Plugin" | grep -o 'user-data-dir=[^ ]*'
```

Report: "CDP <port> running, profile: <user-data-dir>"

### Step 1: Detect Expired Sessions

Check login status by navigating to each platform's check URL.

For each platform:
1. Navigate to the platform's `check_url` via MCP browser tools
2. Take a screenshot or snapshot
3. If redirected to a login page (URL contains "login", "flow/login", or page shows sign-in form) → **expired**
4. If page shows logged-in content (dashboard, feed, profile, account name) → **healthy**
5. **Close the tab immediately** after checking

If all healthy → "All platforms healthy, nothing to do." → STOP.

### Step 2: Present Expired Platforms

```
CDP <port> (<user-data-dir>)

Expired sessions:
  1. RedNote Creator — expired
  2. Twitter/X — expired

Healthy:
  ✓ (none)

Which to fix? (e.g., "1,2" or "all")
```

### Step 3: Sequential Login Loop

For each selected platform:

**a) Open login page** via MCP browser tools.

**b) Notify user:**
- "Opened [platform] login page in Chrome (CDP <port>). Please log in, then say 'done'."
- For QR-code platforms: add "Requires QR code scanning — check your Chrome window."

**c) Wait** for user to say "done", "ok", or "next".

**d) Verify** — navigate to check_url again, confirm logged-in state.

**e) Handle failure** — "Still not logged in. Try again or skip?"

**f) Close tab** — ALWAYS close after each platform.

### Step 4: Summary

```
Login refresh complete (CDP <port>):
  ✓ RedNote Creator — logged in
  ✗ Twitter/X — still expired
```

## Platform Reference

| Platform | Display Name | Check URL | Login URL | QR? |
|----------|-------------|-----------|-----------|-----|
| rednote | RedNote Creator | https://creator.xiaohongshu.com/publish/publish | https://creator.xiaohongshu.com/login | No |
| rednote_www | RedNote Web | https://www.xiaohongshu.com/explore | https://www.xiaohongshu.com/explore | No |
| twitter | Twitter/X | https://x.com/home | https://x.com/i/flow/login | No |
| weibo | Weibo | https://weibo.com | https://weibo.com/login.php | No |
| linkedin | LinkedIn | https://www.linkedin.com/feed | https://www.linkedin.com/login | No |
| wechat_mp | WeChat MP | https://mp.weixin.qq.com/cgi-bin/home | https://mp.weixin.qq.com | Yes |
| bilibili | Bilibili | https://www.bilibili.com/ | https://passport.bilibili.com/login | Yes |
| youtube | YouTube | https://www.youtube.com/ | https://accounts.google.com | No |
| youtube_studio | YouTube Studio | https://studio.youtube.com/ | https://accounts.google.com | No |
| bilibili_creator | Bilibili Creator | https://member.bilibili.com/platform/home | https://passport.bilibili.com/login | Yes |

## MCP Tool Selection

This skill works with either MCP browser tool:
- **Chrome MCP** (`chrome-devtools-mcp`) — connects to existing CDP Chrome via `--browserUrl`
- **Playwright MCP** (`@playwright/mcp`) — needs `--cdp-endpoint` to connect to existing Chrome

Prefer whichever is configured to connect to the target CDP port. If both available, use Chrome MCP (it connects to CDP by default).

## Key Rules

- **Always verify CDP is running first** (Step 0) — don't assume
- **Never launch Chrome yourself** — if CDP is down, print the startup command as a copy-paste bash block and stop. The user runs it in their own terminal.
- **Report the user-data-dir** — so user knows which Chrome profile is being checked
- **No hardcoded selectors** — read snapshots/screenshots to identify login state
- **Always close tabs** — CDP tabs accumulate
- **Sequential only** — one platform at a time
- **Use `--noproxy '*'`** for all curl commands to localhost — machine may have proxy configured
