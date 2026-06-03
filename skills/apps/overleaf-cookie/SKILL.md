---
name: overleaf-cookie
description: "Use when extracting Overleaf login cookies from Chrome on macOS for the VSCode Overleaf Workshop extension. Decrypts Chrome's cookie store and outputs the session cookie in the exact format the extension expects. Triggers: "overleaf cookie", "overleaf login", "overleaf workshop cookie", "get overleaf cookie", "vscode overleaf login"
---

# Overleaf Cookie Extractor

## When to Use

- You need to log into the [Overleaf Workshop](https://github.com/iamhyc/overleaf-workshop) VSCode extension via cookie login
- You're on macOS and use Chrome as your browser
- Overleaf's SSO/captcha blocks normal password login in the extension
- You want the exact `overleaf_session2` cookie string without manually opening DevTools

## Requirements

- macOS (Chrome cookie decryption uses macOS Keychain)
- Chrome browser with an active Overleaf login session
- Python 3 with `browser_cookie3` installed (`pip install browser_cookie3`)

## Quick Use

```bash
python skills/apps/overleaf-cookie/scripts/get_overleaf_cookie.py
```

Output:
```
overleaf_session2=s:XXXXXXXXXXXX.XXXXXXXXXXXX
```

Copy the entire line and paste it into VSCode → `Overleaf Workshop: Login with Cookies` → Server: `https://www.overleaf.com`.

## How It Works

Chrome encrypts cookies using AES-128-CBC (v10) or AES-256-GCM (v11) on macOS. The encryption key lives in the macOS Keychain under "Chrome Safe Storage". The `browser_cookie3` library handles the keychain access and decryption automatically.

The script:
1. Opens Chrome's SQLite cookie DB at `~/Library/Application Support/Google/Chrome/Default/Cookies`
2. Decrypts all cookies via the Keychain-derived key
3. Filters for `overleaf_session2` on `.overleaf.com`
4. URL-decodes the value (Chrome sometimes stores percent-encoded characters)
5. Prints the clean `name=value` string

## Known Extension Bug (As of 2025-06)

The Overleaf Workshop extension's `cookiesLogin()` method parses the `/project` page HTML to extract `user-id`, `user-email`, and `csrf-token` from `<meta>` tags. Overleaf renamed these tags:

| Old (extension expects) | New (Overleaf serves) |
|------------------------|----------------------|
| `name="user-id"` | `name="ol-user_id"` |
| `name="user-email"` | `name="ol-usersEmail"` |
| `name="csrf-token"` | `name="ol-csrfToken"` |

This causes **"Failed to get User ID"** even with a perfectly valid cookie. The cookie itself is correct — the extension's regex is outdated.

**Workarounds:**
1. **Patch locally** — edit the installed extension's `out/api/base.js` (or `src/api/base.ts`) to match the new `ol-*` meta tag names
2. **Use DevTools login** — follow the [official manual method](https://github.com/overleaf-workshop/overleaf-workshop#how-to-login-with-cookies) and copy the raw Cookie header directly; in some cases the extension bypasses the broken parser path
3. **Wait for upstream fix** — file an issue at `iamhyc/Overleaf-Workshop`

## Anti-Patterns

- **Do not** manually copy the cookie from Chrome DevTools → Application → Cookies — the value shown there may be URL-encoded and will fail if pasted directly
- **Do not** use `curl` or `wget` to test the cookie — Overleaf's `/project` endpoint returns different HTML to non-browser user-agents, making verification unreliable
- **Do not** commit cookies or session tokens to git — they are ephemeral credentials

## See Also

- [Overleaf Workshop README — Cookie Login](https://github.com/overleaf-workshop/overleaf-workshop#how-to-login-with-cookies)
- `browser_cookie3` documentation for cross-platform cookie extraction
