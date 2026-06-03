#!/usr/bin/env python3
"""
Extract the overleaf_session2 cookie from Chrome on macOS.

Requires: pip install browser_cookie3
Usage: python get_overleaf_cookie.py
Output: overleaf_session2=s:XXXXXX.XXXXXX
"""

import sys
import urllib.parse


def main() -> int:
    try:
        import browser_cookie3
    except ImportError:
        print("Error: browser_cookie3 is not installed.", file=sys.stderr)
        print("Run: pip install browser_cookie3", file=sys.stderr)
        return 1

    cj = browser_cookie3.chrome()
    found = False

    for cookie in cj:
        if cookie.name == "overleaf_session2" and "overleaf" in cookie.domain:
            # Chrome may store percent-encoded characters; decode them
            raw_value = urllib.parse.unquote(cookie.value)
            print(f"overleaf_session2={raw_value}")
            found = True
            break

    if not found:
        print("Error: overleaf_session2 cookie not found in Chrome.", file=sys.stderr)
        print("Make sure you are logged into https://www.overleaf.com in Chrome.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
