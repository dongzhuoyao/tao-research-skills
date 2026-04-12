"""Stateless full-text section retrieval from parsed papers.

Standalone — no poster_agent dependency. Reads paper_content.json directly.

Usage:
    python section_reader.py <paper_id> Method Experiments
    python section_reader.py <paper_id> --all
    python section_reader.py <paper_id> --keyword attention

    # As library:
    from section_reader import read_sections
    text = read_sections("2403.13802", ["Method", "Experiments"], workspace_dir="workspace")
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("workspace")


# ── LaTeX cleaning (inlined from poster_agent.densifier) ──────────

def clean_latex(text: str) -> str:
    """Clean LaTeX artifacts from text, including accented characters."""
    _accent_map = {"'": "\u0301", '"': "\u0308", "`": "\u0300", "~": "\u0303", "^": "\u0302"}

    def _replace_accent(m: re.Match) -> str:
        acc, char = m.group(1), m.group(2)
        if acc in _accent_map:
            try:
                return unicodedata.normalize("NFC", char + _accent_map[acc])
            except Exception:
                pass
        return char

    text = re.sub(r"\{\\(['\"`~^])([a-zA-Z])\}", _replace_accent, text)
    text = re.sub(r"\\(['\"`~^])\{([a-zA-Z])\}", _replace_accent, text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"~", " ", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    text = re.sub(r"\\citep?\{[^}]*\}", "", text)
    text = re.sub(r"\\cref\{[^}]*\}", "Table/Figure", text)
    text = re.sub(r"\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── paper_content.json loading (standalone) ───────────────────────

def _load_paper(paper_json: Path) -> dict:
    """Load paper_content.json and return raw dict."""
    return json.loads(paper_json.read_text(encoding="utf-8"))


# ── Section reading ──────────────────────────────────────────────

def read_sections(
    paper_id: str,
    sections: list[int] | list[str] | str,
    *,
    workspace_dir: Path | str | None = None,
    clean: bool = True,
) -> list[dict]:
    """Read full text of requested sections from a parsed paper.

    Args:
        paper_id: Arxiv paper ID.
        sections: Section identifier(s):
            - list[int]: by index (0-based)
            - list[str]: by exact name
            - str: "all" to return everything, or a keyword for fuzzy match
        workspace_dir: Override workspace directory.
        clean: If True, clean LaTeX artifacts from text.

    Returns:
        List of dicts with keys: index, name, level, text, figures, tables.
        Text is NEVER truncated.
    """
    ws = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE
    paper_json = ws / paper_id / "paper_content.json"
    if not paper_json.exists():
        raise FileNotFoundError(
            f"paper_content.json not found at {paper_json}. "
            "Run 'poster-agent parse' first."
        )

    paper = _load_paper(paper_json)
    all_sections = paper.get("sections", [])
    figures = paper.get("figures", [])
    tables = paper.get("tables", [])

    # Build figure/table → section mapping
    section_figures: dict[int, list[str]] = {}
    section_tables: dict[int, list[str]] = {}
    for i, sec in enumerate(all_sections):
        sec_text = sec.get("text", "")
        sec_figs = [f["id"] for f in figures if f.get("id") and f["id"] in sec_text]
        sec_tabs = [t["id"] for t in tables if t.get("id") and t["id"] in sec_text]
        section_figures[i] = sec_figs
        section_tables[i] = sec_tabs

    # Resolve which sections to return
    indices = _resolve_sections(all_sections, sections)

    results = []
    for idx in indices:
        sec = all_sections[idx]
        text = clean_latex(sec["text"]) if clean else sec["text"]
        results.append({
            "index": idx,
            "name": sec["name"],
            "level": sec.get("level", 1),
            "text": text,
            "figures": section_figures.get(idx, []),
            "tables": section_tables.get(idx, []),
        })

    logger.info(
        "read_sections(%s, %s): returned %d sections, %d total chars",
        paper_id, sections, len(results), sum(len(r["text"]) for r in results),
    )
    return results


def _resolve_sections(sections: list[dict], identifiers: list[int] | list[str] | str) -> list[int]:
    """Resolve section identifiers to indices."""
    if isinstance(identifiers, str):
        if identifiers.lower() == "all":
            return list(range(len(sections)))
        return _fuzzy_match(sections, identifiers)

    if not identifiers:
        return []

    if isinstance(identifiers[0], int):
        return [i for i in identifiers if 0 <= i < len(sections)]

    # By name — exact match first, then case-insensitive
    result = []
    for name in identifiers:
        name_lower = name.lower()
        for i, sec in enumerate(sections):
            if sec["name"] == name:
                result.append(i)
                break
            elif sec["name"].lower() == name_lower:
                result.append(i)
                break
        else:
            matches = _fuzzy_match(sections, name)
            if matches:
                result.append(matches[0])
    return result


def _fuzzy_match(sections: list[dict], keyword: str) -> list[int]:
    """Find sections whose name contains the keyword (case-insensitive)."""
    kw_lower = keyword.lower()
    return [i for i, sec in enumerate(sections) if kw_lower in sec["name"].lower()]


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Read full sections from paper_content.json")
    parser.add_argument("paper_id", help="Paper ID (e.g., 2403.13802)")
    parser.add_argument("sections", nargs="*", help="Section names or indices")
    parser.add_argument("--all", action="store_true", help="Read all sections")
    parser.add_argument("--keyword", default=None, help="Fuzzy match by keyword")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument("--no-clean", action="store_true", help="Skip LaTeX cleaning")
    args = parser.parse_args()

    if args.all:
        identifiers = "all"
    elif args.keyword:
        identifiers = args.keyword
    elif args.sections:
        try:
            identifiers = [int(s) for s in args.sections]
        except ValueError:
            identifiers = args.sections
    else:
        print("Specify section names, --all, or --keyword", file=sys.stderr)
        sys.exit(1)

    results = read_sections(
        args.paper_id, identifiers,
        workspace_dir=args.workspace, clean=not args.no_clean,
    )
    for sec in results:
        print(f"\n{'=' * 60}")
        print(f"§{sec['index']} {sec['name']} ({len(sec['text']):,} chars)")
        print("=" * 60)
        print(sec["text"])


if __name__ == "__main__":
    main()
