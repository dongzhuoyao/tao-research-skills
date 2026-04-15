"""Stateless full-text retrieval from paper_content.json.

No LLM calls. Reads sections by name, index, or keyword.
Cleans LaTeX artifacts for downstream consumption.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _clean_latex(text: str) -> str:
    """Remove LaTeX commands, citations, and math delimiters."""
    # Remove \cite{...}, \citep{...}, \citet{...}
    text = re.sub(r"\\cite[pt]?\{[^}]*\}", "", text)
    # Remove \ref{...}, \eqref{...}, \label{...}
    text = re.sub(r"\\(?:eq)?ref\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    # Unwrap formatting commands: \textbf{X} → X, \textit{X} → X, \emph{X} → X
    text = re.sub(r"\\(?:textbf|textit|emph|underline|textrm|textsf|texttt)\{([^}]*)\}", r"\1", text)
    # Remove \begin{...} and \end{...}
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    # Remove standalone commands like \noindent, \vspace{...}, \hspace{...}
    text = re.sub(r"\\(?:noindent|vspace|hspace|smallskip|medskip|bigskip)\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:noindent|smallskip|medskip|bigskip)", "", text)
    # Simplify math delimiters: $x$ → x, \( x \) → x
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text)
    # LaTeX accents: \"{o} → o, \'e → e
    text = re.sub(r"""\\['"`^~=.uvHtcdb]\{([^}])\}""", r"\1", text)
    text = re.sub(r"""\\['"`^~]\w""", lambda m: m.group()[-1], text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class SectionReader:
    """Stateless reader for paper_content.json sections."""

    def __init__(self, paper_content_path: str | Path):
        path = Path(paper_content_path)
        with open(path) as f:
            data = json.load(f)
        self._sections = data.get("sections", [])

    def read(self, names: list[str]) -> list[dict]:
        """Read sections by exact or fuzzy name match."""
        results = []
        name_lower = [n.lower() for n in names]
        for sec in self._sections:
            if sec["name"].lower() in name_lower:
                results.append({
                    "name": sec["name"],
                    "text": _clean_latex(sec["text"]),
                    "level": sec.get("level", 1),
                    "char_count": len(sec["text"]),
                })
        return results

    def read_by_index(self, indices: list[int]) -> list[dict]:
        """Read sections by positional index."""
        results = []
        for i in indices:
            if 0 <= i < len(self._sections):
                sec = self._sections[i]
                results.append({
                    "name": sec["name"],
                    "text": _clean_latex(sec["text"]),
                    "level": sec.get("level", 1),
                    "char_count": len(sec["text"]),
                })
        return results

    def read_by_keyword(self, keyword: str) -> list[dict]:
        """Fuzzy match section names against keyword."""
        kw = keyword.lower()
        results = []
        for sec in self._sections:
            if kw in sec["name"].lower():
                results.append({
                    "name": sec["name"],
                    "text": _clean_latex(sec["text"]),
                    "level": sec.get("level", 1),
                    "char_count": len(sec["text"]),
                })
        return results

    def read_all(self) -> list[dict]:
        """Return all sections with cleaned text."""
        return [
            {
                "name": sec["name"],
                "text": _clean_latex(sec["text"]),
                "level": sec.get("level", 1),
                "char_count": len(sec["text"]),
            }
            for sec in self._sections
        ]

    @property
    def section_names(self) -> list[str]:
        """List all section names."""
        return [s["name"] for s in self._sections]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python section_reader.py <paper_id> [section_names...] [--all] [--keyword KW]")
        sys.exit(1)

    paper_id = sys.argv[1]
    workspace = Path("workspace") / paper_id / "paper_content.json"
    reader = SectionReader(workspace)

    if "--all" in sys.argv:
        for sec in reader.read_all():
            print(f"\n{'='*60}\n§ {sec['name']} ({sec['char_count']} chars)\n{'='*60}")
            print(sec["text"])
    elif "--keyword" in sys.argv:
        idx = sys.argv.index("--keyword")
        kw = sys.argv[idx + 1]
        for sec in reader.read_by_keyword(kw):
            print(f"\n§ {sec['name']} ({sec['char_count']} chars)")
            print(sec["text"])
    else:
        names = sys.argv[2:]
        for sec in reader.read(names):
            print(f"\n§ {sec['name']} ({sec['char_count']} chars)")
            print(sec["text"])
