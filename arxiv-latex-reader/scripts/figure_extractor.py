"""Extract figures and tables from arxiv papers as standalone PDFs.

Downloads the LaTeX source, extracts figure/table environments,
wraps each in a minimal standalone LaTeX document, compiles to PDF,
and converts to PNG for easy viewing.

Usage:
    python figure_extractor.py <paper_id>
    python figure_extractor.py <paper_id> --key-only
    python figure_extractor.py <paper_id> --workspace workspace

    # As library:
    from figure_extractor import FigureExtractor
    ext = FigureExtractor("2210.06462")
    figures = ext.extract_all()
    key_figs = ext.extract_key()
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tarfile
import tempfile
import urllib.request
import urllib.error
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = Path("workspace")

# Keywords for auto-categorizing figures
_METHOD_KW = {"method", "approach", "framework", "pipeline", "overview", "architecture", "model", "system", "structure"}
_RESULT_KW = {"result", "comparison", "ablation", "performance", "evaluation", "effect", "impact", "trade-off"}
_TABLE_RESULT_KW = {"comparison", "result", "ablation", "benchmark", "performance"}


@dataclass
class FigureInfo:
    """Metadata for an extracted figure/table."""
    number: int
    env_type: str        # "figure" or "table"
    caption: str
    label: str
    category: str        # "method" | "result" | "qualitative" | "supplementary"
    is_key: bool
    pdf_path: str = ""
    png_path: str = ""
    latex_snippet: str = ""


def _categorize_figure(caption: str, number: int) -> tuple[str, bool]:
    """Auto-categorize a figure by caption."""
    cap = caption.lower()
    if any(kw in cap for kw in _METHOD_KW):
        return "method", True
    if any(kw in cap for kw in _RESULT_KW):
        return "result", True
    if number <= 5:
        return "qualitative", True
    return "supplementary", False


def _categorize_table(caption: str, number: int) -> tuple[str, bool]:
    """Auto-categorize a table by caption."""
    cap = caption.lower()
    if any(kw in cap for kw in _TABLE_RESULT_KW):
        return "result", True
    if number <= 3:
        return "result", True
    return "supplementary", False


def _download_source(paper_id: str, dest_dir: Path) -> Path:
    """Download and extract arxiv LaTeX source tarball."""
    url = f"https://arxiv.org/e-print/{paper_id}"
    tar_path = dest_dir / "source.tar.gz"

    if not tar_path.exists():
        logger.info("Downloading source: %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "arxiv-latex-reader/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tar_path.write_bytes(resp.read())

    src_dir = dest_dir / "source"
    if not src_dir.exists():
        src_dir.mkdir(parents=True)
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(src_dir, filter="data")
        except tarfile.ReadError:
            # Might be a single .tex file, not a tarball
            src_dir.mkdir(exist_ok=True)
            (src_dir / "main.tex").write_bytes(tar_path.read_bytes())

    return src_dir


def _find_main_tex(src_dir: Path) -> Path:
    """Find the main .tex file in the source directory."""
    tex_files = list(src_dir.glob("*.tex"))

    for f in tex_files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        if "\\documentclass" in content:
            return f

    if tex_files:
        return tex_files[0]

    raise FileNotFoundError(f"No .tex files found in {src_dir}")


def _extract_environments(tex_content: str, env_name: str) -> list[dict]:
    """Extract all \\begin{env}...\\end{env} blocks with captions and labels."""
    # Match figure, figure*, table, table* environments
    pattern = rf"\\begin\{{{env_name}\*?\}}(.*?)\\end\{{{env_name}\*?\}}"
    results = []

    for i, match in enumerate(re.finditer(pattern, tex_content, re.DOTALL), 1):
        body = match.group(0)

        # Extract caption
        cap_match = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body)
        caption = cap_match.group(1) if cap_match else ""

        # Extract label
        label_match = re.search(r"\\label\{([^}]+)\}", body)
        label = label_match.group(1) if label_match else ""

        results.append({
            "number": i,
            "body": body,
            "caption": caption,
            "label": label,
        })

    return results


def _build_standalone_tex(
    snippet: str,
    src_dir: Path,
    preamble_lines: list[str],
) -> str:
    """Wrap a figure/table snippet in a standalone LaTeX document.

    Uses a minimal preamble to avoid conflicts with venue .sty files.
    Strips citation commands and unknown macros from the snippet.
    """
    graphics_paths = [str(src_dir)]
    for d in src_dir.iterdir():
        if d.is_dir():
            graphics_paths.append(str(d))
    paths_str = "".join(f"{{{p}}}" for p in graphics_paths)

    # Neutralize commands that fail without venue .sty
    snippet = re.sub(r"\\cite[pt]?\{[^}]*\}", "", snippet)
    snippet = re.sub(r"\\[Cc]ref\{[^}]*\}", "REF", snippet)
    snippet = re.sub(r"\\blfootnote\{[^}]*\}", "", snippet)
    snippet = re.sub(r"\\vspace\{[^}]*\}", "", snippet)
    snippet = re.sub(r"\\vspace\*\{[^}]*\}", "", snippet)

    return (
        "\\documentclass[border=5pt]{standalone}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{amsmath,amssymb,amsfonts,bm}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{multirow}\n"
        "\\usepackage{makecell}\n"
        "\\usepackage{xcolor}\n"
        "\\usepackage{subcaption}\n"
        "\\usepackage{adjustbox}\n"
        "\\newcommand{\\cmark}{\\checkmark}\n"
        "\\newcommand{\\xmark}{$\\times$}\n"
        f"\\graphicspath{{{paths_str}}}\n"
        "\\begin{document}\n"
        f"{snippet}\n"
        "\\end{document}\n"
    )


def _compile_latex(tex_path: Path, output_dir: Path, src_dir: Path | None = None) -> Path | None:
    """Compile a .tex file to PDF. Returns PDF path or None on failure."""
    try:
        env = os.environ.copy()
        if src_dir:
            # Add source dir to TEXINPUTS so custom .sty files are found
            tex_inputs = f"{src_dir}:{src_dir}//:"
            env["TEXINPUTS"] = tex_inputs + env.get("TEXINPUTS", "")

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        pdf_path = output_dir / tex_path.with_suffix(".pdf").name
        if pdf_path.exists():
            return pdf_path
        logger.warning("pdflatex produced no PDF for %s", tex_path.name)
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("pdflatex failed for %s: %s", tex_path.name, e)
        return None


def _pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 300) -> bool:
    """Convert PDF to PNG using sips (macOS) or convert (ImageMagick)."""
    try:
        # Try sips (macOS built-in)
        subprocess.run(
            ["sips", "-s", "format", "png", "--resampleWidth", "2000", str(pdf_path), "--out", str(png_path)],
            capture_output=True,
            timeout=15,
        )
        if png_path.exists():
            return True
    except FileNotFoundError:
        pass

    try:
        # Fallback: ImageMagick convert
        subprocess.run(
            ["convert", "-density", str(dpi), str(pdf_path), str(png_path)],
            capture_output=True,
            timeout=15,
        )
        return png_path.exists()
    except FileNotFoundError:
        logger.warning("Neither sips nor convert available for PDF→PNG")
        return False


class FigureExtractor:
    """Extract figures and tables from arxiv LaTeX source."""

    def __init__(self, paper_id: str, workspace_dir: str | Path | None = None):
        self.paper_id = paper_id
        self.workspace = (Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE) / paper_id
        self._figures_dir = self.workspace / "figures"

    def extract_all(self, force: bool = False) -> list[FigureInfo]:
        """Extract all figures and tables, compile to PDF/PNG."""
        manifest_path = self._figures_dir / "manifest.json"
        if not force and manifest_path.exists():
            return self._load_manifest(manifest_path)

        src_dir = _download_source(self.paper_id, self.workspace)
        main_tex = _find_main_tex(src_dir)
        tex_content = self._expand_inputs(main_tex, src_dir)

        # Extract preamble (before \begin{document})
        doc_start = tex_content.find("\\begin{document}")
        preamble_lines = tex_content[:doc_start].split("\n") if doc_start > 0 else []

        results: list[FigureInfo] = []
        self._figures_dir.mkdir(parents=True, exist_ok=True)

        # Extract figures
        for fig in _extract_environments(tex_content, "figure"):
            cat, is_key = _categorize_figure(fig["caption"], fig["number"])
            info = FigureInfo(
                number=fig["number"],
                env_type="figure",
                caption=fig["caption"],
                label=fig["label"],
                category=cat,
                is_key=is_key,
                latex_snippet=fig["body"],
            )
            self._compile_snippet(info, src_dir, preamble_lines)
            results.append(info)

        # Extract tables
        for tab in _extract_environments(tex_content, "table"):
            cat, is_key = _categorize_table(tab["caption"], tab["number"])
            info = FigureInfo(
                number=tab["number"],
                env_type="table",
                caption=tab["caption"],
                label=tab["label"],
                category=cat,
                is_key=is_key,
                latex_snippet=tab["body"],
            )
            self._compile_snippet(info, src_dir, preamble_lines)
            results.append(info)

        self._save_manifest(results, manifest_path)
        logger.info("Extracted %d figures + tables (%d key)", len(results), sum(1 for r in results if r.is_key))
        return results

    def extract_key(self, force: bool = False) -> list[FigureInfo]:
        """Extract only key figures/tables."""
        return [f for f in self.extract_all(force=force) if f.is_key]

    @staticmethod
    def _expand_inputs(tex_file: Path, src_dir: Path, depth: int = 0) -> str:
        """Recursively expand \\input{} directives in a .tex file."""
        if depth > 10:
            return ""
        content = tex_file.read_text(encoding="utf-8", errors="ignore")

        def _replace_input(m: re.Match) -> str:
            rel_path = m.group(1)
            input_file = src_dir / rel_path
            if not input_file.suffix:
                input_file = input_file.with_suffix(".tex")
            if input_file.exists():
                return FigureExtractor._expand_inputs(input_file, src_dir, depth + 1)
            return m.group(0)  # keep as-is if file not found

        return re.sub(r"\\input\{([^}]+)\}", _replace_input, content)

    def _compile_snippet(
        self,
        info: FigureInfo,
        src_dir: Path,
        preamble_lines: list[str],
    ) -> None:
        """Compile a single figure/table snippet to PDF and PNG."""
        prefix = "fig" if info.env_type == "figure" else "tab"
        tex_name = f"{prefix}{info.number}.tex"
        tex_path = self._figures_dir / tex_name

        standalone_tex = _build_standalone_tex(info.latex_snippet, src_dir, preamble_lines)
        tex_path.write_text(standalone_tex, encoding="utf-8")

        pdf_path = _compile_latex(tex_path, self._figures_dir, src_dir)
        if pdf_path:
            info.pdf_path = str(pdf_path)
            png_path = pdf_path.with_suffix(".png")
            if _pdf_to_png(pdf_path, png_path):
                info.png_path = str(png_path)
            logger.info("Compiled %s%d → %s", prefix, info.number, pdf_path.name)

    def _save_manifest(self, figures: list[FigureInfo], path: Path) -> None:
        data = []
        for f in figures:
            d = asdict(f)
            d.pop("latex_snippet", None)  # Don't store full LaTeX in manifest
            data.append(d)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_manifest(self, path: Path) -> list[FigureInfo]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [FigureInfo(**{**d, "latex_snippet": ""}) for d in data]


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract figures/tables from arxiv LaTeX source")
    parser.add_argument("paper_id", help="Arxiv paper ID (e.g. 2210.06462)")
    parser.add_argument("--key-only", action="store_true", help="Only extract key figures")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ext = FigureExtractor(args.paper_id, workspace_dir=args.workspace)

    if args.key_only:
        figures = ext.extract_key(force=args.force)
        print(f"\nKey figures/tables: {len(figures)}")
    else:
        figures = ext.extract_all(force=args.force)
        print(f"\nAll figures/tables: {len(figures)}")

    for fig in figures:
        key = " ★" if fig.is_key else ""
        status = "✓" if fig.png_path else ("PDF" if fig.pdf_path else "✗")
        print(f"  [{status}] {fig.env_type.title()} {fig.number} [{fig.category}]{key}")
        print(f"       {fig.caption[:100]}{'...' if len(fig.caption) > 100 else ''}")
        if fig.png_path:
            print(f"       → {fig.png_path}")


if __name__ == "__main__":
    main()
