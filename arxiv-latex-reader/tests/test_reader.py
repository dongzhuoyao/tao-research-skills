#!/usr/bin/env python3
"""End-to-end tests for arxiv-latex-reader skill.

Uses paper 2210.06462 (Self-Guided Diffusion Models, CVPR 2023) as fixture.
Queries facts from every section and verifies:
  1. Correct section is retrieved
  2. Expected fact appears in returned text
  3. Content is NEVER truncated
  4. All access patterns work (by name, index, keyword, all)

Run:
    cd arxiv-latex-reader
    python -m pytest tests/test_reader.py -v
    python -m pytest tests/test_reader.py -v -k "not slow"   # skip LLM tests
    python -m pytest tests/test_reader.py -v --run-slow       # include LLM tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to path
SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from fixture_paper import PAPER_ID, QUERIES, SECTIONS, create_fixture
from section_reader import SectionReader, _clean_latex


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Path:
    ws = tmp_path_factory.mktemp("workspace")
    create_fixture(ws)
    return ws


@pytest.fixture(scope="session")
def reader(workspace) -> SectionReader:
    paper_json = workspace / PAPER_ID / "paper_content.json"
    return SectionReader(paper_json)


# ── 1. Read by name ────────────────────────────────────────────

class TestReadByName:
    """Every section retrievable by exact name."""

    @pytest.mark.parametrize(
        "section_name",
        [s["name"] for s in SECTIONS],
        ids=[s["name"] for s in SECTIONS],
    )
    def test_read_section_by_name(self, reader, section_name):
        results = reader.read([section_name])
        assert len(results) == 1, f"Expected 1 result for '{section_name}', got {len(results)}"
        assert results[0]["name"] == section_name

    @pytest.mark.parametrize(
        "section_name",
        [s["name"] for s in SECTIONS],
        ids=[s["name"] for s in SECTIONS],
    )
    def test_section_not_empty(self, reader, section_name):
        results = reader.read([section_name])
        assert len(results[0]["text"]) > 0


# ── 2. Read by index ───────────────────────────────────────────

class TestReadByIndex:

    @pytest.mark.parametrize("idx", range(len(SECTIONS)))
    def test_read_by_index(self, reader, idx):
        results = reader.read_by_index([idx])
        assert len(results) == 1
        assert results[0]["name"] == SECTIONS[idx]["name"]


# ── 3. Read all ────────────────────────────────────────────────

class TestReadAll:
    def test_returns_every_section(self, reader):
        results = reader.read_all()
        assert len(results) == len(SECTIONS)

    def test_section_names_property(self, reader):
        names = reader.section_names
        assert names == [s["name"] for s in SECTIONS]

    def test_total_chars_consistent(self, reader):
        """Total chars from read_all equals sum of individual reads."""
        all_results = reader.read_all()
        total_all = sum(len(r["text"]) for r in all_results)

        individual_total = 0
        for s in SECTIONS:
            r = reader.read([s["name"]])
            individual_total += len(r[0]["text"])

        assert total_all == individual_total


# ── 4. Keyword search ──────────────────────────────────────────

class TestKeywordSearch:
    @pytest.mark.parametrize("keyword,expected", [
        ("introduction", "Introduction"),
        ("related", "Related Work"),
        ("approach", "Approach"),
        ("experiment", "Experiments"),
        ("conclusion", "Conclusion"),
        ("acknowledgment", "Acknowledgments"),
        ("appendix", "Appendix A: Implementation Details"),
        ("implementation", "Appendix A: Implementation Details"),
    ])
    def test_keyword_finds_section(self, reader, keyword, expected):
        results = reader.read_by_keyword(keyword)
        names = [r["name"] for r in results]
        assert expected in names, f"Keyword '{keyword}' should find '{expected}', got {names}"


# ── 5. Never truncates ─────────────────────────────────────────

class TestNeverTruncates:
    """Core invariant: no content is lost."""

    @pytest.mark.parametrize(
        "section_name",
        [s["name"] for s in SECTIONS],
        ids=[s["name"] for s in SECTIONS],
    )
    def test_char_count_matches_original(self, reader, section_name):
        """char_count field should match the original raw text length."""
        results = reader.read([section_name])
        original = next(s for s in SECTIONS if s["name"] == section_name)
        assert results[0]["char_count"] == len(original["text"])

    def test_all_sections_have_substantial_text(self, reader):
        """No section should be suspiciously short (truncated)."""
        for result in reader.read_all():
            assert len(result["text"]) > 50, (
                f"Section '{result['name']}' has only {len(result['text'])} chars — truncated?"
            )


# ── 6. Fact retrieval across the paper ─────────────────────────

class TestFactRetrieval:
    """Query specific facts from different parts of the paper.

    22 queries across all 7 sections. Each verifies the expected
    fact string appears in the cleaned text.
    """

    @pytest.mark.parametrize(
        "description,target_section,expected_fact",
        QUERIES,
        ids=[q[0] for q in QUERIES],
    )
    def test_fact_in_section(self, reader, description, target_section, expected_fact):
        results = reader.read([target_section])
        assert len(results) == 1, f"Section '{target_section}' not found"
        text = results[0]["text"]
        assert expected_fact in text, (
            f"Fact '{expected_fact}' not found in '{target_section}'. "
            f"Query: {description}. Text length: {len(text)}"
        )


# ── 7. Multi-section reads ─────────────────────────────────────

class TestMultiSection:
    def test_read_two_sections(self, reader):
        results = reader.read(["Introduction", "Conclusion"])
        assert len(results) == 2
        assert results[0]["name"] == "Introduction"
        assert results[1]["name"] == "Conclusion"

    def test_read_three_by_index(self, reader):
        results = reader.read_by_index([0, 2, 4])
        assert len(results) == 3
        assert results[0]["name"] == "Introduction"
        assert results[1]["name"] == "Approach"
        assert results[2]["name"] == "Conclusion"

    def test_cross_section_isolation(self, reader):
        """Facts from Appendix should NOT appear in Introduction."""
        intro = reader.read(["Introduction"])
        assert "4 NVIDIA A100" not in intro[0]["text"]
        assert "768-dimensional" not in intro[0]["text"]


# ── 8. LaTeX cleaning ──────────────────────────────────────────

class TestLatexCleaning:
    def test_citations_removed(self):
        assert "foo2024" not in _clean_latex("text \\citep{foo2024} more")

    def test_cite_removed(self):
        assert "bar" not in _clean_latex("see \\cite{bar}")

    def test_math_delimiters_stripped(self):
        result = _clean_latex("loss $L = 0$ done")
        assert "L = 0" in result
        assert "$" not in result

    def test_textbf_unwrapped(self):
        assert "bold text" in _clean_latex("\\textbf{bold text}")

    def test_ref_removed(self):
        result = _clean_latex("see Table \\ref{tab:main}")
        assert "tab:main" not in result

    def test_label_removed(self):
        result = _clean_latex("see \\label{sec:method} here")
        assert "sec:method" not in result


# ── 9. Edge cases ──────────────────────────────────────────────

class TestEdgeCases:
    def test_nonexistent_section_returns_empty(self, reader):
        results = reader.read(["Nonexistent Section"])
        assert len(results) == 0

    def test_empty_keyword_returns_empty(self, reader):
        results = reader.read_by_keyword("zzzznonmatching")
        assert len(results) == 0

    def test_case_insensitive_name_match(self, reader):
        results = reader.read(["introduction"])
        assert len(results) == 1
        assert results[0]["name"] == "Introduction"

    def test_out_of_range_index(self, reader):
        results = reader.read_by_index([999])
        assert len(results) == 0


# ── 10. Section Indexer (requires LLM) ─────────────────────────

def _ask_claude(question: str, context: str) -> str:
    """Ask Claude CLI a question about paper content. No API key needed."""
    import subprocess
    prompt = (
        f"Based ONLY on the following paper excerpt, answer the question in one short sentence.\n\n"
        f"--- EXCERPT ---\n{context}\n--- END ---\n\n"
        f"Question: {question}\nAnswer:"
    )
    result = subprocess.run(
        ["claude", "-p"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"claude CLI failed: {result.stderr}"
    return result.stdout.strip()


class TestLLMQueries:
    """Use Claude CLI to query facts from sections retrieved by the skill.

    Simulates an agent using the skill: read a section, then ask a question.
    Verifies the answer contains the expected fact.

    Run: pytest --run-slow -k slow
    """

    @pytest.mark.slow
    def test_imagenet32_fid(self, reader):
        """Q: What is the FID of self-labeled guidance on ImageNet32?"""
        section = reader.read(["Introduction"])[0]["text"]
        answer = _ask_claude("What FID does self-labeled guidance achieve on ImageNet32?", section)
        assert "7.3" in answer, f"Expected 7.3 in answer: {answer}"

    @pytest.mark.slow
    def test_feature_extractor(self, reader):
        """Q: Which feature extractor performs best?"""
        section = reader.read(["Related Work"])[0]["text"]
        answer = _ask_claude("What is the primary self-supervised feature extractor used?", section)
        assert "dino" in answer.lower(), f"Expected DINO in answer: {answer}"

    @pytest.mark.slow
    def test_cluster_count(self, reader):
        """Q: How many clusters give the best result on ImageNet64?"""
        section = reader.read(["Approach"])[0]["text"]
        answer = _ask_claude("At how many clusters does self-labeled guidance outperform ground-truth on ImageNet64?", section)
        assert "5,000" in answer or "5000" in answer, f"Expected 5000 in answer: {answer}"

    @pytest.mark.slow
    def test_lost_tool(self, reader):
        """Q: What tool is used for unsupervised object localization?"""
        section = reader.read(["Approach"])[0]["text"]
        answer = _ask_claude("What method is used for unsupervised object localization?", section)
        assert "lost" in answer.lower(), f"Expected LOST in answer: {answer}"

    @pytest.mark.slow
    def test_pascal_voc_boxed(self, reader):
        """Q: What is the self-boxed FID on Pascal VOC?"""
        section = reader.read(["Experiments"])[0]["text"]
        answer = _ask_claude("What FID does self-boxed guidance achieve on Pascal VOC?", section)
        assert "18.4" in answer, f"Expected 18.4 in answer: {answer}"

    @pytest.mark.slow
    def test_guidance_strength(self, reader):
        """Q: At what guidance strength w does quality saturate?"""
        section = reader.read(["Experiments"])[0]["text"]
        answer = _ask_claude("At what value of guidance strength w does quality saturate?", section)
        assert "3" in answer, f"Expected 3 in answer: {answer}"

    @pytest.mark.slow
    def test_gpu_count(self, reader):
        """Q: How many GPUs were used for training?"""
        section = reader.read(["Appendix A: Implementation Details"])[0]["text"]
        answer = _ask_claude("How many GPUs were used for training?", section)
        assert "4" in answer, f"Expected 4 in answer: {answer}"

    @pytest.mark.slow
    def test_cross_section_query(self, reader):
        """Query facts that span multiple sections via keyword search."""
        # "guidance" appears in multiple sections
        results = reader.read_by_keyword("approach")
        combined = " ".join(r["text"] for r in results)
        answer = _ask_claude(
            "Name the three types of self-guided approaches (labeled, boxed, segmented tools).",
            combined,
        )
        ans_lower = answer.lower()
        assert "k-means" in ans_lower or "label" in ans_lower, f"Missing labeling: {answer}"

    @pytest.mark.slow
    def test_conclusion_insight(self, reader):
        """Q: What is the key insight from the conclusion?"""
        section = reader.read(["Conclusion"])[0]["text"]
        answer = _ask_claude("What is the key insight about self-supervised features vs categorical labels?", section)
        assert "self-supervised" in answer.lower() or "visual structure" in answer.lower(), (
            f"Expected insight about self-supervised features: {answer}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
