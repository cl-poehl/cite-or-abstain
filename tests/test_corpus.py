"""Corpus identity / manifest tests — no LLM."""
from pathlib import Path

from coa.corpus import Corpus, CorpusManifest, resolve_corpus_text
from coa.scorer import compile_report, score_case
from coa.testing import RoutedBackend
from coa.types import Case, Category

EXAMPLE_CORPUS_DIR = Path(__file__).parent.parent / "examples" / "corpus"


def test_from_text_fingerprint_is_content_pinned():
    c1 = Corpus.from_text("hello world", id="x", version="1")
    c2 = Corpus.from_text("hello world", id="x", version="1")
    c3 = Corpus.from_text("hello WORLD", id="x", version="1")
    assert c1.fingerprint() == c2.fingerprint()
    assert c1.sha != c3.sha  # different text -> different content pin
    fp = c1.fingerprint()
    assert fp["corpus_id"] == "x"
    assert fp["corpus_version"] == "1"
    assert len(fp["corpus_sha"]) == 12


def test_from_path_directory_reads_manifest():
    c = Corpus.from_path(EXAMPLE_CORPUS_DIR)
    assert isinstance(c.manifest, CorpusManifest)
    assert c.manifest.id == "synthetic-demo"
    assert c.manifest.version == "2024"
    assert "§6.4.2" in c.text


def test_from_path_plain_file_gets_content_identity(tmp_path):
    p = tmp_path / "guideline.txt"
    p.write_text("some guideline text", encoding="utf-8")
    c = Corpus.from_path(p)
    assert c.manifest.id == "guideline"  # filename stem
    assert c.manifest.source_document == "guideline.txt"


def test_resolve_corpus_text_accepts_str_corpus_or_none():
    assert resolve_corpus_text(None) is None
    assert resolve_corpus_text("raw text") == "raw text"
    assert resolve_corpus_text(Corpus.from_text("abc")) == "abc"


def test_score_case_accepts_corpus_object_and_report_stamps_fingerprint():
    corpus = Corpus.from_path(EXAMPLE_CORPUS_DIR)

    def route(system, user):
        if "OUTPUT TO CATEGORIZE" in user:
            return (
                '{"category":"cited","citations":[{"source":"EAU 2024","section":"§6.4.2",'
                '"passage":"Intensified systemic therapy at the time of metastatic diagnosis"}],'
                '"rationale":"r"}'
            )
        return "supports"

    backend = RoutedBackend(route)
    cs = score_case(Case(id="a", prompt="q", output="Per EAU 2024 §6.4.2 ..."), backend, corpus)
    assert cs.category == Category.CITED

    report = compile_report([cs], backend.name, corpus=corpus.fingerprint())
    assert report.corpus["corpus_id"] == "synthetic-demo"
    assert report.corpus["corpus_sha"] == corpus.sha
