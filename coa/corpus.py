"""Corpus identity and versioning.

A verification result is only reproducible if you know *which* corpus it was
scored against. A bare `str` of guideline text carries no identity, so a report
scored against "EAU 2024" and one scored against "EAU 2026" are indistinguishable
after the fact. `Corpus` attaches a pinned `(id, version, source_document)` manifest
plus a content hash, and that fingerprint is stamped into every run report.

A corpus on disk is a directory containing `manifest.json` and a text file:

    my_corpus/
      manifest.json   {"id": "eau-pca", "version": "2024", "source_document": "corpus.txt"}
      corpus.txt      <plain-text guideline excerpts>

JSON is used deliberately (no YAML dependency). A plain `.txt` file is also
accepted and gets a content-derived identity, so the simple case stays simple.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


class CorpusManifest(BaseModel):
    """Pinned identity of a corpus. Unknown keys are ignored for forward-compat."""

    id: str = "ad-hoc"
    version: str = ""
    source_document: str = ""
    description: str = ""


class Corpus:
    """Guideline/source text plus its pinned manifest."""

    def __init__(self, text: str, manifest: CorpusManifest | None = None):
        self._text = text
        self.manifest = manifest or CorpusManifest()

    @property
    def text(self) -> str:
        return self._text

    @property
    def sha(self) -> str:
        """First 12 hex chars of the sha256 of the corpus text — the content pin."""
        return hashlib.sha256(self._text.encode("utf-8")).hexdigest()[:12]

    def fingerprint(self) -> dict[str, str]:
        """The corpus identity stamped into a run report."""
        return {
            "corpus_id": self.manifest.id,
            "corpus_version": self.manifest.version,
            "corpus_source_document": self.manifest.source_document,
            "corpus_sha": self.sha,
        }

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        id: str = "ad-hoc",
        version: str = "",
        source_document: str = "",
        description: str = "",
    ) -> Corpus:
        return cls(
            text,
            CorpusManifest(
                id=id, version=version, source_document=source_document, description=description
            ),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> Corpus:
        """Load a corpus from a directory (manifest.json + text file) or a plain .txt file."""
        p = Path(path)
        if p.is_dir():
            manifest = CorpusManifest(**json.loads((p / "manifest.json").read_text("utf-8")))
            src = manifest.source_document
            text_path = (p / src) if src and (p / src).exists() else (p / "corpus.txt")
            if not text_path.exists():
                raise FileNotFoundError(
                    f"corpus dir {p} has no readable text file "
                    f"(looked for source_document={src!r} and corpus.txt)"
                )
            return cls(text_path.read_text("utf-8"), manifest)
        # A plain text file: give it a content-derived identity so runs stay traceable.
        text = p.read_text("utf-8")
        return cls.from_text(text, id=p.stem, source_document=p.name)


def resolve_corpus_text(corpus: str | Corpus | None) -> str | None:
    """Extract the raw text from either a `str` corpus, a `Corpus`, or None."""
    if corpus is None:
        return None
    if isinstance(corpus, Corpus):
        return corpus.text
    return corpus
