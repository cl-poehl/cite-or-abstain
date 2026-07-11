"""Guards for the SDK-free lazy-import contract and Unicode-aware passage matching."""
from __future__ import annotations

import subprocess
import sys
import unicodedata

from coa.verifier import _passage_in_region


def test_import_coa_is_sdk_free():
    # Importing the package must NOT eagerly import the vendor SDKs — they are optional
    # extras, loaded only when a backend is actually accessed. Checked in a fresh
    # subprocess so an earlier test importing openai/anthropic can't mask a regression.
    code = (
        "import sys, coa; "
        "assert 'openai' not in sys.modules, 'openai imported eagerly by `import coa`'; "
        "assert 'anthropic' not in sys.modules, 'anthropic imported eagerly by `import coa`'; "
        "assert coa.SemanticMatcher.__name__ == 'SemanticMatcher'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_lazy_backend_attribute_resolves():
    # The backends are re-exported lazily but must still resolve (openai/anthropic are
    # installed in the dev/test env).
    import coa

    assert coa.OpenAIBackend.__name__ == "OpenAIBackend"
    assert coa.AnthropicBackend.__name__ == "AnthropicBackend"


def test_passage_match_normalizes_unicode_forms():
    # Same text, different Unicode normal forms (composed NFC vs decomposed NFD) must match:
    # a guideline corpus stored in one form should not miss a citation stored in the other.
    passage = "Läsion der Prostata mit Größe unter Schwelle"
    region = unicodedata.normalize("NFD", "Vorher. " + passage + " Nachher.")
    assert passage != region  # decomposed form differs byte-wise
    found, how = _passage_in_region(passage, region, 0.80)
    assert found and how == "substring"


def test_section_token_tier_keeps_non_ascii_words_whole():
    # The scoped token tier uses \w (Unicode) not [a-z0-9], so umlaut/accented content
    # words are kept whole rather than fragmented, and a paraphrase still overlaps.
    passage = "Knochendichte Messung Bisphosphonat Fraktur Risiko"
    region = (
        "§13.2.2 Osteoprotektion. Empfohlen wird eine Messung der Knochendichte, "
        "Bisphosphonat bei erhöhtem Fraktur-Risiko, plus Kalzium."
    )
    found, how = _passage_in_region(passage, region, 0.999, scoped=True)
    assert found and how == "section-token"
