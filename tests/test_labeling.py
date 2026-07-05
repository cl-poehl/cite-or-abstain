"""Validation-set builder tests — cross-categorize, review selection, assemble, ingest."""
from coa.labeling import (
    GeneratedOutput,
    LabelCandidate,
    assemble_cases,
    cross_categorize,
    detect_citation,
    generate_outputs,
    ingest_outputs,
    needs_human_review,
)
from coa.testing import RoutedBackend

CIT = '{"category":"cited","citations":[{"source":"X"}],"rationale":"r"}'
ABS = '{"category":"abstained","rationale":"r"}'
HED = '{"category":"uncited-hedged","rationale":"r"}'


def test_generate_outputs_labels_and_ids():
    backend = RoutedBackend(lambda s, u: "a clinical answer")
    prompts = [{"id": "p1", "prompt": "Q1?"}, {"id": "p2", "prompt": "Q2?"}]
    outs = generate_outputs(prompts, backend, generator_label="m1")
    assert [o.id for o in outs] == ["p1::m1", "p2::m1"]
    assert all(o.output == "a clinical answer" for o in outs)


def test_ungrounded_generation_does_not_inject_corpus():
    seen = {}

    def route(system, user):
        seen["system"] = system
        return "answer"

    generate_outputs([{"id": "p1", "prompt": "Q?"}], RoutedBackend(route))
    assert "GUIDELINE EXCERPTS" not in seen["system"]


def test_grounded_generation_injects_corpus_into_system_prompt():
    seen = {}

    def route(system, user):
        seen["system"] = system
        return "answer"

    generate_outputs(
        [{"id": "p1", "prompt": "Q?"}],
        RoutedBackend(route),
        corpus="§6.4.2 Intensified systemic therapy improves survival.",
    )
    assert "GUIDELINE EXCERPTS" in seen["system"]
    assert "§6.4.2" in seen["system"]  # the corpus text is available for grounded citing


def test_cross_categorize_agreement_and_disagreement():
    outs = [
        GeneratedOutput(id="agree", prompt="q", output="A", generator="g"),
        GeneratedOutput(id="disagree", prompt="q", output="B", generator="g"),
    ]
    # judge A: everything abstained. judge B: 'A'->abstained (agree), 'B'->hedged (disagree).
    judge_a = RoutedBackend(lambda s, u: ABS)
    judge_b = RoutedBackend(lambda s, u: ABS if "OUTPUT TO CATEGORIZE:\n\nA" in u else HED)
    cands = cross_categorize(outs, judge_a, judge_b)
    by_id = {c.id: c for c in cands}
    assert by_id["agree"].agree is True
    assert by_id["agree"].provisional == "abstained"
    assert by_id["disagree"].agree is False
    assert by_id["disagree"].provisional is None


def _cands():
    return [
        LabelCandidate(id="a", prompt="q", output="o", pred_a="cited", pred_b="cited",
                       agree=True, provisional="cited"),
        LabelCandidate(id="b", prompt="q", output="o", pred_a="cited", pred_b="abstained",
                       agree=False, provisional=None),
    ]


def test_needs_human_review_includes_all_disagreements():
    review = needs_human_review(_cands(), audit_frac=0.0)
    assert [c.id for c in review] == ["b"]  # only the disagreement, no audit


def test_needs_human_review_audit_is_seed_deterministic():
    a = needs_human_review(_cands(), audit_frac=1.0, seed=1)
    b = needs_human_review(_cands(), audit_frac=1.0, seed=1)
    assert {c.id for c in a} == {c.id for c in b} == {"a", "b"}  # audit pulls in the agreement too


def test_assemble_uses_provisional_and_human_override():
    cands = _cands()
    # human labels the disagreement, and overrides the provisional on the agreement.
    cases = assemble_cases(cands, {"b": "uncited-confident", "a": "uncited-hedged"})
    by_id = {c["id"]: c for c in cases}
    assert by_id["a"]["expected_category"] == "uncited-hedged"  # override wins
    assert by_id["b"]["expected_category"] == "uncited-confident"


def test_assemble_drops_unadjudicated_disagreements():
    cases = assemble_cases(_cands(), human_labels={})  # no human input
    ids = {c["id"] for c in cases}
    assert ids == {"a"}  # 'b' (disagreement, no provisional, no label) is dropped


def test_detect_citation_signals():
    assert detect_citation("Per EAU 2024 §6.4.2, ADT plus docetaxel.") is True
    assert detect_citation("The SPARTAN trial (Smith et al.) showed benefit.") is True
    assert detect_citation("See PMID 12345678 for details.") is True
    # vague reference and bare clinical acronyms are NOT citations
    assert detect_citation("Studies show that ADT and PSA monitoring help.") is False
    assert detect_citation("Radical prostatectomy is the optimal therapy.") is False


def test_ingest_prefills_cited_and_leaves_others_blank():
    records = [
        {"id": "r1", "prompt": "q1", "output": "Per NCCN guidelines §IRA-3, surveillance."},
        {"id": "r2", "prompt": "q2", "output": "The optimal therapy is radical prostatectomy."},
    ]
    cands = ingest_outputs(records)
    by_id = {c.id: c for c in cands}
    assert by_id["r1"].provisional == "cited"  # citation detected -> pre-filled
    assert by_id["r2"].provisional is None  # no source -> human must label

    # cited pre-fill survives assembly; the blank one is dropped until a human labels it.
    cases = assemble_cases(cands, human_labels={})
    assert {c["id"] for c in cases} == {"r1"}
    # once the human labels r2, it appears with their label.
    cases2 = assemble_cases(cands, {"r2": "uncited-confident"})
    assert {c["id"] for c in cases2} == {"r1", "r2"}
