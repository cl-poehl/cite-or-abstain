"""Wilson interval + Gwet AC1 tests."""
from coa.stats import gwet_ac1, wilson_ci


def test_zero_n_is_degenerate():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_all_success_has_nontrivial_lower_bound():
    lo, hi = wilson_ci(1, 1)
    assert hi == 1.0
    assert 0.0 < lo < 1.0  # Wilson does not collapse to [1, 1] the way Wald would


def test_zero_events_has_nontrivial_upper_bound():
    lo, hi = wilson_ci(0, 137)
    assert lo == 0.0
    assert 0.0 < hi < 0.05  # a 0/137 never-event count still carries a reportable upper bound


def test_interval_brackets_point_estimate():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi


def test_bounds_are_clamped():
    lo, hi = wilson_ci(10, 10)
    assert lo >= 0.0
    assert hi <= 1.0


def test_gwet_ac1_perfect_agreement():
    pairs = [("cited", "cited"), ("abstained", "abstained")]
    assert gwet_ac1(pairs) == 1.0


def test_gwet_ac1_empty_is_zero():
    assert gwet_ac1([]) == 0.0


def test_gwet_ac1_below_one_on_disagreement():
    pairs = [("cited", "cited"), ("cited", "abstained")]
    assert gwet_ac1(pairs) < 1.0


def test_gwet_ac1_robust_to_skew():
    """Under heavy prevalence of one class, AC1 stays high where κ would collapse."""
    # 19/20 agree on 'cited' (skewed), 1 disagreement.
    pairs = [("cited", "cited")] * 19 + [("cited", "abstained")]
    ac1 = gwet_ac1(pairs, categories=["cited", "uncited-confident", "uncited-hedged", "abstained"])
    assert ac1 > 0.9  # high agreement is reflected, not paradoxically deflated
