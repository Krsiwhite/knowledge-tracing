"""Synthetic, CPU-only statistical checks; no training or measured model results."""
from dataclasses import replace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ktlab.statistics import RunPredictions, holm_adjust, paired_auc_bootstrap


def fixture_runs():
    # Unequal student lengths and tied scores exercise the cluster weighting.
    users = ["a", "a", "b", "b", "b", "c", "c", "c", "c"]
    events = [0, 1, 0, 1, 2, 0, 1, 2, 3]
    labels = [0, 1, 1, 0, 1, 0, 1, 0, 1]
    first = np.array([0.1, 0.8, 0.4, 0.4, 0.9, 0.6, 0.6, 0.2, 0.8])
    second = np.array([0.3, 0.7, 0.8, 0.8, 0.6, 0.2, 0.5, 0.4, 0.5])
    candidate, baseline = [], []
    for seed in [42, 2026]:
        for fold in [0, 1]:
            candidate.append(RunPredictions(seed, fold, users, events, labels,
                                           first if seed == 42 else second, f"candidate-{seed}-{fold}"))
            baseline.append(RunPredictions(seed, fold, users, events, labels,
                                          second if fold == 0 else 1 - first, f"baseline-{seed}-{fold}"))
    return candidate, baseline


def explicit_student_bootstrap(candidate, baseline, *, n_resamples, random_state):
    """Reference expands sampled student rows and uses sklearn directly."""
    rng = np.random.default_rng(random_state)
    users = np.asarray(candidate[0].user_ids)
    y = np.asarray(candidate[0].y_true)
    students = np.unique(users)
    draws = []
    for _ in range(n_resamples):
        sampled = rng.integers(len(students), size=len(students))
        rows = np.concatenate([np.flatnonzero(users == students[i]) for i in sampled])
        deltas = [roc_auc_score(y[rows], np.asarray(c.probabilities)[rows])
                  - roc_auc_score(y[rows], np.asarray(b.probabilities)[rows])
                  for c, b in zip(candidate, baseline)]
        draws.append([np.mean(deltas), np.mean(deltas[:2]), np.mean(deltas[2:])])
    return np.asarray(draws)


def test_shared_cluster_bootstrap_matches_explicit_student_replication():
    candidate, baseline = fixture_runs()
    result = paired_auc_bootstrap(candidate, baseline, n_resamples=127, random_state=7, family_size=3)
    draws = explicit_student_bootstrap(candidate, baseline, n_resamples=127, random_state=7)
    alpha = 0.05 / 3
    intervals = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    np.testing.assert_allclose(result["confidence_interval"], intervals[:, 0], atol=1e-14)
    for i, row in enumerate(result["per_seed"]):
        np.testing.assert_allclose(row["confidence_interval"], intervals[:, i + 1], atol=1e-14)
    assert result["n_events_per_run"] == 9
    assert result["n_students"] == 3
    assert result["per_comparison_alpha"] == pytest.approx(alpha)
    assert result["family_alpha"] == pytest.approx(0.05)
    assert result["interval_confidence"] == pytest.approx(1 - alpha)
    assert result["multiplicity_correction"] == "bonferroni_on_interval_alpha"
    assert "not_retraining" in result["bootstrap"]["scope"]


def test_bonferroni_family_alpha_equals_pre_adjusted_single_interval():
    candidate, baseline = fixture_runs()
    family = paired_auc_bootstrap(candidate, baseline, n_resamples=64, confidence=0.95, family_size=3)
    single = paired_auc_bootstrap(candidate, baseline, n_resamples=64, confidence=1 - 0.05 / 3)
    np.testing.assert_allclose(family["confidence_interval"], single["confidence_interval"], atol=1e-14)


def test_row_order_run_order_and_id_storage_do_not_change_results():
    candidate, baseline = fixture_runs()
    expected = paired_auc_bootstrap(candidate, baseline, n_resamples=32)
    shuffled = []
    for run in reversed(baseline):
        order = [8, 0, 4, 2, 6, 3, 1, 7, 5]
        shuffled.append(replace(run, user_ids=[run.user_ids[i] for i in order],
                                event_ids=[str(run.event_ids[i]) for i in order],
                                y_true=[run.y_true[i] for i in order],
                                probabilities=[run.probabilities[i] for i in order]))
    assert paired_auc_bootstrap(list(reversed(candidate)), shuffled, n_resamples=32) == expected


def test_estimand_is_mean_run_auc_not_auc_of_ensemble_predictions():
    common = dict(user_ids=["a", "a", "b", "b"], event_ids=[0, 1, 0, 1], y_true=[0, 1, 0, 1])
    candidate = [RunPredictions(42, 0, probabilities=[0, 1, 0, 1], **common),
                 RunPredictions(2026, 0, probabilities=[0.6, 0.4, 0.6, 0.4], **common)]
    baseline = [replace(run, probabilities=[0.5] * 4) for run in candidate]
    result = paired_auc_bootstrap(candidate, baseline, n_resamples=16)
    assert result["auc_delta"] == 0
    assert result["candidate_auc_mean"] == 0.5
    assert roc_auc_score(common["y_true"], np.mean([r.probabilities for r in candidate], axis=0)) == 1
    assert [row["direction"] for row in result["per_seed"]] == ["positive", "negative"]
    assert result["confidence_interval"] == [0, 0]
    assert not result["all_seed_deltas_positive"]


def test_identical_models_have_zero_delta_for_all_resamples():
    candidate, _ = fixture_runs()
    result = paired_auc_bootstrap(candidate, candidate, n_resamples=32)
    assert result["auc_delta"] == 0
    assert result["confidence_interval"] == [0, 0]
    assert all(row["direction"] == "zero" for row in result["per_seed"])


@pytest.mark.parametrize("field,value,match", [
    ("event_ids", [0] * 9, "Duplicate event"),
    ("event_ids", list(range(9)), "Misaligned test event"),
    ("user_ids", ["a"] * 8, "misaligned arrays"),
    ("user_ids", ["a"] * 8 + [None], "nonempty string or integer"),
    ("user_ids", ["a"] * 8 + [" "], "nonempty string or integer"),
    ("y_true", [0, 0, 1, 0, 1, 0, 1, 0, 1], "Label mismatch"),
    ("y_true", [0] * 9, "both binary classes"),
    ("y_true", [0, 1, 2, 0, 1, 0, 1, 0, 1], "both binary classes"),
    ("probabilities", [float("nan")] * 9, "finite and within"),
    ("probabilities", [1.1] * 9, "finite and within"),
    ("probabilities", [[0.5]] * 9, "non-vector"),
    ("seed", True, "seed must be an integer"),
    ("run_id", None, "run_id must be a string"),
])
def test_rejects_invalid_or_unpaired_predictions(field, value, match):
    candidate, baseline = fixture_runs()
    baseline[0] = replace(baseline[0], **{field: value})
    with pytest.raises(ValueError, match=match):
        paired_auc_bootstrap(candidate, baseline, n_resamples=8)


def test_rejects_duplicate_runs_unmatched_runs_and_unbalanced_grids():
    candidate, baseline = fixture_runs()
    with pytest.raises(ValueError, match="Duplicate candidate seed/fold"):
        paired_auc_bootstrap(candidate + [candidate[0]], baseline, n_resamples=8)
    with pytest.raises(ValueError, match="seed/fold sets"):
        paired_auc_bootstrap(candidate, baseline[:-1], n_resamples=8)
    with pytest.raises(ValueError, match="balanced seed x fold grid"):
        paired_auc_bootstrap(candidate[:-1], baseline[:-1], n_resamples=8)
    with pytest.raises(ValueError, match="at least one run"):
        paired_auc_bootstrap([], [], n_resamples=8)


def test_rejects_different_test_events_between_otherwise_paired_runs():
    candidate, baseline = fixture_runs()
    for runs in [candidate, baseline]:
        runs[-1] = replace(runs[-1], event_ids=[9, *runs[-1].event_ids[1:]])
    with pytest.raises(ValueError, match="Misaligned test event sets"):
        paired_auc_bootstrap(candidate, baseline, n_resamples=8)


def test_single_class_cluster_draws_are_reported_and_attempts_bounded():
    run = RunPredictions(42, 0, ["a", "b"], [0, 0], [0, 1], [0.2, 0.8])
    result = paired_auc_bootstrap([run], [run], n_resamples=32, random_state=0)
    assert result["bootstrap"]["rejected_single_class_draws"] > 0
    assert result["bootstrap"]["attempts"] == 32 + result["bootstrap"]["rejected_single_class_draws"]
    with pytest.raises(ValueError, match="valid two-class student resamples"):
        paired_auc_bootstrap([run], [run], n_resamples=2, random_state=0, max_attempts=2)
    single_student = replace(run, user_ids=["a", "a"], event_ids=[0, 1])
    with pytest.raises(ValueError, match="At least two test students"):
        paired_auc_bootstrap([single_student], [single_student], n_resamples=8)


@pytest.mark.parametrize("kwargs", [
    {"n_resamples": 1}, {"n_resamples": 2.5}, {"confidence": 1},
    {"confidence": float("nan")}, {"family_size": 0}, {"family_size": True},
    {"random_state": -1}, {"max_attempts": 1},
])
def test_rejects_invalid_bootstrap_parameters(kwargs):
    candidate, baseline = fixture_runs()
    with pytest.raises(ValueError):
        paired_auc_bootstrap(candidate, baseline, **kwargs)


def test_holm_adjustment_is_monotone_in_rank_and_restores_original_order():
    # Holm's running maximum prevents the second 0.01 having a smaller adjusted p.
    raw = [0.04, 0.01, 0.01, 0.2]
    np.testing.assert_allclose(holm_adjust(raw), [0.08, 0.04, 0.04, 0.2])
    np.testing.assert_allclose(holm_adjust([0, 0.4, 0.8, 1]), [0, 1, 1, 1])
    assert holm_adjust([0.03])[0] == 0.03
    assert raw == [0.04, 0.01, 0.01, 0.2]


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-0.1], [1.1], [[0.1]]])
def test_holm_rejects_invalid_p_values(values):
    with pytest.raises(ValueError, match="p_values"):
        holm_adjust(values)
