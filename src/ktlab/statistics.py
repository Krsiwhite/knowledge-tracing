"""Paired AUC analysis of frozen models on one shared set of test students.

``paired_auc_bootstrap(candidate_runs, baseline_runs, family_size=3)`` compares
the mean of run-level AUCs on a balanced seed x fold grid. It never averages
predictions. Runs are paired by (seed, fold), and rows by (user_id, event_id).
All runs must cover the same test events and labels. IDs are normalized to text.

The percentile interval resamples students, sharing their multiplicities across
both models and every run. It measures test-student sampling uncertainty
conditional on these already trained models, NOT retraining uncertainty. Seeds
are descriptive repeated fits, not independent samples of test students.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class RunPredictions:
    """One frozen run; event IDs need only be unique within a student.

    Arrays must be one dimensional and aligned. ``probabilities`` are the final
    question-event probabilities, after the evaluation protocol's KC aggregation.
    ``run_id`` is optional provenance; pairing uses ``seed`` and ``fold``.
    """

    seed: int
    fold: int
    user_ids: Sequence[str | int]
    event_ids: Sequence[str | int]
    y_true: Sequence[int]
    probabilities: Sequence[float]
    run_id: str = ""


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _identifier(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{name} must contain nonempty string or integer IDs")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{name} must contain nonempty string or integer IDs")
    return result


def _validated_runs(runs, side):
    validated = {}
    for run in runs:
        if not isinstance(run, RunPredictions):
            raise ValueError(f"{side} entries must be RunPredictions")
        if not isinstance(run.run_id, str):
            raise ValueError("run_id must be a string provenance identifier")
        key = (_integer(run.seed, "seed"), _integer(run.fold, "fold"))
        if key in validated:
            raise ValueError(f"Duplicate {side} seed/fold: {key}")
        users = np.asarray(run.user_ids, dtype=object)
        events = np.asarray(run.event_ids, dtype=object)
        y = np.asarray(run.y_true, dtype=np.float64)
        p = np.asarray(run.probabilities, dtype=np.float64)
        arrays = (users, events, y, p)
        if any(a.ndim != 1 for a in arrays) or not y.size or any(a.size != y.size for a in arrays):
            raise ValueError(f"Empty, non-vector or misaligned arrays in {side} run {key}")
        if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
            raise ValueError(f"Labels in {side} run {key} must contain both binary classes")
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Probabilities in {side} run {key} must be finite and within [0, 1]")
        keys = [(_identifier(u, "user_ids"), _identifier(e, "event_ids"))
                for u, e in zip(users, events)]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate event keys in {side} run {key}")
        order = sorted(range(len(keys)), key=keys.__getitem__)
        validated[key] = (tuple(keys[i] for i in order), y[order], p[order], run.run_id)
    if not validated:
        raise ValueError(f"{side} must contain at least one run")
    return validated


class _RankedAUC:
    """Weighted Mann-Whitney AUC; cache sort/ties for repeated cluster weights."""

    def __init__(self, y, p):
        self.order = np.argsort(p, kind="stable")
        sorted_p = p[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
        self.labels = y[self.order]

    def __call__(self, weights):
        sorted_weights = weights[self.order]
        positives = np.add.reduceat(sorted_weights * self.labels, self.starts)
        negatives = np.add.reduceat(sorted_weights * (1 - self.labels), self.starts)
        # Half credit for equal scores; zero-weight events do not contribute.
        concordance = np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives))
        return float(concordance / (positives.sum() * negatives.sum()))


def paired_auc_bootstrap(
    candidate_runs: Sequence[RunPredictions],
    baseline_runs: Sequence[RunPredictions],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    family_size: int = 1,
    random_state: int = 42,
    max_attempts: int | None = None,
) -> dict:
    """Return JSON-compatible mean AUC differences and student-bootstrap CIs.

    ``confidence`` is the desired family confidence; ``family_size=3`` allocates
    alpha/3 to each predeclared primary dataset comparison (Bonferroni intervals,
    not Holm). Equivalently, one comparison can use confidence=1-0.05/3 and
    family_size=1. The caller must declare the family before inspecting results.
    Percentile bootstrap coverage is approximate, including after adjustment.

    Every seed must have the same folds. All runs share the same test events;
    out-of-fold predictions with different test sets are unsupported. Row order
    and run order may differ between inputs. AUC weights events, while the
    resampling unit is students. Student draws that lack either class have
    undefined AUC and are redrawn, counted, and bounded by ``max_attempts``.
    The resulting interval is conditional on a draw containing both classes.

    Per-seed intervals/directions are descriptive diagnostics, not extra primary
    hypothesis tests. No bootstrap-null p-value or retraining CI is inferred.
    """
    n_resamples = _integer(n_resamples, "n_resamples", 2)
    family_size = _integer(family_size, "family_size", 1)
    random_state = _integer(random_state, "random_state")
    if isinstance(confidence, (bool, np.bool_)) or not isinstance(confidence, Real) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and strictly between 0 and 1")
    max_attempts = (max(1000, n_resamples * 10) if max_attempts is None
                    else _integer(max_attempts, "max_attempts", n_resamples))
    candidate = _validated_runs(candidate_runs, "candidate")
    baseline = _validated_runs(baseline_runs, "baseline")
    if candidate.keys() != baseline.keys():
        raise ValueError("Candidate/baseline seed/fold sets are misaligned")
    run_keys = sorted(candidate)
    seeds = sorted({key[0] for key in run_keys})
    folds = sorted({key[1] for key in run_keys})
    if set(run_keys) != {(seed, fold) for seed in seeds for fold in folds}:
        raise ValueError("Runs must form a complete, balanced seed x fold grid")
    event_keys, y, _, _ = candidate[run_keys[0]]
    for key in run_keys:
        for side, collection in (("candidate", candidate), ("baseline", baseline)):
            keys, labels, _, _ = collection[key]
            if keys != event_keys:
                raise ValueError(f"Misaligned test event sets in {side} run {key}")
            if not np.array_equal(labels, y):
                raise ValueError(f"Label mismatch in {side} run {key}")
    students = sorted({key[0] for key in event_keys})
    if len(students) < 2:
        raise ValueError("At least two test students are required")
    student_lookup = {student: i for i, student in enumerate(students)}
    event_student = np.array([student_lookup[key[0]] for key in event_keys], dtype=np.int64)
    scorers = [(_RankedAUC(y, candidate[key][2]), _RankedAUC(y, baseline[key][2])) for key in run_keys]
    per_run = []
    for seed, fold in run_keys:
        candidate_auc = float(roc_auc_score(y, candidate[(seed, fold)][2]))
        baseline_auc = float(roc_auc_score(y, baseline[(seed, fold)][2]))
        per_run.append({"seed": seed, "fold": fold,
                        "candidate_run_id": candidate[(seed, fold)][3],
                        "baseline_run_id": baseline[(seed, fold)][3],
                        "candidate_auc": candidate_auc, "baseline_auc": baseline_auc,
                        "auc_delta": candidate_auc - baseline_auc})
    seed_columns = [np.array([i for i, key in enumerate(run_keys) if key[0] == seed]) for seed in seeds]
    samples = np.empty((n_resamples, len(seeds) + 1), dtype=np.float64)
    rng = np.random.default_rng(random_state)
    accepted = attempts = 0
    while accepted < n_resamples and attempts < max_attempts:
        attempts += 1
        multiplicities = np.bincount(rng.integers(len(students), size=len(students)), minlength=len(students))
        weights = multiplicities[event_student].astype(np.float64)
        positives = float(np.dot(weights, y))
        if positives == 0 or positives == weights.sum():
            continue
        deltas = np.array([c(weights) - b(weights) for c, b in scorers])
        samples[accepted] = [float(deltas.mean()), *[float(deltas[cols].mean()) for cols in seed_columns]]
        accepted += 1
    if accepted != n_resamples:
        raise ValueError(f"Only {accepted}/{n_resamples} valid two-class student resamples after "
                         f"{attempts} attempts; test student/class support is insufficient")
    family_alpha = 1 - float(confidence)
    per_comparison_alpha = family_alpha / family_size
    intervals = np.quantile(samples, [per_comparison_alpha / 2, 1 - per_comparison_alpha / 2], axis=0, method="linear")
    per_seed = []
    for i, (seed, columns) in enumerate(zip(seeds, seed_columns)):
        delta = float(np.mean([per_run[j]["auc_delta"] for j in columns]))
        per_seed.append({"seed": seed, "n_folds": len(folds), "auc_delta": delta,
                         "direction": "positive" if delta > 0 else "negative" if delta < 0 else "zero",
                         "confidence_interval": intervals[:, i + 1].tolist(),
                         "interpretation": "descriptive_fixed_model_student_sampling"})
    return {
        "estimand": "mean_of_matched_run_auc_differences",
        "candidate_auc_mean": float(np.mean([row["candidate_auc"] for row in per_run])),
        "baseline_auc_mean": float(np.mean([row["baseline_auc"] for row in per_run])),
        "auc_delta": float(np.mean([row["auc_delta"] for row in per_run])),
        "confidence_interval": intervals[:, 0].tolist(),
        "family_confidence": float(confidence), "family_size": family_size,
        "family_alpha": family_alpha, "per_comparison_alpha": per_comparison_alpha,
        "interval_confidence": 1 - per_comparison_alpha,
        "multiplicity_correction": "bonferroni_on_interval_alpha" if family_size > 1 else "none",
        "n_runs": len(run_keys), "n_seeds": len(seeds), "n_folds": len(folds),
        "n_students": len(students), "n_events_per_run": len(event_keys),
        "all_seed_deltas_positive": all(row["auc_delta"] > 0 for row in per_seed),
        "per_run": per_run, "per_seed": per_seed,
        "bootstrap": {
            "method": "percentile", "resampling_unit": "student", "shared_weights_across_runs_and_models": True,
            "n_resamples": n_resamples, "random_state": random_state,
            "attempts": attempts, "rejected_single_class_draws": attempts - accepted,
            "scope": "fixed_trained_models_test_student_sampling_not_retraining_uncertainty",
            "conditioning": "resampled_events_contain_both_binary_classes",
            "seed_variation": "descriptive_repeated_fits_not_independent_test_samples",
        },
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, restored to the supplied order.

    Requires valid raw p-values for a predeclared family. This utility does not
    manufacture p-values from bootstrap percentile intervals or repeated seeds.
    Reject at a chosen family alpha when the returned value is <= alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_values must be a nonempty vector of finite values within [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(p.size, 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result
