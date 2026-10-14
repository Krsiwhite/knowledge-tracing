"""Artificial saved CSVs/receipts only: no ktlab, model, weight or data imports."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("saved_prediction_analysis", ROOT / "scripts/analyze_shared_predictions.py")
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text())


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANALYSIS.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rehash_csv(path):
    sidecar_path = path.with_suffix(".csv.manifest.json")
    sidecar = load_json(sidecar_path)
    sidecar["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(sidecar_path, sidecar)


@pytest.fixture
def artifacts(tmp_path):
    rows = [{"uid": uid, "event_id": f"event-{i}", "question_id": i,
             "label": label, "probability": probability, "kc_count": 1 if i % 2 else 2,
             "target_sha256": sha(f"target-{i}"), "context_sha256": sha(f"context-{i}")}
            for i, (uid, label, probability) in enumerate(zip(
                ["PRIVATE-STUDENT-A"] * 3 + ["PRIVATE-STUDENT-B"] * 2 + ["PRIVATE-STUDENT-C"] * 3,
                [0, 1, 0, 1, 0, 1, 0, 1], [.1, .8, .4, .9, .3, .6, .2, .7]))]
    audit = {"protocol": ANALYSIS.PROTOCOL, "dataset": "artificial", "split": "valid", "fold": 0,
             "source_sha256": sha("unread development source"), "kc_budget": 200,
             "selected_targets": len(rows), "expected_total": len(rows), "eligible_total": len(rows),
             "max_targets": None, "selected_for_debug": False, "selection_limited": False,
             "test_access": False, "hash_scheme": "ktlab-shared-event-compositional-sha256-v1",
             "hash_scope": "selected_targets_in_original_csv_and_event_order"}
    for kind in ("target", "context"):
        h = hashlib.sha256(f"ktlab.shared.{kind}-sequence.v1\0".encode())
        for row in rows:
            h.update(bytes.fromhex(row[kind + "_sha256"]))
        audit[kind + "_sha256"] = h.hexdigest()
    entries = []
    for model in ("akt", "ktst_mean"):
        config = {"model": model, "dataset": "artificial", "fold": 0, "seed": 42,
                  "protocol": ANALYSIS.PROTOCOL, "epochs": 2, "max_train_steps": 20,
                  "initial_validation": True, "debug_smoke": False,
                  "dataset_manifest": "/NOT-READ/data/manifest.json", "model_config": {"d_model": 128}}
        entries.append({"run_id": "B-SYNTHETIC-" + model, "config": config,
                        "dataset_manifest_sha256": sha("unread manifest"),
                        "shared_validation": {k: audit[k] for k in ("protocol", "selected_targets", "source_sha256",
                                                                   "target_sha256", "context_sha256")}})
    batch = {"batch_id": "B-SYNTHETIC", "test_access": False, "source_sha256": sha("unread implementation"),
             "runs": entries}
    batch_path = tmp_path / "batch.json"
    write_json(batch_path, batch)
    batch_sha = ANALYSIS.canonical_hash(batch)
    paths = []
    for index, entry in enumerate(entries):
        directory = tmp_path / entry["run_id"]
        directory.mkdir()
        path = directory / "valid_predictions.csv"
        model_rows = copy.deepcopy(rows)
        if index:
            for row in model_rows:
                row["probability"] = .9 * row["probability"] + .05
        write_csv(path, model_rows)
        config = dict(entry["config"], batch_id=batch["batch_id"], approved_batch_sha256=batch_sha)
        write_json(directory / "resolved_config.json", config)
        metrics = ANALYSIS.metrics_for(np.array([r["label"] for r in model_rows], dtype=float),
                                       np.array([r["probability"] for r in model_rows]))
        metrics.pop("auc_unavailable_reason")
        result = dict(metrics, protocol=ANALYSIS.PROTOCOL, split="valid", model=entry["config"]["model"],
                      source_sha256=audit["source_sha256"], target_sha256=audit["target_sha256"],
                      context_sha256=audit["context_sha256"], quality_metrics_computed=True,
                      selection_limited=False, test_accessed=False, implementation_sha256=batch["source_sha256"],
                      model_source_sha256=sha("unread model " + str(index)),
                      model_training_and_checkpoint_selection_not_verified=True)
        write_json(path.with_suffix(".csv.manifest.json"), {"schema_version": 1, "dataset_audit": audit,
                   "result": result, "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "metric_scope": "development_only_not_sota_or_final_test"})
        write_json(directory / "valid_metrics.json", dict(result, partial_debug=False,
                   checkpoint_selection_policy=ANALYSIS.SELECTION))
        metadata = {"protocol": ANALYSIS.PROTOCOL, "run_id": entry["run_id"], "status": "completed",
                    "test_accessed": False, "partial_debug": False, "initial_validation": True,
                    "checkpoint_selection_policy": ANALYSIS.SELECTION, "batch_id": batch["batch_id"],
                    "approved_batch_sha256": batch_sha, "implementation_sha256": batch["source_sha256"],
                    "model_source_sha256": result["model_source_sha256"], "config_sha256": ANALYSIS.canonical_hash(config),
                    "dataset_manifest_sha256": entry["dataset_manifest_sha256"], "checkpoint_sha256": sha("NO WEIGHT FILE" + str(index)),
                    "optimizer_steps": 10, "best_epoch": 1, "best_epoch_completed": True,
                    "best_valid_auc": metrics["auc"], "valid_event_audit": audit,
                    "valid_file_sha256": audit["source_sha256"], "seed": 42}
        for kind in ("source", "target", "context"):
            metadata[f"validation_{kind}_sha256"] = audit[f"{kind}_sha256"]
        write_json(directory / "metadata.json", metadata)
        paths.append(path)
    return paths, batch_path, entries[0]["run_id"]


def analyze(artifacts, **kwargs):
    paths, batch, reference = artifacts
    return ANALYSIS.analyze(paths, batch, reference, replicates=40, seed=42, **kwargs)


def test_full_artifacts_metrics_and_privacy_without_source_or_weights(artifacts, monkeypatch):
    paths, batch, _ = artifacts
    allowed = {batch.resolve()}
    for path in paths:
        allowed.update((path.resolve(), path.with_suffix(".csv.manifest.json").resolve()))
        allowed.update((path.parent / name).resolve() for name in ("metadata.json", "resolved_config.json", "valid_metrics.json"))
    original_open = Path.open
    opened = []
    def checked_open(path, *args, **kwargs):
        assert path.resolve() in allowed, f"Unexpected file read: {path.name}"
        opened.append(path.resolve())
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    report = analyze(artifacts)
    assert set(opened) == allowed
    assert report["events"] == 8 and report["students"] == 3
    assert report["runs"][0]["overall"]["auc"] == 1
    assert report["runs"][0]["by_query_kc_count"]["single_kc"]["n"] == 4
    assert report["runs"][0]["by_query_kc_count"]["multiple_kc"]["auc"] is None  # all negatives
    assert report["receipt_scope"]["training_identity_fully_proven"] is False
    assert not report["test_accessed"] and report["model_forward_calls"] == 0
    encoded = json.dumps(report, allow_nan=False)
    assert "PRIVATE-STUDENT" not in encoded and "event-0" not in encoded and "/NOT-READ" not in encoded
    assert "checkpoint_sha256_recorded" in encoded


def test_pair_by_ids_accepts_reordered_reference(artifacts):
    paths, _, _ = artifacts
    before = analyze(artifacts)
    write_csv(paths[0], list(reversed(read_csv(paths[0]))))
    rehash_csv(paths[0])
    after = analyze(artifacts)
    assert before["bootstrap"] == after["bootstrap"]
    assert before["labels_sha256"] == after["labels_sha256"]
    assert before["runs"][0]["input_files"]["prediction"] != after["runs"][0]["input_files"]["prediction"]


def test_valid_epoch_zero_selection_is_retained(artifacts):
    path = artifacts[0][0].parent / "metadata.json"
    metadata = load_json(path)
    metadata.update(best_epoch=0, best_epoch_completed=False)
    write_json(path, metadata)
    report = analyze(artifacts)
    assert report["runs"][0]["selected_checkpoint_stage"] == "initial_untrained"
    assert report["runs"][0]["optimizer_steps_recorded"] == 10


@pytest.mark.parametrize("key,value", [("status", "running"), ("test_accessed", True),
    ("partial_debug", True), ("optimizer_steps", 0), ("optimizer_steps", 21),
    ("approved_batch_sha256", "a" * 64), ("config_sha256", "b" * 64),
    ("checkpoint_sha256", None), ("best_valid_auc", .51), ("seed", 99), ("seed", 42.0),
    ("checkpoint_selection_policy", "old-v1"), ("best_epoch", 3), ("protocol", "old-v1")])
def test_reject_invalid_training_receipts(artifacts, key, value):
    path = artifacts[0][0].parent / "metadata.json"
    metadata = load_json(path)
    metadata[key] = value
    write_json(path, metadata)
    with pytest.raises(ValueError):
        analyze(artifacts)


@pytest.mark.parametrize("key,value", [("split", "test"), ("split", "train"), ("protocol", "v1"),
    ("selection_limited", True), ("selected_for_debug", True), ("max_targets", 8),
    ("expected_total", 9), ("eligible_total", 9), ("kc_budget", 199), ("test_access", True)])
def test_reject_non_full_v2_valid_before_csv_read(artifacts, monkeypatch, key, value):
    path = artifacts[0][0].with_suffix(".csv.manifest.json")
    metadata = load_json(path)
    metadata["dataset_audit"][key] = value
    write_json(path, metadata)
    original = ANALYSIS.Receipts.read
    def guard(self, path):
        assert Path(path).suffix != ".csv", "Scope must be rejected before predictions are read"
        return original(self, path)
    monkeypatch.setattr(ANALYSIS.Receipts, "read", guard)
    with pytest.raises(ValueError):
        analyze(artifacts)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "label", "context", "question", "kc", "nan", "outside", "stale_hash"])
def test_reject_changed_or_misaligned_csv(artifacts, mutation):
    path = artifacts[0][1]
    rows = read_csv(path)
    if mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "missing":
        rows.pop()
    else:
        field, value = {"label": ("label", "1"), "context": ("context_sha256", "0" * 64),
                        "question": ("question_id", "999"), "kc": ("kc_count", "3"),
                        "nan": ("probability", "NaN"), "outside": ("probability", "1.1"),
                        "stale_hash": ("probability", ".5")}[mutation]
        rows[0][field] = value
    write_csv(path, rows)
    if mutation != "stale_hash":
        rehash_csv(path)
    with pytest.raises(ValueError):
        analyze(artifacts)


def test_aggregate_context_hash_and_saved_metrics_are_checked(artifacts):
    # Equal per-row tampering in both models still cannot bypass the frozen digest.
    for path in artifacts[0]:
        rows = read_csv(path)
        rows[0]["context_sha256"] = "0" * 64
        write_csv(path, rows)
        rehash_csv(path)
    with pytest.raises(ValueError, match="Aggregate"):
        analyze(artifacts)


def test_no_input_retains_frozen_sequence_is_rejected(artifacts):
    for path in artifacts[0]:
        write_csv(path, list(reversed(read_csv(path))))
        rehash_csv(path)
    with pytest.raises(ValueError, match="no input retains the frozen order"):
        analyze(artifacts)


def test_all_single_kc_keeps_empty_multi_kc_stratum(artifacts):
    for path in artifacts[0]:
        rows = read_csv(path)
        for row in rows:
            row["kc_count"] = "1"
        write_csv(path, rows)
        rehash_csv(path)
    report = analyze(artifacts)
    for run in report["runs"]:
        assert run["by_query_kc_count"]["multiple_kc"]["n"] == 0
        assert run["by_query_kc_count"]["multiple_kc"]["auc"] is None


def test_saved_metrics_must_match_probabilities(artifacts):
    path = artifacts[0][0]
    rows = read_csv(path)
    rows[0]["probability"] = ".99"
    write_csv(path, rows)
    rehash_csv(path)
    with pytest.raises(ValueError, match="Saved"):
        analyze(artifacts)


def test_reference_and_minimum_inputs(artifacts):
    paths, batch, reference = artifacts
    for inputs, ref in ((paths[:1], reference), (paths, "unknown"), ([paths[0]] * 2, reference)):
        with pytest.raises(ValueError):
            ANALYSIS.analyze(inputs, batch, ref, replicates=2)


@pytest.mark.parametrize("field,value,expected", [("seed", 2026, "same training seed"), ("model", "akt", "distinct models")])
def test_otherwise_consistent_different_seeds_or_duplicate_models_rejected(artifacts, field, value, expected):
    paths, batch_path, _ = artifacts
    batch = load_json(batch_path)
    batch["runs"][1]["config"][field] = value
    write_json(batch_path, batch)
    batch_sha = ANALYSIS.canonical_hash(batch)
    for index, path in enumerate(paths):
        config_path = path.parent / "resolved_config.json"
        config = load_json(config_path)
        config["approved_batch_sha256"] = batch_sha
        if index == 1:
            config[field] = value
        write_json(config_path, config)
        metadata_path = path.parent / "metadata.json"
        metadata = load_json(metadata_path)
        metadata.update(approved_batch_sha256=batch_sha, config_sha256=ANALYSIS.canonical_hash(config))
        if field == "seed" and index == 1:
            metadata["seed"] = value
        write_json(metadata_path, metadata)
        if field == "model" and index == 1:
            sidecar_path = path.with_suffix(".csv.manifest.json")
            sidecar = load_json(sidecar_path)
            sidecar["result"]["model"] = value
            write_json(sidecar_path, sidecar)
            metrics_path = path.parent / "valid_metrics.json"
            metrics = load_json(metrics_path)
            metrics["model"] = value
            write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match=expected):
        analyze(artifacts)


def test_known_metrics_and_probability_endpoints():
    m = ANALYSIS.metrics_for(np.array([0., 1., 0., 1.]), np.array([.1, .9, .8, .2]))
    assert m["auc"] == .75 and m["accuracy"] == .5
    assert m["brier"] == pytest.approx(.325)
    assert m["nll"] == pytest.approx(-(2 * np.log(.9) + 2 * np.log(.2)) / 4)
    assert m["ece_15bins"] == pytest.approx(.45)
    endpoints = ANALYSIS.metrics_for(np.array([0., 1.]), np.array([1., 0.]))
    assert endpoints["ece_15bins"] == 1 and np.isfinite(endpoints["nll"])
    assert ANALYSIS.metrics_for(np.array([]), np.array([]))["auc_unavailable_reason"] == "empty_stratum"


def test_ranked_auc_matches_independently_expanded_ties():
    y, p, weights = np.array([0., 1., 1., 0., 1.]), np.array([.5, .5, .9, .1, .1]), np.array([2, 3, 0, 1, 2])
    actual = ANALYSIS.RankedAUC(y, p)(weights)
    assert actual == pytest.approx(roc_auc_score(np.repeat(y, weights), np.repeat(p, weights)))


def test_cluster_bootstrap_matches_explicit_student_resampling():
    y = np.array([0., 1., 1., 0., 1., 0.])
    students = np.array([0, 0, 0, 1, 2, 2])  # unequal student event counts
    scores = {"reference": np.array([.2, .6, .6, .7, .8, .1]),
              "candidate": np.array([.5, .8, .8, .2, .7, .3])}
    report = ANALYSIS.paired_bootstrap(y, scores, students, "reference", replicates=80, seed=7)
    rng = np.random.default_rng(7)
    deltas = {key: [] for key in ("auc", "nll", "brier")}
    missing = 0
    for _ in range(80):
        drawn = rng.integers(3, size=3)
        indices = np.concatenate([np.flatnonzero(students == uid) for uid in drawn])
        yy = y[indices]
        if len(np.unique(yy)) == 2:
            deltas["auc"].append(roc_auc_score(yy, scores["candidate"][indices]) - roc_auc_score(yy, scores["reference"][indices]))
        else:
            missing += 1
        for metric in ("nll", "brier"):
            means = []
            for model in ("candidate", "reference"):
                pp = scores[model][indices]
                means.append(float(np.mean((pp - yy) ** 2)) if metric == "brier" else float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log1p(-pp))))
            deltas[metric].append(means[0] - means[1])
    assert report["single_class_auc_draws"] == missing
    for metric in deltas:
        observed = report["comparisons"]["candidate"][metric]
        assert observed["percentile_95_ci"] == pytest.approx(np.quantile(deltas[metric], [.025, .975]))
        assert observed["usable_draws"] == len(deltas[metric])


def test_single_class_draws_counted_and_identical_models_have_zero_deltas():
    y, p = np.array([0., 0., 1.]), np.array([.1, .2, .7])
    scores = {"a": p, "b": p.copy()}
    report = ANALYSIS.paired_bootstrap(y, scores, np.array([0, 0, 1]), "a", replicates=100, seed=42)
    assert 0 < report["single_class_auc_draws"] < 100
    for metric, value in report["comparisons"]["b"].items():
        assert value["percentile_95_ci"] == [0., 0.]
        assert value["usable_draws"] == (100 - report["single_class_auc_draws"] if metric == "auc" else 100)
    assert report == ANALYSIS.paired_bootstrap(y, scores, np.array([0, 0, 1]), "a", replicates=100, seed=42)


def test_no_two_class_resamples_do_not_manufacture_auc_interval():
    y, p = np.array([0., 0.]), np.array([.1, .2])
    report = ANALYSIS.paired_bootstrap(y, {"a": p, "b": p + .1}, np.array([0, 1]), "a", replicates=10)
    assert report["single_class_auc_draws"] == 10
    auc = report["comparisons"]["b"]["auc"]
    assert auc["percentile_95_ci"] is None and auc["usable_draws"] == 0 and auc["delta"] is None
    assert report["comparisons"]["b"]["nll"]["usable_draws"] == 10


def test_bootstrap_work_limit_fails_without_silent_subsampling(monkeypatch):
    monkeypatch.setattr(ANALYSIS, "MAX_BOOTSTRAP_EVENT_VISITS", 1)
    with pytest.raises(ValueError, match="work limit"):
        ANALYSIS.paired_bootstrap(np.array([0., 1.]), {"a": np.array([.1, .9]), "b": np.array([.2, .8])},
                                  np.array([0, 1]), "a", replicates=2)


def test_input_race_is_detected(artifacts, monkeypatch):
    original = ANALYSIS.paired_bootstrap
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = artifacts[0][0].parent / "metadata.json"
        path.write_text(path.read_text() + " ")
        return result
    monkeypatch.setattr(ANALYSIS, "paired_bootstrap", changed)
    with pytest.raises(ValueError, match="changed during"):
        analyze(artifacts)


def test_cli_writes_only_aggregate_and_never_overwrites(artifacts, tmp_path):
    paths, batch, reference = artifacts
    output = tmp_path / "aggregate.json"
    args = ["--predictions", *map(str, paths), "--batch", str(batch), "--reference", reference,
            "--bootstrap-replicates", "20", "--output", str(output)]
    assert ANALYSIS.main(args) == 0
    report = output.read_bytes()
    assert ANALYSIS.main(args) == 2 and output.read_bytes() == report
    invalid = tmp_path / "must-not-exist.json"
    args[-1] = str(invalid)
    paths[0].write_text("tampered")
    assert ANALYSIS.main(args) == 2 and not invalid.exists()
