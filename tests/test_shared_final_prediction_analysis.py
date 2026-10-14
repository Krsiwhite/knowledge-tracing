"""Artificial saved files only. No ktlab model/evaluator/torch imports or calls."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("final_saved_analysis", ROOT / "scripts/analyze_shared_final_predictions.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
MSPEC = importlib.util.spec_from_file_location("final_fixture_metrics", ROOT / "scripts/analyze_shared_predictions.py")
M = importlib.util.module_from_spec(MSPEC)
MSPEC.loader.exec_module(M)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value, *, ascii=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=ascii, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=A.FIELDS)
        writer.writeheader(); writer.writerows(rows)


def metrics(rows):
    value = M.metrics_for(np.array([row["label"] for row in rows], dtype=float),
                          np.array([row["probability"] for row in rows], dtype=float))
    value.pop("auc_unavailable_reason")
    return value


def make_fixture(tmp_path, *, datasets=1, epoch=1, single_class=False, two_comparisons=False, pure_class_students=False):
    root = tmp_path.resolve()
    f = SimpleNamespace(root=root, runs={}, manifests={}, row_templates={},
                        plan_path=root / "plan.json", review_path=root / "review.json", release_path=root / "release.json")
    s = {"schema_version": 1, "method": A.METHOD, "reader_sha256": file_sha(Path(A.__file__)),
         "statistics_sha256": file_sha(ROOT / "src/ktlab/statistics.py"),
         "metric_helpers_sha256": file_sha(ROOT / "scripts/analyze_shared_predictions.py"),
         "folds": list(range(5)), "seeds": [42, 2026, 3407], "metrics": A.METRICS,
         "auxiliary_intervals": "none", "dispersion": "sample_std_ddof1_descriptive",
         "multiple_comparisons": "bonferroni_auc_family", "methods": {}, "cohort": [], "comparisons": []}
    for d in range(datasets):
        dataset = f"artificial-{d}"
        base = root / "prepared" / dataset
        # Sentinels exist solely to make accidental raw/mapping/weight reads fail the open allowlist.
        base.mkdir(parents=True)
        (base / "test.csv").write_text("DO NOT READ RAW TEST")
        (base / "keyid2idx.json").write_text('{"DO_NOT_READ_MAPPING":true}')
        manifest = {"dataset": dataset, "data_dir": str(base), "split_audit": {"train_test_disjoint": True},
                    "data_config": {"num_q": 20, "num_c": 8, "maxlen": 200, "folds": list(range(5)), "test_original_file": "test.csv"},
                    "files": {"test.csv": {"sha256": sha("DO NOT READ RAW TEST"), "bytes": 20},
                              "keyid2idx.json": {"sha256": sha('{"DO_NOT_READ_MAPPING":true}'), "bytes": 28}}}
        write(base / "manifest.json", manifest)
        f.manifests[dataset] = manifest
        rows = []
        for student in range(3):
            for ordinal in (1, 2):
                label = 0 if single_class else student % 2 if pure_class_students else ordinal - 1
                uid, event_id = f"PRIVATE-STUDENT-{student}", f"{dataset}:PRIVATE-STUDENT-{student}:{ordinal}"
                rows.append({"uid": uid, "event_id": event_id, "question_id": ordinal, "label": label,
                             "kc_count": ordinal, "target_sha256": sha(dataset + event_id + "target"),
                             "context_sha256": sha(dataset + event_id + "context"), "probability": .5})
        f.row_templates[dataset] = rows
        method_names = ["candidate", "reference", "second"] if two_comparisons else ["candidate", "reference"]
        for method in method_names:
            method_id = f"{dataset}-{method}"
            model = {"candidate": "akt", "reference": "simplekt", "second": "ktst_mean"}[method]
            for seed in s["seeds"]:
                for fold in s["folds"]:
                    run_id = f"{dataset}-{method}-f{fold}-s{seed}"
                    directory = root / "runs" / run_id
                    config = {"model": model, "dataset": dataset, "dataset_manifest": str(base / "manifest.json"),
                              "protocol": A.DEV, "fold": fold, "seed": seed, "initial_validation": True,
                              "debug_smoke": False, "epochs": 50, "patience": 3, "learning_rate": .001,
                              "batch_size": 32, "weight_decay": 0., "max_train_steps": 100 + fold,
                              "max_wall_seconds": 3600, "model_config": {"d_model": 16, "dropout": 0.}, "device": "cpu"}
                    valid = {"protocol": A.DEV, "dataset": dataset, "fold": fold, "split": "valid", "kc_budget": 200,
                             "selected_targets": 3, "eligible_total": 3, "expected_total": 3, "max_targets": None,
                             "selection_limited": False, "selected_for_debug": False, "test_access": False,
                             **{kind + "_sha256": sha(f"{dataset}-{fold}-valid-{kind}") for kind in ("source", "target", "context")}}
                    metadata = {"run_id": run_id, "status": "completed", "protocol": A.DEV, "test_accessed": False,
                                "partial_debug": False, "initial_validation": True, "seed": seed, "valid_event_audit": valid,
                                "checkpoint_selection_policy": A.SELECTION, "implementation_sha256": sha("original trainer"),
                                "model_source_sha256": sha(model), "dataset_manifest_sha256": file_sha(base / "manifest.json"),
                                "checkpoint_sha256": sha("UNREAD WEIGHT " + run_id), "best_epoch": epoch, "best_valid_auc": .75,
                                "best_epoch_completed": epoch != 0, "optimizer_steps": 10,
                                **{"validation_" + kind + "_sha256": valid[kind + "_sha256"] for kind in ("source", "target", "context")}}
                    entry = {"run_id": run_id, "run_dir": str(directory), "dataset": dataset, "method_id": method_id, "fold": fold, "seed": seed,
                             "max_train_steps": config["max_train_steps"]}
                    s["cohort"].append(entry)
                    s["methods"][method_id] = {"label": method, **A._recipe(config)}
                    probabilities = []
                    for row in rows:
                        if method == "reference": p = .5
                        elif seed in (42, 2026): p = .9 if row["label"] else .1
                        else: p = .4 if row["label"] else .6
                        if method == "second": p = 1 - p
                        probabilities.append(p)
                    f.runs[run_id] = {"entry": entry, "directory": directory, "config": config, "metadata": metadata,
                                      "rows": [dict(row, probability=p) for row, p in zip(rows, probabilities)]}
            if method == "reference": continue
            s["comparisons"].append({"comparison_id": f"{dataset}-{method}-vs-reference", "dataset": dataset,
                "candidate_method": method_id, "reference_method": f"{dataset}-reference",
                "pairs": [{"fold": fold, "seed": seed, "candidate_run_id": f"{dataset}-{method}-f{fold}-s{seed}",
                           "reference_run_id": f"{dataset}-reference-f{fold}-s{seed}"} for seed in s["seeds"] for fold in s["folds"]]})
    s["bootstrap"] = {"n_resamples": 2000, "seed": 42, "seed_scheme": "reset_42_per_dataset", "confidence": .95,
                       "family_size": len(s["comparisons"]), "max_attempts": 20000,
                       "single_class_draws": "bounded_redraw_condition_on_two_classes"}
    f.plan = {"schema_version": 1, "protocol": A.FINAL, "run_ids": list(f.runs), "metrics": A.METRICS,
              "accuracy_threshold": .5, "ece_bins": 15, "kc_budget": 200,
              "test_selection": "all_eligible_complete_event_kc200", "batch_size": 32, "max_rows_per_forward": 32,
              "final_statistics": s}
    f.release = {"schema_version": 1, "protocol": A.FINAL, "status": "approved", "user_confirmation": "fictional fixture",
                 "approved_at": "2000-01-01T00:00:00+00:00", "source_sha256": sha("separate evaluation source"),
                 "analysis_plan_file": str(f.plan_path), "compatibility_review_file": str(f.review_path)}
    freeze(f)
    return f


def freeze(f):
    records = {}
    for run_id, run in f.runs.items():
        directory, config, metadata = run["directory"], run["config"], run["metadata"]
        write(directory / "resolved_config.json", config)
        metadata["config_sha256"] = A.canonical(config)
        write(directory / "metadata.json", metadata)
        (directory / "best.pt").write_text("DO NOT READ WEIGHTS")
        records[run_id] = {key: metadata[key] for key in A.RECORD_KEYS}
        records[run_id].update(training_protocol=A.DEV, metadata_sha256=file_sha(directory / "metadata.json"))
    f.release["runs"] = records
    write(f.plan_path, f.plan)
    review = {"schema_version": 1, "status": "approved", "reviewed_by": "fictional reviewer", "statement": "fixture only",
              "training_protocol": A.DEV, "evaluation_protocol": A.FINAL,
              "evaluation_source_sha256": f.release["source_sha256"], "candidates": records}
    write(f.review_path, review)
    f.release.update(analysis_plan_sha256=file_sha(f.plan_path), compatibility_review_sha256=file_sha(f.review_path))
    write(f.release_path, f.release)
    release_sha = file_sha(f.release_path)
    for run_id, run in f.runs.items():
        entry, record = run["entry"], records[run_id]
        manifest = f.manifests[entry["dataset"]]
        base = Path(manifest["data_dir"]); output = run["directory"] / "final_test"
        test_identity = {"run_id": run_id, "checkpoint_sha256": record["checkpoint_sha256"],
                         "dataset_manifest_sha256": record["dataset_manifest_sha256"], "test_source_sha256": manifest["files"]["test.csv"]["sha256"]}
        dataset_identity = {"dataset": entry["dataset"], "test_source_sha256": test_identity["test_source_sha256"],
                            "mapping_sha256": manifest["files"]["keyid2idx.json"]["sha256"]}
        registry = base / ".final-test-attempts"
        run["attempt_path"] = registry / (A.canonical(test_identity) + ".json")
        run["lock_path"] = registry / ("dataset-" + A.canonical(dataset_identity) + ".json")
        write(run["lock_path"], {"schema_version": 1, "dataset_identity": dataset_identity, "release_sha256": release_sha,
            "analysis_plan_sha256": f.release["analysis_plan_sha256"], "compatibility_review_sha256": f.release["compatibility_review_sha256"],
            "evaluation_source_sha256": f.release["source_sha256"], "candidates_sha256": A.canonical(records)})
        start = {"schema_version": 1, "status": "started", "created_at": "2000-01-01T00:00:01+00:00",
                 "test_identity": test_identity, "release_sha256": release_sha,
                 "evaluation_source_sha256": f.release["source_sha256"], "output_directory": str(output),
                 "test_access_attempted": True, "automatic_retry_allowed": False}
        write(run["attempt_path"], start, ascii=True)
        start_sha = file_sha(run["attempt_path"])
        rows = run["rows"]
        audit = {"protocol": A.FINAL, "dataset": entry["dataset"], "split": "test", "fold": -1,
                 "source_path": str(base / "test.csv"), "source_sha256": test_identity["test_source_sha256"],
                 "kc_budget": 200, "selected_targets": len(rows), "eligible_total": len(rows), "expected_total": len(rows),
                 "max_targets": None, "selection_limited": False, "selected_for_debug": False, "test_access": True,
                 "hash_scheme": "ktlab-shared-event-compositional-sha256-v1",
                 "hash_scope": "selected_targets_in_original_csv_and_event_order",
                 "source_bytes": manifest["files"]["test.csv"]["bytes"], "source_students": 3, "split_students": 3, "indexed_students": 3,
                 "canonical_events": 9, "duplicate_kc_positions_removed": 0, "exclusion_counts": {"target_over_budget": 0, "no_history": 3},
                 "no_history_reasons": {"first_event": 3, "immediate_predecessor_exceeds_remaining_budget": 0}, "optimizer_steps": 0,
                 "hash_labels": "target labels excluded; permitted historical responses included in context only",
                 "event_id_scheme": "dataset:uid:original_student_event_ordinal",
                 "question_missing_policy": "None; any model-specific dummy belongs to its adapter",
                 "kc_order": "original occurrence order with within-event duplicates removed",
                 "history_policy": "longest complete contiguous event suffix; history plus query KC count <= 200"}
        for kind in ("target", "context"):
            h = hashlib.sha256(f"ktlab.shared.{kind}-sequence.v1\0".encode())
            for row in f.row_templates[entry["dataset"]]: h.update(bytes.fromhex(row[kind + "_sha256"]))
            audit[kind + "_sha256"] = h.hexdigest()
        result = {**metrics(rows), "protocol": A.FINAL, "split": "test", "quality_metrics_computed": True,
            "selection_limited": False, "test_accessed": True, "evaluation_seconds": .1,
            "model": run["config"]["model"], "implementation_sha256": f.release["source_sha256"],
            "model_source_sha256": record["model_source_sha256"], "model_training_and_checkpoint_selection_not_verified": True,
            **{kind + "_sha256": audit[kind + "_sha256"] for kind in ("source", "target", "context")},
            "run_id": run_id, "training_identity": record, "training_fold": entry["fold"], "training_seed": entry["seed"],
            "evaluation_protocol": A.FINAL, "evaluation_implementation_sha256": f.release["source_sha256"],
            "test_identity": test_identity, "test_release_sha256": release_sha,
            "analysis_plan_sha256": f.release["analysis_plan_sha256"], "compatibility_review_sha256": f.release["compatibility_review_sha256"],
            "attempt_file": str(run["attempt_path"]), "attempt_start_sha256": start_sha,
            "dataset_release_lock_file": str(run["lock_path"]), "dataset_release_lock_sha256": file_sha(run["lock_path"]),
            "original_candidate_receipts_verified": True, "compatibility_review_required_not_independently_proven": True}
        csv_write(output / "predictions.csv", rows)
        sidecar = {"schema_version": 1, "created_at": "2000-01-01T00:00:02+00:00", "dataset_audit": audit,
                   "result": result, "prediction_sha256": file_sha(output / "predictions.csv"), "metric_scope": "released_final_test"}
        write(output / "predictions.csv.manifest.json", sidecar)
        final = dict(result, prediction_file=str(output / "predictions.csv"), prediction_sha256=sidecar["prediction_sha256"],
                     prediction_manifest_sha256=file_sha(output / "predictions.csv.manifest.json"), auc_available=result["auc"] is not None,
                     auc_unavailable_reason=None if result["auc"] is not None else "single_response_class")
        write(output / "result.json", final)
        write(run["attempt_path"], {**start, "status": "completed", "finished_at": "2000-01-01T00:00:03+00:00",
              "attempt_start_sha256": start_sha, "result_sha256": file_sha(output / "result.json"),
              "prediction_sha256": final["prediction_sha256"], "prediction_manifest_sha256": final["prediction_manifest_sha256"]})


def refresh_output(run, *, change=None, rows=None):
    output = run["directory"] / "final_test"
    sidecar, result, attempt = read(output / "predictions.csv.manifest.json"), read(output / "result.json"), read(run["attempt_path"])
    if rows is not None:
        csv_write(output / "predictions.csv", rows)
        sidecar["result"].update(metrics(rows))
    if change: change(sidecar, result, attempt)
    sidecar["prediction_sha256"] = file_sha(output / "predictions.csv")
    write(output / "predictions.csv.manifest.json", sidecar)
    result.update(sidecar["result"], prediction_sha256=sidecar["prediction_sha256"], prediction_manifest_sha256=file_sha(output / "predictions.csv.manifest.json"))
    result.update(auc_available=result["auc"] is not None, auc_unavailable_reason=None if result["auc"] is not None else "single_response_class")
    write(output / "result.json", result)
    attempt.update(result_sha256=file_sha(output / "result.json"), prediction_sha256=result["prediction_sha256"],
                   prediction_manifest_sha256=result["prediction_manifest_sha256"])
    write(run["attempt_path"], attempt)


def forbid_raw_reads(monkeypatch, f):
    original = Path.open
    opened = set()
    forbidden = {Path(m["data_dir"]) / name for m in f.manifests.values() for name in ("test.csv", "keyid2idx.json")}
    forbidden |= {run["directory"] / "best.pt" for run in f.runs.values()}
    def guarded(path, *args, **kwargs):
        assert path.resolve() not in forbidden, "Reader opened raw/mapping/weight sentinel"
        opened.add(path.resolve())
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    return opened


def test_full_three_dataset_15_pair_grid_uses_mean_auc_not_ensemble_and_releases_arrays(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, datasets=3, epoch=0)
    opened = forbid_raw_reads(monkeypatch, f)
    original = A._read_final
    previous, refs = [None], []
    def observe(entry, *args, **kwargs):
        if previous[0] is not None and entry["dataset"] != previous[0]:
            assert all(reference() is None for reference in refs), "Previous dataset score arrays retained"
            refs.clear()
        result = original(entry, *args, **kwargs)
        previous[0] = entry["dataset"]; refs.append(weakref.ref(result[1]))
        return result
    monkeypatch.setattr(A, "_read_final", observe)
    report = A.analyze(f.release_path)
    assert report["complete_release_candidates"] == 90 and len(report["datasets"]) == 3
    for dataset in report["datasets"]:
        assert len(dataset["runs"]) == 30 and dataset["students"] == 3
        comparison = dataset["comparisons"][0]
        result = comparison["primary_auc"]
        assert len(comparison["pairs"]) == 15 and result["n_seeds"] == 3 and result["n_folds"] == 5
        assert result["candidate_auc_mean"] == pytest.approx(2 / 3)
        assert result["auc_delta"] == pytest.approx(1 / 6)
        assert result["family_size"] == 3 and result["per_comparison_alpha"] == pytest.approx(.05 / 3)
        assert result["bootstrap"]["n_resamples"] == 2000
        assert result["bootstrap"]["attempts"] == 2000 and result["bootstrap"]["rejected_single_class_draws"] == 0
        assert result["all_seed_deltas_positive"] is False
        assert comparison["descriptive"]["candidate"]["auc"]["sample_std"] > 0
        assert all(run["selected_checkpoint_stage"] == "initial_untrained" for run in dataset["runs"])
        assert all(run["training_identity"]["best_epoch"] == 0 for run in dataset["runs"])
        # Actual per-fit scores differ: their ensemble would have AUC=1, not 2/3.
        ensemble = np.mean([[.1, .9]] * 10 + [[.6, .4]] * 5, axis=0)
        assert M.metrics_for(np.array([0, 1]), ensemble)["auc"] == 1
        assert dataset["cost_estimate"]["accepted_draw_event_fit_visits"] == 6 * 30 * 2000
    encoded = json.dumps(report, allow_nan=False)
    assert "PRIVATE-STUDENT" not in encoded and "PRIVATE-EVENT" not in encoded
    assert report["model_forward_calls"] == report["gradient_calls"] == report["optimizer_steps"] == 0
    assert report["receipt_scope"]["weights_read_or_verified"] is False
    assert len(opened) > 90


def test_same_dataset_comparisons_share_actual_bounded_redraws(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, two_comparisons=True, pure_class_students=True)
    original_load, original_rng, draw_receipts = A._load_pure, np.random.default_rng, []
    def tracked_rng(seed):
        rng = original_rng(seed)
        receipt = {"seed": seed, "draws": 0, "hash": hashlib.sha256()}
        draw_receipts.append(receipt)
        class Traced:
            def integers(self, *args, **kwargs):
                values = rng.integers(*args, **kwargs)
                receipt["draws"] += 1
                receipt["hash"].update(values.astype(np.int64).tobytes())
                return values
        return Traced()
    class NumpyProxy:
        random = SimpleNamespace(default_rng=tracked_rng)
        def __getattr__(self, key):
            return getattr(np, key)
    def load(*args):
        module = original_load(*args)
        if args[-1] == "_ktlab_frozen_final_statistics": module.np = NumpyProxy()
        return module
    monkeypatch.setattr(A, "_load_pure", load)
    report = A.analyze(f.release_path)
    first, second = [c["primary_auc"] for c in report["datasets"][0]["comparisons"]]
    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["rejected_single_class_draws"] > 0
    assert first["bootstrap"]["attempts"] <= 20000
    assert len(draw_receipts) == 2 and all(receipt["seed"] == 42 for receipt in draw_receipts)
    assert all(receipt["draws"] == first["bootstrap"]["attempts"] for receipt in draw_receipts)
    assert draw_receipts[0]["hash"].digest() == draw_receipts[1]["hash"].digest()
    assert first["auc_delta"] == pytest.approx(-second["auc_delta"])
    np.testing.assert_allclose(first["confidence_interval"], -np.array(second["confidence_interval"])[::-1], atol=1e-14)


def test_single_class_dataset_keeps_all_runs_auxiliary_metrics_and_family(tmp_path):
    f = make_fixture(tmp_path, datasets=2)
    for run in f.runs.values():
        if run["entry"]["dataset"] == "artificial-1":
            for row in run["rows"]: row["label"] = 0
    freeze(f)
    report = A.analyze(f.release_path)
    assert report["bootstrap_plan"]["family_size"] == 2
    first = report["datasets"][0]["comparisons"][0]["primary_auc"]
    assert first["family_size"] == 2 and first["per_comparison_alpha"] == pytest.approx(.05 / 2)
    for dataset in report["datasets"][1:]:
        comparison = dataset["comparisons"][0]
        assert comparison["primary_auc"] is None and comparison["auc_interval_unavailable_reason"] == "single_response_class"
        assert not comparison["positive_adjusted_auc_interval"]
        assert comparison["descriptive"]["delta"]["auc"]["mean"] is None
        assert math_isfinite(comparison["descriptive"]["delta"]["nll"]["mean"])
        assert len(dataset["runs"]) == 30


def math_isfinite(value):
    return bool(np.isfinite(value))


@pytest.mark.parametrize("failure", ["missing_run", "duplicate_run", "missing_pair", "duplicate_pair", "wrong_pair", "missing_dataset",
    "different_seed", "single_fold", "family_shrink", "bootstrap_override", "reader_sha", "method_recipe", "method_config", "training_recipe"])
def test_bad_frozen_plan_fails_before_saved_prediction_read(tmp_path, monkeypatch, failure):
    f = make_fixture(tmp_path)
    s = f.plan["final_statistics"]
    if failure == "missing_run": s["cohort"].pop()
    elif failure == "duplicate_run": s["cohort"].append(copy.deepcopy(s["cohort"][0]))
    elif failure == "missing_pair": s["comparisons"][0]["pairs"].pop()
    elif failure == "duplicate_pair": s["comparisons"][0]["pairs"][-1] = copy.deepcopy(s["comparisons"][0]["pairs"][0])
    elif failure == "wrong_pair": s["comparisons"][0]["pairs"][0]["reference_run_id"] = s["comparisons"][0]["pairs"][1]["reference_run_id"]
    elif failure == "missing_dataset": s["comparisons"][0]["dataset"] = "omitted-dataset"
    elif failure == "different_seed": s["cohort"][0]["seed"] = 7
    elif failure == "single_fold": s["folds"] = [0]
    elif failure == "family_shrink": s["bootstrap"]["family_size"] = 0
    elif failure == "bootstrap_override": s["bootstrap"]["n_resamples"] = 20
    elif failure == "reader_sha": s["reader_sha256"] = "0" * 64
    elif failure == "method_recipe": s["methods"][s["cohort"][0]["method_id"]]["ukt_training_recipe"] = "ukt-no-cl"
    elif failure == "method_config": s["methods"][s["cohort"][0]["method_id"]]["model_config_sha256"] = "0" * 64
    elif failure == "training_recipe": next(iter(f.runs.values()))["config"]["learning_rate"] = .005
    freeze(f)
    original = Path.open
    def guard(path, *args, **kwargs):
        assert path.name != "predictions.csv", "Invalid plan must fail before saved scores"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError): A.analyze(f.release_path)


@pytest.mark.parametrize("failure", ["metadata", "config", "release", "review", "plan", "result", "sidecar", "csv", "lock"])
def test_changed_artifact_bytes_are_rejected(tmp_path, failure):
    f = make_fixture(tmp_path)
    run = next(iter(f.runs.values())); output = run["directory"] / "final_test"
    path = {"metadata": run["directory"] / "metadata.json", "config": run["directory"] / "resolved_config.json",
            "release": f.release_path, "review": f.review_path, "plan": f.plan_path, "result": output / "result.json",
            "sidecar": output / "predictions.csv.manifest.json", "csv": output / "predictions.csv", "lock": run["lock_path"]}[failure]
    if failure == "config":
        value = read(path); value["learning_rate"] = .004; write(path, value)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): A.analyze(f.release_path)


@pytest.mark.parametrize("failure", ["started", "failed", "attempt_path", "output_path", "attempt_start", "source", "training_protocol",
    "partial", "debug_audit", "wrong_fold", "epoch0_completed", "selection_policy", "label", "question", "kc", "context", "target", "order", "missing_event",
    "source_bytes", "students", "event_counts", "history_semantics", "event_id"])
def test_semantic_and_chain_negative_controls(tmp_path, failure):
    f = make_fixture(tmp_path, epoch=0)
    run = list(f.runs.values())[1]  # A preceding valid run supplies the alignment reference.
    if failure in {"started", "failed", "attempt_start", "output_path"}:
        value = read(run["attempt_path"])
        if failure in {"started", "failed"}: value["status"] = failure
        elif failure == "attempt_start": value["attempt_start_sha256"] = "0" * 64
        else: value["output_directory"] = str(f.root / "other-output")
        write(run["attempt_path"], value)
    elif failure in {"epoch0_completed", "selection_policy", "training_protocol"}:
        if failure == "epoch0_completed": run["metadata"]["best_epoch_completed"] = True
        elif failure == "selection_policy": run["metadata"]["checkpoint_selection_policy"] = "last_epoch"
        else: run["metadata"]["protocol"] = A.FINAL
        freeze(f)
    elif failure in {"label", "question", "kc", "context", "target", "order", "missing_event", "event_id"}:
        rows = copy.deepcopy(run["rows"])
        if failure == "label": rows[0]["label"] = 1
        elif failure == "question": rows[0]["question_id"] = 4
        elif failure == "kc": rows[0]["kc_count"] = 3
        elif failure == "context": rows[0]["context_sha256"] = "0" * 64
        elif failure == "target": rows[0]["target_sha256"] = "0" * 64
        elif failure == "event_id": rows[0]["event_id"] = "different-dataset:student:1"
        elif failure == "order": rows.reverse()
        else: rows.pop()
        refresh_output(run, rows=rows)
    else:
        def change(sidecar, result, attempt):
            if failure == "attempt_path": sidecar["result"]["attempt_file"] = str(f.root / "fake-attempt.json")
            elif failure == "source": sidecar["result"]["evaluation_implementation_sha256"] = "0" * 64
            elif failure == "partial": sidecar["dataset_audit"]["selection_limited"] = True
            elif failure == "debug_audit": sidecar["dataset_audit"]["selected_for_debug"] = True
            elif failure == "wrong_fold": sidecar["dataset_audit"]["fold"] = 0
            elif failure == "source_bytes": sidecar["dataset_audit"]["source_bytes"] += 1
            elif failure == "students": sidecar["dataset_audit"]["indexed_students"] = 2
            elif failure == "event_counts": sidecar["dataset_audit"]["canonical_events"] += 1
            elif failure == "history_semantics": sidecar["dataset_audit"]["history_policy"] = "different-window"
        refresh_output(run, change=change)
    with pytest.raises(ValueError): A.analyze(f.release_path)


def test_completed_attempt_cannot_be_replaced_by_a_copied_registry(tmp_path):
    f = make_fixture(tmp_path)
    run = next(iter(f.runs.values()))
    copy_path = f.root / "copied-attempt.json"
    write(copy_path, read(run["attempt_path"]))
    run["attempt_path"].unlink()
    with pytest.raises(FileNotFoundError): A.analyze(f.release_path)


def test_first_csv_order_hash_cannot_be_replaced_with_reordered_bytes(tmp_path):
    f = make_fixture(tmp_path)
    run = next(iter(f.runs.values()))
    refresh_output(run, rows=list(reversed(run["rows"])))
    with pytest.raises(ValueError, match="Ordered target/context digest"): A.analyze(f.release_path)


@pytest.mark.parametrize("model", ["ukt", "mockt"])
def test_frozen_training_variants_keep_recipe_identity_without_models(tmp_path, model):
    f = make_fixture(tmp_path, single_class=True)
    for run in f.runs.values():
        is_candidate = run["entry"]["method_id"].endswith("-candidate")
        config = run["config"]; config["model"] = model
        if model == "ukt":
            config["model_config"].update(use_CL=is_candidate, use_uncertainty_aug=True)
            config["ukt_training_recipe"] = A.UKT_AUG if is_candidate else "ukt-no-cl"
        else:
            config["model_config"].update(emb_type="qid_conv_ker_noexp", kernel_size=4, seq_len=200)
            config["mockt_training_recipe"] = "mockt-target-prefix-branches-v1" if is_candidate else "mockt-three-pass-ground-seq-control-v1"
        run["metadata"]["model_source_sha256"] = sha(model)
        f.plan["final_statistics"]["methods"][run["entry"]["method_id"]] = {"label": run["entry"]["method_id"], **A._recipe(config)}
    freeze(f)
    report = A.analyze(f.release_path)
    assert {method["model"] for method in report["frozen_methods"].values()} == {model}
    field = "ukt_training_recipe" if model == "ukt" else "mockt_training_recipe"
    assert len({method[field] for method in report["frozen_methods"].values()}) == 2


def test_concept_only_saved_question_is_empty_and_fold_step_caps_remain_explicit(tmp_path):
    f = make_fixture(tmp_path, single_class=True)
    for manifest in f.manifests.values():
        manifest["data_config"]["num_q"] = 0
        write(Path(manifest["data_dir"]) / "manifest.json", manifest)
    for run in f.runs.values():
        for row in run["rows"]: row["question_id"] = ""
        run["metadata"]["dataset_manifest_sha256"] = file_sha(Path(run["config"]["dataset_manifest"]))
    freeze(f)
    report = A.analyze(f.release_path)
    assert {run["max_train_steps"] for run in report["datasets"][0]["runs"]} == set(range(100, 105))
    run = next(iter(f.runs.values()))
    run["config"]["max_train_steps"] = 99
    freeze(f)
    with pytest.raises(ValueError, match="step budget"): A.analyze(f.release_path)


def test_windows_initial_attempt_crlf_receipt_and_partial_selected_epoch_are_valid(tmp_path):
    f = make_fixture(tmp_path, single_class=True)
    for run in f.runs.values(): run["metadata"]["best_epoch_completed"] = False
    freeze(f)
    for run in f.runs.values():
        attempt = read(run["attempt_path"])
        start = {key: value for key, value in attempt.items() if key not in {"finished_at", "attempt_start_sha256", "result_sha256", "prediction_sha256", "prediction_manifest_sha256"}}
        start["status"] = "started"
        digest = sha((json.dumps(start, indent=2) + "\n").replace("\n", "\r\n"))
        def change(sidecar, result, terminal):
            sidecar["result"]["attempt_start_sha256"] = digest
            terminal["attempt_start_sha256"] = digest
        refresh_output(run, change=change)
    report = A.analyze(f.release_path)
    assert all(run["selected_checkpoint_stage"] == "after_partial_training_epoch" for run in report["datasets"][0]["runs"])


def test_file_race_is_rejected_before_report_return(tmp_path, monkeypatch):
    f = make_fixture(tmp_path, single_class=True)
    original = A.Receipts.verify
    def race(receipts):
        f.review_path.write_bytes(f.review_path.read_bytes() + b" ")
        original(receipts)
    monkeypatch.setattr(A.Receipts, "verify", race)
    with pytest.raises(ValueError, match="changed during analysis"): A.analyze(f.release_path)


def test_cli_has_no_posthoc_analysis_switches_and_does_not_overwrite(tmp_path):
    f = make_fixture(tmp_path, single_class=True)
    output = tmp_path / "report.json"
    args = ["--release", str(f.release_path), "--output", str(output)]
    assert A.main(args) == 0
    before = output.read_bytes()
    assert A.main(args) == 2 and output.read_bytes() == before
    with pytest.raises(SystemExit): A.main(args + ["--reference", "different"])


def test_loader_executes_captured_verified_bytes_not_changed_disk(tmp_path):
    path = tmp_path / "pure.py"
    path.write_text("VALUE = 'captured'\n")
    receipts = A.Receipts(); original = receipts.read
    def mutate(filename, expected=None):
        content = original(filename, expected)
        path.write_text("raise RuntimeError('unverified disk bytes must never execute')\n")
        return content
    receipts.read = mutate
    module = A._load_pure(path, file_sha(path), receipts, "_pure_capture_fixture")
    assert module.VALUE == "captured"
    with pytest.raises(ValueError, match="changed"): receipts.verify()
