"""Saved artificial files only; no torch, model, checkpoint or real dataset."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("shared_batch_summary", ROOT / "scripts/summarize_shared_batch.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
A = SUMMARY.analysis


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_history(directory, rows):
    (directory / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_history(directory):
    return [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]


def change_json(path, **changes):
    value = read_json(path)
    value.update(changes)
    write_json(path, value)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=A.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def alter_csv(directory, mutate, *, rehash=True):
    path = directory / "valid_predictions.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    write_csv(path, rows)
    if rehash:
        change_json(path.with_suffix(".csv.manifest.json"), prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def artifacts(tmp_path):
    rows = [{"uid": "PRIVATE-STUDENT-" + str(i // 2), "event_id": f"PRIVATE-EVENT-{i}",
             "question_id": i, "label": label, "probability": probability, "kc_count": 1 + i % 2,
             "target_sha256": hashed(f"target-{i}"), "context_sha256": hashed(f"context-{i}")}
            for i, (label, probability) in enumerate(zip([0, 1, 0, 1], [.1, .8, .7, .6]))]
    audit = {"protocol": A.PROTOCOL, "dataset": "artificial", "split": "valid", "fold": 0,
             "source_sha256": hashed("unread original source"), "kc_budget": 200,
             "selected_targets": 4, "expected_total": 4, "eligible_total": 4,
             "max_targets": None, "selected_for_debug": False, "selection_limited": False, "test_access": False,
             "hash_scheme": "ktlab-shared-event-compositional-sha256-v1",
             "hash_scope": "selected_targets_in_original_csv_and_event_order"}
    for kind in ("target", "context"):
        hasher = hashlib.sha256(f"ktlab.shared.{kind}-sequence.v1\0".encode())
        for row in rows:
            hasher.update(bytes.fromhex(row[kind + "_sha256"]))
        audit[kind + "_sha256"] = hasher.hexdigest()
    entries = []
    for model in ("akt", "ktst_mean"):
        config = {"protocol": A.PROTOCOL, "model": model, "dataset": "artificial", "fold": 0, "seed": 42,
                  "epochs": 2, "patience": 3, "early_stop_min_delta": 0.0, "max_train_steps": 10,
                  "max_wall_seconds": 60, "initial_validation": True, "debug_smoke": False,
                  "dataset_manifest": "/DO-NOT-READ/private-data/manifest.json", "model_config": {"d_model": 16}}
        entries.append({"run_id": "B-SYNTHETIC-" + model, "config": config,
                        "dataset_manifest_sha256": hashed("unread manifest"),
                        "shared_validation": {key: audit[key] for key in ("protocol", "selected_targets", "source_sha256",
                                                                          "target_sha256", "context_sha256")}})
    batch = {"batch_id": "B-SYNTHETIC", "test_access": False, "source_sha256": hashed("unread implementation"),
             "limits": {"total_optimizer_steps_upper_bound": 20}, "runs": entries,
             "private_note": "PRIVATE-AUTHORIZATION-TEXT"}
    batch_path = tmp_path / "batch.json"
    write_json(batch_path, batch)
    batch_sha = A.canonical_hash(batch)
    directories = []
    for index, entry in enumerate(entries):
        directory = tmp_path / "runs" / entry["run_id"]
        directory.mkdir(parents=True)
        config = dict(entry["config"], batch_id=batch["batch_id"], approved_batch_sha256=batch_sha)
        write_json(directory / "resolved_config.json", config)
        path = directory / "valid_predictions.csv"
        model_rows = copy.deepcopy(rows)
        if index:
            for row in model_rows:
                row["probability"] = .9 * row["probability"] + .05
        write_csv(path, model_rows)
        scores = A.metrics_for(np.array([r["label"] for r in model_rows], dtype=float),
                               np.array([r["probability"] for r in model_rows]))
        scores.pop("auc_unavailable_reason")
        result = dict(scores, protocol=A.PROTOCOL, split="valid", model=config["model"],
                      source_sha256=audit["source_sha256"], target_sha256=audit["target_sha256"],
                      context_sha256=audit["context_sha256"], quality_metrics_computed=True,
                      selection_limited=False, test_accessed=False, implementation_sha256=batch["source_sha256"],
                      model_source_sha256=hashed("unread model " + str(index)),
                      model_training_and_checkpoint_selection_not_verified=True)
        write_json(path.with_suffix(".csv.manifest.json"), {"schema_version": 1, "dataset_audit": audit,
                   "result": result, "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "metric_scope": "development_only_not_sota_or_final_test"})
        metrics = dict(result, partial_debug=False, checkpoint_selection_policy=A.SELECTION)
        write_json(directory / "valid_metrics.json", metrics)
        initial = dict(metrics, auc=.5, candidate_epoch=0, optimizer_steps=0, training_started=False,
                       epoch_completed=False, validation_seconds=1.)
        write_json(directory / "initial_valid_metrics.json", initial)
        write_history(directory, [{"epoch": epoch, "optimizer_steps": epoch * 2, "train_loss": .6,
                                   "valid": metrics, "elapsed_seconds": epoch * 4., "epoch_completed": True,
                                   "train_seconds": 2., "validation_seconds": 1.} for epoch in (1, 2)])
        metadata = {"protocol": A.PROTOCOL, "run_id": entry["run_id"], "status": "completed",
                    "test_accessed": False, "partial_debug": False, "initial_validation": True,
                    "checkpoint_selection_policy": A.SELECTION, "batch_id": batch["batch_id"],
                    "approved_batch_sha256": batch_sha, "implementation_sha256": batch["source_sha256"],
                    "model_source_sha256": result["model_source_sha256"], "config_sha256": A.canonical_hash(config),
                    "dataset_manifest_sha256": entry["dataset_manifest_sha256"],
                    "checkpoint_sha256": hashed("NO CHECKPOINT FILE " + str(index)),
                    "optimizer_steps": 4, "best_epoch": 1, "best_epoch_completed": True,
                    "best_valid_auc": scores["auc"], "valid_event_audit": audit,
                    "valid_file_sha256": audit["source_sha256"], "seed": 42,
                    "initial_validation_seconds": 1., "elapsed_seconds": 10., "stop_reason": "epochs_completed",
                    "started_at": f"2026-10-15T00:0{index}:00+00:00", "finished_at": f"2026-10-15T00:0{index}:12+00:00",
                    "peak_gpu_allocated_bytes": 128, "private_note": "PRIVATE-AUTHORIZATION-TEXT"}
        if index == 0:  # KTST legitimately lacks reserved-peak / ended_at fields.
            metadata.update(peak_gpu_reserved_bytes=256, ended_at=metadata["finished_at"])
        for kind in ("source", "target", "context"):
            metadata[f"validation_{kind}_sha256"] = audit[f"{kind}_sha256"]
        write_json(directory / "metadata.json", metadata)
        directories.append(directory)
    return tmp_path, batch_path, directories


def summary(artifacts):
    root, batch, _ = artifacts
    return SUMMARY.summarize(root, batch)


def update_configs(artifacts, **changes):
    _, batch_path, directories = artifacts
    batch = read_json(batch_path)
    for entry in batch["runs"]:
        entry["config"].update(changes)
    write_json(batch_path, batch)
    batch_sha = A.canonical_hash(batch)
    for directory in directories:
        config_path = directory / "resolved_config.json"
        config = read_json(config_path)
        config.update(changes, approved_batch_sha256=batch_sha)
        write_json(config_path, config)
        change_json(directory / "metadata.json", approved_batch_sha256=batch_sha, config_sha256=A.canonical_hash(config))


def test_complete_batch_public_allowlist_and_only_named_saved_files_read(artifacts, monkeypatch):
    _, batch, directories = artifacts
    allowed = {batch.resolve()}
    for directory in directories:
        allowed.update(path.resolve() for path in directory.iterdir())
    original = Path.open
    opened = set()
    def guarded(path, *args, **kwargs):
        assert path.resolve() in allowed, f"Unexpected input read: {path.name}"
        opened.add(path.resolve())
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    report = summary(artifacts)
    assert opened == allowed
    assert report["runs_expected"] == report["runs_completed"] == 2
    assert report["total_optimizer_steps"] == 8 and report["sum_loop_elapsed_seconds"] == 20
    assert report["recorded_batch_time_window_seconds"] == 72
    assert report["groups"][0]["paired_events"] == 4 and report["groups"][0]["paired_students"] == 2
    assert report["runs"][0]["selected_epoch"] == 1 and report["ece_bins"] == 15
    assert report["runs"][1]["peak_gpu_reserved_bytes"] is None
    assert report["receipt_scope"]["training_identity_fully_proven"] is False
    encoded = json.dumps(report)
    for private in ("PRIVATE-STUDENT", "PRIVATE-EVENT", "PRIVATE-AUTHORIZATION", "/DO-NOT-READ", str(artifacts[0])):
        assert private not in encoded
    assert report["model_loads"] == report["model_forward_calls"] == report["new_optimizer_steps"] == 0


def test_epoch_zero_remains_selected_on_all_ties(artifacts):
    for directory in artifacts[2]:
        metadata = read_json(directory / "metadata.json")
        change_json(directory / "initial_valid_metrics.json", auc=metadata["best_valid_auc"])
        change_json(directory / "metadata.json", best_epoch=0, best_epoch_completed=False)
    report = summary(artifacts)
    assert all(run["selected_epoch"] == 0 and run["selected_checkpoint_stage"] == "initial_untrained" for run in report["runs"])
    assert all(run["terminal_stale_epochs"] == 2 for run in report["runs"])


def test_tied_later_epoch_cannot_replace_earliest_strict_best(artifacts):
    change_json(artifacts[2][0] / "metadata.json", best_epoch=2)
    with pytest.raises(ValueError, match="earliest strict best"):
        summary(artifacts)


@pytest.mark.parametrize("failure", ["running", "missing_run", "missing_initial", "missing_history"])
def test_unfinished_or_missing_evidence_does_not_emit_report(artifacts, failure):
    root, batch, directories = artifacts
    directory = directories[-1]
    if failure == "running":
        change_json(directory / "metadata.json", status="running")
    else:
        filename = {"missing_run": "metadata.json", "missing_initial": "initial_valid_metrics.json",
                    "missing_history": "history.jsonl"}[failure]
        (directory / filename).unlink()
    output = root / "public-report.json"
    assert SUMMARY.main(["--repo-root", str(root), "--batch", str(batch), "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", ["csv_bytes", "label", "context", "missing_target", "duplicate_target", "no_frozen_order"])
def test_tampered_or_unpaired_predictions_fail(artifacts, failure):
    directory = artifacts[2][1]
    if failure == "csv_bytes":
        alter_csv(directory, lambda rows: rows[0].update(probability=".4"), rehash=False)
    elif failure == "no_frozen_order":
        for directory in artifacts[2]:
            alter_csv(directory, lambda rows: rows.reverse())
    else:
        def change(rows):
            if failure == "label":
                rows[0]["label"] = "1"
            elif failure == "context":
                rows[0]["context_sha256"] = "0" * 64
            elif failure == "missing_target":
                rows.pop()
            else:
                rows[-1] = rows[0]
        alter_csv(directory, change)
    with pytest.raises(ValueError):
        summary(artifacts)


@pytest.mark.parametrize("failure", ["terminal_steps", "epoch_gap", "history_context", "selected_metric", "partial_without_cap", "premature_stop"])
def test_history_and_terminal_receipts_fail_closed(artifacts, failure):
    directory = artifacts[2][0]
    history = read_history(directory)
    if failure == "terminal_steps":
        change_json(directory / "metadata.json", optimizer_steps=5)
    elif failure == "premature_stop":
        update_configs(artifacts, epochs=3)
    else:
        if failure == "epoch_gap":
            history[-1]["epoch"] = 3
        elif failure == "history_context":
            history[-1]["valid"]["context_sha256"] = "0" * 64
        elif failure == "selected_metric":
            history[0]["valid"]["nll"] += .1
        else:
            history[-1]["epoch_completed"] = False
        write_history(directory, history)
    with pytest.raises(ValueError):
        summary(artifacts)


def test_full_final_epoch_at_step_cap_is_not_truncation(artifacts):
    update_configs(artifacts, max_train_steps=4)
    for directory in artifacts[2]:
        change_json(directory / "metadata.json", stop_reason="max_train_steps")
    report = summary(artifacts)
    assert all(run["epochs_completed"] == 2 and run["stop_reason"] == "max_train_steps" for run in report["runs"])


def test_early_stopping_has_priority_over_simultaneous_step_cap(artifacts):
    update_configs(artifacts, epochs=50, patience=1, max_train_steps=4)
    for directory in artifacts[2]:
        change_json(directory / "metadata.json", stop_reason="early_stopping")
    assert all(run["stop_reason"] == "early_stopping" for run in summary(artifacts)["runs"])
    change_json(artifacts[2][0] / "metadata.json", stop_reason="max_train_steps")
    with pytest.raises(ValueError, match="stop reason"):
        summary(artifacts)


def test_soft_wall_excess_is_described_without_rejecting_completed_evidence(artifacts):
    change_json(artifacts[2][0] / "metadata.json", elapsed_seconds=60.01)
    run = summary(artifacts)["runs"][0]
    assert run["soft_wall_limit_exceeded"] is True and run["soft_wall_seconds_upper_bound"] == 60


def test_duplicate_manifest_run_and_insufficient_total_budget_fail(artifacts):
    _, batch_path, _ = artifacts
    batch = read_json(batch_path)
    batch["runs"].append(copy.deepcopy(batch["runs"][0]))
    write_json(batch_path, batch)
    with pytest.raises(ValueError, match="Duplicate run"):
        summary(artifacts)
    batch["runs"].pop()
    batch["limits"]["total_optimizer_steps_upper_bound"] = 19
    write_json(batch_path, batch)
    with pytest.raises(ValueError, match="budgets exceed"):
        summary(artifacts)


def test_cli_writes_once_and_refuses_overwrite(artifacts):
    root, batch, _ = artifacts
    output = root / "public-report.json"
    arguments = ["--repo-root", str(root), "--batch", str(batch), "--output", str(output)]
    assert SUMMARY.main(arguments) == 0
    before = output.read_bytes()
    assert SUMMARY.main(arguments) == 2 and output.read_bytes() == before


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("model", ["ukt", "mockt"])
def test_training_recipes_keep_distinct_bound_identities(artifacts, duplicate, model):
    _, batch_path, directories = artifacts
    batch = read_json(batch_path)
    for index, entry in enumerate(batch["runs"]):
        use_cl = index == 0 or duplicate
        if model == "ukt":
            entry["config"].update(model=model, ukt_training_recipe=A.UKT_AUTHOR_AUGMENTATION if use_cl else "ukt-no-cl",
                                   model_config={"d_model": 16, "use_CL": use_cl, "use_uncertainty_aug": True})
        else:
            entry["config"].update(model=model, mockt_training_recipe=("mockt-target-prefix-branches-v1" if use_cl else
                                   "mockt-three-pass-ground-seq-control-v1"),
                                   model_config={"d_model": 16, "emb_type": "qid_conv_ker_noexp", "kernel_size": 4, "seq_len": 200})
    write_json(batch_path, batch)
    batch_sha = A.canonical_hash(batch)
    for entry, directory in zip(batch["runs"], directories):
        config = dict(entry["config"], batch_id=batch["batch_id"], approved_batch_sha256=batch_sha)
        write_json(directory / "resolved_config.json", config)
        change_json(directory / "metadata.json", approved_batch_sha256=batch_sha, config_sha256=A.canonical_hash(config))
        for name in ("initial_valid_metrics.json", "valid_metrics.json"):
            change_json(directory / name, model=model)
        sidecar_path = directory / "valid_predictions.csv.manifest.json"
        sidecar = read_json(sidecar_path)
        sidecar["result"]["model"] = model
        write_json(sidecar_path, sidecar)
        history = read_history(directory)
        for row in history:
            row["valid"]["model"] = model
        write_history(directory, history)
    if duplicate:
        with pytest.raises(ValueError, match="Duplicate model/recipe"):
            summary(artifacts)
    else:
        report = summary(artifacts)
        if model == "ukt":
            assert {run["ukt_training_recipe"] for run in report["runs"]} == {A.UKT_AUTHOR_AUGMENTATION, "ukt-no-cl"}
        else:
            assert {run["mockt_training_recipe"] for run in report["runs"]} == A.MOCKT_RECIPES
        assert len({run["comparison_label"] for run in report["runs"]}) == 2
        assert len({run["run_id"] for run in report["runs"]}) == 2


def shift_selected_history(directory, metric, delta):
    history = read_history(directory)
    history[0]["valid"][metric] += delta
    if metric == "auc":
        # Keep the two history boundaries tied and the true first-best identity
        # internally consistent; only the independently exported score differs.
        history[1]["valid"][metric] += delta
        change_json(directory / "metadata.json", best_valid_auc=history[0]["valid"][metric])
    write_history(directory, history)


@pytest.mark.parametrize("metric", SUMMARY.METRICS)
@pytest.mark.parametrize("delta", [5e-7, 2e-6])
def test_selected_export_predeclared_drift_tolerance_and_disclosure(artifacts, metric, delta):
    shift_selected_history(artifacts[2][0], metric, delta)
    if delta > 1e-6:
        with pytest.raises(ValueError, match=metric):
            summary(artifacts)
    else:
        report = summary(artifacts)
        check = report["runs"][0]["selected_to_export_consistency"]
        assert check["absolute_tolerance"] == 1e-6 and check["relative_tolerance"] == 0
        assert check["delta_direction"] == "exported_minus_selected_history"
        assert check["metrics"][metric]["delta"] == pytest.approx(-delta, abs=1e-15)
        assert all(item["delta"] == 0 for key, item in check["metrics"].items() if key != metric)


def test_metadata_to_strict_best_history_remains_strict_below_export_tolerance(artifacts):
    path = artifacts[2][0] / "metadata.json"
    change_json(path, best_valid_auc=read_json(path)["best_valid_auc"] + 5e-7)
    with pytest.raises(ValueError, match="best validation AUC"):
        summary(artifacts)


def test_csv_metric_recomputation_remains_strict_below_export_tolerance(artifacts):
    alter_csv(artifacts[2][0], lambda rows: rows[0].update(probability=str(float(rows[0]["probability"]) + 5e-7)))
    with pytest.raises(ValueError, match="saved CSV"):
        summary(artifacts)
