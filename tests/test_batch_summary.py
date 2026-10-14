"""Opaque synthetic artifacts only; no model/data package is imported."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("batch_summary", ROOT / "scripts/summarize_batch.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _reapprove(root, batch):
    batch_sha = SUMMARY.canonical_hash(batch)
    _write(root / "batch.json", batch)
    _write(root / "approvals/B-fixture.json", {"status": "approved", "approved_at": "2026-10-14T00:00:00+00:00",
           "batch_sha256": batch_sha, "user_confirmation": "PRIVATE APPROVAL WORDS"})
    for run_id in {entry["run_id"] for entry in batch["runs"]}:
        directory = root / "runs" / run_id
        if not (directory / "metadata.json").exists():
            continue
        config = SUMMARY.read_json(directory / "resolved_config.json")
        config["approved_batch_sha256"] = batch_sha
        metadata = SUMMARY.read_json(directory / "metadata.json")
        metadata.update(approved_batch_sha256=batch_sha, config_sha256=SUMMARY.canonical_hash(config))
        _write(directory / "resolved_config.json", config)
        _write(directory / "metadata.json", metadata)


@pytest.fixture
def saved_batch(tmp_path):
    root = tmp_path / "repo"
    (root / "src/ktlab").mkdir(parents=True)
    (root / "src/ktlab/adapters.py").write_text('def build_model():\n    settings = {"d_model": 8, "emb_type": "qid"}\n')
    (root / "src/ktlab/ktst_model.py").write_text('_DEFAULT_MODEL = {"bias_before_agg": False}\n')
    (root / "third_party/pykt/pykt/models").mkdir(parents=True)
    (root / "third_party/pykt/pykt/models/akt.py").write_text("# Opaque source fixture, never imported\n")
    _write(root / "third_party/ktst.lock.json", {"fixture": True})
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    data = {"num_q": 0, "num_c": 5, "max_concepts": 1, "maxlen": 200,
            "train_valid_file": "train_valid_sequences.csv"}
    manifest = {"dataset": "synthetic", "data_config": data,
                "files": {"train_valid_sequences.csv": {"sha256": "a" * 64, "bytes": 100}}}
    manifest_path = root / "data/prepared/synthetic/manifest.json"
    _write(manifest_path, manifest)  # No prepared CSV exists: summary must not scan it.
    entries = []
    for model in ("akt", "ktst_mean"):
        config = {"model": model, "dataset": "synthetic", "dataset_manifest": str(manifest_path),
                  "fold": 0, "seed": 42, "epochs": 1, "max_train_steps": 2, "max_wall_seconds": 60,
                  "model_config": {}}
        if model == "ktst_mean":
            config.update(protocol=SUMMARY.KTST, initial_validation=True)
        entries.append({"run_id": "B-" + model, "config": config,
                        "dataset_manifest_sha256": SUMMARY.file_hash(manifest_path)})
    batch = {"batch_id": "B-fixture", "test_access": False, "source_sha256": SUMMARY.source_fingerprint(root),
             "limits": {"total_optimizer_steps_upper_bound": 4}, "runs": entries}
    batch_sha = SUMMARY.canonical_hash(batch)
    for entry in entries:
        config = dict(entry["config"], batch_id=batch["batch_id"], approved_batch_sha256=batch_sha)
        config["model_config"] = SUMMARY.expected_model_config(root, entry["config"], data)
        run_dir = root / "runs" / entry["run_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_bytes(b"not-a-deserializable-checkpoint")
        (run_dir / "valid_predictions.csv").write_bytes(b"PRIVATE STUDENT RECORD\n")
        ktst = config["model"] == "ktst_mean"
        protocol = SUMMARY.KTST if ktst else SUMMARY.V1
        metrics = {"n": 4, "auc": .7, "accuracy": .6, "brier": .2, "nll": .6, "ece_15bins": .1,
                   "evaluated_batches": 1, "partial_debug": False, "prediction_unit": "question_event",
                   "aggregation": "native_set_event" if ktst else "late_mean_full_precision"}
        metadata = {"protocol": protocol, "run_id": entry["run_id"], "batch_id": batch["batch_id"],
                    "status": "completed", "approved_batch_sha256": batch_sha, "test_accessed": False,
                    "implementation_sha256": batch["source_sha256"], "config_sha256": SUMMARY.canonical_hash(config),
                    "dataset_manifest_sha256": entry["dataset_manifest_sha256"],
                    "model_source_sha256": SUMMARY.file_hash(root / ("third_party/ktst.lock.json" if ktst else
                                                                   "third_party/pykt/pykt/models/akt.py")),
                    "train_file_sha256": "a" * 64, "valid_file_sha256": "a" * 64,
                    "optimizer_steps": 1, "best_epoch": 0 if ktst else 1, "best_valid_auc": .7,
                    "partial_debug": False, "checkpoint_sha256": SUMMARY.file_hash(run_dir / "best.pt"),
                    "started_at": "2026-10-14T00:00:10+00:00", "finished_at": "2026-10-14T00:00:13+00:00",
                    "elapsed_seconds": 2.5, "peak_gpu_allocated_bytes": 1024, "train_rows": 3, "valid_rows": 2,
                    "stop_reason": "epochs_completed", "private_note": "PRIVATE METADATA WORDS"}
        _write(run_dir / "resolved_config.json", config)
        _write(run_dir / "metadata.json", metadata)
        _write(run_dir / "valid_metrics.json", metrics)
        _write(run_dir / "initial_valid_metrics.json", metrics)
        _write(run_dir / "history.jsonl", {"epoch": 1, "optimizer_steps": 1, "valid": metrics})
    _reapprove(root, batch)
    return root, batch


def test_summary_uses_saved_artifacts_only_and_sanitizes_output(saved_batch, tmp_path):
    root, _ = saved_batch
    output = tmp_path / "report.json"
    before_imports = set(sys.modules)
    assert SUMMARY.main(["--repo-root", str(root), "--batch", "batch.json", "--output", str(output)]) == 0
    assert "torch" not in set(sys.modules) - before_imports
    text = output.read_text()
    assert "PRIVATE" not in text and str(root) not in text
    report = json.loads(text)
    assert report["runs_completed"] == 2 and report["total_optimizer_steps"] == 2
    assert report["sum_elapsed_seconds"] == 5 and report["recorded_batch_time_window_seconds"] == 3
    assert report["runs"][1]["selected_epoch"] == 0
    assert report["evidence_scope"]["checkpoint_deserialized"] is False
    assert report["evidence_scope"]["prepared_data_records_read"] is False


@pytest.mark.parametrize("failure", ["running", "missing", "checkpoint", "approval", "steps", "total_steps",
                                      "config", "source", "test", "duplicate", "extra", "metrics"])
def test_invalid_or_incomplete_batch_never_writes_final_report(saved_batch, tmp_path, failure):
    root, batch = saved_batch
    run = root / "runs/B-akt"
    metadata = SUMMARY.read_json(run / "metadata.json")
    if failure == "running":
        metadata["status"] = "running"
    elif failure == "missing":
        (run / "valid_metrics.json").unlink()
    elif failure == "checkpoint":
        (run / "best.pt").write_bytes(b"changed")
    elif failure == "approval":
        approval = SUMMARY.read_json(root / "approvals/B-fixture.json")
        approval["batch_sha256"] = "f" * 64
        _write(root / "approvals/B-fixture.json", approval)
    elif failure == "steps":
        metadata["optimizer_steps"] = 3
    elif failure == "total_steps":
        batch["limits"]["total_optimizer_steps_upper_bound"] = 1
        _reapprove(root, batch)
        metadata = SUMMARY.read_json(run / "metadata.json")
    elif failure == "config":
        config = SUMMARY.read_json(run / "resolved_config.json")
        config["seed"] = 999
        metadata["config_sha256"] = SUMMARY.canonical_hash(config)
        _write(run / "resolved_config.json", config)
    elif failure == "source":
        (root / "src/ktlab/adapters.py").write_text("# changed source\n")
    elif failure == "test":
        metadata["test_accessed"] = True
    elif failure == "duplicate":
        batch["runs"].append(copy.deepcopy(batch["runs"][0]))
        _reapprove(root, batch)
        metadata = SUMMARY.read_json(run / "metadata.json")
    elif failure == "extra":
        _write(root / "runs/unapproved-extra/metadata.json", {"batch_id": batch["batch_id"]})
    elif failure == "metrics":
        metrics = SUMMARY.read_json(run / "valid_metrics.json")
        metrics["auc"] = .8
        _write(run / "valid_metrics.json", metrics)
    _write(run / "metadata.json", metadata)
    output = tmp_path / "must-not-exist.json"
    assert SUMMARY.main(["--repo-root", str(root), "--batch", "batch.json", "--output", str(output)]) != 0
    assert not output.exists()
