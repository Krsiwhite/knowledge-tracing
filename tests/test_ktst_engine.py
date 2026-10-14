"""Reference-track contract tests; no backward call or optimizer construction."""
import csv
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from ktlab import ktst_engine as e
from ktlab.evaluation import metrics_from_predictions


def config():
    return {"protocol": e.PROTOCOL, "model": "ktst_mean", "learning_rate": 1e-4,
            "theta_learning_rate": 5e-4}


def test_ktst_batch_approval_and_protocol_are_independent_gates(tmp_path):
    with pytest.raises(PermissionError, match="per-batch"):
        e.train_ktst(config(), tmp_path / "forbidden")
    with pytest.raises(ValueError, match="explicit protocol"):
        e._check_config({**config(), "protocol": "ktlab-v1"})
    with pytest.raises(ValueError, match="theta_learning_rate"):
        e._check_config({**config(), "theta_learning_rate": 0})
    with pytest.raises(ValueError, match="Partial validation"):
        e._check_config({**config(), "max_validation_batches": 3})
    for key in ("epochs", "patience", "max_train_steps", "batch_size", "fold", "seed"):
        with pytest.raises(ValueError, match="integer"):
            e._check_config({**config(), key: .5})
    with pytest.raises(PermissionError, match="not implemented"):
        e.evaluate_ktst_run(tmp_path / "absent", split="test")
    assert not (tmp_path / "forbidden").exists()


def batch(*, empty=False):
    return {"score_mask": torch.tensor([[False, not empty, not empty]]),
            "uid": ["student"], "event_ids": torch.tensor([[0, 1, 2]]),
            "question_ids": torch.tensor([[2, 3, 4]]), "labels": torch.tensor([[0, 0, 1]]),
            "kc_counts": torch.tensor([[2, 1, 2]])}


def test_native_event_metrics_skip_empty_batches_and_never_fuse(tmp_path, monkeypatch):
    calls = []
    def forward(model, actual):
        calls.append(actual)
        assert actual["score_mask"].any()
        return torch.tensor([[.2, .3, .8]]), torch.tensor(.5)
    monkeypatch.setattr(e, "_components", lambda: (None, None, None, forward, None))
    prediction_path = tmp_path / "predictions.csv"
    result = e.evaluate_ktst_loader(SimpleNamespace(eval=lambda: None), [batch(empty=True), batch()],
                                    torch.device("cpu"), prediction_path)
    assert len(calls) == 1
    assert result["n"] == 2 and result["auc"] == 1
    assert result["aggregation"] == "native_set_event" and result["evaluated_batches"] == 2
    with prediction_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["event_id"] for row in rows] == ["1", "2"]
    assert [row["kc_count"] for row in rows] == ["1", "2"]
    assert metrics_from_predictions(prediction_path)["auc"] == result["auc"]
    with pytest.raises(ValueError, match="Duplicate KTST target"):
        e.evaluate_ktst_loader(SimpleNamespace(eval=lambda: None), [batch(), batch()], torch.device("cpu"))


def test_initial_checkpoint_preserves_reference_protocol(tmp_path):
    path = tmp_path / "initial.pt"
    model = SimpleNamespace(state_dict=lambda: {"fixture": torch.tensor(7)}, ktlab_model_config={"fixture": True},
                            ktlab_source_sha256="fixture-only")
    e._save_checkpoint(model, path, .55, 0, "fixture-data")
    restored = torch.load(path, weights_only=True)
    assert restored["epoch"] == 0 and restored["protocol"] == e.PROTOCOL
    assert restored["state_dict"]["fixture"].item() == 7


def fixture_manifest(tmp_path):
    rows = []
    for fold in (0, 1):
        rows.append({"fold": fold, "uid": str(fold), "questions": "0,0,1,2,3",
                     "concepts": "0,1,2,3,4", "responses": "0,0,1,0,1", "is_repeat": "0,1,0,0,0"})
    original = tmp_path / "train_valid.csv"
    sequences = tmp_path / "train_valid_sequences.csv"
    for path in (original, sequences):
        fields = list(rows[0]) + (["selectmasks"] if path == sequences else [])
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for raw in rows:
                row = raw.copy()
                if path == sequences:
                    for key in ("questions", "concepts", "responses", "is_repeat"):
                        row[key] += ",-1" * 195
                    row["selectmasks"] = "1,1,1,1,1" + ",-1" * 195
                writer.writerow(row)
    manifest = {"dataset": "synthetic", "data_dir": str(tmp_path),
                "data_config": {"num_q": 4, "num_c": 5, "max_concepts": 2, "maxlen": 200,
                                "min_seq_len": 3, "folds": [0, 1], "train_valid_file": sequences.name,
                                "train_valid_original_file": original.name},
                "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                          for path in (original, sequences)}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


SMALL_MODEL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2, "num_layers_tf_enc": 1,
               "num_layers_tf_dec": 1, "num_layers_agg": 1, "dim_feedforward": 32,
               "dropout": 0., "dim_classifier": 32}


def test_ktst_real_setup_stops_before_optimizer_and_records_zero_steps(tmp_path, monkeypatch):
    manifest_path, _ = fixture_manifest(tmp_path)
    class BeforeOptimizer(Exception):
        pass
    def refuse(*args, **kwargs):
        raise BeforeOptimizer("Fixture: optimizer was not constructed")
    monkeypatch.setattr(torch.optim, "Adam", refuse)
    with pytest.raises(BeforeOptimizer):
        e.train_ktst({**config(), "_batch_approved": True, "dataset_manifest": str(manifest_path),
                      "model_config": SMALL_MODEL, "device": "cpu", "fold": 0}, tmp_path / "setup_only")
    metadata = json.loads((tmp_path / "setup_only/metadata.json").read_text())
    assert metadata["optimizer_steps"] == 0 and metadata["protocol"] == e.PROTOCOL
    assert not (tmp_path / "setup_only/best.pt").exists()


def test_ktst_checkpoint_validation_roundtrip_without_training(tmp_path):
    from ktlab.engine import evaluate_run
    from ktlab.ktst_model import build_ktst_model
    manifest_path, manifest = fixture_manifest(tmp_path)
    run = tmp_path / "forward_only"
    run.mkdir()
    model = build_ktst_model("ktst_mean", manifest["data_config"], SMALL_MODEL, "cpu")
    resolved = {**config(), "dataset_manifest": str(manifest_path), "model_config": model.ktlab_model_config,
                "device": "cpu", "fold": 0, "validation_batch_size": 2}
    (run / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    e._save_checkpoint(model, run / "best.pt", .5, 0, hashlib.sha256(manifest_path.read_bytes()).hexdigest(), resolved)
    metrics = evaluate_run(run, split="valid")
    assert metrics["n"] == 3 and metrics["protocol"] == e.PROTOCOL
    (run / "resolved_config.json").write_text(json.dumps({**resolved, "fold": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration mismatch"):
        evaluate_run(run, split="valid")
    with pytest.raises(PermissionError, match="not implemented"):
        evaluate_run(run, split="test", allow_test=True)
    assert not (run / "test_predictions.csv").exists()
