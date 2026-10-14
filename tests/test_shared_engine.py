"""Shared checkpoint-selection integration; optimizers and gradients forbidden."""
import csv
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from ktlab import engine as e
from ktlab import ktst_engine as k


KC_MODEL = {"d_model": 16, "n_blocks": 1, "d_ff": 32, "num_attn_heads": 2,
            "final_fc_dim": 32, "dropout": 0.0}
SET_MODEL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2,
             "num_layers_tf_enc": 1, "num_layers_tf_dec": 1, "num_layers_agg": 1,
             "dim_feedforward": 32, "dim_classifier": 32, "dropout": 0.0}


class OptimizerNotConstructed(Exception):
    pass


@pytest.fixture(autouse=True)
def prohibit_updates_and_gpu(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Shared engine tests never differentiate or construct an optimizer")
    monkeypatch.setattr(torch.Tensor, "backward", deny)
    monkeypatch.setattr(torch.autograd, "backward", deny)
    monkeypatch.setattr(torch.autograd, "grad", deny)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", deny)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture_config(tmp_path, name="akt", **extra):
    root = tmp_path / "data"
    root.mkdir(exist_ok=True)
    original, sequences = root / "train_valid.csv", root / "train_valid_sequences.csv"
    for path in (original, sequences):
        fields = ["fold", "uid", "questions", "concepts", "responses", "is_repeat"]
        if path == sequences:
            fields.append("selectmasks")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for fold in (0, 1):
                row = {"fold": fold, "uid": str(fold), "questions": "0,0,1,2,3",
                       "concepts": "0,1,2,3,4", "responses": "0,0,1,0,1", "is_repeat": "0,1,0,0,0"}
                if path == sequences:
                    for key in ("questions", "concepts", "responses", "is_repeat"):
                        row[key] += ",-1" * 195
                    row["selectmasks"] = "1,1,1,1,1" + ",-1" * 195
                writer.writerow(row)
    manifest = {"dataset": "synthetic", "data_dir": str(root),
                "data_config": {"num_q": 4, "num_c": 5, "max_concepts": 2, "maxlen": 200,
                                "min_seq_len": 3, "folds": [0, 1], "train_valid_file": sequences.name,
                                "train_valid_original_file": original.name},
                "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "bytes": path.stat().st_size} for path in (original, sequences)}}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"protocol": e.SHARED_PROTOCOL, "model": name, "dataset": "synthetic",
            "dataset_manifest": str(manifest_path), "device": "cpu", "fold": 0, "seed": 42,
            "learning_rate": 1e-4, "theta_learning_rate": 5e-4, "epochs": 1,
            "model_config": SET_MODEL if name.startswith("ktst_") else KC_MODEL, **extra}


def setup_until_optimizer(tmp_path, monkeypatch, name="akt", **extra):
    config = fixture_config(tmp_path, name, **extra)
    seen = []
    def refuse(*args, **kwargs):
        seen.append((args, kwargs))
        raise OptimizerNotConstructed("Stopped before actual optimizer construction")
    monkeypatch.setattr(torch.optim, "Adam", refuse)
    run = tmp_path / "run"
    with pytest.raises(OptimizerNotConstructed):
        e.train({**config, "_batch_approved": True}, run)
    assert len(seen) == 1
    return run, json.loads((run / "resolved_config.json").read_text()), seen[0]


@pytest.mark.parametrize("name", ["akt", "simplekt", "ktst_mean", "ktst_mhsa"])
def test_shared_epoch0_is_selected_before_optimizer_and_keeps_original_training(tmp_path, monkeypatch, name):
    run, config, optimizer_arguments = setup_until_optimizer(tmp_path, monkeypatch, name)
    checkpoint = torch.load(run / "best.pt", weights_only=True)
    metadata = json.loads((run / "metadata.json").read_text())
    initial = json.loads((run / "initial_valid_metrics.json").read_text())
    assert checkpoint["epoch"] == 0 and checkpoint["checkpoint_stage"] == "initial_untrained"
    assert checkpoint["epoch_completed"] is False and metadata["best_epoch_completed"] is False
    assert metadata["optimizer_steps"] == 0 and metadata["best_epoch"] == 0
    assert initial["candidate_epoch"] == 0 and initial["training_started"] is False
    assert initial["optimizer_steps"] == 0 and initial["split"] == "valid"
    assert initial["epoch_completed"] is False
    assert initial["validation_seconds"] >= 0
    assert metadata["initial_validation_seconds"] == initial["validation_seconds"]
    assert initial["model_training_and_checkpoint_selection_not_verified"] is True
    assert initial["checkpoint_selection_policy"] == e.SHARED_SELECTION
    assert config["validation_batch_size"] == config["max_rows_per_forward"] == 32
    assert config["initial_validation"] is True
    assert not (run / "history.jsonl").exists()
    for key in ("protocol", "dataset_manifest_sha256", "config_sha256", "implementation_sha256",
                "model_source_sha256", "validation_source_sha256", "validation_target_sha256",
                "validation_context_sha256", "checkpoint_selection_policy"):
        assert checkpoint[key] == metadata[key]
    assert checkpoint["validation_target_sha256"] == initial["target_sha256"]
    assert checkpoint["validation_context_sha256"] == initial["context_sha256"]
    assert metadata["checkpoint_sha256"] == e._sha(run / "best.pt")
    if name.startswith("ktst_"):
        assert metadata["training_representation"] == "original_ktst_set_dense"
        groups = optimizer_arguments[0][0]
        assert [group["lr"] for group in groups] == [5e-4, 1e-4]
        assert metadata["loss"] == "upstream_BCEWithLogits_excluding_segment_first_event"
    else:
        assert metadata["training_representation"] == "original_pykt_repeated_kc"


@pytest.mark.parametrize("name", ["akt", "ktst_mean"])
def test_bound_shared_checkpoint_can_be_evaluated_but_never_tested(tmp_path, monkeypatch, name):
    run, config, _ = setup_until_optimizer(tmp_path, monkeypatch, name)
    metrics = e.evaluate_run(run)
    assert metrics["n"] == 3 and metrics["protocol"] == e.SHARED_PROTOCOL
    initial = json.loads((run / "initial_valid_metrics.json").read_text())
    assert metrics["target_sha256"] == initial["target_sha256"]
    assert metrics["context_sha256"] == initial["context_sha256"]
    with pytest.raises(PermissionError, match="not implemented"):
        e.evaluate_run(run, "test", allow_test=True)
    assert not (run / "test_predictions.csv").exists()
    config["fold"] = 1
    (run / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        e.evaluate_run(run)


def test_checkpoint_is_strict_best_including_initial_candidate(tmp_path):
    model = SimpleNamespace(state_dict=lambda: {"fixture": torch.tensor(1)}, ktlab_model_config={})
    path = tmp_path / "best.pt"
    identity = {"checkpoint_selection_policy": e.SHARED_SELECTION, "config_sha256": "test-only"}
    assert e.save_best_checkpoint(model, path, .5, -float("inf"), 0, "data", protocol=e.SHARED_PROTOCOL, identity=identity)
    initial_bytes = path.read_bytes()
    assert not e.save_best_checkpoint(model, path, .5, .5, 1, "data", protocol=e.SHARED_PROTOCOL, identity=identity)
    assert path.read_bytes() == initial_bytes
    assert e.save_best_checkpoint(model, path, .50001, .5, 2, "data", protocol=e.SHARED_PROTOCOL, identity=identity)
    assert torch.load(path, weights_only=True)["epoch"] == 2


@pytest.mark.parametrize("field,value", [
    ("protocol", e.PROTOCOL), ("model_source_sha256", "changed"),
    ("validation_context_sha256", "changed"), ("validation_target_sha256", "changed"),
])
def test_shared_evaluation_rejects_relabelled_or_changed_checkpoint(tmp_path, monkeypatch, field, value):
    run, _, _ = setup_until_optimizer(tmp_path, monkeypatch)
    checkpoint = torch.load(run / "best.pt", weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, run / "best.pt")
    metadata = json.loads((run / "metadata.json").read_text())
    # Even agreement between a rewritten checkpoint and metadata cannot replace
    # comparison to the actual implementation/data/selection at evaluation.
    metadata[field] = value
    metadata["checkpoint_sha256"] = e._sha(run / "best.pt")
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        e.evaluate_run(run)


def test_source_change_is_rejected_before_validation(tmp_path, monkeypatch):
    run, _, _ = setup_until_optimizer(tmp_path, monkeypatch)
    monkeypatch.setattr(e, "source_fingerprint", lambda: "changed-implementation")
    with pytest.raises(ValueError, match="implementation_sha256"):
        e.evaluate_run(run)


def test_debug_limit_is_constructed_before_hashing_and_bound(tmp_path, monkeypatch):
    run, config, _ = setup_until_optimizer(tmp_path, monkeypatch, max_validation_targets=2, debug_smoke=True)
    metadata = json.loads((run / "metadata.json").read_text())
    initial = json.loads((run / "initial_valid_metrics.json").read_text())
    assert metadata["partial_debug"] and initial["partial_debug"] and initial["n"] == 2
    assert metadata["valid_event_audit"]["selected_targets"] == 2
    assert metadata["valid_event_audit"]["eligible_total"] == 3
    assert metadata["validation_target_sha256"] == initial["target_sha256"]
    config["max_validation_targets"] = 3
    (run / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        e.evaluate_run(run)


def test_shared_config_and_validation_only_guards():
    base = {"protocol": e.SHARED_PROTOCOL, "model": "akt"}
    for change, message in [({"max_validation_batches": 1, "debug_smoke": True}, "max_validation_batches"),
                            ({"max_validation_targets": 2}, "debug_smoke"),
                            ({"initial_validation": False}, "epoch0"),
                            ({"validation_batch_size": None}, "positive integer"),
                            ({"max_rows_per_forward": False}, "positive integer"),
                            *[({"fold": value}, "fold") for value in (True, .5, "0", -1, 5)]]:
        with pytest.raises(ValueError, match=message):
            e._shared_config({**base, **change})
    wrong = SimpleNamespace(audit={"protocol": e.SHARED_PROTOCOL, "split": "train", "fold": 0})
    with pytest.raises(ValueError, match="validation fold"):
        e._shared_identity(base, None, wrong)


@pytest.mark.parametrize("completed", [False, True])
def test_selected_checkpoint_reload_binds_weight_bytes_candidate_and_epoch_completion(tmp_path, completed):
    model = SimpleNamespace(state_dict=lambda: {"fixture": torch.tensor(1)}, ktlab_model_config={"test": 1})
    path = tmp_path / "best.pt"
    identity = {"checkpoint_selection_policy": e.SHARED_SELECTION, "config_sha256": "fixture"}
    e.save_best_checkpoint(model, path, .6, .5, 1, "data", protocol=e.SHARED_PROTOCOL,
                           identity=identity, epoch_completed=completed)
    metadata = {"protocol": e.SHARED_PROTOCOL, "dataset_manifest_sha256": "data", **identity,
                "best_epoch": 1, "best_valid_auc": .6, "best_epoch_completed": completed,
                "checkpoint_sha256": e._sha(path)}
    checkpoint = e._selected_shared_checkpoint(path, metadata, identity, "data", model.ktlab_model_config)
    assert checkpoint["epoch_completed"] is completed
    assert checkpoint["checkpoint_stage"] == ("after_training_epoch" if completed else "after_partial_training_epoch")
    for changed, message in [({"best_epoch": 2}, "recorded best"),
                             ({"best_valid_auc": .7}, "recorded best"),
                             ({"best_epoch_completed": not completed}, "completion state"),
                             ({"config_sha256": "changed"}, "config_sha256")]:
        with pytest.raises(ValueError, match=message):
            e._selected_shared_checkpoint(path, {**metadata, **changed}, identity, "data", model.ktlab_model_config)
    with path.open("ab") as stream:
        stream.write(b"changed-after-selection")
    with pytest.raises(ValueError, match="file changed"):
        e._selected_shared_checkpoint(path, metadata, identity, "data", model.ktlab_model_config)


def test_legacy_protocol_checkpoint_defaults_remain_unchanged(tmp_path):
    model = SimpleNamespace(state_dict=lambda: {}, ktlab_model_config={}, ktlab_source_sha256="fixture")
    e.save_best_checkpoint(model, tmp_path / "v1.pt", .5, 0, 1, "data")
    k._save_checkpoint(model, tmp_path / "reference.pt", .5, 0, "data")
    assert torch.load(tmp_path / "v1.pt", weights_only=True)["protocol"] == e.PROTOCOL
    assert torch.load(tmp_path / "reference.pt", weights_only=True)["protocol"] == k.PROTOCOL
    assert "checkpoint_stage" not in torch.load(tmp_path / "v1.pt", weights_only=True)
    assert "epoch_completed" not in torch.load(tmp_path / "reference.pt", weights_only=True)
