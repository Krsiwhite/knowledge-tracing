"""Forward-only correctness checks: this suite never calls backward/step."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ktlab.adapters import (ModelUnavailableError, SequenceDataset, build_model,
                             forward_model, move_batch)
from ktlab.engine import PROTOCOL, dataset_for_split, evaluate_run, read_manifest, train
from ktlab.evaluation import binary_metrics, evaluate_loader, metrics_from_predictions


DATA_CONFIG = {"num_c": 5, "num_q": 7, "maxlen": 16, "folds": [0, 1]}
SMALL_MODEL = {"d_model": 16, "d_ff": 32, "n_blocks": 1, "dropout": 0.,
               "num_attn_heads": 2, "final_fc_dim": 32}


def write_sequences(path, *, question=False, pad_to=8, flip_at=None):
    rows = []
    for row_id in range(2):
        concepts = [0, 1, 2, 1, 3, 2]
        questions = [0, 1, 2, 3, 4, 5]
        responses = [0, 1, row_id, 0, 1, 0]
        if flip_at is not None:
            responses[flip_at] = 1 - responses[flip_at]
        padding = [-1] * (pad_to - len(concepts))
        row = {"fold": row_id, "uid": str(row_id),
               "concepts": concepts + padding, "questions": questions + padding,
               "responses": responses + padding, "selectmasks": [1] * 6 + padding}
        if question:
            row.update(qidxs=list(range(6)) + padding, rest=[0] * 6 + padding, orirow=[row_id] * 6 + padding)
        rows.append({key: ",".join(map(str, value)) if isinstance(value, list) else value for key, value in row.items()})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batch_at(tmp_path, *, pad_to=8, flip_at=None):
    path = tmp_path / f"data_{pad_to}_{flip_at}.csv"
    write_sequences(path, pad_to=pad_to, flip_at=flip_at)
    dataset = SequenceDataset(path, {0, 1})
    return next(iter(DataLoader(dataset, batch_size=2)))


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_original_forward_is_causal_and_padding_invariant(tmp_path, name):
    torch.manual_seed(42)
    model = build_model(name, DATA_CONFIG, SMALL_MODEL, "cpu").eval()
    batch = batch_at(tmp_path)
    changed = batch_at(tmp_path, flip_at=3)
    padded = batch_at(tmp_path, pad_to=12)
    with torch.inference_mode():
        original, _ = forward_model(model, batch)
        altered, _ = forward_model(model, changed)
        longer, _ = forward_model(model, padded)
    assert original.shape == (2, 7)
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original[:, :3], altered[:, :3], rtol=0, atol=1e-7)
    torch.testing.assert_close(original[:, :5], longer[:, :5], rtol=0, atol=1e-7)
    assert batch["cseqs"].device.type == "cpu"
    assert not batch["smasks"][:, 5:].any()


@pytest.mark.parametrize("name", ["akt", "simplekt"])
def test_original_checkpoint_reload_and_device_parity(tmp_path, name):
    torch.manual_seed(77)
    cpu = build_model(name, DATA_CONFIG, SMALL_MODEL, "cpu").eval()
    path = tmp_path / f"{name}.pt"
    torch.save(cpu.state_dict(), path)
    restored = build_model(name, DATA_CONFIG, SMALL_MODEL, "cpu").eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    batch = batch_at(tmp_path)
    with torch.inference_mode():
        expected, _ = forward_model(cpu, batch)
        actual, _ = forward_model(restored, batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if torch.cuda.is_available():
            gpu = build_model(name, DATA_CONFIG, SMALL_MODEL, "cuda").eval()
            gpu.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
            accelerated, _ = forward_model(gpu, move_batch(batch, torch.device("cuda")))
            torch.testing.assert_close(accelerated.cpu(), expected, rtol=2e-4, atol=2e-5)


def test_no_optional_models_eager_import(tmp_path):
    import sys
    build_model("akt", DATA_CONFIG, SMALL_MODEL, "cpu")
    assert not any(key.endswith(".cdfkt") for key in sys.modules)


def test_mockt_requires_gpu_without_silent_patch():
    with pytest.raises(ModelUnavailableError, match="hard-codes CUDA"):
        build_model("mockt", DATA_CONFIG, SMALL_MODEL, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MoC-KT original requires CUDA/ROCm")
def test_mockt_original_gpu_forward_and_causality(tmp_path):
    torch.manual_seed(23)
    model = build_model("mockt", DATA_CONFIG, SMALL_MODEL, "cuda").eval()
    batch = move_batch(batch_at(tmp_path), torch.device("cuda"))
    changed = move_batch(batch_at(tmp_path, flip_at=3), torch.device("cuda"))
    with torch.inference_mode():
        expected, _ = forward_model(model, batch)
        altered, _ = forward_model(model, changed)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(expected[:, :3], altered[:, :3], rtol=2e-5, atol=2e-6)


def test_fa_kt_reports_missing_mamba_without_substitution():
    import importlib.util
    if importlib.util.find_spec("mamba_ssm") is not None:
        pytest.skip("Mamba installed; feasibility must be checked on this machine")
    with pytest.raises(ModelUnavailableError, match="No expert substitution"):
        build_model("fa_kt", DATA_CONFIG, SMALL_MODEL, "cpu")


def test_training_and_test_are_separately_gated(tmp_path):
    with pytest.raises(PermissionError, match="per-batch"):
        train({}, tmp_path / "forbidden")
    with pytest.raises(PermissionError, match="sealed"):
        evaluate_run(tmp_path / "not-even-created", "test")
    assert not (tmp_path / "forbidden").exists()


def test_mask_zero_is_excluded(tmp_path):
    path = tmp_path / "mask.csv"
    write_sequences(path)
    text = path.read_text().replace('"1,1,1,1,1,1,-1,-1"', '"1,0,1,1,1,1,-1,-1"')
    path.write_text(text)
    data = SequenceDataset(path, {0})
    assert not bool(data[0]["smasks"][0])


def test_question_group_fusion_and_saved_metrics_are_full_precision(tmp_path):
    path = tmp_path / "question.csv"
    fields = ["fold", "uid", "concepts", "questions", "responses", "selectmasks", "qidxs", "rest", "orirow"]
    rows = [
        [0, "a", "0,1,-1", "0,1,-1", "1,0,-1", "-1,1,-1", "0,1,-1", "0,1,-1", "0,0,-1"],
        [0, "a", "0,2,-1", "0,1,-1", "1,0,-1", "-1,1,-1", "0,1,-1", "0,0,-1", "0,0,-1"],
        [0, "b", "0,3,-1", "0,2,-1", "0,1,-1", "-1,1,-1", "2,3,-1", "0,0,-1", "1,1,-1"],
    ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    dataset = SequenceDataset(path, {0}, question_level=True)
    model = build_model("akt", DATA_CONFIG, SMALL_MODEL, "cpu").eval()
    loader = DataLoader(dataset, batch_size=1)
    output = tmp_path / "predictions.csv"
    result = evaluate_loader(model, loader, torch.device("cpu"), output)
    assert result["n"] == 2 and result["concept_predictions"] == 3
    with output.open() as stream:
        saved = list(csv.DictReader(stream))
    with torch.inference_mode():
        separate = [float(forward_model(model, b)[0][0, 0]) for b in DataLoader(dataset, batch_size=1)]
    assert float(saved[0]["probability"]) == np.mean(separate[:2], dtype=np.float64)
    assert metrics_from_predictions(output) == {k: result[k] for k in ("n", "auc", "accuracy", "brier", "nll", "ece_15bins")}


def test_metric_input_failures_and_single_class():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.2, float("nan")])
    assert binary_metrics([1, 1], [0.5, 0.6])["auc"] is None
    assert binary_metrics([0, 1], [0.50001, 0.50002])["auc"] == 1.0
    assert binary_metrics([0, 1], [0, 1])["ece_15bins"] == 0.0
    assert binary_metrics([0, 1], [0.8, 0.8])["ece_15bins"] == pytest.approx(0.3)


def test_checkpoint_keeps_small_mid_run_improvement_not_last_snapshot(tmp_path):
    from types import SimpleNamespace
    from ktlab.engine import save_best_checkpoint
    best = float("-inf")
    path = tmp_path / "best.pt"
    for epoch, auc in enumerate([0.7, 0.7004, 0.69], start=1):
        # Synthetic serialized snapshots, no model updates or training.
        model = SimpleNamespace(state_dict=lambda epoch=epoch: {"fixture": torch.tensor(epoch)}, ktlab_model_config={})
        if save_best_checkpoint(model, path, auc, best, epoch, "fixture-manifest"):
            best = auc
    saved = torch.load(path, weights_only=True)
    assert saved["epoch"] == 2
    assert saved["state_dict"]["fixture"].item() == 2
    assert saved["valid_auc"] == 0.7004


def test_train_setup_reaches_optimizer_boundary_without_constructing_optimizer(tmp_path, monkeypatch):
    """Exercise real setup/provenance; stop before creating any optimizer."""
    path = tmp_path / "fixture.csv"
    write_sequences(path)
    data_config = {**DATA_CONFIG, "num_q": 0, "train_valid_file": path.name}
    manifest = {"dataset": "fixture", "data_dir": str(tmp_path), "data_config": data_config,
                "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    class BeforeOptimizer(Exception):
        pass
    def refuse_optimizer(*args, **kwargs):
        raise BeforeOptimizer("No optimizer was created")
    monkeypatch.setattr(torch.optim, "Adam", refuse_optimizer)
    with pytest.raises(BeforeOptimizer):
        train({"_batch_approved": True, "dataset_manifest": str(manifest_path), "model": "akt",
               "model_config": SMALL_MODEL, "device": "cpu", "fold": 0, "batch_size": 2}, tmp_path / "fixture-run")
    metadata = json.loads((tmp_path / "fixture-run/metadata.json").read_text())
    assert metadata["optimizer_steps"] == 0
    assert metadata["config_sha256"] and metadata["implementation_sha256"]
    assert not (tmp_path / "fixture-run/best.pt").exists()


def test_manifest_hash_check_and_causal_valid_requirement(tmp_path):
    path = tmp_path / "data.csv"
    write_sequences(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"dataset": "fixture", "data_dir": str(tmp_path), "data_config": {**DATA_CONFIG, "train_valid_file": path.name},
                "files": {path.name: {"sha256": digest, "bytes": path.stat().st_size}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    checked = read_manifest({"dataset_manifest": str(manifest_path)})
    with pytest.raises(ValueError, match="Causal"):
        dataset_for_split(checked, 0, "valid")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        read_manifest({"dataset_manifest": str(manifest_path)})


def test_evaluate_run_restores_untrained_checkpoint_without_test_access(tmp_path):
    """End-to-end evaluation API using a fresh, never-optimized checkpoint."""
    path = tmp_path / "train_valid.csv"
    write_sequences(path)
    data_config = {**DATA_CONFIG, "num_q": 0, "train_valid_file": path.name}
    manifest = {"dataset": "fixture", "data_dir": str(tmp_path), "data_config": data_config,
                "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    run_dir = tmp_path / "forward_only"
    run_dir.mkdir()
    config = {"dataset_manifest": str(manifest_path), "model": "akt", "model_config": SMALL_MODEL,
              "device": "cpu", "fold": 0, "batch_size": 2}
    (run_dir / "resolved_config.json").write_text(json.dumps(config))
    model = build_model("akt", data_config, SMALL_MODEL, "cpu")
    torch.save({"state_dict": model.state_dict(), "model_config": model.ktlab_model_config,
                "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "protocol": PROTOCOL}, run_dir / "best.pt")
    result = evaluate_run(run_dir)
    assert result["n"] == 5
    assert result["auc"] is not None
    assert (run_dir / "valid_predictions.csv").is_file()
    assert not (run_dir / "test_predictions.csv").exists()
    with pytest.raises(Exception, match="sealed"):
        evaluate_run(run_dir, "test", allow_test=True)
