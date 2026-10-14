"""Synthetic shared-evaluation identities and output checks; no training."""
import copy
import csv
import json
from types import SimpleNamespace

import pytest

from ktlab.provenance import sha256_file
from ktlab.shared_evaluation import evaluate_shared, assert_shared_prediction_alignment


def make_dataset(tmp_path):
    from ktlab.shared_events import SharedEventDataset
    path = tmp_path / "train_valid.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "uid", "questions", "concepts", "responses", "is_repeat"])
        writer.writeheader()
        for fold in (0, 1):
            writer.writerow({"fold": fold, "uid": str(fold), "questions": "0,0,1,2,2,3",
                             "concepts": "0,1,2,3,4,1", "responses": "0,0,1,0,0,1", "is_repeat": "0,1,0,0,1,0"})
    manifest = {"dataset": "fixture", "data_dir": str(tmp_path),
                "data_config": {"num_q": 4, "num_c": 5, "max_concepts": 2, "maxlen": 200,
                                "folds": [0, 1], "train_valid_original_file": path.name},
                "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}}}
    return SharedEventDataset(manifest, 0, "valid")


def fixture_predictor(monkeypatch):
    from ktlab import shared_adapters
    calls = []
    def predict(model, contexts, device, **kwargs):
        calls.extend(contexts)
        return [.8 if context["target_index"] % 2 else .2 for context in contexts]
    monkeypatch.setattr(shared_adapters, "predict_shared", predict)
    return SimpleNamespace(model_name="fixture_model", ktlab_source_sha256="fixture-source"), calls


def test_saved_targets_and_histories_align_without_quality_metrics(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    model, calls = fixture_predictor(monkeypatch)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    result = evaluate_shared(model, dataset, "cpu", first, batch_size=2, compute_metrics=False)
    assert result["n"] == 3 and "auc" not in result and not result["quality_metrics_computed"]
    assert len(calls) == 3
    evaluate_shared(model, dataset, "cpu", second, batch_size=1, compute_metrics=False)
    assert assert_shared_prediction_alignment([first, second])["matched_events"] == 3
    with pytest.raises(FileExistsError):
        evaluate_shared(model, dataset, "cpu", first)
    with first.open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["context_sha256"] and row["target_sha256"] for row in rows)
    assert len({row["event_id"] for row in rows}) == 3


def test_changed_prediction_or_labels_cannot_silently_join_comparison(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    model, _ = fixture_predictor(monkeypatch)
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for path in paths:
        evaluate_shared(model, dataset, "cpu", path)
    sidecar = paths[1].with_suffix(".csv.manifest.json")
    with paths[1].open() as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), reader.fieldnames
    rows[0]["label"] = str(1 - int(rows[0]["label"]))
    with paths[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="no longer matches"):
        assert_shared_prediction_alignment(paths)
    metadata = json.loads(sidecar.read_text())
    metadata["prediction_sha256"] = sha256_file(paths[1])
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="labels differ"):
        assert_shared_prediction_alignment(paths)


def test_context_tampering_and_test_split_fail_before_predictions(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    model, calls = fixture_predictor(monkeypatch)
    class Changed:
        protocol = dataset.protocol
        audit = dataset.audit
        path = dataset.path
        def __len__(self):
            return len(dataset)
        def __getitem__(self, index):
            value = copy.deepcopy(dataset[index])
            value["history"][0]["response"] = 1 - value["history"][0]["response"]
            return value
    with pytest.raises(ValueError, match="context changed"):
        evaluate_shared(model, Changed(), "cpu", tmp_path / "failed.csv")
    assert not calls and not (tmp_path / "failed.csv.tmp").exists()
    Changed.audit = {**dataset.audit, "split": "test"}
    with pytest.raises(PermissionError, match="test remains sealed"):
        evaluate_shared(model, Changed(), "cpu")


def test_source_changed_after_index_is_rejected_before_prediction(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    model, calls = fixture_predictor(monkeypatch)
    dataset.path.write_bytes(dataset.path.read_bytes().replace(b'0,0,1,0,0,1', b'0,0,1,0,0,0'))
    with pytest.raises(ValueError, match="source changed after selection"):
        evaluate_shared(model, dataset, "cpu")
    assert calls == []


def test_reordered_self_consistent_contexts_do_not_match_frozen_selection(tmp_path, monkeypatch):
    dataset = make_dataset(tmp_path)
    model, _ = fixture_predictor(monkeypatch)
    class Reordered:
        protocol = dataset.protocol
        audit = dataset.audit
        path = dataset.path
        def __len__(self):
            return len(dataset)
        def __getitem__(self, index):
            return dataset[len(dataset) - 1 - index]
    output = tmp_path / "reordered.csv"
    with pytest.raises(ValueError, match="differs from the frozen selection"):
        evaluate_shared(model, Reordered(), "cpu", output)
    assert not output.exists() and not output.with_suffix(".csv.tmp").exists()
