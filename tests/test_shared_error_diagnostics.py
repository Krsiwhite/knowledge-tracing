"""Artificial development CSV and saved predictions; no model or torch imports."""
from collections import Counter
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("error_diagnostics", ROOT / "scripts/diagnose_shared_errors.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
A, E = D.A, D.E


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path):
    def create(has_questions=True):
        base = tmp_path / ("questions" if has_questions else "concepts")
        base.mkdir()
        records = [
            ("PRIVATE-VALID-A", 0, [(0, [0, 1, 1], 0), (0, [1, 2], 1), (1, [3], 0), (2, [2], 1)]),
            ("PRIVATE-VALID-B", 0, [(3, [3], 1), (4, [4], 0)]),
            ("PRIVATE-TRAIN", 1, [(1, [1, 2], 0), (0, [0, 1, 2], 1), (0, [1], 0)]),
        ]
        rows = []
        for uid, fold, events in records:
            question, concept, response, repeat = [], [], [], []
            for q, cs, r in events:
                if not has_questions:
                    cs = cs[:1]
                question.extend([q] * len(cs))
                concept.extend(cs)
                response.extend([r] * len(cs))
                repeat.extend([0] + [1] * (len(cs) - 1))
            rows.append({"uid": uid, "fold": fold, **{name: ",".join(map(str, values)) for name, values in
                         (("questions", question), ("concepts", concept), ("responses", response), ("is_repeat", repeat))}})
        source = base / "train_valid.csv"
        write_csv(source, ["uid", "fold", "questions", "concepts", "responses", "is_repeat"], rows)
        manifest = {"dataset": "artificial", "data_dir": str(base),
                    "data_config": {"folds": [0, 1], "num_c": 5, "num_q": 5 if has_questions else 0,
                                    "maxlen": 200, "train_valid_original_file": "train_valid.csv"},
                    "files": {"train_valid.csv": {"sha256": sha(source), "bytes": source.stat().st_size},
                              "test.csv": {"sha256": token("DO NOT READ TEST"), "bytes": 999}}}
        manifest_path = base / "manifest.json"
        write_json(manifest_path, manifest)
        dataset = E.SharedEventDataset(manifest, 0)
        contexts = [dataset[index] for index in range(len(dataset))]
        audit = copy.deepcopy(dataset.audit)
        dataset.close()
        config = {"model": "akt", "dataset": "artificial", "fold": 0, "seed": 42, "protocol": A.PROTOCOL,
                  "epochs": 2, "max_train_steps": 4, "initial_validation": True, "debug_smoke": False,
                  "model_config": {"d_model": 16}, "dataset_manifest": str(manifest_path)}
        entry = {"run_id": "B-ARTIFICIAL-akt", "config": config, "dataset_manifest_sha256": sha(manifest_path),
                 "shared_validation": {key: audit[key] for key in ("protocol", "selected_targets", "source_sha256", "target_sha256", "context_sha256")}}
        batch = {"batch_id": "B-ARTIFICIAL", "test_access": False, "source_sha256": token("different frozen training implementation"),
                 "runs": [entry], "private_note": "PRIVATE-APPROVAL"}
        batch_path = base / "batch.json"
        write_json(batch_path, batch)
        batch_sha = A.canonical_hash(batch)
        directory = base / entry["run_id"]
        directory.mkdir()
        resolved = dict(config, batch_id=batch["batch_id"], approved_batch_sha256=batch_sha)
        write_json(directory / "resolved_config.json", resolved)
        predictions = [{"uid": context["uid"], "event_id": context["event_id"], "question_id": context["query"]["question"],
                        "label": context["label"], "probability": .8 if context["label"] else .2,
                        "kc_count": len(context["query"]["concepts"]),
                        "target_sha256": context["target_sha256"], "context_sha256": context["context_sha256"]} for context in contexts]
        path = directory / "valid_predictions.csv"
        write_csv(path, A.FIELDS, predictions)
        calculated = A.metrics_for(np.array([row["label"] for row in predictions], dtype=float),
                                   np.array([row["probability"] for row in predictions]))
        calculated.pop("auc_unavailable_reason")
        result = {**calculated, "protocol": A.PROTOCOL, "split": "valid", "model": "akt",
                  **{key: audit[key] for key in ("source_sha256", "target_sha256", "context_sha256")},
                  "quality_metrics_computed": True, "selection_limited": False, "test_accessed": False,
                  "implementation_sha256": batch["source_sha256"], "model_source_sha256": token("model not read"),
                  "model_training_and_checkpoint_selection_not_verified": True}
        write_json(path.with_suffix(".csv.manifest.json"), {"schema_version": 1, "dataset_audit": audit, "result": result,
                   "prediction_sha256": sha(path), "metric_scope": "development_only_not_sota_or_final_test"})
        write_json(directory / "valid_metrics.json", {**result, "partial_debug": False, "checkpoint_selection_policy": A.SELECTION})
        metadata = {"protocol": A.PROTOCOL, "run_id": directory.name, "status": "completed", "test_accessed": False,
                    "partial_debug": False, "initial_validation": True, "checkpoint_selection_policy": A.SELECTION,
                    "batch_id": batch["batch_id"], "approved_batch_sha256": batch_sha, "config_sha256": A.canonical_hash(resolved),
                    "implementation_sha256": batch["source_sha256"], "model_source_sha256": result["model_source_sha256"],
                    "dataset_manifest_sha256": sha(manifest_path), "checkpoint_sha256": token("no weight file"),
                    "optimizer_steps": 2, "best_epoch": 1, "best_epoch_completed": True, "best_valid_auc": result["auc"],
                    "valid_event_audit": audit, "valid_file_sha256": audit["source_sha256"], "seed": 42}
        metadata.update({f"validation_{kind}_sha256": audit[f"{kind}_sha256"] for kind in ("source", "target", "context")})
        write_json(directory / "metadata.json", metadata)
        return directory, batch_path, manifest_path, manifest, contexts
    return create


def alter_prediction(directory, field, value, *, reorder=False):
    path = directory / "valid_predictions.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if reorder:
        rows.reverse()
    else:
        rows[0][field] = value
    write_csv(path, A.FIELDS, rows)
    sidecar_path = path.with_suffix(".csv.manifest.json")
    sidecar = read_json(sidecar_path)
    sidecar["prediction_sha256"] = sha(path)
    write_json(sidecar_path, sidecar)


def test_canonical_join_train_counts_privacy_and_read_scope(fixture, monkeypatch):
    directory, batch, manifest_path, manifest, _ = fixture()
    source = manifest_path.parent / "train_valid.csv"
    frequencies, scope = D.training_question_counts(source, manifest, 0)
    assert frequencies == Counter({0: 2, 1: 1})  # includes first event; multi-KC events count once
    assert scope["canonical_events"] == 3 and scope["students"] == 1
    allowed = set(D.CODE_RECEIPTS) | {batch, manifest_path, source} | set(directory.iterdir())
    real_open = Path.open
    def checked(path, *args, **kwargs):
        assert path.resolve() in allowed, f"Unexpected read: {path.name}"
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked)
    report = D.diagnose(directory, batch)
    assert report["overall"]["n"] == 4 and report["overall"]["students"] == 2
    assert report["overall"]["accuracy"] == 1 and report["overall"]["positive_rate"] == .5
    assert report["strata"]["question_train_event_frequency"]["bins"]["1-5"]["n"] == 2
    assert report["strata"]["question_train_event_frequency"]["bins"]["0"]["n"] == 2
    same_q = report["strata"]["visible_history_same_question"]["bins"][">=1"]
    assert same_q["n"] == 1 and same_q["auc"] is None and same_q["auc_unavailable_reason"] == "single_class"
    assert report["strata"]["query_kc_visible_history_support"]["bins"][">=1"]["n"] == 2
    encoded = json.dumps(report)
    for private in ("PRIVATE-VALID", "PRIVATE-TRAIN", "PRIVATE-APPROVAL", str(manifest_path.parent)):
        assert private not in encoded
    assert report["development_data_accessed"] is True and report["test_accessed"] is False
    assert report["model_loads"] == report["model_forward_calls"] == report["new_optimizer_steps"] == 0


def test_reordered_predictions_join_by_student_event_not_question(fixture):
    directory, batch, _, _, _ = fixture()
    before = D.diagnose(directory, batch)
    alter_prediction(directory, None, None, reorder=True)
    after = D.diagnose(directory, batch)
    assert before["strata"] == after["strata"] and before["overall"] == after["overall"]


@pytest.mark.parametrize("field,value", [("label", "0"), ("event_id", "other-event"), ("question_id", "4"),
                                        ("kc_count", "3"), ("context_sha256", "0" * 64)])
def test_canonical_per_target_mismatch_rejected_even_after_csv_rehash(fixture, field, value):
    directory, batch, _, _, _ = fixture()
    alter_prediction(directory, field, value)
    with pytest.raises(ValueError, match="Canonical target identity|do not match canonical"):
        D.diagnose(directory, batch)


def test_fixed_boundaries_and_visible_event_semantics():
    context = {"query": {"question": 0, "concepts": [1, 2]},
               "history": [{"question": 0, "concepts": [1, 2, 3], "response": 0}]}
    for count, expected in ((1, "1-5"), (5, "1-5"), (6, "6-20"), (20, "6-20"), (21, "21-50"), (50, "21-50"), (51, ">50")):
        value = D.stratum_values({**context, "history": context["history"] * count}, Counter(), True)
        assert value["visible_history_events"] == expected
        assert value["query_kc_visible_history_support"] == value["visible_history_same_question"] == ">=1"
    for count, expected in ((0, "0"), (1, "1-5"), (5, "1-5"), (6, "6-20"), (20, "6-20"), (21, "21-100"), (100, "21-100"), (101, ">100")):
        assert D.stratum_values(context, Counter({0: count}), True)["question_train_event_frequency"] == expected
    context["history"] = [{"question": 4, "concepts": [4], "response": 1}]
    assert D.stratum_values(context, Counter(), True)["query_kc_visible_history_support"] == "0"


def test_concept_only_question_layers_are_unsupported(fixture):
    directory, batch, _, _, _ = fixture(False)
    report = D.diagnose(directory, batch)
    for name in ("question_train_event_frequency", "visible_history_same_question"):
        assert report["strata"][name]["supported"] is False and report["strata"][name]["bins"] is None
    assert report["strata"]["visible_history_events"]["supported"] is True
    assert report["strata"]["query_kc_visible_history_support"]["supported"] is True


def test_manifest_receipt_source_tamper_and_mid_analysis_race_fail(fixture, monkeypatch):
    directory, batch, manifest_path, manifest, _ = fixture()
    override = manifest_path.with_name("changed_manifest.json")
    write_json(override, {**manifest, "extra": "changed"})
    with pytest.raises(ValueError, match="manifest byte receipt"):
        D.diagnose(directory, batch, dataset_manifest=override)
    original = D.stratum_values
    changed = False
    def mutate(context, frequencies, has_questions):
        nonlocal changed
        if not changed:
            source = manifest_path.parent / "train_valid.csv"
            with source.open("ab") as stream:
                stream.write(b"\n")
            changed = True
        return original(context, frequencies, has_questions)
    monkeypatch.setattr(D, "stratum_values", mutate)
    with pytest.raises(ValueError, match="Input changed"):
        D.diagnose(directory, batch)
    with pytest.raises(ValueError, match="source receipt mismatch"):
        D.diagnose(directory, batch)


def test_cli_never_overwrites_and_invalid_run_emits_no_report(fixture):
    directory, batch, _, _, _ = fixture()
    output = directory.parent / "diagnostic.json"
    args = ["--run-dir", str(directory), "--batch", str(batch), "--output", str(output)]
    assert D.main(args) == 0
    content = output.read_bytes()
    assert D.main(args) == 2 and output.read_bytes() == content
    bad_output = directory.parent / "bad.json"
    metadata_path = directory / "metadata.json"
    metadata = read_json(metadata_path)
    metadata["status"] = "running"
    write_json(metadata_path, metadata)
    assert D.main(args[:-1] + [str(bad_output)]) == 2 and not bad_output.exists()
