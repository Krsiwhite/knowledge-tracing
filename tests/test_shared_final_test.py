"""Synthetic files and fixture predictions only: no model, weights or training."""
import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ktlab import final_test as ft
from ktlab.provenance import canonical_hash, config_fingerprint, read_json, sha256_file, write_json
from ktlab.shared_events import FINAL_PROTOCOL, PROTOCOL, SharedEventDataset, context_digests
from ktlab.engine import SHARED_SELECTION

EVALUATOR, TRAINER, MODEL = "a" * 64, "b" * 64, "c" * 64


def freeze(f):
    """Create a fictional freeze before any synthetic evaluation; never real data."""
    write_json(f.run / "resolved_config.json", f.config)
    f.metadata.update(config_sha256=config_fingerprint(f.config))
    f.checkpoint.update(config_sha256=config_fingerprint(f.config), model_config=f.config["model_config"])
    write_json(f.run / "best.pt", f.checkpoint)  # Deliberately JSON, never torch.save/load.
    f.metadata["checkpoint_sha256"] = sha256_file(f.run / "best.pt")
    write_json(f.run / "metadata.json", f.metadata)
    record = {k: f.metadata[k] for k in ft._IDENTITY_KEYS}
    record.update({k: f.metadata[k] for k in ("dataset_manifest_sha256", "checkpoint_sha256",
                                            "best_epoch", "best_valid_auc", "best_epoch_completed")})
    record.update(training_protocol=PROTOCOL, metadata_sha256=sha256_file(f.run / "metadata.json"))
    f.release["runs"] = {f.run.name: record}
    f.plan["run_ids"] = [f.run.name]
    f.review["candidates"] = copy.deepcopy(f.release["runs"])
    write_json(f.plan_path, f.plan)
    write_json(f.review_path, f.review)
    f.release.update(analysis_plan_sha256=sha256_file(f.plan_path), compatibility_review_sha256=sha256_file(f.review_path))
    write_json(f.release_path, f.release)


def fixture(tmp_path, *, epoch=1, fold=0, concept_only=False, events=None, source_fold=-1):
    root = tmp_path.resolve()
    run = root / "runs" / "fictional-run"
    run.mkdir(parents=True)
    data = root / "prepared"
    data.mkdir()
    events = events or [(4, [2, 2, 1], 0), (4, [3], 1), (5, [4, 5], 0), (6, [2], 1)]
    flat = {"concepts": [], "responses": []}
    if not concept_only:
        flat.update(questions=[], is_repeat=[])
    for question, concepts, response in events:
        for i, concept in enumerate(concepts):
            flat["concepts"].append(concept)
            flat["responses"].append(response)
            if not concept_only:
                flat["questions"].append(question)
                flat["is_repeat"].append(int(i > 0))
    row = {"uid": "student", "fold": source_fold, **{k: ",".join(map(str, v)) for k, v in flat.items()}}
    source = data / "test.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    write_json(data / "keyid2idx.json", {"synthetic_only": True})
    manifest = {"dataset": "synthetic", "data_dir": str(data),
                "data_config": {"num_q": 0 if concept_only else 1000, "num_c": 512, "maxlen": 200,
                                "max_concepts": 200, "folds": [0, 1, 2, 3, 4], "test_original_file": "test.csv"},
                "split_audit": {"train_test_disjoint": True},
                "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                          for p in (source, data / "keyid2idx.json")}}
    manifest_path = data / "manifest.json"
    write_json(manifest_path, manifest)
    config = {"model": "akt", "dataset": "synthetic", "dataset_manifest": str(manifest_path),
              "protocol": PROTOCOL, "fold": fold, "seed": 42, "initial_validation": True,
              "model_config": {"synthetic_config": True}, "device": "cpu"}
    identity = {"config_sha256": config_fingerprint(config), "implementation_sha256": TRAINER,
                "model_source_sha256": MODEL, "validation_source_sha256": "1" * 64,
                "validation_target_sha256": "2" * 64, "validation_context_sha256": "3" * 64,
                "checkpoint_selection_policy": SHARED_SELECTION}
    audit = {"protocol": PROTOCOL, "split": "valid", "fold": fold, "dataset": "synthetic", "kc_budget": 200,
             "max_targets": None, "selection_limited": False, "selected_for_debug": False,
             "selected_targets": 3, "eligible_total": 3, "expected_total": 3,
             **{f"{name}_sha256": identity[f"validation_{name}_sha256"] for name in ("source", "target", "context")}}
    metadata = {**identity, "run_id": run.name, "protocol": PROTOCOL, "status": "completed", "test_accessed": False,
                "partial_debug": False, "initial_validation": True, "seed": 42, "valid_event_audit": audit,
                "dataset_manifest_sha256": sha256_file(manifest_path), "best_epoch": epoch, "best_valid_auc": .7,
                "best_epoch_completed": epoch > 0, "optimizer_steps": 10}
    checkpoint = {**identity, "protocol": PROTOCOL, "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
                  "epoch": epoch, "valid_auc": .7, "epoch_completed": epoch > 0,
                  "checkpoint_stage": "initial_untrained" if epoch == 0 else "after_training_epoch",
                  "model_config": config["model_config"], "state_dict": {}}
    plan_path, review_path, release_path = root / "plan.json", root / "review.json", root / "release.json"
    plan = {"schema_version": 1, "protocol": FINAL_PROTOCOL, "metrics": list(ft._METRICS), "accuracy_threshold": .5,
            "ece_bins": 15, "kc_budget": 200, "test_selection": "all_eligible_complete_event_kc200",
            "batch_size": 2, "max_rows_per_forward": 32}
    review = {"schema_version": 1, "status": "approved", "reviewed_by": "synthetic fixture", "statement": "fiction only",
              "training_protocol": PROTOCOL, "evaluation_protocol": FINAL_PROTOCOL, "evaluation_source_sha256": EVALUATOR}
    release = {"schema_version": 1, "protocol": FINAL_PROTOCOL, "status": "approved", "user_confirmation": "synthetic fixture",
               "approved_at": "2000-01-01", "source_sha256": EVALUATOR,
               "analysis_plan_file": str(plan_path), "compatibility_review_file": str(review_path)}
    f = SimpleNamespace(**locals())
    freeze(f)
    return f


@pytest.fixture
def stubs(monkeypatch):
    from ktlab import provenance, shared_adapters, shared_evaluation
    calls = {"build": 0, "checkpoint": 0, "contexts": []}
    monkeypatch.setattr(ft, "source_fingerprint", lambda: EVALUATOR)
    monkeypatch.setattr(provenance, "source_fingerprint", lambda: EVALUATOR)
    monkeypatch.setattr(shared_evaluation, "source_fingerprint", lambda: EVALUATOR)
    monkeypatch.setattr(ft, "_current_model_source", lambda name: MODEL)
    def checkpoint(path):
        calls["checkpoint"] += 1
        return read_json(path)
    monkeypatch.setattr(ft, "_load_checkpoint", checkpoint)
    class FixtureModel:
        def __init__(self, config):
            self.model_name = config["model"]
            self.ktlab_source_sha256 = MODEL
            self.ktlab_model_config = config["model_config"]
        def load_state_dict(self, value):
            assert value == {}
    def build(config, data_config):
        calls["build"] += 1
        return FixtureModel(config), "cpu"
    def predict(model, contexts, device, **kwargs):
        calls["contexts"].extend(copy.deepcopy(contexts))
        # A fixed fixture probability, never label-derived and never a model call.
        return [.8 if c["target_index"] % 2 else .2 for c in contexts]
    monkeypatch.setattr(ft, "_build_model", build)
    monkeypatch.setattr(shared_adapters, "predict_shared", predict)
    return calls


def protect_data(monkeypatch, f, *, require_attempt=False):
    original = Path.open
    seen = []
    def checked(path, *args, **kwargs):
        if path.resolve() in {f.source.resolve(), (f.data / "keyid2idx.json").resolve()}:
            seen.append(path.name)
            if not require_attempt:
                raise AssertionError("Data was read before the gate rejected the run")
            attempts = [p for p in (f.data / ".final-test-attempts").glob("*.json") if not p.name.startswith("dataset-")]
            assert len(attempts) == 1
            assert read_json(attempts[0])["status"] == "started"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked)
    return seen


@pytest.mark.parametrize("epoch", [0, 1])
def test_complete_export_and_original_identity_unchanged(tmp_path, monkeypatch, stubs, epoch):
    f = fixture(tmp_path, epoch=epoch)
    (f.run / "valid_predictions.csv").write_text("unchanged synthetic historical artifact\n")
    (f.run / "history.jsonl").write_text('{"synthetic_only": true, "optimizer_steps": 10}\n')
    before = {p.name: p.read_bytes() for p in f.run.iterdir()}
    opened = protect_data(monkeypatch, f, require_attempt=True)
    result = ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    assert opened and stubs["build"] == 1 and result["n"] == 3
    assert result["training_identity"]["implementation_sha256"] == TRAINER != EVALUATOR
    assert result["evaluation_implementation_sha256"] == EVALUATOR
    assert result["training_identity"]["best_epoch"] == epoch
    assert result["training_fold"] == 0 and result["training_seed"] == 42
    assert result["model_training_and_checkpoint_selection_not_verified"] is True
    assert result["original_candidate_receipts_verified"] is True
    assert all((f.run / name).read_bytes() == value for name, value in before.items())
    sidecar = read_json(f.run / "final_test/predictions.csv.manifest.json")
    attempt = read_json(result["attempt_file"])
    assert attempt["status"] == "completed" and attempt["attempt_start_sha256"] == result["attempt_start_sha256"]
    assert attempt["result_sha256"] == sha256_file(f.run / "final_test/result.json")
    assert attempt["prediction_sha256"] == sidecar["prediction_sha256"] == result["prediction_sha256"]
    assert attempt["prediction_manifest_sha256"] == sha256_file(f.run / "final_test/predictions.csv.manifest.json")
    assert sidecar["dataset_audit"]["fold"] == -1 and sidecar["metric_scope"] == "released_final_test"
    assert sidecar["result"]["training_identity"] == result["training_identity"]
    assert all(c["fold"] == -1 and "response" not in c["query"] for c in stubs["contexts"])


@pytest.mark.parametrize("failure", ["permission", "missing_release", "status", "source", "plan_hash", "review_hash",
    "config", "metadata", "checkpoint", "protocol", "model_source", "valid_identity", "debug", "partial",
    "epoch_stage", "candidate_set", "plan_data_path", "release_mapping_path", "checkpoint_data_path"])
def test_negative_gates_do_not_open_test_or_build_model(tmp_path, monkeypatch, stubs, failure):
    f = fixture(tmp_path)
    allow = True
    if failure == "permission": allow = False
    elif failure == "missing_release": f.release_path.unlink()
    elif failure == "status": f.release["status"] = "pending"; write_json(f.release_path, f.release)
    elif failure == "source": monkeypatch.setattr(ft, "source_fingerprint", lambda: "d" * 64)
    elif failure in {"plan_hash", "review_hash"}:
        (f.plan_path if failure == "plan_hash" else f.review_path).write_text("{}")
    elif failure == "config": f.config["seed"] = 43; write_json(f.run / "resolved_config.json", f.config)
    elif failure == "metadata": f.metadata["status"] = "running"; freeze(f)
    elif failure == "checkpoint": (f.run / "best.pt").write_text("{}")
    elif failure == "protocol": f.checkpoint["protocol"] = "old-v1"; freeze(f)
    elif failure == "model_source": monkeypatch.setattr(ft, "_current_model_source", lambda name: "e" * 64)
    elif failure == "valid_identity": f.metadata["valid_event_audit"]["context_sha256"] = "9" * 64; freeze(f)
    elif failure == "debug": f.config["debug_smoke"] = True; freeze(f)
    elif failure == "partial": f.metadata["valid_event_audit"]["selected_targets"] = 2; freeze(f)
    elif failure == "epoch_stage": f.checkpoint["checkpoint_stage"] = "initial_untrained"; freeze(f)
    elif failure == "candidate_set":
        f.plan["run_ids"] = [f.run.name, "new-candidate"]; write_json(f.plan_path, f.plan)
        f.release["analysis_plan_sha256"] = sha256_file(f.plan_path); write_json(f.release_path, f.release)
    elif failure == "plan_data_path":
        f.release["analysis_plan_file"] = str(f.source); write_json(f.release_path, f.release)
    elif failure == "release_mapping_path": f.release_path = f.data / "keyid2idx.json"
    elif failure == "checkpoint_data_path":
        (f.run / "best.pt").unlink(); (f.run / "best.pt").symlink_to(f.source)
    protect_data(monkeypatch, f)
    with pytest.raises((PermissionError, ft.ApprovalError, ValueError)):
        ft.evaluate_shared_test(f.run, f.release_path, allow_test=allow)
    assert stubs["build"] == 0 and not (f.run / "final_test").exists()
    assert not (f.data / ".final-test-attempts").exists()


def test_failure_consumes_candidate_and_different_release_name_does_not_retry(tmp_path, monkeypatch, stubs):
    from ktlab import shared_adapters
    f = fixture(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic interrupted prediction")
    monkeypatch.setattr(shared_adapters, "predict_shared", fail)
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    attempt = next(p for p in (f.data / ".final-test-attempts").glob("*.json") if not p.name.startswith("dataset-"))
    assert read_json(attempt)["status"] == "failed"
    assert not (f.run / "final_test/result.json").exists()
    renamed = f.root / "different-name.json"
    renamed.write_bytes(f.release_path.read_bytes())
    # A copied run/output location still uses the canonical dataset's candidate ledger.
    copied = f.root / "copied" / f.run.name
    copied.mkdir(parents=True)
    for name in ("resolved_config.json", "metadata.json", "best.pt"):
        (copied / name).write_bytes((f.run / name).read_bytes())
    protect_data(monkeypatch, f)
    for directory in (f.run, copied):
        with pytest.raises((ft.ApprovalError, FileExistsError)):
            ft.evaluate_shared_test(directory, renamed, allow_test=True)
    assert stubs["build"] == 1 and read_json(attempt)["status"] == "failed"


@pytest.mark.parametrize("change", ["candidate", "plan"])
def test_test_source_freezes_whole_release_before_first_candidate(tmp_path, monkeypatch, stubs, change):
    f = fixture(tmp_path)
    ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    if change == "candidate": f.metadata["best_valid_auc"] = .71; f.checkpoint["valid_auc"] = .71
    else: f.plan["batch_size"] = 3
    freeze(f)
    protect_data(monkeypatch, f)
    with pytest.raises(ft.ApprovalError, match="different frozen release"):
        ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    assert stubs["build"] == 1


def test_one_frozen_set_allows_two_training_folds_with_identical_test_identity(tmp_path, stubs):
    a, b = fixture(tmp_path / "a", fold=0), fixture(tmp_path / "b", fold=1)
    b.run.rename(b.run.with_name("second-fictional-run"))
    b.run = b.run.with_name("second-fictional-run")
    b.metadata["run_id"] = b.run.name
    b.config["dataset_manifest"] = str(a.manifest_path)
    b.metadata["dataset_manifest_sha256"] = b.checkpoint["dataset_manifest_sha256"] = sha256_file(a.manifest_path)
    freeze(b)
    a.release["runs"].update(b.release["runs"])
    a.plan["run_ids"] = list(a.release["runs"])
    a.review["candidates"] = copy.deepcopy(a.release["runs"])
    write_json(a.plan_path, a.plan)
    write_json(a.review_path, a.review)
    a.release.update(analysis_plan_sha256=sha256_file(a.plan_path), compatibility_review_sha256=sha256_file(a.review_path))
    write_json(a.release_path, a.release)
    first = ft.evaluate_shared_test(a.run, a.release_path, allow_test=True)
    second = ft.evaluate_shared_test(b.run, a.release_path, allow_test=True)
    assert (first["training_fold"], second["training_fold"]) == (0, 1)
    assert first["target_sha256"] == second["target_sha256"]
    assert first["context_sha256"] == second["context_sha256"]
    assert first["dataset_release_lock_file"] == second["dataset_release_lock_file"]
    assert first["attempt_file"] != second["attempt_file"]
    assert stubs["build"] == 2


def test_active_attempt_reentry_cannot_start_a_second_prediction(tmp_path, monkeypatch, stubs):
    f = fixture(tmp_path)
    build = ft._build_model
    def reentry(config, data_config):
        # First call owns a durable started attempt but has no output directory yet.
        assert not (f.run / "final_test").exists()
        with pytest.raises(FileExistsError):
            ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
        return build(config, data_config)
    monkeypatch.setattr(ft, "_build_model", reentry)
    result = ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    assert result["n"] == 3 and stubs["build"] == 1 and len(stubs["contexts"]) == 3


@pytest.mark.parametrize("mutated", ["release", "plan", "review", "config", "checkpoint", "source", "mapping"])
def test_mid_evaluation_receipt_change_never_completes(tmp_path, monkeypatch, stubs, mutated):
    from ktlab import shared_adapters
    f = fixture(tmp_path)
    paths = {"release": f.release_path, "plan": f.plan_path, "review": f.review_path,
             "config": f.run / "resolved_config.json", "checkpoint": f.run / "best.pt",
             "mapping": f.data / "keyid2idx.json"}
    def predict(model, contexts, device, **kwargs):
        if mutated == "source": monkeypatch.setattr(ft, "source_fingerprint", lambda: "d" * 64)
        else:
            with paths[mutated].open("a") as stream: stream.write(" ")
        return [.5] * len(contexts)
    monkeypatch.setattr(shared_adapters, "predict_shared", predict)
    with pytest.raises(ft.ApprovalError, match="changed"):
        ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    attempt = next(p for p in (f.data / ".final-test-attempts").glob("*.json") if not p.name.startswith("dataset-"))
    assert read_json(attempt)["status"] == "failed" and not (f.run / "final_test/result.json").exists()


def test_complete_event_boundary_and_current_label_isolation(tmp_path, stubs):
    import torch
    from ktlab.shared_adapters import collate_shared_kc, collate_shared_ktst, audit_shared_projection
    events = [(1, list(range(100)), 0), (1, list(range(100)), 1), (2, [100, 101], 0), (3, [1], 1)]
    first = fixture(tmp_path / "a", events=events)
    ft.evaluate_shared_test(first.run, first.release_path, allow_test=True)
    first_contexts = copy.deepcopy(stubs["contexts"])
    stubs["contexts"].clear()
    events[2] = (2, [100, 101], 1)  # Current label is never part of query or its context digest.
    second = fixture(tmp_path / "b", fold=1, events=events)
    ft.evaluate_shared_test(second.run, second.release_path, allow_test=True)
    after = stubs["contexts"]
    assert [c["event_id"] for c in after] == [c["event_id"] for c in first_contexts]
    assert after[0]["history_kc_count"] == 100 and after[0]["context_kc_count"] == 200
    assert after[1]["history_start_index"] == 1 and after[1]["history_kc_count"] == 100
    assert len(after[1]["history"]) == 1 and len(after[1]["query"]["concepts"]) == 2
    assert first_contexts[1]["label"] != after[1]["label"]
    assert context_digests(first_contexts[1]) == context_digests(after[1])
    assert first_contexts[1]["query"] == after[1]["query"] and first_contexts[1]["history"] == after[1]["history"]
    # CPU tensor collation only: the two model representations cannot receive
    # a target label through any tensor, even though their saved labels differ.
    for collate in (collate_shared_kc, lambda c: collate_shared_ktst(c, max_concepts=100)):
        left, right = collate([first_contexts[1]]), collate([after[1]])
        for key, value in left.items():
            assert torch.equal(value, right[key]) if isinstance(value, torch.Tensor) else value == right[key]
    projection = audit_shared_projection(after, dict(first.manifest["data_config"], max_concepts=100))
    assert projection["protocol"] == FINAL_PROTOCOL and projection["passed"]
    assert first_contexts[2]["context_sha256"] != after[2]["context_sha256"]  # Later allowed history changes.
    assert all(c["fold"] == -1 for c in after + first_contexts)


def test_as15_and_single_class_auc_is_explicitly_unavailable(tmp_path, stubs):
    from ktlab.shared_adapters import collate_shared_kc, collate_shared_ktst
    f = fixture(tmp_path, concept_only=True, events=[(0, [0], 1), (0, [1], 1), (0, [0], 1)])
    result = ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    assert result["auc"] is None and result["auc_available"] is False
    assert result["auc_unavailable_reason"] == "single_response_class"
    assert all(c["query"]["question"] is None for c in stubs["contexts"])
    assert stubs["contexts"][0]["event_id"] != stubs["contexts"][1]["event_id"]
    kcs = collate_shared_kc(stubs["contexts"])
    sets = collate_shared_ktst(stubs["contexts"], 1)
    assert not kcs["questions"].any()
    assert (sets["data"][:, :, -2][sets["padding_mask"]] == 0).all()


def test_non_test_fold_fails_after_registration_before_model(tmp_path, stubs):
    f = fixture(tmp_path, source_fold=0)
    with pytest.raises(ValueError, match="membership"):
        ft.evaluate_shared_test(f.run, f.release_path, allow_test=True)
    assert stubs["build"] == 0
    attempt = next(p for p in (f.data / ".final-test-attempts").glob("*.json") if not p.name.startswith("dataset-"))
    assert read_json(attempt)["status"] == "failed"


def test_original_dev_api_cannot_read_test_and_cli_routes_without_config_edit(tmp_path, monkeypatch, stubs):
    from ktlab.engine import evaluate_run
    f = fixture(tmp_path)
    with pytest.raises(ValueError, match="test remains sealed"):
        SharedEventDataset(f.manifest, 0, split="test")
    result = evaluate_run(f.run, split="test", allow_test=True, test_release_file=f.release_path)
    assert result["protocol"] == FINAL_PROTOCOL
    assert "test_release_file" not in read_json(f.run / "resolved_config.json")
