"""S008 artificial CPU acceptance: exactly 5 models / 85 forwards / 0 grads.

Real models require KTLAB_RUN_PREFIX_GROUPS_CPU_ACCEPTANCE=1 after the frozen
S008 manifest is registered. Other tests use file fixtures and fake predictors.
No prepared data, existing weights, accelerator, or optimizer may be accessed.
"""
import copy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ktlab import adapters, engine, ktst_engine, shared_adapters, shared_evaluation
from ktlab import shared_prefix_groups as pg
from ktlab.ktst_model import build_ktst_model, forward_ktst
from ktlab.provenance import sha256_file
from ktlab.shared_events import SharedEventDataset, context_digests

OPT_IN = "KTLAB_RUN_PREFIX_GROUPS_CPU_ACCEPTANCE"
DATA = {"num_q": 0, "num_c": 7, "max_concepts": 1, "maxlen": 200}
KC_MODEL = {"d_model": 16, "n_blocks": 2, "num_attn_heads": 2, "d_ff": 32,
            "final_fc_dim": 32, "dropout": 0.0}
SET_MODEL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2, "num_layers_tf_enc": 2,
             "num_layers_tf_dec": 2, "num_layers_agg": 1, "dim_feedforward": 32,
             "dim_classifier": 32, "dropout": 0.0}


@pytest.fixture(scope="module", autouse=True)
def cpu_budget(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    counts = {"models": 0, "forwards": 0, "gradients": 0, "optimizers": 0}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    temporary = tmp_path_factory.getbasetemp().resolve()
    original_load, original_open = torch.load, Path.open

    def forbidden(*args, **kwargs):
        raise AssertionError("S008 forbids gradients, optimizer construction, and accelerators")

    def load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(temporary), "Only artificial temporary checkpoints are allowed"
        return original_load(path, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        if path.suffix.lower() == ".csv":
            assert path.resolve().is_relative_to(temporary), "Only artificial CSV fixtures are allowed"
        return original_open(path, *args, **kwargs)

    patch.setattr(torch.autograd, "grad", forbidden)
    patch.setattr(torch.autograd, "backward", forbidden)
    patch.setattr(torch.Tensor, "backward", forbidden)
    patch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    patch.setattr(torch.cuda, "is_available", lambda: False)
    patch.setattr(torch.cuda, "_lazy_init", forbidden)
    patch.setattr(torch.Tensor, "cuda", forbidden)
    patch.setattr(torch.nn.Module, "cuda", forbidden)
    patch.setattr(torch, "load", load)
    patch.setattr(Path, "open", open_file)
    try:
        yield counts
    finally:
        patch.undo()
        torch.set_num_threads(old_threads)
        print("S008_ACTUAL_COUNTS=" + json.dumps(counts, sort_keys=True))
        if os.environ.get(OPT_IN) == "1":
            assert counts == {"models": 5, "forwards": 85, "gradients": 0, "optimizers": 0}
        else:
            assert counts == {"models": 0, "forwards": 0, "gradients": 0, "optimizers": 0}


def artificial_contexts(lengths=(35, 202), *, response_changes=None, concept_changes=None):
    response_changes, concept_changes = response_changes or {}, concept_changes or {}
    contexts = []
    for student, length in enumerate(lengths):
        uid = f"student:{student}"
        events = [{"event_id": f"assist2015:{uid}:{i}", "question": None,
                   "concepts": [concept_changes.get((student, i), i % 7)],
                   "response": response_changes.get((student, i), i % 2)} for i in range(length)]
        for i in range(1, length):
            start = max(0, i - 199)
            context = {"protocol": pg.PROTOCOL, "uid": uid, "fold": 0,
                       "event_id": events[i]["event_id"], "target_index": i,
                       "history_start_index": start, "history": copy.deepcopy(events[start:i]),
                       "query": {"question": None, "concepts": events[i]["concepts"].copy()},
                       "label": events[i]["response"], "history_kc_count": i - start,
                       "target_kc_count": 1, "context_kc_count": i - start + 1}
            context.update(context_digests(context))
            contexts.append(context)
    return contexts


def fixture_dataset(tmp_path, lengths=(6, 7)):
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "train_valid.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "uid", "concepts", "responses"])
        writer.writeheader()
        for student, length in enumerate(lengths):
            writer.writerow({"fold": 0, "uid": f"student:{student}",
                             "concepts": ",".join(str(i % 7) for i in range(length)),
                             "responses": ",".join(str(i % 2) for i in range(length))})
    manifest = {"dataset": "assist2015", "data_dir": str(tmp_path),
                "data_config": {**DATA, "folds": [0, 1], "train_valid_original_file": path.name},
                "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SharedEventDataset(manifest, 0, "valid"), manifest_path


def fake_predictors(monkeypatch):
    calls = []

    def groups(model, selected, device, **kwargs):
        calls.append({"groups": len(selected), "targets": sum(len(group.targets) for group in selected),
                      "max_rows": kwargs["max_rows_per_forward"]})
        return [0.2 + 0.6 * (int(event_id.rsplit(":", 1)[1]) % 2)
                for group in selected for event_id, _ in group.targets]

    def original(model, contexts, device, **kwargs):
        return [0.2 + 0.6 * (context["target_index"] % 2) for context in contexts]

    monkeypatch.setattr(pg, "predict_prefix_groups", groups)
    monkeypatch.setattr(shared_adapters, "predict_shared", original)
    return SimpleNamespace(model_name="akt", ktlab_source_sha256="synthetic-model"), calls


def test_planner_crosses_old_batch_boundaries_and_splits_window_201():
    contexts = artificial_contexts()
    groups = list(pg.iter_prefix_groups(contexts))
    assert [len(group.targets) for group in groups] == [34, 199, 1, 1]
    assert [group.context["history_start_index"] for group in groups] == [0, 0, 1, 2]
    assert [event_id for group in groups for event_id, _ in group.targets] == [c["event_id"] for c in contexts]
    assert len(groups[1].context["history"]) == 199
    assert groups[2].context["target_index"] == 200  # 201st original event


def test_labels_are_unreadable_to_planner_and_collation():
    class NoLabel(dict):
        def __getitem__(self, key):
            assert key != "label", "Planner/collator read a target label"
            return super().__getitem__(key)
        def get(self, key, *args):
            assert key != "label", "Planner/collator read a target label"
            return super().get(key, *args)
    contexts = artificial_contexts((8,))
    before = list(pg.iter_prefix_groups([NoLabel(c) for c in contexts]))
    flipped = copy.deepcopy(contexts)
    for context in flipped:
        context["label"] = 1 - context["label"]
    after = list(pg.iter_prefix_groups(flipped))
    assert before == after
    for collate in (shared_adapters.collate_shared_kc, lambda c: shared_adapters.collate_shared_ktst(c, 1)):
        first, second = collate([g.context for g in before]), collate([g.context for g in after])
        for key, value in first.items():
            if torch.is_tensor(value):
                assert torch.equal(value, second[key])
    assert "label" not in before[0].context and "response" not in before[0].context["query"]


@pytest.mark.parametrize("fault", ["multi_kc", "question", "history_response", "history_concept", "query",
                                    "history_order", "nonconsecutive", "wrong_start", "wrong_count", "response_in_query"])
def test_planner_rejects_incompatible_or_forged_prefix_before_prediction(fault):
    contexts = artificial_contexts((6,))
    current = contexts[2]
    if fault == "multi_kc":
        current["query"]["concepts"].append(6)
    elif fault == "question":
        current["query"]["question"] = 0
    elif fault == "history_response":
        current["history"][0]["response"] ^= 1
    elif fault == "history_concept":
        current["history"][0]["concepts"] = [6]
    elif fault == "query":
        contexts[1]["query"]["concepts"] = [6]
    elif fault == "history_order":
        current["history"].reverse()
    elif fault == "nonconsecutive":
        contexts.pop(1)
    elif fault == "wrong_start":
        current["history_start_index"] = 1
    elif fault == "wrong_count":
        current["history_kc_count"] += 1
    else:
        current["query"]["response"] = 0
    with pytest.raises(ValueError):
        list(pg.iter_prefix_groups(contexts))


@pytest.mark.parametrize("change", [{"shared_validation_execution": None}, {"shared_validation_execution": "unknown"},
                                     {"model": "mockt"}, {"protocol": engine.PROTOCOL}, {"dataset": "assist2009"}])
def test_execution_is_explicit_and_rejects_unsupported_config(change):
    config = {"protocol": pg.PROTOCOL, "dataset": "assist2015", "model": "akt",
              "shared_validation_execution": pg.EXECUTION}
    with pytest.raises(ValueError):
        pg.check_execution({**config, **change})
    assert pg.check_execution({"model": "akt"}) is None
    assert "shared_validation_execution" not in engine._shared_config(
        {"protocol": pg.PROTOCOL, "model": "akt"})


@pytest.mark.parametrize("field,value", [("num_q", 1), ("max_concepts", 2), ("maxlen", 199)])
def test_manifest_dimensions_cannot_be_silently_treated_as_as15(field, value):
    config = {"protocol": pg.PROTOCOL, "dataset": "assist2015", "model": "akt",
              "shared_validation_execution": pg.EXECUTION}
    manifest = {"dataset": "assist2015", "data_config": {**DATA, field: value}}
    with pytest.raises(ValueError, match="no question IDs"):
        pg.check_execution(config, manifest)


@pytest.mark.parametrize("model,settings", [
    ("akt", {"emb_type": "qid_pdiff"}), ("simplekt", {"emb_type": "qid_scalar"}),
    ("ukt", {"emb_type": "stoc_qid_scalar"}), ("ukt", {"atten_type": "dot"}),
    ("ktst_mean", {"attn_variant": "standard"}), ("ktst_mhsa", {"attn_variant": "akt_monotonic"}),
])
def test_unaudited_embedding_or_attention_variant_is_rejected(model, settings):
    with pytest.raises(ValueError, match="audited"):
        pg.check_execution({"protocol": pg.PROTOCOL, "dataset": "assist2015", "model": model,
                            "model_config": settings, "shared_validation_execution": pg.EXECUTION})


@pytest.mark.parametrize("entry", ["pykt_train", "ktst_train", "pykt_evaluate", "ktst_evaluate"])
def test_nonshared_entry_rejects_execution_before_model_or_data(tmp_path, monkeypatch, entry):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported config reached model/data construction")
    monkeypatch.setattr(engine, "read_manifest", forbidden)
    monkeypatch.setattr(engine, "build_model", forbidden)
    config = {"model": "ktst_mean" if entry.startswith("ktst") else "akt", "dataset": "assist2015",
              "protocol": ktst_engine.PROTOCOL if entry.startswith("ktst") else engine.PROTOCOL,
              "shared_validation_execution": pg.EXECUTION, "_batch_approved": True}
    (tmp_path / "resolved_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="shared development"):
        if entry == "pykt_train":
            engine.train(config, tmp_path)
        elif entry == "ktst_train":
            ktst_engine.train_ktst(config, tmp_path)
        elif entry == "pykt_evaluate":
            engine.evaluate_run(tmp_path)
        else:
            ktst_engine.evaluate_ktst_run(tmp_path)


def test_saved_output_order_hash_labels_and_group_batch_limit(tmp_path, monkeypatch):
    dataset, _ = fixture_dataset(tmp_path / "fixture", (35, 202) + (3,) * 35)
    model, calls = fake_predictors(monkeypatch)
    first, second = tmp_path / "old.csv", tmp_path / "grouped.csv"
    original = shared_evaluation.evaluate_shared(model, dataset, "cpu", first)
    result = shared_evaluation.evaluate_shared(model, dataset, "cpu", second, batch_size=2,
                                               max_rows_per_forward=32, shared_validation_execution=pg.EXECUTION)
    assert calls == [{"groups": 32, "targets": 291, "max_rows": 32},
                     {"groups": 7, "targets": 14, "max_rows": 32}]
    assert first.read_bytes() == second.read_bytes()
    for key in ("n", "auc", "brier", "nll", "target_sha256", "context_sha256"):
        assert result[key] == original[key]
    assert result["shared_validation_execution"] == pg.EXECUTION
    assert result["execution"]["group_rows"] == 39
    assert result["execution"]["forward_calls"] == 2
    assert result["execution"]["validation_batch_size"] == 2
    sidecar = json.loads(second.with_suffix(".csv.manifest.json").read_text())
    assert sidecar["result"]["execution"] == result["execution"]
    assert sidecar["prediction_sha256"] == sha256_file(second)
    assert "execution" not in original and "shared_validation_execution" not in original


@pytest.mark.parametrize("fault", ["digest", "label", "source", "expired", "dataset", "model"])
def test_saved_evaluation_rejects_changes_and_cleans_pending(tmp_path, monkeypatch, fault):
    dataset, _ = fixture_dataset(tmp_path / "fixture")
    model, calls = fake_predictors(monkeypatch)
    original = type(dataset).__getitem__
    if fault in {"digest", "label"}:
        def changed(self, index):
            context = original(self, index)
            if index == 0:
                if fault == "digest":
                    context["history"][0]["response"] ^= 1
                else:
                    context["label"] = 2
            return context
        monkeypatch.setattr(type(dataset), "__getitem__", changed)
    elif fault == "source":
        dataset.path.write_bytes(dataset.path.read_bytes() + b"\n")
    elif fault == "dataset":
        dataset.audit["dataset"] = "assist2009"
    elif fault == "model":
        model.model_name = "mockt"
    path = tmp_path / "rejected.csv"
    with pytest.raises((ValueError, TimeoutError)):
        shared_evaluation.evaluate_shared(model, dataset, "cpu", path,
            shared_validation_execution=pg.EXECUTION, deadline=0 if fault == "expired" else None)
    assert not calls
    assert not path.exists() and not path.with_suffix(".csv.tmp").exists()


def test_grouping_checks_deadline_while_reading_and_source_after_prediction(tmp_path, monkeypatch):
    dataset, _ = fixture_dataset(tmp_path / "fixture")
    model, calls = fake_predictors(monkeypatch)
    count = 0
    def deadline(_):
        nonlocal count
        count += 1
        if count == 4:
            raise TimeoutError("synthetic construct deadline")
    with monkeypatch.context() as patch:
        patch.setattr(pg, "check_deadline", deadline)
        with pytest.raises(TimeoutError):
            shared_evaluation.evaluate_shared(model, dataset, "cpu", shared_validation_execution=pg.EXECUTION)
    assert calls == []
    def mutate(model, groups, device, **kwargs):
        dataset.path.write_bytes(dataset.path.read_bytes() + b"\n")
        return [.5] * sum(len(g.targets) for g in groups)
    monkeypatch.setattr(pg, "predict_prefix_groups", mutate)
    path = tmp_path / "race.csv"
    with pytest.raises(ValueError, match="changed during"):
        shared_evaluation.evaluate_shared(model, dataset, "cpu", path, shared_validation_execution=pg.EXECUTION)
    assert not path.exists() and not path.with_suffix(".csv.tmp").exists()


class FixtureModel(torch.nn.Module):
    """Identity-only engine stub; its forward is never invoked."""
    def __init__(self, name):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.model_name = name
        self.ktlab_model_config = {}
        self.ktlab_source_sha256 = "synthetic-model"
        self.ktlab_upstream_commit = "synthetic-commit"
        self.ktlab_import_binding_only = True
        self.n_pid = 0
    def forward(self, *args, **kwargs):
        raise AssertionError("Identity stub must never be forwarded")


def test_forged_selector_rejected_without_forward_and_training_state_restored():
    group, = list(pg.iter_prefix_groups(artificial_contexts((5,))))
    model = FixtureModel("akt")
    model.train()
    bad = pg.PrefixGroup(group.context, (("assist2015:student:0:4", 1),))
    with pytest.raises(ValueError, match="selector"):
        pg.predict_prefix_groups(model, [bad], "cpu")
    assert model.training
    model.model_name = "mockt"
    with pytest.raises(ValueError, match="MoC"):
        pg.predict_prefix_groups(model, [group], "cpu")


@pytest.mark.parametrize("name", ["akt", "ktst_mean"])
def test_engine_epoch0_config_metadata_checkpoint_and_reload_bind_execution(tmp_path, monkeypatch, name):
    dataset, manifest_path = fixture_dataset(tmp_path / "fixture")
    _, calls = fake_predictors(monkeypatch)
    class Training:
        path, audit = dataset.path, {}
        def __init__(self, *args, **kwargs):
            pass
        def __len__(self):
            return 1
        def __getitem__(self, _):
            raise AssertionError("No training iteration is allowed")
        def close(self):
            pass
    def build(*args, **kwargs):
        return FixtureModel(name)
    monkeypatch.setattr(engine, "build_model", build)
    monkeypatch.setattr(engine, "dataset_for_split", lambda *args, **kwargs: Training())
    monkeypatch.setattr(ktst_engine, "_components", lambda: (Training, None, build, None, lambda *args: []))
    from ktlab import ktst_model
    monkeypatch.setattr(ktst_model, "build_ktst_model", build)
    class StoppedBeforeOptimizer(Exception):
        pass
    def stop(*args, **kwargs):
        raise StoppedBeforeOptimizer()
    monkeypatch.setattr(torch.optim, "Adam", stop)
    config = {"protocol": pg.PROTOCOL, "dataset": "assist2015", "model": name,
              "dataset_manifest": str(manifest_path), "shared_validation_execution": pg.EXECUTION,
              "device": "cpu", "fold": 0, "seed": 42, "epochs": 1, "patience": 3,
              "learning_rate": 1e-4, "theta_learning_rate": 1e-4, "_batch_approved": True}
    run = tmp_path / "run"
    with pytest.raises(StoppedBeforeOptimizer):
        engine.train(config, run)
    metadata = json.loads((run / "metadata.json").read_text())
    resolved = json.loads((run / "resolved_config.json").read_text())
    checkpoint = torch.load(run / "best.pt", weights_only=True)
    initial = json.loads((run / "initial_valid_metrics.json").read_text())
    for record in (metadata, resolved, checkpoint, initial):
        assert record["shared_validation_execution"] == pg.EXECUTION
    assert checkpoint["epoch"] == 0 and checkpoint["checkpoint_stage"] == "initial_untrained"
    assert metadata["optimizer_steps"] == 0 and not checkpoint["epoch_completed"]
    result = engine.evaluate_run(run)
    assert result["shared_validation_execution"] == pg.EXECUTION
    assert result["execution"]["group_rows"] == 2 and len(calls) == 2
    with pytest.raises(PermissionError, match="not implemented"):
        engine.evaluate_run(run, "test", allow_test=True)
    checkpoint["shared_validation_execution"] = "forged"
    torch.save(checkpoint, run / "best.pt")
    metadata["checkpoint_sha256"] = sha256_file(run / "best.pt")
    (run / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shared_validation_execution"):
        engine.evaluate_run(run)
    checkpoint["shared_validation_execution"] = pg.EXECUTION
    torch.save(checkpoint, run / "best.pt")
    metadata["checkpoint_sha256"] = sha256_file(run / "best.pt")
    (run / "metadata.json").write_text(json.dumps(metadata))
    # Removing a config field is a changed config, never a valid migration.
    resolved.pop("shared_validation_execution")
    (run / "resolved_config.json").write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="config_sha256"):
        engine.evaluate_run(run)


def _real_model(name, budget):
    assert os.environ.get(OPT_IN) == "1"
    assert budget["models"] < 5
    budget["models"] += 1  # reject excess before construction
    torch.manual_seed(42)
    if name.startswith("ktst_"):
        model = build_ktst_model(name, DATA, SET_MODEL, "cpu")
    else:
        settings = dict(KC_MODEL)
        if name == "ukt":
            settings.update(final_fc_dim2=16, use_CL=True, use_uncertainty_aug=True)
        model = adapters.build_model(name, DATA, settings, "cpu")
    def before_forward(module, inputs):
        assert not torch.is_grad_enabled(), "S008 permits no grad-enabled forward"
        assert budget["forwards"] < 128, "S008 forward ceiling exceeded before call"
        budget["forwards"] += 1
    model.register_forward_pre_hook(before_forward)
    return model


@pytest.mark.parametrize("name", ["akt", "simplekt", "ukt", "ktst_mean", "ktst_mhsa"])
def test_real_cpu_original_oracle_causality_padding_reload_and_evaluator(tmp_path, cpu_budget, name):
    if os.environ.get(OPT_IN) != "1":
        pytest.skip("S008 frozen opt-in is required before real model construction")
    model = _real_model(name, cpu_budget)
    start_count = cpu_budget["forwards"]
    contexts = artificial_contexts()
    groups = list(pg.iter_prefix_groups(contexts))
    model.train()
    original = np.asarray(shared_adapters.predict_shared(model, contexts, "cpu"))  # 8 calls
    packed = np.asarray(pg.predict_prefix_groups(model, groups, "cpu"))  # 1
    assert model.training
    np.testing.assert_allclose(packed, original, rtol=0, atol=1e-6)
    assert original.shape == packed.shape == (235,)
    singleton = pg.predict_prefix_groups(model, [groups[-1]], "cpu")  # 1
    np.testing.assert_allclose(singleton, original[-1:], rtol=0, atol=1e-6)

    changed_response = artificial_contexts(response_changes={(0, 10): 1})
    response_predictions = np.asarray(pg.predict_prefix_groups(model, list(pg.iter_prefix_groups(changed_response)), "cpu"))  # 1
    np.testing.assert_allclose(response_predictions[:10], packed[:10], rtol=0, atol=1e-6)
    assert np.max(np.abs(response_predictions[10:34] - packed[10:34])) > 1e-8
    future = artificial_contexts(response_changes={(0, 20): 1}, concept_changes={(0, 20): 1})
    future_predictions = np.asarray(pg.predict_prefix_groups(model, list(pg.iter_prefix_groups(future)), "cpu"))  # 1
    np.testing.assert_allclose(future_predictions[:19], packed[:19], rtol=0, atol=1e-6)
    current_kc = artificial_contexts(concept_changes={(0, 10): 6})
    current_predictions = np.asarray(pg.predict_prefix_groups(model, list(pg.iter_prefix_groups(current_kc)), "cpu"))  # 1
    np.testing.assert_allclose(current_predictions[:9], packed[:9], rtol=0, atol=1e-6)
    assert abs(current_predictions[9] - packed[9]) > 1e-8

    # The final dummy response itself is also causally excluded, not merely
    # held constant to conceal a wrong shift. Original wrapper loss is ignored.
    final = groups[-1].context
    model.eval()
    with torch.no_grad():
        if name.startswith("ktst_"):
            batch = shared_adapters.collate_shared_ktst([final], 1)
            batch["data"][0, batch["target_positions"][0], -1] = 1
            scores, _ = forward_ktst(model, {"data": batch["data"], "padding_mask": batch["padding_mask"]})  # 1
        else:
            batch = shared_adapters.collate_shared_kc([final])
            position = int(batch["target_positions"][0])
            batch["responses"][0, position] = 1
            batch["shft_rseqs"][0, position - 1] = 1
            if name == "akt":
                scores, _ = model(batch["concepts"], batch["responses"], batch["questions"])  # 1
            else:
                scores = model(batch)  # 1
    assert abs(float(scores[0, batch["target_positions"][0]]) - singleton[0]) <= 1e-6
    assert all(parameter.grad is None for parameter in model.parameters())

    path = tmp_path / "artificial-weights.pt"
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path, weights_only=True))
    reloaded = pg.predict_prefix_groups(model, groups, "cpu")  # 1
    np.testing.assert_allclose(reloaded, packed, rtol=0, atol=1e-6)

    dataset, _ = fixture_dataset(tmp_path / "fixture")
    old_path, group_path = tmp_path / "old.csv", tmp_path / "group.csv"
    old = shared_evaluation.evaluate_shared(model, dataset, "cpu", old_path, compute_metrics=False)  # 1
    grouped = shared_evaluation.evaluate_shared(model, dataset, "cpu", group_path, compute_metrics=False,
                                                shared_validation_execution=pg.EXECUTION)  # 1
    with old_path.open() as stream:
        rows_old = list(csv.DictReader(stream))
    with group_path.open() as stream:
        rows_group = list(csv.DictReader(stream))
    for a, b in zip(rows_old, rows_group):
        assert {k: v for k, v in a.items() if k != "probability"} == {k: v for k, v in b.items() if k != "probability"}
        assert abs(float(a["probability"]) - float(b["probability"])) <= 1e-6
    assert old["target_sha256"] == grouped["target_sha256"]
    assert old["context_sha256"] == grouped["context_sha256"]
    assert "auc" not in old and "auc" not in grouped
    assert grouped["execution"]["forward_calls"] == 1
    assert cpu_budget["forwards"] - start_count == 17
    print("S008_MODEL=" + json.dumps({"model": name, "forwards": 17,
        "oracle_max_abs_error": float(np.max(np.abs(packed - original))),
        "history_positive_max_abs_change": float(np.max(np.abs(response_predictions[10:34] - packed[10:34]))),
        "current_kc_abs_change": float(abs(current_predictions[9] - packed[9]))}, sort_keys=True))
