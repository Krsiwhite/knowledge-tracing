"""S010 synthetic acceptance; execution requires the separately bound opt-in.

No existing data, checkpoints, models, or GPU are read.  Real calls below are
CPU d16 random initializations; optimizer construction and backward are banned.
Static exact budget: 6 models, 36 full forwards, 8 autograd.grad calls, 0 updates.
The runner must freeze this file and the implementation before invoking pytest.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ktlab import engine, ktst_engine
from ktlab import ktst_history_training as h
from ktlab.ktst_model import build_ktst_model, load_ktst_reference
from ktlab.shared_adapters import PROTOCOL, collate_shared_ktst


SMALL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2,
         "num_layers_tf_enc": 2, "num_layers_tf_dec": 2,
         "num_layers_agg": 1, "dim_feedforward": 32,
         "dim_classifier": 24, "dropout": 0.0}
RECIPES = ("sampled_chunk_prefix_v1", "sampled_complete_suffix_v1")
COUNTS = {"models": 0, "forwards": 0, "gradients": 0, "optimizers": 0}
LIMITS = {"models": 12, "forwards": 128, "gradients": 12}
ALLOW_REAL = False
STARTED = None


def _charge(kind):
    assert ALLOW_REAL, "A pure fixture unexpectedly called a real model or derivative"
    assert time.monotonic() - STARTED < 300, "S010 CPU elapsed budget exceeded before call"
    assert COUNTS[kind] < LIMITS[kind], f"S010 {kind} budget exceeded before call"
    COUNTS[kind] += 1


@pytest.fixture(scope="module", autouse=True)
def execution_budget():
    global STARTED
    if os.environ.get("KTLAB_RUN_KTST_HISTORY_CPU_ACCEPTANCE") != "1":
        pytest.skip("S010 requires the hash-bound CPU acceptance runner")
    STARTED = time.monotonic()
    patch = pytest.MonkeyPatch()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    original = load_ktst_reference("pkg.model.KTST").Model
    constructor, autograd_grad = original.__init__, torch.autograd.grad

    def guarded_init(self, *args, **kwargs):
        _charge("models")
        constructor(self, *args, **kwargs)
        self.register_forward_pre_hook(lambda *unused: _charge("forwards"))

    def guarded_grad(*args, **kwargs):
        _charge("gradients")
        return autograd_grad(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("S010 forbids optimizer construction, backward, and GPU")

    patch.setattr(original, "__init__", guarded_init)
    patch.setattr(torch.autograd, "grad", guarded_grad)
    patch.setattr(torch.Tensor, "backward", forbidden)
    patch.setattr(torch.autograd, "backward", forbidden)
    patch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    patch.setattr(torch.cuda, "is_available", lambda: False)
    patch.setattr(torch.nn.Module, "cuda", forbidden)
    patch.setattr(torch.Tensor, "cuda", forbidden)
    yield
    patch.undo()
    torch.set_num_threads(previous_threads)
    print("S010_COUNTS=" + json.dumps(COUNTS, sort_keys=True))
    assert COUNTS == {"models": 6, "forwards": 36, "gradients": 8, "optimizers": 0}


@pytest.fixture
def real_cpu():
    global ALLOW_REAL
    ALLOW_REAL = True
    try:
        yield
    finally:
        ALLOW_REAL = False


def _event(concepts, response=0, question=0):
    return (list(concepts), response, question)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seal(manifest):
    root = Path(manifest["data_dir"])
    manifest["files"] = {
        name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
               "bytes": (root / name).stat().st_size}
        for name in ("train_valid.csv", "train_valid_sequences.csv")}
    return manifest


def _fixture(tmp_path, students, *, concept_only=False):
    """Write artificial events then literal nonoverlapping 200-raw-KC chunks.

    This is only fixture serialization, not the eligibility/history algorithm.
    Each student's events are specified by hand by the individual test.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    originals, chunks = [], []
    for uid, fold, events in students:
        columns = {key: [] for key in ("concepts", "responses")}
        if not concept_only:
            columns.update(questions=[], is_repeat=[])
        for concepts, response, question in events:
            columns["concepts"].extend(concepts)
            columns["responses"].extend([response] * len(concepts))
            if not concept_only:
                columns["questions"].extend([question] * len(concepts))
                columns["is_repeat"].extend([0] + [1] * (len(concepts) - 1))
        original = {"uid": uid, "fold": fold}
        original.update({key: ",".join(map(str, values)) for key, values in columns.items()})
        originals.append(original)
        size = len(columns["responses"])
        for start in range(0, size, 200):
            length = min(200, size - start)
            if length < 3:
                continue
            row = {"uid": uid, "fold": fold}
            row.update({key: ",".join(map(str, values[start:start + length] + [-1] * (200 - length)))
                        for key, values in columns.items()})
            row["selectmasks"] = ",".join(map(str, [1] * length + [-1] * (200 - length)))
            chunks.append(row)
    _write_csv(tmp_path / "train_valid.csv", originals)
    _write_csv(tmp_path / "train_valid_sequences.csv", chunks)
    return _seal({"dataset": "synthetic", "data_dir": str(tmp_path.resolve()),
                  "data_config": {"num_q": 0 if concept_only else 8, "num_c": 8,
                                  "max_concepts": 1 if concept_only else 3,
                                  "maxlen": 200, "min_seq_len": 3, "folds": [0, 1, 2, 3, 4],
                                  "train_valid_file": "train_valid_sequences.csv",
                                  "train_valid_original_file": "train_valid.csv"}})


def _dataset(manifest, recipe=RECIPES[0], *, k=2, seed=42, epoch=1):
    data = h.SampledKTSTHistoryDataset(manifest, 0, recipe=recipe,
                                     samples_per_chunk=k, seed=seed)
    data.set_epoch(epoch)
    return data


def _context(*, concept_only=False, ordinal=3, uid="artificial"):
    return {"protocol": PROTOCOL, "event_id": f"synthetic:{uid}:{ordinal}",
            "uid": uid, "fold": 1, "target_index": ordinal,
            "history_start_index": 0,
            "history": [{"event_id": f"synthetic:{uid}:{i}",
                         "question": None if concept_only else i % 8,
                         "concepts": [i % 8] if concept_only else [i % 8, (i + 2) % 8],
                         "response": i % 2} for i in range(ordinal)],
            "query": {"question": None if concept_only else ordinal % 8,
                      "concepts": [ordinal % 8] if concept_only else [ordinal % 8, (ordinal + 1) % 8]}}


def test_stateless_sampling_is_paired_with_replacement_and_does_not_touch_rng():
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    chunk_id = hashlib.sha256(b"fixture-chunk").hexdigest()
    first = h.sample_target_indices(42, 1, chunk_id, 2, 8)
    assert len(first) == 8 and set(first) <= {0, 1}
    assert len(set(first)) < len(first), "K repeats must not be silently deduplicated"
    assert h.sample_target_indices(42, 1, chunk_id, 2, 8) == first
    assert h.sample_target_indices(42, 2, chunk_id, 1, 8) == (0,) * 8
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    # The two recipes are deliberately absent from this pure sampler's inputs.
    draws = {tuple(h.sample_target_indices(42, epoch, chunk_id, 19, 8))
             for epoch in (1, 2, 3)}
    assert len(draws) > 1, "A frozen per-chunk target repeated forever is not epoch sampling"


@pytest.mark.parametrize("change", [
    {"ktst_history_training_recipe": "unknown"},
    {"ktst_history_samples_per_chunk": True},
    {"ktst_history_samples_per_chunk": 0},
    {"ktst_history_samples_per_chunk": 9},
    {"ktst_history_samples_per_chunk": 1.0},
    {"ktst_history_samples_per_chunk": "1"},
    {"ktst_history_samples_per_chunk": None}, {"batch_size": 31},
    {"ktst_history_training_source_sha256": "0" * 64},
    {"model": "ktst_mhsa"}, {"model": "akt"},
    {"protocol": "ktlab-ktst-reference-dev-v1"}, {"fold": True}, {"fold": 1},
])
def test_history_config_rejects_unsupported_or_ambiguous_contract(change):
    config = {"model": "ktst_mean", "protocol": PROTOCOL, "fold": 0, "seed": 42,
              "ktst_history_training_recipe": RECIPES[0], "ktst_history_samples_per_chunk": 2}
    with pytest.raises(ValueError):
        h.validate_history_config({**config, **change})


def test_default_path_and_explicit_history_identity_are_distinct():
    original = {"model": "ktst_mean", "protocol": PROTOCOL, "fold": 0, "seed": 42}
    before = copy.deepcopy(original)
    assert h.validate_history_config(original) is None
    assert original == before
    with pytest.raises(ValueError):
        h.validate_history_config({**original, "ktst_history_samples_per_chunk": 2})
    with pytest.raises(ValueError):
        h.validate_history_config({**original, "ktst_history_training_source_sha256": "0" * 64})
    identities = []
    for recipe, k in ((RECIPES[0], 1), (RECIPES[0], 2), (RECIPES[1], 2)):
        config = {**original, "ktst_history_training_recipe": recipe,
                  "ktst_history_samples_per_chunk": k}
        h.validate_history_config(config)
        identities.append(h.history_training_identity(config))
    assert len({json.dumps(identity, sort_keys=True) for identity in identities}) == 3


def test_complete_targets_partial_history_tail_and_fold_membership(tmp_path):
    singles = [_event([i % 8], i % 2, i % 8) for i in range(203)]
    cross = singles[:199] + [_event([1, 2, 3], 1, 7)] + singles[:3]
    tail = singles[:199] + [_event([1, 2, 3], 1, 7)]
    duplicate = [_event([0, 0, 1], 1, 0), _event([2], 0, 1), _event([3], 1, 2)]
    manifest = _fixture(tmp_path, [("plain", 1, singles), ("cross", 2, cross),
                                   ("tail", 3, tail), ("dedup", 4, duplicate),
                                   ("held-out", 0, singles[:3])])
    short, long = [_dataset(manifest, recipe, k=8) for recipe in RECIPES]
    # plain contributes 199+2, cross 198+0, tail 198, dedup 2.
    assert len(short) == len(long) == 5
    assert short.audit["eligible_unique_targets"] == long.audit["eligible_unique_targets"] == 599
    assert short.audit["unconsumed_kc_positions"] == 2
    expected = {"plain": set(range(1, 200)) | {201, 202}, "cross": set(range(1, 199)),
                "tail": set(range(1, 199)), "dedup": {1, 2}}
    expanded = False
    for index in range(len(short)):
        left, right = short[index], long[index]
        assert torch.equal(left["labels"], right["labels"])
        assert torch.equal(left["weights"], right["weights"])
        assert left["sampling_sha256"] == right["sampling_sha256"]
        for a, b in zip(left["contexts"], right["contexts"]):
            assert a["event_id"] == b["event_id"] and a["query"] == b["query"]
            uid, ordinal = a["uid"], a["target_index"]
            assert uid in expected and ordinal in expected[uid]
            assert "label" not in a and "label" not in b
            assert len(a["history"]) >= 1
            assert sum(len(event["concepts"]) for event in b["history"]) + len(b["query"]["concepts"]) <= 200
            if uid == "plain" and ordinal >= 201:
                assert a["history"][0]["event_id"] == "synthetic:plain:200"
                assert b["history"][0]["event_id"] == f"synthetic:plain:{ordinal - 199}"
                expanded = True
            else:
                assert a["history"] == b["history"]
            if uid == "dedup":
                assert a["history"][0]["concepts"] == [0, 1]
    assert expanded, "The fixture must exercise genuinely different allowed histories"


def test_concept_only_events_and_repeated_sample_weights(tmp_path):
    manifest = _fixture(tmp_path, [("u", 1, [_event([2], 1), _event([2], 0), _event([3], 1)]),
                                   ("v", 0, [_event([0], 0)] * 3)], concept_only=True)
    dataset = _dataset(manifest, k=8)
    item = dataset[0]
    assert len(dataset) == 1 and len(item["contexts"]) == 8
    assert item["weights"].tolist() == [2 / 8] * 8
    assert float(item["weights"].sum()) == 2
    assert len({context["event_id"] for context in item["contexts"]}) <= 2
    assert all(context["query"]["question"] is None for context in item["contexts"])
    for context in item["contexts"]:
        assert context["target_index"] in (1, 2)
        assert context["history"][0]["response"] == 1
    packed = h.collate_history_chunks([item, item])
    assert packed["chunk_count"] == 2 and len(packed["contexts"]) == 16
    assert packed["denominator"] == 4


def test_source_change_and_wrong_chunk_slice_are_rejected(tmp_path):
    events = [_event([0], 0, 0), _event([1], 1, 1), _event([2], 0, 2)]
    manifest = _fixture(tmp_path, [("u", 1, events), ("v", 0, events)])
    source = tmp_path / "train_valid_sequences.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    responses = rows[0]["responses"].split(",")
    responses[0] = "1"
    rows[0]["responses"] = ",".join(responses)
    _write_csv(source, rows)
    with pytest.raises(ValueError):
        _dataset(manifest)
    _seal(manifest)
    with pytest.raises(ValueError):
        _dataset(manifest)


def test_weighted_bce_uses_global_denominator_and_extreme_logits():
    logits = torch.tensor([-80., 3., .7], dtype=torch.float64)
    labels = torch.tensor([0., 1., 1.], dtype=torch.float64)
    weights = torch.tensor([2., .5, 1.], dtype=torch.float64)
    expected = ((torch.nn.functional.softplus(logits) - labels * logits) * weights).sum() / 3.5
    actual = h.weighted_target_loss(logits, labels, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    micro = sum(h.weighted_target_loss(logits[start:stop], labels[start:stop], weights[start:stop], denominator=3.5)
                for start, stop in ((0, 2), (2, 3)))
    torch.testing.assert_close(micro, expected, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(actual)


def _flat(contexts, *, labels=None, weights=None):
    n = len(contexts)
    weights = torch.tensor(weights or [1.] * n, dtype=torch.float64)
    return {"contexts": contexts, "labels": torch.tensor(labels or [0.] * n, dtype=torch.float32),
            "weights": weights, "denominator": weights.sum(), "chunk_count": n}


def test_microbatch_normalization_and_label_isolation_without_model(monkeypatch):
    calls, inputs = [], []

    def fake_logits(model, projected):
        rows = torch.arange(projected["data"].shape[0])
        target = projected["target_positions"]
        calls.append(len(rows))
        inputs.append({key: value.clone() for key, value in projected.items() if isinstance(value, torch.Tensor)})
        assert projected["data"].shape[1] == 200
        assert torch.count_nonzero(projected["data"][rows, target, -1]) == 0
        return projected["data"][rows, target, -2].float() - 3.

    monkeypatch.setattr(h, "target_logits", fake_logits)
    contexts = [_context(ordinal=i) for i in (2, 3, 4)]
    batch = _flat(contexts, labels=[1., 0., 1.], weights=[3., 2., .5])
    model = SimpleNamespace(max_concepts=3)
    actual = sum(h.iter_weighted_target_losses(model, batch, "cpu", 2))
    logits = torch.tensor([-1., 0., 1.])
    expected = ((torch.nn.functional.softplus(logits) - batch["labels"] * logits)
                * batch["weights"]).sum() / 5.5
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
    assert calls == [2, 1]
    original_inputs = inputs[:]
    calls.clear(); inputs.clear()
    changed = {**batch, "labels": 1 - batch["labels"]}
    changed_loss = sum(h.iter_weighted_target_losses(model, changed, "cpu", 2))
    assert float(changed_loss) != float(actual)
    for first, second in zip(original_inputs, inputs):
        assert first.keys() == second.keys()
        assert all(torch.equal(first[key], second[key]) for key in first)
    with pytest.raises(ValueError, match="denominator"):
        list(h.iter_weighted_target_losses(model, {**batch, "denominator": 1.}, "cpu", 2))
    calls.clear()
    with pytest.raises(TimeoutError):
        list(h.iter_weighted_target_losses(model, batch, "cpu", 2, deadline=0))
    assert calls == []


def _raw_logits(model, projected):
    """Independent original classifier observation, not the production helper.

    Deliberately capture every position, then select in the test. This detects
    an off-by-one selection even if the production helper produces finite BCE.
    """
    captured = []
    handle = model.clf[-1].register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        model(data=projected["data"], padding_mask=projected["padding_mask"])
    finally:
        handle.remove()
    assert len(captured) == 1
    return captured[0].squeeze(-1)


@pytest.mark.parametrize("concept_only", [False, True])
@pytest.mark.parametrize("interaction", [None, "signed_response", "response_independent"])
def test_original_logits_causality_and_optional_interaction(real_cpu, concept_only, interaction):
    data_config = {"num_q": 0 if concept_only else 8, "num_c": 8,
                   "max_concepts": 1 if concept_only else 3, "maxlen": 200}
    torch.manual_seed(4242)
    model = build_ktst_model("ktst_mean", data_config, SMALL, "cpu", interaction_recipe=interaction)
    model.eval()
    if interaction is not None:
        with torch.no_grad():
            model.interaction_vectors.copy_(torch.arange(128).reshape(8, 16).sin() * .2)
    contexts = [_context(concept_only=concept_only, ordinal=i, uid=f"row{i}") for i in (3, 5)]
    projected = collate_shared_ktst(contexts, data_config["max_concepts"])
    assert projected["data"].shape == (2, 200, data_config["max_concepts"] + 2)
    labels, weights = torch.tensor([1., 0.]), torch.tensor([2., .5], dtype=torch.float64)
    actual = h.target_logits(model, projected)
    oracle_all = _raw_logits(model, projected)
    oracle = torch.stack([oracle_all[0, 3], oracle_all[1, 5]])
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
    actual_loss = h.weighted_target_loss(actual, labels, weights)
    oracle_loss = ((torch.nn.functional.softplus(oracle) - labels * oracle) * weights).sum() / 2.5
    torch.testing.assert_close(actual_loss, oracle_loss, rtol=1e-7, atol=1e-8)

    gradient_case = not concept_only and interaction in (None, "signed_response")
    if gradient_case:
        parameters = tuple(model.parameters())
        actual_gradient = torch.autograd.grad(actual_loss, parameters, allow_unused=True)
        oracle_gradient = torch.autograd.grad(oracle_loss, parameters, allow_unused=True)
        for first, second in zip(actual_gradient, oracle_gradient):
            assert (first is None) == (second is None)
            if first is not None:
                assert torch.isfinite(first).all()
                torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-7)
        assert any(g is not None and g.abs().sum() > 0 for g in actual_gradient)
        if interaction == "signed_response":
            index = next(i for i, p in enumerate(parameters) if p is model.interaction_vectors)
            assert actual_gradient[index] is not None and actual_gradient[index].abs().sum() > 0

    with torch.no_grad():
        changed = {key: value.clone() if isinstance(value, torch.Tensor) else value
                   for key, value in projected.items()}
        changed["data"][0, 3, -1] = 1
        changed["data"][1, 5, -1] = 1
        altered = _raw_logits(model, changed)
        torch.testing.assert_close(torch.stack([altered[0, 3], altered[1, 5]]), actual.detach(), rtol=0, atol=0)
        with pytest.raises(ValueError, match="dummy zero"):
            h.target_logits(model, changed)

        future = {key: value.clone() if isinstance(value, torch.Tensor) else value
                  for key, value in projected.items()}
        for row, position in ((0, 4), (1, 6)):
            future["data"][row, position] = -1
            future["data"][row, position, 0] = 7
            future["data"][row, position, -2:] = torch.tensor([0 if concept_only else 7, 1])
            future["padding_mask"][row, position] = True
        after = _raw_logits(model, future)
        torch.testing.assert_close(torch.stack([after[0, 3], after[1, 5]]), actual.detach(), rtol=0, atol=0)
        past = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in projected.items()}
        past["data"][:, 1, -1] = 1 - past["data"][:, 1, -1]
        past_logits = h.target_logits(model, past)
        assert (past_logits - actual.detach()).abs().max() > 1e-8, "Historical answer positive control is ineffective"

    if gradient_case:
        full_batch = _flat(contexts, labels=[1., 0.], weights=[2., .5])
        full_loss = sum(h.iter_weighted_target_losses(model, full_batch, "cpu", 2))
        full_gradient = torch.autograd.grad(full_loss, parameters, allow_unused=True)
        micro_loss = sum(h.iter_weighted_target_losses(model, full_batch, "cpu", 1))
        micro_gradient = torch.autograd.grad(micro_loss, parameters, allow_unused=True)
        torch.testing.assert_close(micro_loss, full_loss, rtol=1e-6, atol=1e-7)
        for full, micro in zip(full_gradient, micro_gradient):
            assert (full is None) == (micro is None)
            if full is not None:
                torch.testing.assert_close(full, micro, rtol=3e-5, atol=3e-7)
        assert all(parameter.grad is None for parameter in parameters)


@pytest.mark.parametrize("recipe,partial,timeout", [(None, False, False), (RECIPES[0], False, False),
                                                  (RECIPES[1], True, False), (RECIPES[1], False, True)])
def test_engine_one_step_per_chunk_batch_default_and_identity(tmp_path, monkeypatch, recipe, partial, timeout):
    """Run only engine control flow with inert objects; never a real optimizer.

    A two-contribution history batch must produce zero, backward, backward,
    clip, step exactly once, not one step per microbatch or sampled target.
    """
    calls = []
    source = tmp_path / "fixture-only.csv"
    source.write_text("artificial control-flow fixture\n", encoding="utf-8")

    class FakeLoss:
        ndim = 0

        def __init__(self, value):
            self.value = value

        def backward(self):
            calls.append("fake_backward")

        def detach(self):
            return self.value

    class FakeModel:
        model_name = "ktst_mean"
        ktlab_model_config = dict(SMALL)
        ktlab_source_sha256 = "synthetic-model-source"
        ktlab_upstream_commit = "synthetic-upstream"
        ktlab_import_binding_only = True

        def __init__(self):
            self.parameter = torch.tensor([0.])

        def parameters(self):
            return (self.parameter,)

        def state_dict(self):
            return {"fixture": self.parameter}

        def load_state_dict(self, state):
            assert set(state) == {"fixture"}

        def train(self):
            return self

    class FakeDataset:
        path = source

        def __init__(self, manifest=None, fold=0, split="train", **kwargs):
            self.audit = {"protocol": PROTOCOL, "split": split, "fold": fold,
                          "source_sha256": "source", "target_sha256": "targets", "context_sha256": "contexts"}

        def __len__(self):
            return 2

        def set_epoch(self, epoch):
            calls.append(("set_epoch", epoch))

        def epoch_sampling_audit(self):
            return {"sampled_rows": 8, "scope": "planned full epoch"}

        def verify_source(self):
            calls.append("verify_source")

        def close(self):
            pass

    class FakeOptimizer:
        def __init__(self, *args, **kwargs):
            calls.append("fake_optimizer")

        def zero_grad(self, **kwargs):
            calls.append("zero")

        def step(self):
            calls.append("step")

    model = FakeModel()
    fake_batch = {"score_mask": torch.tensor([True]), "contexts": [{}, {}, {}, {}],
                  "chunk_count": 2, "sampling_sha256": "12" * 32, "sampled_unique_count": 3}

    def fake_contributions(*args, **kwargs):
        assert args[3] == 2
        yield FakeLoss(.25)
        if timeout:
            raise TimeoutError("Artificial timeout before the second microbatch")
        yield FakeLoss(.75)

    def fake_forward(*args, **kwargs):
        calls.append("legacy_forward")
        return None, FakeLoss(1.)

    validation_calls = []

    def fake_valid(*args, **kwargs):
        validation_calls.append(1)
        return {"auc": .5 if len(validation_calls) == 1 else .6, "n": 2, "protocol": PROTOCOL}

    real_isfinite = torch.isfinite
    monkeypatch.setattr(torch, "isfinite", lambda value: torch.tensor(True) if isinstance(value, FakeLoss) else real_isfinite(value))
    monkeypatch.setattr(torch.optim, "Adam", FakeOptimizer)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args: calls.append("clip"))
    monkeypatch.setattr(ktst_engine, "_components", lambda: (FakeDataset, None, lambda *args, **kwargs: model,
                                                           fake_forward, lambda *args: []))
    monkeypatch.setattr(ktst_engine, "_loader", lambda *args, **kwargs: [fake_batch, fake_batch])
    monkeypatch.setattr(ktst_engine, "move_batch", lambda batch, device: batch)
    monkeypatch.setattr(ktst_engine, "source_fingerprint", lambda: "synthetic-implementation")
    monkeypatch.setattr(ktst_engine, "git_revision", lambda: "synthetic")
    monkeypatch.setattr(h, "SampledKTSTHistoryDataset", FakeDataset)
    monkeypatch.setattr(h, "iter_weighted_target_losses", fake_contributions)
    monkeypatch.setattr(engine, "source_fingerprint", lambda: "synthetic-implementation")
    monkeypatch.setattr(engine, "read_manifest", lambda config: {"dataset": "synthetic", "data_config": {},
                                                               "_manifest_sha256": "manifest"})
    monkeypatch.setattr(engine, "_shared_dataset", lambda *args: FakeDataset(split="valid"))
    monkeypatch.setattr(engine, "_shared_evaluate", fake_valid)
    config = {"_batch_approved": True, "model": "ktst_mean", "protocol": PROTOCOL,
              "dataset": "synthetic", "dataset_manifest": str(tmp_path / "manifest.json"),
              "fold": 0, "seed": 42, "device": "cpu", "epochs": 1, "patience": 3,
              "batch_size": 32, "learning_rate": .001, "theta_learning_rate": .001,
              "max_rows_per_forward": 2, "model_config": dict(SMALL)}
    if recipe is not None:
        config.update(ktst_history_training_recipe=recipe, ktst_history_samples_per_chunk=2)
    if partial:
        config["max_train_steps"] = 1
    result = ktst_engine.train_ktst(config, tmp_path / "run")
    metadata = result["metadata"]
    resolved = json.loads((tmp_path / "run/resolved_config.json").read_text())
    if recipe is not None:
        for key, value in h.history_training_identity(config).items():
            assert resolved[key] == metadata[key] == value
    if timeout:
        assert metadata["status"] == "budget_exhausted" and metadata["optimizer_steps"] == 0
        assert result["valid"] is None
        assert [entry for entry in calls if entry in ("zero", "fake_backward", "clip", "step")] == ["zero", "fake_backward"]
        consumption = metadata["history_training_consumption"]
        assert consumption["completed_microbatches"] == 1
        assert consumption["completed_forward_rows"] == 2
        assert consumption["committed_chunks"] == consumption["committed_sampled_rows"] == 0
        assert not (tmp_path / "run/history.jsonl").exists()
        assert torch.load(tmp_path / "run/best.pt", weights_only=True)["epoch"] == 0
        return
    updates = 1 if partial else 2
    assert metadata["status"] == "completed" and metadata["optimizer_steps"] == updates
    relevant = [entry for entry in calls if entry in ("zero", "fake_backward", "clip", "step")]
    pattern = (["zero", "fake_backward", "fake_backward", "clip", "step"] if recipe else
               ["zero", "fake_backward", "clip", "step"])
    assert relevant == pattern * updates
    checkpoint = torch.load(tmp_path / "run/best.pt", weights_only=True)
    history = json.loads((tmp_path / "run/history.jsonl").read_text())
    assert history["epoch_completed"] is (not partial)
    assert history["train_loss"] == 1.
    assert checkpoint["epoch_completed"] is (not partial)
    if recipe is None:
        assert calls.count("legacy_forward") == 2
        assert "ktst_history_training_recipe" not in checkpoint
        assert "ktst_history_training_recipe" not in metadata
        assert metadata["training_representation"] == "original_ktst_set_dense"
        assert metadata["loss"] == "upstream_BCEWithLogits_excluding_segment_first_event"
        assert "training_sampling" not in history
    else:
        assert "legacy_forward" not in calls
        assert ("set_epoch", 1) in calls
        for key, expected in h.history_training_identity(config).items():
            assert metadata[key] == checkpoint[key] == expected
        sampling = history["training_sampling"]
        assert sampling["consumed_chunks"] == updates * 2
        assert sampling["consumed_rows"] == updates * 4
        assert sampling["consumed_unique_targets"] == updates * 3
        assert sampling["epoch_completed"] is (not partial)
        assert metadata["training_representation"] == "sampled_complete_event_target_only"
        valid = FakeDataset(split="valid")
        with pytest.raises(ValueError, match="identity mismatch"):
            engine._shared_identity({**config, "ktst_history_samples_per_chunk": 3}, model, valid)
        unlabelled = {key: value for key, value in config.items()
                     if key not in ("ktst_history_training_recipe", "ktst_history_samples_per_chunk")}
        with pytest.raises(ValueError, match="relabeled"):
            engine._shared_identity(unlabelled, model, valid)
