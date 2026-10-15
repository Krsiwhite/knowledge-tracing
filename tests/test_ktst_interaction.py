"""S007 artificial CPU checks. Do not execute before the bound check is recorded.

Planned total: 10 d16 models, 52 complete forwards, 8 autograd.grad calls,
0 optimizers/steps, no real data/checkpoints/GPU. Guards reject over-budget
calls before they start. All saved tensors are created here in tmp_path.
"""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ktlab import engine, ktst_engine
from ktlab.ktst_interaction import RECIPES, interaction_source_identity
from ktlab.ktst_model import build_ktst_model, forward_ktst, load_ktst_reference, optimizer_parameter_groups
from ktlab.shared_adapters import PROTOCOL, collate_shared_ktst, predict_shared


SMALL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2,
         "num_layers_tf_dec": 1, "num_layers_tf_enc": 1, "num_layers_agg": 1,
         "dim_feedforward": 32, "dim_classifier": 24, "dropout": .1}
COUNTS = {"models": 0, "forwards": 0, "gradients": 0, "optimizers": 0}
LIMITS = {"models": 12, "forwards": 96, "gradients": 8}


def charge(kind):
    assert COUNTS[kind] < LIMITS[kind], f"S007 {kind} budget exceeded before call"
    COUNTS[kind] += 1


@pytest.fixture(scope="module", autouse=True)
def execution_budget():
    if os.environ.get("KTLAB_RUN_KTST_INTERACTION_CPU_ACCEPTANCE") != "1":
        pytest.skip("S007 real synthetic model/gradient checks require the hash-bound opt-in runner")
    patch = pytest.MonkeyPatch()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    original = load_ktst_reference("pkg.model.KTST").Model
    constructor, grad = original.__init__, torch.autograd.grad

    def guarded_init(self, *args, **kwargs):
        charge("models")
        constructor(self, *args, **kwargs)
        self.register_forward_pre_hook(lambda *unused: charge("forwards"))

    def guarded_grad(*args, **kwargs):
        charge("gradients")
        return grad(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("S007 forbids optimizer construction, backward, and GPU placement")

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
    torch.set_num_threads(previous)
    print("S007_COUNTS=" + json.dumps(COUNTS, sort_keys=True))
    assert COUNTS == {"models": 10, "forwards": 52, "gradients": 8, "optimizers": 0}


def batch_for(concept_only):
    width = 1 if concept_only else 3
    data = torch.full((2, 8, width + 2), -1, dtype=torch.long)
    sets = [[0, 2], [1], [2, 3, 4], [0, 5], [4, 6], [1, 6]]
    for row, length in ((0, 6), (1, 4)):
        for time in range(length):
            cs = [sets[time][0]] if concept_only else sets[time]
            data[row, time, :len(cs)] = torch.tensor(cs)
            data[row, time, -2:] = torch.tensor([0 if concept_only else time % 5, (time + row) % 2])
    return {"data": data, "padding_mask": data[..., -1] >= 0}


def forward(model, batch):
    return forward_ktst(model, batch)


def clone(batch):
    return {key: value.clone() for key, value in batch.items()}


def oracle(model, batch):
    """Pinned original full forward, injecting an independently looped KC mean.

The original calls its Transformer.forward directly, so this local temporary
replacement observes the exact pre-shift source without changing its math.
The root original forward bypasses __call__, hence its explicit budget charge.
    """
    charge("forwards")
    original = load_ktst_reference("pkg.model.KTST").Model.forward
    transformer_forward = model.model.forward
    data, mask = batch["data"], batch["padding_mask"]
    additions = []
    for row in range(len(data)):
        events = []
        for time in range(data.shape[1]):
            if mask[row, time]:
                cs = data[row, time, :-2]
                vector = torch.stack([model.interaction_vectors[int(c)] for c in cs if c >= 0]).mean(0)
                if model.ktlab_interaction_recipe == "signed_response":
                    vector = vector * (2 * int(data[row, time, -1]) - 1)
            else:
                vector = model.interaction_vectors.sum(0) * 0
            events.append(vector)
        additions.append(torch.stack(events))
    delta = torch.stack(additions)

    def inject(*, tgt, src):
        return transformer_forward(tgt=tgt, src=src + delta)

    model.model.forward = inject
    try:
        return original(model, **batch)
    finally:
        model.model.forward = transformer_forward


@pytest.mark.parametrize("concept_only", [False, True])
def test_paired_models_causality_gradients_and_reload(concept_only, tmp_path):
    data_config = {"num_q": 0 if concept_only else 5, "num_c": 7,
                   "max_concepts": 1 if concept_only else 3, "maxlen": 200}
    models, rngs = [], []
    for recipe in (None, *RECIPES):
        torch.manual_seed(42)
        models.append(build_ktst_model("ktst_mean", data_config, SMALL, "cpu", interaction_recipe=recipe))
        rngs.append(torch.get_rng_state().clone())
    base, signed, independent = models
    for other, rng in zip(models[1:], rngs[1:]):
        assert torch.equal(rng, rngs[0]), "Zero parameter construction must not consume RNG"
        for name, value in base.state_dict().items():
            assert torch.equal(value, other.state_dict()[name])
        assert set(other.state_dict()) - set(base.state_dict()) == {"interaction_vectors"}
        assert other.interaction_vectors.shape == (7, 16)
        assert torch.count_nonzero(other.interaction_vectors) == 0
        assert other.ktlab_import_binding_only is False
        source, expected = interaction_source_identity(base.ktlab_source_sha256, other.ktlab_interaction_recipe)
        assert other.ktlab_model_source_identity == source and other.ktlab_source_sha256 == expected
        groups = optimizer_parameter_groups(other, .001, .002)
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        assert len(parameter_ids) == len(set(parameter_ids)) == len(list(other.parameters()))
        assert id(other.interaction_vectors) in {id(p) for p in groups[-1]["params"]}
    assert len({m.ktlab_source_sha256 for m in models}) == 3
    assert base.ktlab_import_binding_only is True
    batch = batch_for(concept_only)
    for training in (False, True):
        outputs = []
        for model in models:
            model.train(training)
            torch.manual_seed(123)
            with torch.no_grad():
                outputs.append(forward(model, batch))
        for output in outputs[1:]:
            for actual, expected in zip(output, outputs[0]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert signed.ktlab_model_config == base.ktlab_model_config == independent.ktlab_model_config

    nonzero_outputs, vector_gradients = [], []
    for model in models[1:]:
        model.eval()
        with torch.no_grad():
            model.interaction_vectors.copy_(torch.arange(112).reshape(7, 16).float().sin() * .2)
        questions, responses, concepts = model._unpack_data(**batch)
        delta = model.history_residual(concepts, responses, batch["padding_mask"])
        assert torch.count_nonzero(delta[~batch["padding_mask"]]) == 0
        assert torch.count_nonzero(delta[batch["padding_mask"]]) > 0
        with torch.no_grad():
            empty = torch.full_like(concepts, -1)
            padding_delta = model.history_residual(empty, responses, torch.zeros_like(batch["padding_mask"]))
            assert torch.isfinite(padding_delta).all() and torch.count_nonzero(padding_delta) == 0
            with pytest.raises(ValueError, match="KC on every valid event"):
                model.history_residual(empty, responses, batch["padding_mask"])
        captured = []
        original_tf = model.model.forward

        def capture(*, tgt, src):
            captured.append((tgt.detach().clone(), src.detach().clone()))
            return original_tf(tgt=tgt, src=src)

        model.model.forward = capture
        try:
            probabilities, loss = forward(model, batch)
        finally:
            model.model.forward = original_tf
        expected_probabilities, expected_loss, _ = oracle(model, batch)
        torch.testing.assert_close(probabilities, expected_probabilities, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-7)
        # Explicit query/source contract, independently using the original mean.
        with torch.no_grad():
            q = model.q_emb(questions + 1)
            c = model.c_emb(concepts + 1)
            present = concepts != -1
            x = q + (c * present.unsqueeze(-1)).sum(2) / present.sum(2, keepdim=True)
            x = x + model.b_emb.weight[0]
            torch.testing.assert_close(captured[0][0], x)
            torch.testing.assert_close(captured[0][1], x + model.r_emb(responses + 1) + delta)
        parameters = tuple(model.parameters())
        actual_grad = torch.autograd.grad(loss, parameters, allow_unused=True)
        expected_grad = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
        for actual, expected in zip(actual_grad, expected_grad):
            assert (actual is None) == (expected is None)
            if actual is not None:
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
        vector_index = next(i for i, parameter in enumerate(parameters) if parameter is model.interaction_vectors)
        assert actual_grad[vector_index].abs().sum() > 0
        vector_gradients.append(actual_grad[vector_index])
        assert all(parameter.grad is None for parameter in parameters)
        reference = probabilities.detach()
        nonzero_outputs.append(reference)
        with torch.no_grad():
            current = clone(batch); current["data"][:, 3, -1] = 1 - current["data"][:, 3, -1]
            torch.testing.assert_close(forward(model, current)[0][:, :4], reference[:, :4], rtol=0, atol=0)
            future = clone(batch); future["data"][0, 4:6, -1] = 1 - future["data"][0, 4:6, -1]
            torch.testing.assert_close(forward(model, future)[0][:, :4], reference[:, :4], rtol=0, atol=0)
            past = clone(batch); past["data"][:, 1, -1] = 1 - past["data"][:, 1, -1]
            past_output = forward(model, past)[0]
            assert (past_output[:, 3] - reference[:, 3]).abs().max() > 1e-7
            permuted = clone(batch); permuted["data"][..., :-2] = permuted["data"][..., :-2].flip(-1)
            torch.testing.assert_close(forward(model, permuted)[0], reference, rtol=1e-6, atol=1e-7)
            padded = {"data": torch.cat([batch["data"], torch.full((2, 4, batch["data"].shape[-1]), -1)], 1),
                      "padding_mask": torch.cat([batch["padding_mask"], torch.zeros((2, 4), dtype=torch.bool)], 1)}
            torch.testing.assert_close(forward(model, padded)[0][:, :8], reference, rtol=1e-6, atol=1e-7)
            singleton = {key: value[:1] for key, value in batch.items()}
            torch.testing.assert_close(forward(model, singleton)[0], reference[:1], rtol=1e-6, atol=1e-7)
        history = [{"event_id": f"synthetic:u:{i}", "question": None if concept_only else int(batch["data"][0, i, -2]),
                    "concepts": [int(c) for c in batch["data"][0, i, :-2] if c >= 0],
                    "response": int(batch["data"][0, i, -1])} for i in range(3)]
        context = {"protocol": PROTOCOL, "event_id": "synthetic:u:3", "history": history,
                   "query": {"question": None if concept_only else 3,
                             "concepts": [int(c) for c in batch["data"][0, 3, :-2] if c >= 0]}, "label": 1}
        hidden_label = copy.deepcopy(context); hidden_label["label"] = 0
        for key in ("data", "padding_mask", "target_positions"):
            assert torch.equal(collate_shared_ktst([context], model.max_concepts)[key],
                               collate_shared_ktst([hidden_label], model.max_concepts)[key])
        model.train()
        result = predict_shared(model, [context], "cpu")
        assert model.training, "Shared validation restores caller training mode"
        assert result == pytest.approx([float(reference[0, 3])], abs=1e-6)
        model.eval()

        config = {"model": "ktst_mean", "model_config": model.ktlab_model_config,
                  "ktst_interaction_recipe": model.ktlab_interaction_recipe, "protocol": PROTOCOL, "fold": 0}
        fake_dataset = SimpleNamespace(audit={"protocol": PROTOCOL, "split": "valid", "fold": 0,
                                             "source_sha256": "1" * 64, "target_sha256": "2" * 64, "context_sha256": "3" * 64})
        identity = engine._shared_identity(config, model, fake_dataset)
        path = tmp_path / f"{model.ktlab_interaction_recipe}.pt"
        ktst_engine._save_checkpoint(model, path, .75, 0, "4" * 64, config, protocol=PROTOCOL, identity=identity)
        metadata = {**identity, "protocol": PROTOCOL, "dataset_manifest_sha256": "4" * 64,
                    "checkpoint_sha256": engine._sha(path), "best_epoch": 0, "best_valid_auc": .75, "best_epoch_completed": False}
        checkpoint = engine._selected_shared_checkpoint(path, metadata, identity, "4" * 64, model.ktlab_model_config)
        restored = build_ktst_model("ktst_mean", data_config, checkpoint["model_config"], "cpu",
                                    interaction_recipe=checkpoint["ktst_interaction_recipe"]).eval()
        restored.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            torch.testing.assert_close(forward(restored, batch)[0], reference, rtol=0, atol=0)
        wrong = dict(identity, ktst_interaction_recipe=next(r for r in RECIPES if r != model.ktlab_interaction_recipe))
        with pytest.raises(ValueError, match="recipe mismatch"):
            engine._selected_shared_checkpoint(path, metadata, wrong, "4" * 64, model.ktlab_model_config)
        with pytest.raises(ValueError, match="recipe mismatch"):
            engine._shared_identity(dict(config, ktst_interaction_recipe=None), model, fake_dataset)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            base.load_state_dict(checkpoint["state_dict"])
        legacy = tmp_path / f"reference-{model.ktlab_interaction_recipe}.pt"
        ktst_engine._save_checkpoint(model, legacy, .75, 1, "4" * 64, config)
        assert torch.load(legacy, weights_only=True)["ktst_interaction_recipe"] == model.ktlab_interaction_recipe
    assert (nonzero_outputs[0] - nonzero_outputs[1]).abs().max() > 1e-7
    assert (vector_gradients[0] - vector_gradients[1]).abs().max() > 1e-7


@pytest.mark.parametrize("name,recipe", [("ktst_mhsa", "signed_response"), ("akt", "signed_response"),
                                      ("ktst_mean", "renamed"), ("ktst_mean", True), ("ktst_mean", {})])
def test_invalid_recipe_rejected_without_model(name, recipe):
    before = dict(COUNTS)
    with pytest.raises(ValueError):
        build_ktst_model(name, {"num_q": 0, "num_c": 7, "max_concepts": 1}, SMALL,
                         interaction_recipe=recipe)
    with pytest.raises(ValueError):
        engine._shared_config({"model": name, "protocol": PROTOCOL, "ktst_interaction_recipe": recipe})
    assert COUNTS == before


def test_unapproved_or_wrong_model_recipe_cannot_reach_optimizer(tmp_path):
    before = dict(COUNTS)
    with pytest.raises(PermissionError):
        ktst_engine.train_ktst({"model": "ktst_mean", "ktst_interaction_recipe": "signed_response"}, tmp_path)
    with pytest.raises(ValueError, match="ktst_interaction_recipe"):
        engine.train({"model": "akt", "ktst_interaction_recipe": "signed_response", "_batch_approved": True}, tmp_path)
    with pytest.raises(PermissionError):
        ktst_engine.evaluate_ktst_run(tmp_path, split="test")
    assert COUNTS == before


@pytest.mark.parametrize("route", ["train", "shared_reload", "reference_reload"])
def test_engine_routes_pass_explicit_recipe_before_any_optimizer(route, tmp_path, monkeypatch):
    """Real routing and identity checks; fake constructor stops before a model."""
    import ktlab.ktst_model as model_module
    recipe = "signed_response"
    config = {"model": "ktst_mean", "protocol": PROTOCOL if route != "reference_reload" else ktst_engine.PROTOCOL,
              "ktst_interaction_recipe": recipe, "dataset_manifest": str(tmp_path / "manifest.json"),
              "model_config": SMALL, "device": "cpu", "fold": 0, "learning_rate": .001,
              "theta_learning_rate": .002, "initial_validation": True, "shuffle": True}
    if config["protocol"] == PROTOCOL:
        config = engine._shared_config(config)
    manifest = {"_manifest_sha256": "4" * 64, "data_config": {"num_q": 0, "num_c": 7, "max_concepts": 1}}
    monkeypatch.setattr(engine, "read_manifest", lambda unused: manifest)
    received = []

    class ReachedConstructor(Exception):
        pass

    def fake_build(name, dimensions, settings, device, **kwargs):
        received.append(kwargs)
        assert name == "ktst_mean" and settings == SMALL and dimensions == manifest["data_config"]
        assert kwargs == {"interaction_recipe": recipe}
        raise ReachedConstructor

    monkeypatch.setattr(model_module, "build_ktst_model", fake_build)
    before = dict(COUNTS)
    if route == "train":
        with pytest.raises(ReachedConstructor):
            ktst_engine.train_ktst(dict(config, _batch_approved=True), tmp_path / "train")
    else:
        (tmp_path / "resolved_config.json").write_text(json.dumps(config))
        checkpoint = {"protocol": config["protocol"], "dataset_manifest_sha256": "4" * 64,
                      "config_sha256": engine.config_fingerprint(config), "implementation_sha256": engine.source_fingerprint(),
                      "checkpoint_selection_policy": engine.SHARED_SELECTION, "model_config": SMALL,
                      "ktst_interaction_recipe": recipe}
        (tmp_path / "best.pt").write_bytes(b"SYNTHETIC-NOT-A-WEIGHT-FILE")
        (tmp_path / "metadata.json").write_text(json.dumps(dict(checkpoint, checkpoint_sha256=engine._sha(tmp_path / "best.pt"))))
        monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
        with pytest.raises(ReachedConstructor):
            ktst_engine.evaluate_ktst_run(tmp_path)
        checkpoint["ktst_interaction_recipe"] = "response_independent"
        with pytest.raises(ValueError, match="recipe mismatch"):
            ktst_engine.evaluate_ktst_run(tmp_path)
    assert len(received) == 1 and COUNTS == before
