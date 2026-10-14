"""S006: synthetic CPU only; real-model section requires explicit execution flag.

Two original d16 MoC instances, <=64 original forwards, <=4 autograd.grad,
zero backward/optimizers. No data loader, saved weights or accelerator use.
"""
import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ktlab import adapters, engine
from ktlab import mockt_training as mt


BOUNDARIES = (49, 50, 51, 149, 150, 151)
ORIGINAL_AUTOGRAD_GRAD = torch.autograd.grad


def config(recipe=mt.PREFIX_RECIPE, **changes):
    result = {"model": "mockt", "protocol": mt.PROTOCOL, "mockt_training_recipe": recipe,
              "model_config": {"emb_type": "qid_conv_ker_noexp", "kernel_size": 4,
                               "seq_len": 200, "sequence1": 50, "sequence2": 150}}
    result.update(changes)
    return result


def batch_fixture(n_pid=5):
    positions = torch.arange(200).repeat(2, 1)
    full = {"cseqs": (positions + torch.tensor([[0], [2]])) % 7,
            "qseqs": positions % n_pid if n_pid else torch.zeros_like(positions),
            "rseqs": ((positions // 3 + positions) % 2).float()}
    lengths = torch.tensor([60, 200])
    mask = torch.arange(199).unsqueeze(0) < (lengths - 1).unsqueeze(1)
    batch = {"masks": mask, "smasks": torch.zeros_like(mask),
             "ground_seq": lengths.unsqueeze(1).float()}
    for name, value in full.items():
        batch[name] = value[:, :-1] * mask
        batch["shft_" + name] = value[:, 1:] * mask
    # Shifted index j is original position j+1, hence prefix length j+2.
    batch["smasks"][:, torch.tensor(BOUNDARIES) - 2] = True
    batch["smasks"] &= mask
    return batch


def full_inputs(batch):
    return {name: torch.cat((batch[name][:, :1], batch["shft_" + name]), 1).long()
            for name in ("cseqs", "qseqs", "rseqs")}


class FakeMoC:
    model_name, emb_type, sequence1, sequence2 = "mockt", "qid_conv_ker_noexp", 50, 150
    training = True

    def __init__(self):
        self.calls = []

    def __call__(self, ground, concepts, responses, questions):
        self.calls.append((ground.clone(), concepts.clone(), responses.clone(), questions.clone()))
        group = (ground > 50).long() + (ground > 150).long()
        return (group.float() + 1).expand_as(concepts) / 4, torch.tensor(2.5)


@pytest.fixture(autouse=True)
def no_updates_or_accelerator(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("S006 forbids backward, optimizers, weights and accelerators")
    monkeypatch.setattr(torch.Tensor, "backward", deny)
    monkeypatch.setattr(torch.autograd, "backward", deny)
    monkeypatch.setattr(torch.autograd, "grad", deny)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", deny)
    monkeypatch.setattr(torch.optim.Optimizer, "step", deny)
    monkeypatch.setattr(torch, "load", deny)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", deny)
    monkeypatch.setattr(torch.Tensor, "cuda", deny)
    monkeypatch.setattr(torch.nn.Module, "cuda", deny)


@pytest.mark.parametrize("recipe", sorted(mt.RECIPES))
def test_routing_boundaries_one_regularizer_and_no_mutation(recipe):
    batch, model = batch_fixture(), FakeMoC()
    before = {key: value.clone() for key, value in batch.items()}
    probabilities, regularizer = mt.forward_mockt_training(model, batch, recipe)
    assert len(model.calls) == 3 and regularizer.item() == 2.5
    assert probabilities.shape == (2, 199)
    expected = torch.tensor([.25, .25, .5, .5, .5, .75])
    if recipe == mt.PREFIX_RECIPE:
        torch.testing.assert_close(probabilities[1, torch.tensor(BOUNDARIES) - 2], expected)
        assert [call[0][0].item() for call in model.calls] == [1, 51, 151]
    else:
        assert all(torch.equal(call[0], batch["ground_seq"]) for call in model.calls)
        torch.testing.assert_close(probabilities[0], torch.full((199,), .5))
        torch.testing.assert_close(probabilities[1], torch.full((199,), .75))
    assert all(torch.equal(batch[key], value) for key, value in before.items())
    for call in model.calls:
        for actual, key in zip(call[1:], ("cseqs", "rseqs", "qseqs")):
            assert torch.equal(actual, full_inputs(batch)[key])


@pytest.mark.parametrize("kind", ["hole", "empty", "ground", "boolean_ground", "selection", "padding", "response", "shape"])
def test_bad_batches_fail_before_any_model_call(kind):
    batch, model = batch_fixture(), FakeMoC()
    if kind == "hole": batch["masks"][1, 3] = False
    elif kind == "empty": batch["masks"][0] = False
    elif kind == "ground": batch["ground_seq"][0] = 200
    elif kind == "boolean_ground": batch["ground_seq"] = batch["ground_seq"].bool()
    elif kind == "selection": batch["smasks"][0, -1] = True
    elif kind == "padding": batch["shft_rseqs"][0, -1] = 1
    elif kind == "response": batch["shft_rseqs"][1, 4] = 2
    elif kind == "shape": batch["masks"] = batch["masks"][:, :-1]
    with pytest.raises(ValueError):
        mt.forward_mockt_training(model, batch, mt.PREFIX_RECIPE)
    assert not model.calls


@pytest.mark.parametrize("change", [
    {"mockt_training_recipe": "renamed"}, {"mockt_training_recipe": True},
    {"model": "akt"}, {"model": "ktst_mean"}, {"protocol": "ktlab-v1"},
    {"model_config": {"emb_type": "qid"}}, {"model_config": {"kernel_size": 5}},
    {"model_config": {"seq_len": 100}}, {"model_config": {"sequence1": 49}},
    {"model_config": {"sequence2": 150.0}}, {"model_config": []}])
def test_config_rejected_before_data_model_or_optimizer(tmp_path, monkeypatch, change):
    def deny(*args, **kwargs):
        raise AssertionError("Configuration must be rejected before data/model construction")
    monkeypatch.setattr(engine, "read_manifest", deny)
    monkeypatch.setattr(engine, "build_model", deny)
    with pytest.raises(ValueError, match="MoC"):
        engine.train(dict(config(**change), _batch_approved=True), tmp_path / "unused")


def test_dispatch_preserves_default_and_evaluation_path_and_config_identity():
    batch, model = batch_fixture(), FakeMoC()
    assert mt.validate_recipe_config({"model": "akt"}) is None
    adapters.forward_model(model, batch)
    assert len(model.calls) == 1
    mt.bind_training_recipe(model, config())
    adapters.forward_model(model, batch)
    assert len(model.calls) == 4
    model.training = False
    adapters.forward_model(model, batch)
    assert len(model.calls) == 5
    assert torch.equal(model.calls[-1][0], batch["ground_seq"])
    from ktlab.provenance import config_fingerprint
    assert len({config_fingerprint(config(recipe)) for recipe in (None, *sorted(mt.RECIPES))}) == 3
    with pytest.raises(ValueError, match="conflicts"):
        mt.bind_training_recipe(SimpleNamespace(model_name="mockt", emb_type="qid", sequence1=50, sequence2=150), config())


class RealBudget:
    def __init__(self):
        self.models = self.forwards = self.gradients = 0

    def construct(self):
        assert self.models < 2, "S006 two-model budget exceeded before construction"
        self.models += 1

    def before_forward(self, model, inputs):
        assert self.forwards < 64, "S006 forward budget exceeded before call"
        self.forwards += 1

    def grad(self, loss, parameters):
        assert self.gradients < 4, "S006 autograd budget exceeded before call"
        self.gradients += 1
        return ORIGINAL_AUTOGRAD_GRAD(loss, parameters, allow_unused=True)


@pytest.fixture(scope="module")
def real_budget():
    budget = RealBudget()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield budget
    torch.set_num_threads(old_threads)
    print("S006_COUNTS " + json.dumps({"original_models": budget.models, "original_forwards": budget.forwards,
                                     "autograd_grad_calls": budget.gradients, "backward_calls": 0,
                                     "optimizer_constructions": 0, "optimizer_steps": 0}))
    assert budget.models == 2 and budget.gradients == 4 and budget.forwards <= 64


def construct_cpu_original(n_pid, monkeypatch, budget):
    budget.construct()
    torch.manual_seed(42)
    original = adapters._load_original("mockt", torch.device("cpu"))
    # Placement-only accommodation, exclusively while constructing this fixture.
    # Production build_model's CUDA requirement and all original math stay intact.
    with monkeypatch.context() as placement:
        def cpu_tensor(tensor, *args, **kwargs):
            assert not args and not kwargs
            return tensor.to("cpu")
        placement.setattr(torch.Tensor, "cuda", cpu_tensor)
        model = original.MocKT(n_question=7, n_pid=n_pid, d_model=16, n_blocks=1, dropout=0.,
                              d_ff=32, final_fc_dim=32, num_attn_heads=2, seq_len=200,
                              kernel_size=4, emb_type="qid_conv_ker_noexp", l2=.01)
    model.register_forward_pre_hook(budget.before_forward)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    with torch.no_grad():
        if n_pid:
            model.difficult_param.weight.copy_(torch.linspace(.05, .2, n_pid + 1).unsqueeze(1))
        # Put the original lower-bound clamp in a stable, non-boundary state.
        for name, parameter in model.named_parameters():
            if name.endswith(("bias_p", "bias_a")):
                parameter.clamp_(min=.02)
    return model


@pytest.mark.skipif(os.environ.get("KTLAB_RUN_MOCKT_PREFIX_CPU_ACCEPTANCE") != "1",
                    reason="Real model calls require the recorded S006 source/test receipt")
@pytest.mark.parametrize("n_pid", [0, 5])
def test_s006_original_cpu_prefix_probability_bce_gradient_and_causality(n_pid, monkeypatch, real_budget):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("HIP_VISIBLE_DEVICES") == ""
    assert os.environ.get("ROCR_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(torch.autograd, "grad", real_budget.grad)
    model = construct_cpu_original(n_pid, monkeypatch, real_budget)
    batch = batch_fixture(n_pid)
    full = full_inputs(batch)
    selected = batch["smasks"]
    labels = batch["shft_rseqs"].double()

    def original(length, *, ground=None):
        supplied = torch.full((2, 1), float(length)) if ground is None else ground
        return model(supplied, full["cseqs"][:, :length], full["rseqs"][:, :length], full["qseqs"][:, :length])

    # no_grad cache -> different length -> same T200 -> grad-enabled training.
    model.eval()
    with torch.no_grad():
        original(200, ground=batch["ground_seq"])
        original(49)
        original(200, ground=batch["ground_seq"])
    cached = [module.cached_matrix for module in model.modules() if hasattr(module, "cached_matrix")]
    assert cached and all(not torch.is_inference(value) for value in cached if value is not None)
    model.train()  # dropout remains zero by the frozen fixture settings.
    fast, regularizer = mt.forward_mockt_training(model, batch, mt.PREFIX_RECIPE)
    assert fast.requires_grad and fast.grad_fn is not None
    oracle = torch.stack([original(length)[0][:, -1] for length in BOUNDARIES], 1)
    locations = torch.tensor(BOUNDARIES) - 2
    oracle_selected = selected[:, locations]
    torch.testing.assert_close(fast[:, locations][oracle_selected], oracle[oracle_selected], atol=1e-6, rtol=0)
    expected_reg = (model.difficult_param(full["qseqs"]) ** 2).sum() * model.l2 if n_pid else fast.new_zeros(())
    torch.testing.assert_close(regularizer, expected_reg, atol=0, rtol=0)
    if n_pid:
        assert regularizer.item() > 0 and regularizer.requires_grad
        # Including repeated padding qid=0 is intentional original L2 behavior.
        assert expected_reg.item() > (model.difficult_param(full["qseqs"])[torch.cat((batch["masks"][:, :1], batch["masks"]), 1)] ** 2).sum().item() * model.l2
    fast_bce = F.binary_cross_entropy(fast[selected].double(), labels[selected])
    oracle_bce = F.binary_cross_entropy(oracle[oracle_selected].double(), labels[:, locations][oracle_selected])
    torch.testing.assert_close(fast_bce, oracle_bce, atol=1e-7, rtol=0)
    parameters = tuple(model.parameters())
    fast_grad = torch.autograd.grad(fast_bce + regularizer, parameters)
    oracle_grad = torch.autograd.grad(oracle_bce + expected_reg, parameters)
    gradient_max_error = 0.
    for actual, expected in zip(fast_grad, oracle_grad):
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            gradient_max_error = max(gradient_max_error, (actual - expected).abs().max().item())
            torch.testing.assert_close(actual, expected, atol=5e-6, rtol=1e-4)
    named_grad = dict(zip(dict(model.named_parameters()), fast_grad))
    for prefix in ("q_embed.", "model.filter_layer.causal_convs.", "out."):
        matching = [g for name, g in named_grad.items() if name.startswith(prefix) and g is not None]
        assert matching and any(g.abs().sum() > 0 for g in matching), prefix

    with torch.no_grad():
        control, control_reg = mt.forward_mockt_training(model, batch, mt.CONTROL_RECIPE)
        baseline, baseline_reg = original(200, ground=batch["ground_seq"])
        torch.testing.assert_close(control, baseline[:, 1:], atol=1e-7, rtol=0)
        torch.testing.assert_close(control_reg, torch.as_tensor(baseline_reg), atol=0, rtol=0)
        # The shorter row's padding must not affect its final valid target.
        short, _ = original(60)
        torch.testing.assert_close(fast[0, 58], short[0, -1], atol=1e-6, rtol=0)
        # Vendor FrequencyLayer has a separate batch-size-one indexing branch.
        singleton = {key: value[:1] for key, value in batch.items()}
        single, single_reg = mt.forward_mockt_training(model, singleton, mt.PREFIX_RECIPE)
        torch.testing.assert_close(single[0, :59], fast[0, :59], atol=1e-6, rtol=0)
        expected_single_reg = ((model.difficult_param(full["qseqs"][:1]) ** 2).sum() * model.l2
                               if n_pid else single.new_zeros(()))
        torch.testing.assert_close(single_reg, expected_single_reg, atol=0, rtol=0)
        causal_target = 51
        predictions = {}
        for change in ("current", "future", "history"):
            perturbed = copy.deepcopy(batch)
            altered = full["rseqs"].clone()
            if change == "current": altered[:, causal_target - 1] = 1 - altered[:, causal_target - 1]
            elif change == "future": altered[:, causal_target:] = 1 - altered[:, causal_target:]
            else: altered[:, causal_target - 2] = 1 - altered[:, causal_target - 2]
            perturbed["rseqs"] = altered[:, :-1].float() * batch["masks"]
            perturbed["shft_rseqs"] = altered[:, 1:].float() * batch["masks"]
            predictions[change] = mt.forward_mockt_training(model, perturbed, mt.PREFIX_RECIPE)[0][:, causal_target - 2]
        target = fast[:, causal_target - 2]
        for change in ("current", "future"):
            torch.testing.assert_close(predictions[change], target, atol=1e-6, rtol=0)
        assert (predictions["history"] - target).abs().max().item() > 1e-8
    print("S006_MODEL " + json.dumps({"num_q": n_pid, "gradient_max_abs_error": gradient_max_error,
                                     "prefix_bce_abs_error": abs((fast_bce - oracle_bce).item()),
                                     "full_batch_l2": regularizer.item(),
                                     "history_change_max": (predictions["history"] - target).abs().max().item()}))
