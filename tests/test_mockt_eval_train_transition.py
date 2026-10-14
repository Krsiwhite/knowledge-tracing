"""Fresh random MoC models only: shared epoch0 evaluation must allow training.

Updates require KTLAB_RUN_MOCKT_TRANSITION_UPDATES=1. The full file performs
exactly two backward calls and two Adam steps, with hard ceilings eight/four.
Unmodified MoC requires CUDA/ROCm. No datasets or existing weights are loaded.
"""
from copy import deepcopy
import inspect
import os
import time

import pytest
import torch

from ktlab.adapters import build_model, forward_model
from ktlab import shared_adapters as shared


DATA = {"num_q": 9, "num_c": 8, "maxlen": 200}
MODEL = {"d_model": 16, "n_blocks": 2, "d_ff": 32, "num_attn_heads": 2,
         "final_fc_dim": 32, "dropout": 0.0, "seq_len": 200,
         "emb_type": "qid_conv_ker_noexp", "kernel_size": 4}


@pytest.fixture(scope="module", autouse=True)
def synthetic_gpu_budget():
    enabled = os.environ.get("KTLAB_RUN_MOCKT_TRANSITION_UPDATES") == "1"
    if not torch.cuda.is_available():
        if enabled:
            pytest.fail("Requested MoC transition acceptance requires CUDA/ROCm")
        pytest.skip("Pinned original MoC requires CUDA/ROCm")
    patch = pytest.MonkeyPatch()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    counts = {"backward": 0, "optimizer_steps": 0}
    actual_backward, actual_step = torch.autograd.backward, torch.optim.Adam.step

    def backward(*args, **kwargs):
        assert enabled, "Record synthetic update authorization before enabling this case"
        counts["backward"] += 1
        assert counts["backward"] <= 8, "Synthetic backward budget exceeded"
        return actual_backward(*args, **kwargs)

    def step(*args, **kwargs):
        assert enabled, "Record synthetic update authorization before enabling this case"
        counts["optimizer_steps"] += 1
        assert counts["optimizer_steps"] <= 4, "Synthetic optimizer budget exceeded"
        return actual_step(*args, **kwargs)

    def forbid_load(*args, **kwargs):
        raise AssertionError("This acceptance never reads existing checkpoints")

    patch.setattr(torch.autograd, "backward", backward)
    patch.setattr(torch.optim.Adam, "step", step)
    patch.setattr(torch, "load", forbid_load)
    torch.cuda.synchronize()
    started = time.monotonic()
    try:
        yield counts
    finally:
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        patch.undo()
        torch.set_num_threads(old_threads)
        assert elapsed <= 300, "Synthetic GPU soft budget exceeded"


def contexts():
    return [{"protocol": shared.PROTOCOL, "uid": str(uid), "event_id": f"synthetic:{uid}:2",
             "history": [{"event_id": f"synthetic:{uid}:0", "question": 1, "concepts": [0, 2], "response": uid % 2},
                         {"event_id": f"synthetic:{uid}:1", "question": 4, "concepts": [1], "response": 1 - uid % 2}],
             "query": {"question": 4, "concepts": [3, 4]}, "label": uid % 2} for uid in (0, 1)]


def model():
    torch.manual_seed(37)
    result = build_model("mockt", DATA, MODEL, "cuda")
    with torch.no_grad():
        result.difficult_param.weight.normal_(mean=.2, std=.1)
        # Avoid the upstream minimum clamp changing this artificial initial
        # state; all compared models receive precisely the same parameters.
        for module in result.modules():
            if type(module).__name__ == "ParallelKerpleLog":
                module.bias_a.clamp_(min=.1)
                module.bias_p.clamp_(min=.1)
    return result


def kernels(model):
    found = [module for module in model.modules() if type(module).__name__ == "ParallelKerpleLog"]
    assert found
    return found


def batch():
    return shared._move(shared.collate_shared_kc(contexts()), torch.device("cuda"))


def test_old_inference_cache_negative_control_and_prediction_equivalence():
    old_model, fixed_model = model(), model()
    fixed_model.load_state_dict(old_model.state_dict(), strict=True)
    assert all(module.cached_matrix is None for module in kernels(old_model))
    # Unwrap only the public grad-mode decorator to recreate the old entry
    # behavior; the unchanged projection and vendor model execute normally.
    old_predict = torch.inference_mode()(inspect.unwrap(shared.predict_shared))
    expected = old_predict(old_model, contexts(), "cuda", max_rows_per_forward=4)
    actual = shared.predict_shared(fixed_model, contexts(), "cuda", max_rows_per_forward=4)
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=2e-5, atol=2e-6)
    assert all(torch.is_inference(module.cached_matrix) for module in kernels(old_model))
    assert all(not torch.is_inference(module.cached_matrix) for module in kernels(fixed_model))
    assert old_model.training and fixed_model.training
    assert all(parameter.grad is None for parameter in fixed_model.parameters())
    ordinary_batch = batch()
    with pytest.raises(RuntimeError, match="Inference tensors cannot be saved for backward"):
        forward_model(old_model, ordinary_batch)  # Fails in train forward, before any backward call.


def test_fresh_eval_train_eval_train_two_finite_updates(synthetic_gpu_budget):
    if os.environ.get("KTLAB_RUN_MOCKT_TRANSITION_UPDATES") != "1":
        pytest.skip("Two synthetic updates require the explicit recorded execution switch")
    fixed_model = model().train()
    optimizer = torch.optim.Adam(fixed_model.parameters(), lr=1e-3)
    training_batch = batch()
    training_batch["smasks"] = training_batch["masks"].clone()
    before_counts = dict(synthetic_gpu_budget)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        shared.predict_shared(fixed_model, contexts(), "cuda", max_rows_per_forward=2)
        assert fixed_model.training
        assert all(parameter.grad is None for parameter in fixed_model.parameters())
        assert all(module.cached_seq_len == 200 and not torch.is_inference(module.cached_matrix)
                   for module in kernels(fixed_model))
        for module in fixed_model.modules():
            assert all(not torch.is_inference(value) for value in vars(module).values() if torch.is_tensor(value))
        predictions, regularizer = forward_model(fixed_model, training_batch)
        mask = training_batch["smasks"]
        loss = torch.nn.functional.binary_cross_entropy(predictions[mask].double(),
                                                        training_batch["shft_rseqs"][mask].double()) + regularizer
        assert torch.isfinite(loss)
        before = {name: parameter.detach().clone() for name, parameter in fixed_model.named_parameters()}
        loss.backward()
        gradients = [parameter.grad for parameter in fixed_model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        optimizer.step()
        assert any(not torch.equal(parameter, before[name]) for name, parameter in fixed_model.named_parameters())
        assert all(torch.isfinite(parameter).all() for parameter in fixed_model.parameters())
    assert synthetic_gpu_budget["backward"] - before_counts["backward"] == 2
    assert synthetic_gpu_budget["optimizer_steps"] - before_counts["optimizer_steps"] == 2


def test_nonzero_difficulty_causality_multikc_and_chunking():
    fixed_model = model().eval()
    assert torch.count_nonzero(fixed_model.difficult_param.weight) == fixed_model.difficult_param.weight.numel()
    items = contexts()
    expected = shared.predict_shared(fixed_model, items, "cuda", max_rows_per_forward=4)
    assert not fixed_model.training
    changed = deepcopy(items)
    for item in changed:
        item["label"] = 1 - item["label"]
        item["query"]["response"] = 1
        item["future_responses"] = [0, 1] * 50
    for chunk in (1, 3):
        actual = shared.predict_shared(fixed_model, changed, "cuda", max_rows_per_forward=chunk)
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=2e-5, atol=2e-6)
    original = batch()
    changed_raw = {key: value.clone() if torch.is_tensor(value) else value for key, value in original.items()}
    indices = torch.arange(4, device="cuda")
    positions = original["target_positions"]
    changed_raw["responses"][indices, positions] = 1
    for row, position in enumerate(positions.tolist()):
        changed_raw["responses"][row, position + 1:] = 1
        changed_raw["concepts"][row, position + 1:] = 7
        changed_raw["questions"][row, position + 1:] = 8
    with torch.no_grad():
        first, _ = fixed_model(original["ground_seq"], original["concepts"], original["responses"], original["questions"])
        second, _ = fixed_model(changed_raw["ground_seq"], changed_raw["concepts"], changed_raw["responses"], changed_raw["questions"])
    torch.testing.assert_close(first[indices, positions], second[indices, positions], rtol=2e-5, atol=2e-6)
    independent = first[indices, positions].double().reshape(2, 2).mean(1).cpu()
    torch.testing.assert_close(torch.tensor(expected, dtype=torch.float64), independent, rtol=2e-5, atol=2e-6)
    past_changed = deepcopy(items)
    for item in past_changed:
        for event in item["history"]:
            event["response"] = 1 - event["response"]
    positive_control = shared.predict_shared(fixed_model, past_changed, "cuda", max_rows_per_forward=4)
    assert max(abs(a - b) for a, b in zip(positive_control, expected)) > 1e-8
