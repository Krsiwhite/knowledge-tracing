"""KTST forward-only verification: no backward or optimizer may be invoked."""
import builtins
import shutil
import sys
from types import ModuleType

import pytest
import torch
from torch.nn import functional as F

from ktlab.ktst_model import (
    KTST_COMMIT, KTSTSourceError, build_ktst_model, forward_ktst,
    load_ktst_reference, optimizer_parameter_groups, verify_ktst_source,
)


DATA = {"num_q": 9, "num_c": 7, "max_concepts": 3, "maxlen": 16}
SMALL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2,
         "num_layers_tf_dec": 2, "num_layers_tf_enc": 1, "num_layers_agg": 2,
         "dim_feedforward": 32, "dim_classifier": 24, "dropout": 0.0}


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("KTST preapproval checks must never create an optimizer or differentiate")
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_count():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def event_batch(length=8, device="cpu"):
    data = torch.full((2, length, 5), -1, dtype=torch.int64, device=device)
    data[0, :6] = torch.tensor([
        [0, 2, -1, 0, 0], [1, -1, -1, 1, 1], [2, 3, 4, 2, 1],
        [0, 5, -1, 3, 0], [4, 6, -1, 4, 1], [1, 6, -1, 5, 0],
    ], device=device)
    data[1, :4] = torch.tensor([
        [3, -1, -1, 4, 1], [0, 1, -1, 3, 0], [2, 6, -1, 2, 1], [5, -1, -1, 1, 0],
    ], device=device)
    mask = data[:, :, -1] >= 0
    return {"data": data, "padding_mask": mask}


def clone_batch(batch):
    return {key: value.clone() for key, value in batch.items()}


def copied_vendor(tmp_path):
    source = verify_ktst_source()
    destination = tmp_path / "ktst"
    shutil.copytree(source["vendor_root"], destination)
    shutil.copyfile(source["lock_path"], destination.with_suffix(".lock.json"))
    return destination


def test_vendor_bytes_and_commit_are_verified():
    verified = verify_ktst_source()
    assert verified["commit"] == KTST_COMMIT
    assert verified["license"] == "MIT"
    assert len(verified["source_sha256"]) == 64
    assert "pkg/data/pykt/__init__.py" in verified["files"]


def test_private_import_does_not_replace_generic_pkg(tmp_path, monkeypatch):
    root = copied_vendor(tmp_path)
    sentinel = ModuleType("pkg")
    sentinel_model = ModuleType("pkg.model")
    monkeypatch.setitem(sys.modules, "pkg", sentinel)
    monkeypatch.setitem(sys.modules, "pkg.model", sentinel_model)
    before = {key: value for key, value in sys.modules.items() if key == "pkg" or key.startswith("pkg.")}
    original_import = builtins.__import__
    model = build_ktst_model("ktst_mean", DATA, SMALL, vendor_root=root)
    reference = load_ktst_reference("pkg.data.pykt", vendor_root=root)
    after = {key: value for key, value in sys.modules.items() if key == "pkg" or key.startswith("pkg.")}
    assert after == before
    assert builtins.__import__ is original_import
    assert callable(reference.format_question_set_dense)
    assert type(model).__module__.startswith("_ktlab_ktst_")
    assert reference.__name__.startswith(model.ktlab_import_namespace)
    normal = build_ktst_model("ktst_mean", DATA, SMALL)
    assert type(normal) is not type(model), "Different vendor roots must not reuse modules"


def test_source_tampering_is_rejected_even_after_import_cache(tmp_path):
    root = copied_vendor(tmp_path)
    build_ktst_model("ktst_mean", DATA, SMALL, vendor_root=root)
    source = root / "pkg/model/KTST.py"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(KTSTSourceError, match="bytes changed"):
        build_ktst_model("ktst_mean", DATA, SMALL, vendor_root=root)


@pytest.mark.parametrize("name", ["ktst_mean", "ktst_mhsa"])
def test_exact_original_loss_and_forward_contract(name):
    torch.manual_seed(19)
    model = build_ktst_model(name, DATA, SMALL).eval()
    batch = event_batch()
    parameters_before = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
    captured = []
    # Upstream calls Sequential.forward directly; hook its final Linear, whose
    # normal __call__ is used by Sequential, without modifying the computation.
    handle = model.clf[-1].register_forward_hook(lambda module, inputs, output: captured.append(output))
    with torch.inference_mode():
        probabilities, loss = forward_ktst(model, batch)
        logits = captured[-1].squeeze(-1)
        mask = batch["padding_mask"][:, 1:]
        expected_loss = F.binary_cross_entropy_with_logits(
            logits[:, 1:][mask], batch["data"][:, 1:, -1][mask].float(), reduction="mean")
        direct_probabilities, direct_loss, extra = model(**batch)
    handle.remove()
    assert probabilities.shape == (2, 8)
    assert torch.isfinite(probabilities).all() and bool(torch.isfinite(loss))
    assert loss.ndim == 0 and extra == 0.0
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(loss, direct_loss, rtol=0, atol=0)
    torch.testing.assert_close(probabilities, direct_probabilities, rtol=0, atol=0)
    assert not probabilities.requires_grad and not loss.requires_grad
    for key, parameter in model.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, parameters_before[key], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["ktst_mean", "ktst_mhsa"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_causality_set_order_future_content_and_padding(name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU/ROCm unavailable")
    torch.manual_seed(23)
    model = build_ktst_model(name, DATA, SMALL, device).eval()
    batch = event_batch(device=device)
    changed_answer = clone_batch(batch)
    changed_answer["data"][:, 2, -1] = 1 - changed_answer["data"][:, 2, -1]
    changed_future = clone_batch(batch)
    future_mask = changed_future["padding_mask"][:, 3:]
    suffix = changed_future["data"][:, 3:]
    suffix[..., -2][future_mask] = (suffix[..., -2][future_mask] + 2) % DATA["num_q"]
    suffix[..., -1][future_mask] = 1 - suffix[..., -1][future_mask]
    suffix[..., :3] = torch.where(suffix[..., :3] >= 0,
                                 (suffix[..., :3] + 1) % DATA["num_c"], suffix[..., :3])
    permuted = clone_batch(batch)
    permuted["data"][..., :3] = permuted["data"][..., [2, 0, 1]]
    prefix = {key: value[:, :3].clone() for key, value in batch.items()}
    altered_history = clone_batch(batch)
    altered_history["data"][:, 0, -1] = 1 - altered_history["data"][:, 0, -1]
    with torch.inference_mode():
        expected, original_loss = forward_ktst(model, batch)
        answer_scores, _ = forward_ktst(model, changed_answer)
        future_scores, _ = forward_ktst(model, changed_future)
        order_scores, order_loss = forward_ktst(model, permuted)
        longer_scores, longer_loss = forward_ktst(model, event_batch(length=12, device=device))
        prefix_scores, _ = forward_ktst(model, prefix)
        history_scores, _ = forward_ktst(model, altered_history)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(expected[:, :3], answer_scores[:, :3], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(expected[:, :3], future_scores[:, :3], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(expected[:, :3], prefix_scores, rtol=2e-5, atol=2e-6)
    valid = batch["padding_mask"]
    torch.testing.assert_close(expected[valid], order_scores[valid], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original_loss, order_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(expected[valid], longer_scores[:, :8][valid], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(original_loss, longer_loss, rtol=2e-5, atol=2e-6)
    assert bool((expected[:, 1] - history_scores[:, 1]).abs().max() > 1e-7), "Non-vacuous history test"


@pytest.mark.parametrize("name", ["ktst_mean", "ktst_mhsa"])
def test_checkpoint_reload_cpu_gpu_parity(name, tmp_path):
    torch.manual_seed(42)
    model = build_ktst_model(name, DATA, SMALL).eval()
    path = tmp_path / "random_initialization.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": model.ktlab_model_config}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = build_ktst_model(name, DATA, checkpoint["model_config"]).eval()
    restored.load_state_dict(checkpoint["state_dict"])
    with torch.inference_mode():
        expected, expected_loss = forward_ktst(model, event_batch())
        actual, actual_loss = forward_ktst(restored, event_batch())
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(expected_loss, actual_loss, rtol=0, atol=0)
        if torch.cuda.is_available():
            gpu = build_ktst_model(name, DATA, checkpoint["model_config"], "cuda").eval()
            gpu.load_state_dict(checkpoint["state_dict"])
            scores, loss = forward_ktst(gpu, event_batch(device="cuda"))
            torch.testing.assert_close(expected, scores.cpu(), rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(expected_loss, loss.cpu(), rtol=2e-4, atol=2e-5)


def test_dummy_question_and_parameter_groups_without_optimizer():
    data_config = {"num_q": 0, "num_c": 7, "max_concepts": 1, "maxlen": 8}
    model = build_ktst_model("ktst_mean", data_config, SMALL).eval()
    assert model.ktlab_dummy_question and model.num_questions == 1
    batch = event_batch()
    batch["data"] = batch["data"][..., [0, 3, 4]]
    batch["data"][..., -2][batch["padding_mask"]] = 0
    with torch.inference_mode():
        scores, loss = forward_ktst(model, batch)
    assert torch.isfinite(scores).all() and bool(torch.isfinite(loss))
    groups = optimizer_parameter_groups(model, lr=1e-4, theta_lr=1e-3)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    assert len(groups) == 2 and groups[0]["lr"] == 1e-3 and groups[1]["lr"] == 1e-4
    assert all(names[id(parameter)].endswith("thetas") for parameter in groups[0]["params"])
    assert all(not names[id(parameter)].endswith("thetas") for parameter in groups[1]["params"])
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped_ids) == len(set(grouped_ids)) == len(names)
    assert set(grouped_ids) == set(names)
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="theta_lr"):
        optimizer_parameter_groups(model, lr=1e-4, theta_lr=None)


def test_explicit_model_and_data_restrictions():
    with pytest.raises(ValueError, match="SharedTheta"):
        build_ktst_model("ktst_mean", DATA, {**SMALL, "attn_variant": "learnable_alibi_monotonic_shared_q_k"})
    with pytest.raises(ValueError, match="aggregation"):
        build_ktst_model("ktst_mean", DATA, {**SMALL, "aggregation": "self_attn_all"})
    with pytest.raises(ValueError, match="num_questions"):
        build_ktst_model("ktst_mean", DATA, {**SMALL, "num_questions": 123})
    model = build_ktst_model("ktst_mean", DATA, SMALL).eval()
    duplicate = event_batch()
    duplicate["data"][0, 0, 1] = duplicate["data"][0, 0, 0]
    with pytest.raises(ValueError, match="deduplicated"):
        forward_ktst(model, duplicate)
    empty = event_batch()
    empty["data"][:, 1:] = -1
    empty["padding_mask"][:, 1:] = False
    with pytest.raises(ValueError, match="no scored target"):
        forward_ktst(model, empty)


def test_reference_batch_metadata_cannot_silently_change_targets():
    model = build_ktst_model("ktst_mean", DATA, SMALL).eval()
    batch = event_batch()
    batch["score_mask"] = batch["padding_mask"].clone()
    batch["score_mask"][:, 0] = False
    batch["labels"] = batch["data"][..., -1].float()
    batch["question_ids"] = batch["data"][..., -2].clone()
    batch["kc_counts"] = (batch["data"][..., :-2] >= 0).sum(-1)
    with torch.inference_mode():
        scores, loss = forward_ktst(model, batch)
    assert torch.isfinite(scores).all() and bool(torch.isfinite(loss))
    for key in ("score_mask", "labels", "question_ids", "kc_counts"):
        changed = clone_batch(batch)
        changed[key][0, 1] = 0 if key == "score_mask" else changed[key][0, 1] + 1
        with pytest.raises(ValueError, match=key):
            forward_ktst(model, changed)
