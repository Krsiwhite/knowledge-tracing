"""Artificial CPU UKT acceptance; never uses prepared data or existing weights.

The update cases require KTLAB_RUN_UKT_CPU_UPDATES=1 after recording their
authorization. A complete run uses five backward calls and three Adam steps,
guarded by hard ceilings of eight and four. Run this file alone, on one CPU
thread, with a 30-second process timeout. Author sources are hash-checked;
their pickle caches and all engine artifacts are confined to pytest tmp_path.
"""
import ast
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import default_collate

from ktlab import adapters, engine, ukt_training
from ktlab.shared_adapters import PROTOCOL, collate_shared_kc, predict_shared


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent / "knowledge-tracing" / ".cache" / "research"
LOADER_SHA = "e6af191d9e814db82d5c4db63e77cf34ac07ee9cb737ec0980c2e112ac322579"
AUTHOR_SHAS = {
    "ukt.py": "8b092537174bbe459d4d2e93333d66b16c77bd94456b2b06dc549af6d1183651",
    "utils.py": "b7967a4897742af0a581b0cfb0206776c40ce4605c5c69656e1f1b342ace731e",
    "train_model.py": "d3c9f654d7d40eb437efe53d86a6dcd66b7a5c3eaa2f6bd0430899e485553328",
}
DATA = {"num_c": 7, "num_q": 5, "maxlen": 200}
MODEL = {"d_model": 16, "n_blocks": 2, "num_attn_heads": 2,
         "d_ff": 32, "final_fc_dim": 32, "final_fc_dim2": 16,
         "dropout": 0.0, "seq_len": 200, "cl_weight": 0.07}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def cpu_only_budget(tmp_path_factory):
    """Deny accelerator initialization and non-temporary checkpoint reads."""
    patch = pytest.MonkeyPatch()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    counts = {"backward": 0, "optimizer_steps": 0}
    actual_backward, actual_step, actual_load = torch.autograd.backward, torch.optim.Adam.step, torch.load
    temporary_root = tmp_path_factory.getbasetemp().resolve()

    def deny_cuda(*args, **kwargs):
        raise AssertionError("CPU-UKT-001 forbids CUDA/ROCm access")

    def backward(*args, **kwargs):
        assert os.environ.get("KTLAB_RUN_UKT_CPU_UPDATES") == "1", "Record CPU authorization before updates"
        counts["backward"] += 1
        assert counts["backward"] <= 8, "CPU backward budget exceeded"
        return actual_backward(*args, **kwargs)

    def step(*args, **kwargs):
        assert os.environ.get("KTLAB_RUN_UKT_CPU_UPDATES") == "1", "Record CPU authorization before updates"
        counts["optimizer_steps"] += 1
        assert counts["optimizer_steps"] <= 4, "CPU optimizer budget exceeded"
        return actual_step(*args, **kwargs)

    def load(path, *args, **kwargs):
        assert isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(temporary_root), \
            "Only this test's temporary checkpoints may be read"
        return actual_load(path, *args, **kwargs)

    patch.setattr(torch.cuda, "is_available", lambda: False)
    patch.setattr(torch.cuda, "_lazy_init", deny_cuda)
    patch.setattr(torch.Tensor, "cuda", deny_cuda)
    patch.setattr(torch.nn.Module, "cuda", deny_cuda)
    patch.setattr(torch.autograd, "backward", backward)
    patch.setattr(torch.optim.Adam, "step", step)
    patch.setattr(torch, "load", load)
    try:
        yield counts
    finally:
        patch.undo()
        torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def author_loader(cpu_only_budget):
    path = Path(os.environ.get("KTLAB_UKT_AUTHOR_LOADER",
                RESEARCH / "ukt_86ad9d8_training" / "datasets" / "ukt_dataloader.py"))
    if not path.is_file() and "KTLAB_UKT_AUTHOR_LOADER" not in os.environ:
        pytest.skip("Public author cache absent; set KTLAB_UKT_AUTHOR_LOADER for mandatory oracle verification")
    assert path.is_file(), "Provide the pinned public author loader via KTLAB_UKT_AUTHOR_LOADER"
    assert sha(path) == LOADER_SHA == ukt_training.AUTHOR_DATASET_SHA256
    spec = importlib.util.spec_from_file_location("_test_pinned_ukt_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UKTDataset


@pytest.fixture(scope="module")
def author_model_and_loss(cpu_only_budget):
    directory = Path(os.environ.get("KTLAB_UKT_AUTHOR_MODELS", RESEARCH / "ukt_86ad9d8"))
    if any(not (directory / name).is_file() for name in AUTHOR_SHAS):
        if "KTLAB_UKT_AUTHOR_MODELS" not in os.environ:
            pytest.skip("Public author model cache absent; set KTLAB_UKT_AUTHOR_MODELS for mandatory verification")
        pytest.fail("Explicit KTLAB_UKT_AUTHOR_MODELS must contain all pinned author files")
    for name, expected in AUTHOR_SHAS.items():
        assert sha(directory / name) == expected, f"Pinned author {name} changed"
    namespace = "_test_pinned_author_ukt_86ad9d8"
    package = types.ModuleType(namespace)
    package.__path__ = [str(directory)]
    sys.modules[namespace] = package
    spec = importlib.util.spec_from_file_location(namespace + ".ukt", directory / "ukt.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Compile the unchanged original function only: unrelated training imports
    # would otherwise import optional datasets/configuration and are unnecessary.
    parsed = ast.parse((directory / "train_model.py").read_text(encoding="utf-8"))
    function, = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "cal_loss"]
    scope = {"torch": torch, "binary_cross_entropy": binary_cross_entropy}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(directory / "train_model.py"), "exec"), scope)
    yield module.UKT, scope["cal_loss"]
    for key in list(sys.modules):
        if key == namespace or key.startswith(namespace + "."):
            del sys.modules[key]


def write_sequences(path, sequences, width, *, zero_selection=False):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "uid", "questions", "concepts", "responses", "selectmasks"])
        writer.writeheader()
        for uid, responses in enumerate(sequences):
            size = len(responses)
            assert 2 <= size <= width
            selected = [1] * size
            if zero_selection:
                selected[1] = 0
            row = {"fold": 1, "uid": uid}
            for key, values in (("questions", [i % 5 for i in range(size)]),
                                ("concepts", [i % 7 for i in range(size)]),
                                ("responses", list(responses)), ("selectmasks", selected)):
                row[key] = ",".join(map(str, values + [-1] * (width - size)))
            writer.writerow(row)


@pytest.mark.parametrize("length", [2, 3, 4, 5])
@pytest.mark.parametrize("padding", [0, 1, 3])
def test_exhaustive_binary_sequences_match_pinned_author_without_mutation(tmp_path, author_loader, length, padding):
    sequences = list(itertools.product((0, 1), repeat=length))
    path = tmp_path / "artificial.csv"
    write_sequences(path, sequences, length + padding)
    reference = author_loader(str(path), ["questions", "concepts"], {1})
    dataset = adapters.SequenceDataset(path, {1})
    try:
        for index in range(len(sequences)):
            original = default_collate([dataset[index]])
            snapshots = {key: value.clone() for key, value in original.items() if torch.is_tensor(value)}
            augmented = ukt_training.author_augmented_responses(original)
            expected = reference[index]
            for key in ("qseqs", "shft_qseqs", "cseqs", "shft_cseqs", "rseqs", "shft_rseqs",
                        "masks", "smasks", "r_aug", "shft_r_aug"):
                assert torch.equal(augmented[key][0], expected[key]), (length, padding, sequences[index], key)
            assert "r_aug" not in original and "shft_r_aug" not in original
            for key, snapshot in snapshots.items():
                assert torch.equal(original[key], snapshot), key
            assert augmented["r_aug"].data_ptr() != original["rseqs"].data_ptr()
            assert augmented["shft_r_aug"].data_ptr() != original["shft_rseqs"].data_ptr()
    finally:
        dataset.close()
    assert (tmp_path / "artificial.csv_1.pkl").is_file()


def test_full_200_and_padding_and_zero_selectmask_difference_are_explicit(tmp_path, author_loader):
    for length in (2, 199, 200):
        path = tmp_path / f"length-{length}.csv"
        write_sequences(path, [[i % 2 for i in range(length)], [1 - i % 2 for i in range(length)]], 200)
        reference = author_loader(str(path), ["questions", "concepts"], {1})
        dataset = adapters.SequenceDataset(path, {1})
        try:
            for index in range(2):
                actual = ukt_training.author_augmented_responses(default_collate([dataset[index]]))
                for key in ("rseqs", "shft_rseqs", "masks", "smasks", "r_aug", "shft_r_aug"):
                    assert torch.equal(actual[key][0], reference[index][key])
        finally:
            dataset.close()
    path = tmp_path / "zero-selection.csv"
    write_sequences(path, [[1, 0, 1]], 200, zero_selection=True)
    reference = author_loader(str(path), ["questions", "concepts"], {1})[0]
    dataset = adapters.SequenceDataset(path, {1})
    try:
        actual = ukt_training.author_augmented_responses(default_collate([dataset[0]]))
        # Deliberate project semantics: author != -1 includes 0; project == 1
        # excludes 0. We claim loader equality only for ordinary 1/-1 training.
        assert reference["smasks"][0].item() is True
        assert actual["smasks"][0, 0].item() is False
        assert torch.equal(actual["smasks"][0, 1:], reference["smasks"][1:])
        for key in ("rseqs", "shft_rseqs", "masks", "r_aug", "shft_r_aug"):
            assert torch.equal(actual[key][0], reference[key])
    finally:
        dataset.close()


@pytest.mark.parametrize("mask,response,shifted", [
    ([], [], []), ([False, False], [0, 0], [0, 0]),
    ([False, True], [0, 1], [0, 0]), ([True, False, True], [0, 0, 1], [1, 0, 0]),
    ([True], [2], [0]), ([True], [0], [-1]), ([True], [float("nan")], [0]),
    ([True], [0], [float("inf")]), ([True, False], [0, 1], [1, 0]),
])
def test_augmentation_rejects_empty_nonprefix_nonbinary_and_dirty_padding(mask, response, shifted):
    batch = {"masks": torch.tensor([mask], dtype=torch.bool),
             "rseqs": torch.tensor([response], dtype=torch.float32),
             "shft_rseqs": torch.tensor([shifted], dtype=torch.float32)}
    with pytest.raises(ValueError):
        ukt_training.author_augmented_responses(batch)


def small_model(**switches):
    torch.manual_seed(142)
    model = adapters.build_model("ukt", DATA, {**MODEL, **switches}, "cpu")
    with torch.no_grad():
        model.difficult_param.weight.normal_(mean=0.2, std=0.3)
    assert torch.count_nonzero(model.difficult_param.weight) == model.difficult_param.weight.numel()
    return model


def require_update_authorization():
    if os.environ.get("KTLAB_RUN_UKT_CPU_UPDATES") != "1":
        pytest.skip("CPU update cases require the recorded synthetic-only authorization")


@pytest.mark.parametrize("use_cl,augmentation,recipe", [
    (False, True, "ukt-no-cl"), (True, False, "ukt-cl-same-responses"),
    (True, True, ukt_training.AUTHOR_AUGMENTATION),
])
def test_three_training_recipes_match_author_model_loss_and_have_finite_gradients(
        tmp_path, author_loader, author_model_and_loss, use_cl, augmentation, recipe):
    require_update_authorization()
    path = tmp_path / "loss-input.csv"
    write_sequences(path, [[1, 1, 0, 1, 0], [0, 1, 1, 0]], 200)
    reference = author_loader(str(path), ["questions", "concepts"], {1})
    author_batch = default_collate([reference[0], reference[1]])
    dataset = adapters.SequenceDataset(path, {1})
    try:
        batch = default_collate([dataset[0], dataset[1]])
    finally:
        dataset.close()
    model = small_model(use_CL=use_cl, use_uncertainty_aug=augmentation).train()
    author_class, cal_loss = author_model_and_loss
    author = author_class(n_question=DATA["num_c"], n_pid=DATA["num_q"], **model.ktlab_model_config).train()
    author.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        original = author(author_batch, train=True)
        expected_loss = cal_loss(author, [original[0][:, 1:], *original[1:]],
                                 author_batch["rseqs"], author_batch["shft_rseqs"], author_batch["smasks"])
    predictions, regularizer = adapters.forward_model(model, batch)
    actual_loss = binary_cross_entropy(predictions[batch["smasks"]].double(),
                                      batch["shft_rseqs"][batch["smasks"]].double()) + regularizer
    torch.testing.assert_close(predictions, original[0][:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(regularizer, model.cl_weight * original[1] if use_cl else torch.zeros(()))
    assert ukt_training.training_recipe(model) == recipe
    assert "r_aug" not in batch
    if use_cl:
        assert regularizer.requires_grad, "NCE must remain attached to autograd"
    actual_loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    if use_cl and augmentation:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer.step()
        assert any(not torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def contexts():
    result = []
    for student, concepts in enumerate(([3, 4], [5], [1, 6])):
        result.append({"protocol": PROTOCOL, "uid": str(student), "fold": 0,
                       "event_id": f"toy:{student}:2", "target_index": 2,
                       "history": [{"event_id": f"toy:{student}:0", "question": 1,
                                    "concepts": [0, 2], "response": student % 2},
                                   {"event_id": f"toy:{student}:1", "question": 2,
                                    "concepts": [1], "response": 1 - student % 2}],
                       "query": {"question": 4, "concepts": list(concepts)}, "label": student % 2})
    return result


def test_eval_does_not_augment_and_shared_isolation_mean_and_chunks(monkeypatch):
    model = small_model().train()
    items = contexts()

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not construct an augmented response view")

    monkeypatch.setattr(ukt_training, "author_augmented_responses", forbidden)
    batch = collate_shared_kc(items)
    model.eval()
    with torch.no_grad():
        full, regularizer = ukt_training.forward_ukt(model, batch)
    assert full.shape == (5, 200) and regularizer.item() == 0
    per_kc = full[torch.arange(5), batch["target_positions"]].double()
    expected = torch.stack([per_kc[:2].mean(), per_kc[2], per_kc[3:].mean()])
    model.train()
    baseline = predict_shared(model, items, "cpu", max_rows_per_forward=5)
    assert model.training, "Prediction must restore caller mode"
    torch.testing.assert_close(torch.as_tensor(baseline, dtype=torch.float64), expected, rtol=2e-6, atol=2e-7)
    for chunk in (1, 2, 3):
        actual = predict_shared(model, items, "cpu", max_rows_per_forward=chunk)
        torch.testing.assert_close(torch.as_tensor(actual, dtype=torch.float64),
                                   torch.as_tensor(baseline, dtype=torch.float64), rtol=2e-6, atol=2e-7)
    changed = copy.deepcopy(items)
    for item in changed:
        item["label"] = 1 - item["label"]
        item["query"]["response"] = 1
        item["future_responses"] = [1, 0, 1]
    assert torch.equal(collate_shared_kc(changed)["responses"], batch["responses"])
    assert predict_shared(model, changed, "cpu", max_rows_per_forward=5) == baseline
    # Positive control: the model can use legitimate past responses even with
    # nonzero randomized item difficulty; isolation is not a constant predictor.
    history_changed = copy.deepcopy(items)
    for item in history_changed:
        for event in item["history"]:
            event["response"] = 1 - event["response"]
    altered = predict_shared(model, history_changed, "cpu", max_rows_per_forward=5)
    assert max(abs(a - b) for a, b in zip(altered, baseline)) > 1e-8


def engine_config(tmp_path):
    original, sequences = tmp_path / "train_valid.csv", tmp_path / "train_valid_sequences.csv"
    for path in (original, sequences):
        fields = ["fold", "uid", "questions", "concepts", "responses", "is_repeat"]
        if path == sequences:
            fields.append("selectmasks")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for fold in (0, 1):
                for student in (0, 1):
                    row = {"fold": fold, "uid": str(10 * fold + student),
                           "questions": "0,0,1,2,3", "concepts": "0,1,2,3,4",
                           "responses": "0,0,1,0,1" if student == 0 else "1,1,0,1,0",
                           "is_repeat": "0,1,0,0,0"}
                    if path == sequences:
                        for key in ("questions", "concepts", "responses", "is_repeat"):
                            row[key] += ",-1" * 195
                        row["selectmasks"] = "1,1,1,1,1" + ",-1" * 195
                    writer.writerow(row)
    manifest = {"dataset": "synthetic", "data_dir": str(tmp_path),
                "data_config": {**DATA, "max_concepts": 2, "folds": [0, 1],
                                "train_valid_file": sequences.name, "train_valid_original_file": original.name},
                "files": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                          for path in (original, sequences)}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"protocol": PROTOCOL, "model": "ukt", "dataset": "synthetic", "dataset_manifest": str(manifest_path),
            "device": "cpu", "fold": 0, "seed": 42, "model_config": dict(MODEL),
            "batch_size": 2, "num_workers": 0, "epochs": 2, "patience": 3,
            "max_train_steps": 2, "max_wall_seconds": 20, "learning_rate": 1e-3,
            "validation_batch_size": 6, "max_rows_per_forward": 8,
            "batch_id": "CPU-UKT-001", "_batch_approved": True}


def test_tiny_real_engine_saves_actual_strict_best_and_bound_recipe(tmp_path, monkeypatch, cpu_only_budget):
    require_update_authorization()
    candidates = []
    real_save = engine.save_best_checkpoint

    def observe(model, path, auc, best, epoch, manifest_sha256, **kwargs):
        candidates.append((epoch, float(auc), {key: value.detach().clone() for key, value in model.state_dict().items()}))
        return real_save(model, path, auc, best, epoch, manifest_sha256, **kwargs)

    monkeypatch.setattr(engine, "save_best_checkpoint", observe)
    before = dict(cpu_only_budget)
    run = tmp_path / "run"
    result = engine.train(engine_config(tmp_path), run)
    assert result["metadata"]["status"] == "completed"
    assert result["metadata"]["optimizer_steps"] == 2
    assert cpu_only_budget["backward"] - before["backward"] == 2
    assert cpu_only_budget["optimizer_steps"] - before["optimizer_steps"] == 2
    assert [entry[0] for entry in candidates] == [0, 1, 2]
    # Actual validation AUC, without replacing scores, labels or model calls.
    # Ties keep the earliest checkpoint, including a legitimate epoch-0 best.
    best_epoch, best_auc, best_state = max(candidates, key=lambda entry: entry[1])
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    resolved = json.loads((run / "resolved_config.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())
    assert checkpoint["epoch"] == metadata["best_epoch"] == best_epoch
    assert checkpoint["valid_auc"] == metadata["best_valid_auc"] == best_auc
    assert checkpoint["state_dict"].keys() == best_state.keys()
    assert all(torch.equal(checkpoint["state_dict"][key], value) for key, value in best_state.items())
    assert any(not torch.equal(candidates[0][2][key], candidates[2][2][key]) for key in best_state)
    assert resolved["ukt_training_recipe"] == ukt_training.AUTHOR_AUGMENTATION
    assert checkpoint["config_sha256"] == metadata["config_sha256"] == engine.config_fingerprint(resolved)
    assert metadata["checkpoint_sha256"] == sha(run / "best.pt")
    assert result["valid"]["auc"] == best_auc
    assert metadata["test_accessed"] is False and metadata["device"] == "cpu"
    assert metadata["peak_gpu_allocated_bytes"] == metadata["peak_gpu_reserved_bytes"] == 0
    assert (run / "valid_predictions.csv").is_file()
    assert (run / "valid_predictions.csv.manifest.json").is_file()
    assert not list(run.glob("test*"))
