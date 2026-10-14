"""Shared-context projection and real-model inference, without differentiation."""
from copy import deepcopy

import pytest
import torch

from ktlab.adapters import build_model
from ktlab.ktst_model import build_ktst_model
from ktlab import shared_adapters as shared


DATA = {"num_q": 9, "num_c": 8, "max_concepts": 3, "maxlen": 200}
KC_SMALL = {"d_model": 16, "n_blocks": 1, "d_ff": 32, "num_attn_heads": 2,
            "final_fc_dim": 32, "dropout": 0.0, "seq_len": 200}
SET_SMALL = {"d_model": 16, "nhead_tf": 2, "nhead_agg": 2,
             "num_layers_tf_dec": 1, "num_layers_tf_enc": 1, "num_layers_agg": 1,
             "dim_feedforward": 32, "dim_classifier": 32, "dropout": 0.0}


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("Shared evaluation is forward-only")
    monkeypatch.setattr(torch.Tensor, "backward", deny)
    monkeypatch.setattr(torch.autograd, "backward", deny)
    monkeypatch.setattr(torch.autograd, "grad", deny)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", deny)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def contexts():
    first = {"protocol": shared.PROTOCOL, "uid": "7", "fold": 0, "event_id": "toy:7:2",
             "target_index": 2, "history_start_index": 0,
             "history": [{"event_id": "toy:7:0", "question": 4, "concepts": [0, 2], "response": 1},
                         {"event_id": "toy:7:1", "question": 4, "concepts": [1], "response": 0}],
             "query": {"question": 4, "concepts": [3, 4]}, "label": 1,
             "history_kc_count": 3, "target_kc_count": 2, "context_kc_count": 5}
    second = deepcopy(first)
    second.update(event_id="toy:7:3", target_index=3, label=0, history_kc_count=5, context_kc_count=7)
    second["history"].append({"event_id": "toy:7:2", "question": 4, "concepts": [3, 4], "response": 1})
    return [first, second]


def build(name, device="cpu"):
    torch.manual_seed(37)
    if name.startswith("ktst_"):
        return build_ktst_model(name, DATA, SET_SMALL, device)
    settings = dict(KC_SMALL)
    if name == "akt":
        settings.pop("seq_len")
    return build_model(name, DATA, settings, device)


def test_actual_projections_preserve_history_and_sibling_isolation():
    items = contexts()
    result = shared.audit_shared_projection(items, DATA)
    assert result["passed"] and result["events"] == 2 and result["kc_rows"] == 4
    kc = shared.collate_shared_kc(items)
    sets = shared.collate_shared_ktst(items, max_concepts=3)
    assert kc["concepts"].shape == (4, 200) and sets["data"].shape == (2, 200, 5)
    assert kc["concepts"][0, :4].tolist() == [0, 2, 1, 3]
    assert kc["concepts"][1, :4].tolist() == [0, 2, 1, 4]
    assert kc["responses"][0, :4].tolist() == [1, 1, 0, 0]
    assert torch.equal(kc["responses"][0, :3], kc["responses"][1, :3])
    assert kc["ground_seq"].flatten().tolist() == [4.0, 4.0, 6.0, 6.0]
    assert kc["event_ids"] == ["toy:7:2", "toy:7:2", "toy:7:3", "toy:7:3"]
    assert sets["history_event_ids"][0] == ["toy:7:0", "toy:7:1"]  # same question, distinct attempts
    assert "labels" not in sets and "label" not in kc and "score_mask" not in sets


def test_label_and_unknown_future_fields_do_not_affect_projection():
    original, altered = contexts(), contexts()
    for item in altered:
        item["label"] = 1 - item["label"]
        item["future_responses"] = [1, 1, 0, 0] * 100
        item["student_final_sequence_length"] = 100000
        item["query"]["response"] = 1  # Even this extraneous current answer is never read.
    for collator in (lambda c: shared.collate_shared_ktst(c, 3), shared.collate_shared_kc):
        before, after = collator(original), collator(altered)
        for key, value in before.items():
            if torch.is_tensor(value):
                assert torch.equal(value, after[key]), key
            else:
                assert value == after[key]
    assert shared.audit_shared_projection(original, DATA) == shared.audit_shared_projection(altered, DATA)


def test_projection_audit_checks_tensors_not_only_source_hash(monkeypatch):
    real_collator = shared.collate_shared_kc
    def wrong_history(*args, **kwargs):
        result = real_collator(*args, **kwargs)
        result["responses"][0, 0] = 1 - result["responses"][0, 0]
        return result
    monkeypatch.setattr(shared, "collate_shared_kc", wrong_history)
    with pytest.raises(ValueError, match="differs"):
        shared.audit_shared_projection(contexts(), DATA)


def test_projection_audit_checks_event_boundaries_not_just_flat_history(monkeypatch):
    items = contexts()[:1]
    items[0]["history"][1]["response"] = 1  # Same q/r, so flattened equality cannot detect regrouping.
    real_collator = shared.collate_shared_ktst
    def wrong_boundary(*args, **kwargs):
        result = real_collator(*args, **kwargs)
        result["data"][0, 0, :3] = torch.tensor([0, -1, -1])
        result["data"][0, 1, :3] = torch.tensor([2, 1, -1])
        return result
    monkeypatch.setattr(shared, "collate_shared_ktst", wrong_boundary)
    with pytest.raises(ValueError, match="per-event boundary"):
        shared.audit_shared_projection(items, DATA)


def test_concept_only_dummy_question_and_exact_capacity():
    items = contexts()
    for item in items:
        item["query"]["question"] = None
        for event in item["history"]:
            event["question"] = None
    assert shared.audit_shared_projection(items, {**DATA, "num_q": 0})["passed"]
    assert shared.collate_shared_kc(items)["questions"].eq(0).all()
    boundary = {"protocol": shared.PROTOCOL, "event_id": "toy:9:198",
                "history": [{"event_id": f"toy:9:{i}", "question": 0, "concepts": [i % 8], "response": i % 2}
                            for i in range(198)], "query": {"question": 1, "concepts": [3, 4]}, "label": 1}
    rows = shared.collate_shared_kc([boundary])
    assert rows["target_positions"].tolist() == [198, 198]
    assert rows["ground_seq"].flatten().tolist() == [199, 199]
    boundary["query"]["concepts"].append(5)
    with pytest.raises(ValueError, match="complete query"):
        shared.collate_shared_kc([boundary])  # Each separate KC would fit, full query does not.


def test_malformed_or_target_contaminated_contexts_are_rejected():
    item = contexts()[0]
    bad = deepcopy(item)
    bad["history"][0]["event_id"] = bad["event_id"]
    with pytest.raises(ValueError, match="exclude the target"):
        shared.collate_shared_kc([bad])
    bad = deepcopy(item)
    bad["query"]["concepts"] = [3, 3]
    with pytest.raises(ValueError, match="deduplicated"):
        shared.collate_shared_ktst([bad], 3)
    bad = deepcopy(item)
    bad["history"] = []
    with pytest.raises(ValueError, match="nonempty allowed history"):
        shared.collate_shared_kc([bad])


class _NoLabelRead(dict):
    def __getitem__(self, key):
        if key == "label":
            raise AssertionError("Predictor tried to read the current label")
        return super().__getitem__(key)


@pytest.mark.parametrize("name", ["akt", "simplekt", "ktst_mean", "ktst_mhsa", "mockt"])
def test_original_models_target_only_causality_and_chunk_alignment(name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mockt" and device != "cuda":
        pytest.skip("Unmodified MoC-KT requires CUDA/ROCm")
    model = build(name, device)
    original_training = model.training
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    items = contexts()
    altered = deepcopy(items)
    for item in altered:
        item["label"] = 1 - item["label"]
        item["future_responses"] = [1 - (i % 2) for i in range(100)]
        item["student_final_sequence_length"] = 9999
    expected = shared.predict_shared(model, [_NoLabelRead(i) for i in items], device, max_rows_per_forward=1)
    actual = shared.predict_shared(model, altered, device, max_rows_per_forward=3)
    torch.testing.assert_close(torch.tensor(expected), torch.tensor(actual), rtol=2e-5, atol=2e-6)
    assert isinstance(expected, list) and len(expected) == 2 and all(type(p) is float for p in expected)
    assert model.training == original_training
    model.eval()
    # Independently perturb the raw current response after projection, proving
    # model causality rather than merely checking that the adapter uses a dummy.
    if name.startswith("ktst_"):
        batch = shared._move(shared.collate_shared_ktst(items, model.max_concepts), torch.device(device))
        changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        row_indices = torch.arange(2, device=device)
        changed["data"][row_indices, batch["target_positions"], -1] = 1
        with torch.inference_mode():
            raw_a, _, _ = model(batch["data"], batch["padding_mask"])
            raw_b, _, _ = model(changed["data"], changed["padding_mask"])
    else:
        batch = shared._move(shared.collate_shared_kc(items), torch.device(device))
        changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
        row_indices = torch.arange(4, device=device)
        changed["responses"][row_indices, batch["target_positions"]] = 1
        changed["shft_rseqs"][row_indices, batch["target_positions"] - 1] = 1
        if name == "simplekt":
            with torch.inference_mode():
                raw_a, raw_b = model(batch), model(changed)
        else:
            with torch.inference_mode():
                arguments = [batch["concepts"], batch["responses"], batch["questions"]]
                altered_arguments = [changed["concepts"], changed["responses"], changed["questions"]]
                if name == "mockt":
                    arguments.insert(0, batch["ground_seq"])
                    altered_arguments.insert(0, changed["ground_seq"])
                raw_a, _ = model(*arguments)
                raw_b, _ = model(*altered_arguments)
    selected_a = raw_a[row_indices, batch["target_positions"]]
    selected_b = raw_b[row_indices, batch["target_positions"]]
    torch.testing.assert_close(selected_a, selected_b, rtol=2e-5, atol=2e-6)
    if not name.startswith("ktst_"):
        independent = [selected_a[:2].double().mean().item(), selected_a[2:].double().mean().item()]
    else:
        independent = selected_a.double().tolist()
    torch.testing.assert_close(torch.tensor(expected), torch.tensor(independent), rtol=2e-5, atol=2e-6)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_expanded_forward_rows_are_bounded_and_full_precision_mean_retained():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model_name = "akt"
            self.batch_sizes = []
        def forward(self, concepts, responses, questions):
            assert not torch.is_grad_enabled()
            self.batch_sizes.append(len(concepts))
            return 0.123456789 + concepts.double() / 100.0, torch.tensor(float("nan"))
    model = Model()
    result = shared.predict_shared(model, contexts(), "cpu", max_rows_per_forward=3)
    assert model.batch_sizes == [3, 1]
    assert result == pytest.approx([0.158456789, 0.158456789], abs=1e-15)
    assert result[0] != round(result[0], 4)


def test_ktst_dummy_loss_is_never_returned_or_used():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model_name = "ktst_mean"
            self.max_concepts, self.max_len = 3, 200
            self.num_questions, self.num_concepts = 9, 8
        def forward(self, data, padding_mask):
            assert not torch.is_grad_enabled()
            return torch.full(data.shape[:2], 0.75), torch.tensor(float("nan")), 0.0
    assert shared.predict_shared(Model(), contexts(), "cpu") == [0.75, 0.75]


def test_invalid_individual_kc_scores_cannot_be_hidden_by_late_mean():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.model_name = "akt"
        def forward(self, concepts, responses, questions):
            scores = torch.where(concepts == 3, -0.2, 1.2)
            return scores, torch.tensor(0.0)
    with pytest.raises(ValueError, match="per-KC"):
        shared.predict_shared(Model(), contexts(), "cpu")
