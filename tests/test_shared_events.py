"""CPU-only canonical-event protocol tests; no test split or model execution."""
from array import array
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from ktlab.shared_events import PROTOCOL, SharedEventDataset, context_digests


ROOT = Path(__file__).resolve().parents[1]


def _event(question, concepts, response=0):
    return {"question": question, "concepts": concepts, "response": response}


def _fixture(path, students, *, concept_only=False):
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid, fold, events in students:
        row = {"uid": str(uid), "fold": fold}
        flat = {"concepts": [], "responses": []}
        if not concept_only:
            flat.update(questions=[], is_repeat=[])
        for event in events:
            for index, concept in enumerate(event["concepts"]):
                flat["concepts"].append(concept)
                flat["responses"].append(event["response"])
                if not concept_only:
                    flat["questions"].append(event["question"])
                    flat["is_repeat"].append(int(index > 0))
        row.update({key: ",".join(map(str, values)) for key, values in flat.items()})
        rows.append(row)
    source = path / "train_valid.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"dataset": "synthetic", "data_dir": str(path.resolve()),
            "data_config": {"num_q": 0 if concept_only else 1000, "num_c": 512,
                            "folds": sorted({fold for _, fold, _ in students}),
                            "train_valid_original_file": "train_valid.csv"},
            "files": {"train_valid.csv": {"bytes": source.stat().st_size,
                                          "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}}


def test_canonical_merge_preserves_kc_order_and_separate_attempts(tmp_path):
    manifest = _fixture(tmp_path, [("7", 0, [_event(10, [2, 2, 1], 1),
                                           _event(10, [3], 0), _event(11, [4, 5], 1)])])
    dataset = SharedEventDataset(manifest, 0)
    assert len(dataset) == 2
    first, second = dataset[0], dataset[1]
    assert first["event_id"] == "synthetic:7:1"
    assert first["history"] == [{"event_id": "synthetic:7:0", "question": 10,
                                 "concepts": [2, 1], "response": 1}]
    assert first["query"] == {"question": 10, "concepts": [3]}
    assert first["label"] == 0 and "response" not in first["query"]
    assert second["target_index"] == 2 and len(second["history"]) == 2
    assert second["context_kc_count"] == 5
    assert dataset.audit["duplicate_kc_positions_removed"] == 1
    assert dataset.audit["exclusion_counts"] == {"target_over_budget": 0, "no_history": 1}


def test_complete_contiguous_suffix_at_actual_200_kc_boundary(tmp_path):
    manifest = _fixture(tmp_path, [("7", 0, [_event(1, list(range(100))),
                                           _event(2, list(range(100, 199))),
                                           _event(3, [200, 201])])])
    dataset = SharedEventDataset(manifest, 0)
    assert dataset[0]["context_kc_count"] == 199
    target = dataset[1]
    assert target["history_start_index"] == 1
    assert target["context_kc_count"] == 101
    assert target["history"][0]["concepts"] == list(range(100, 199))
    assert len(target["history"]) == 1  # never retain a partial piece of event 0


def test_oversize_and_no_history_exclusions_do_not_skip_nearest_event(tmp_path):
    events = [_event(1, [1]), _event(2, list(range(200))), _event(3, [2]),
              _event(4, list(range(201))), _event(5, [3]), _event(6, [4])]
    dataset = SharedEventDataset(_fixture(tmp_path, [("7", 0, events)]), 0)
    assert len(dataset) == 1 and dataset[0]["target_index"] == 5
    assert dataset[0]["history_start_index"] == 4
    assert dataset.audit["exclusion_counts"] == {"target_over_budget": 1, "no_history": 4}
    assert dataset.audit["no_history_reasons"] == {
        "first_event": 1, "immediate_predecessor_exceeds_remaining_budget": 3}
    assert dataset.audit["canonical_events"] == 6
    assert dataset.audit["eligible_total"] == dataset.audit["expected_total"] == 1


def test_current_label_is_separate_from_its_digests_but_can_be_future_history(tmp_path):
    original_events = [_event(1, [1], 0), _event(2, [2, 3], 0), _event(3, [4], 1)]
    changed_events = copy.deepcopy(original_events)
    changed_events[1]["response"] = 1
    original = SharedEventDataset(_fixture(tmp_path / "a", [("7", 0, original_events)]), 0)
    changed = SharedEventDataset(_fixture(tmp_path / "b", [("7", 0, changed_events)]), 0)
    assert original[0]["label"] != changed[0]["label"]
    assert context_digests(original[0]) == context_digests(changed[0])
    assert original[1]["target_sha256"] == changed[1]["target_sha256"]
    assert original[1]["context_sha256"] != changed[1]["context_sha256"]
    assert original.audit["source_sha256"] != changed.audit["source_sha256"]
    assert original.audit["target_sha256"] == changed.audit["target_sha256"]
    assert original.audit["context_sha256"] != changed.audit["context_sha256"]
    assert original.audit["exclusion_counts"] == changed.audit["exclusion_counts"]
    last_changed = copy.deepcopy(original_events)
    last_changed[-1]["response"] = 0
    future = SharedEventDataset(_fixture(tmp_path / "c", [("7", 0, last_changed)]), 0)
    assert original.audit["context_sha256"] == future.audit["context_sha256"]


def test_prefix_selection_keeps_full_eligibility_and_selected_hash_scope(tmp_path):
    events = [_event(i, [i]) for i in range(6)]
    manifest = _fixture(tmp_path, [("7", 0, events), ("8", 0, events)])
    dataset = SharedEventDataset(manifest, 0, max_targets=2)
    assert dataset.audit["eligible_total"] == dataset.audit["expected_total"] == 10
    assert dataset.audit["selected_targets"] == len(dataset) == 2
    assert dataset.audit["selection_limited"] and dataset.audit["selected_for_debug"]
    assert dataset.audit["exclusion_counts"]["no_history"] == 2
    assert dataset.audit["hash_scope"] == "selected_targets_in_original_csv_and_event_order"
    target_hash = hashlib.sha256(b"ktlab.shared.target-sequence.v1\0")
    context_hash = hashlib.sha256(b"ktlab.shared.context-sequence.v1\0")
    for context in dataset:
        target_hash.update(bytes.fromhex(context["target_sha256"]))
        context_hash.update(bytes.fromhex(context["context_sha256"]))
    assert dataset.audit["target_sha256"] == target_hash.hexdigest()
    assert dataset.audit["context_sha256"] == context_hash.hexdigest()


def test_assist2015_has_unknown_question_and_preserves_repeated_observations(tmp_path):
    events = [_event(None, [2], 0), _event(None, [2], 1), _event(None, [3], 0)]
    dataset = SharedEventDataset(_fixture(tmp_path, [("7", 0, events)], concept_only=True), 0)
    assert len(dataset) == 2
    assert dataset[0]["query"] == {"question": None, "concepts": [2]}
    assert dataset[0]["history"][0]["question"] is None
    assert dataset[1]["history"][0]["event_id"] != dataset[1]["history"][1]["event_id"]


def test_compact_index_bounded_cache_and_returned_contexts_are_independent(tmp_path):
    manifest = _fixture(tmp_path, [(str(i), 0, [_event(1, [1]), _event(2, [2])]) for i in range(4)])
    dataset = SharedEventDataset(manifest, 0)
    assert not dataset._cache
    assert all(isinstance(value, array) for value in (dataset._row_ids, dataset._target_ordinals, dataset._history_starts))
    assert dataset.audit["index_bytes"] == 16 * len(dataset)
    lengths = dataset.audit["length_statistics"]
    assert lengths["scope"] == "selected_targets"
    assert lengths["history_kc_min"] == lengths["history_kc_max"] == 1
    assert lengths["history_events_min"] == lengths["history_events_max"] == 1
    assert sum(lengths["history_kc_bins"].values()) == len(dataset)
    assert lengths["query_kc_counts"] == {"1": 4}
    for i in range(4):
        assert dataset[i]["uid"] == str(i)
        assert len(dataset._cache) <= 2
    context = dataset[3]
    context["history"][0]["concepts"].append(99)
    context["query"]["concepts"].append(98)
    assert dataset[3]["history"][0]["concepts"] == [1]
    assert dataset[3]["query"]["concepts"] == [2]
    dataset.close()
    assert not dataset._cache


def test_fold_selection_uses_existing_development_membership(tmp_path):
    events = [_event(1, [1]), _event(2, [2])]
    manifest = _fixture(tmp_path, [("7", 0, events), ("8", 1, events)])
    assert SharedEventDataset(manifest, 0, "valid")[0]["uid"] == "7"
    assert SharedEventDataset(manifest, 0, "train")[0]["uid"] == "8"
    invalid = copy.deepcopy(manifest)
    invalid["data_config"]["folds"] = [-1, 0]
    with pytest.raises(ValueError, match="development folds"):
        SharedEventDataset(invalid, 0)


def test_no_eligible_targets_is_an_explicit_empty_dataset(tmp_path):
    dataset = SharedEventDataset(_fixture(tmp_path, [("7", 0, [_event(1, [1])])]), 0)
    assert len(dataset) == dataset.audit["expected_total"] == 0
    assert dataset.audit["exclusion_counts"]["no_history"] == 1
    with pytest.raises(IndexError):
        dataset[0]


@pytest.mark.parametrize("split", ["test", "test_window", "all"])
def test_non_development_split_is_rejected_before_source_access(split):
    with pytest.raises(ValueError, match="test remains sealed"):
        SharedEventDataset({}, 0, split)


def test_wrong_budget_source_and_changed_file_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires kc_budget=200"):
        SharedEventDataset({}, 0, kc_budget=199)
    manifest = _fixture(tmp_path, [("7", 0, [_event(1, [1]), _event(2, [2])])])
    bad = copy.deepcopy(manifest)
    bad["data_config"]["train_valid_original_file"] = "test.csv"
    with pytest.raises(ValueError, match="canonical train_valid.csv"):
        SharedEventDataset(bad, 0)
    source = tmp_path / "train_valid.csv"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        SharedEventDataset(manifest, 0)


@pytest.mark.parametrize("fold", [0.5, True, "0"])
def test_fold_type_cannot_be_silently_coerced(fold):
    with pytest.raises(ValueError, match="integer development fold"):
        SharedEventDataset({}, fold)


def test_source_mutation_after_indexing_is_rejected(tmp_path):
    manifest = _fixture(tmp_path, [("7", 0, [_event(1, [1], 0), _event(2, [2], 0)])])
    dataset = SharedEventDataset(manifest, 0)
    source = tmp_path / "train_valid.csv"
    source.write_bytes(source.read_bytes().replace(b'"0,0"', b'"0,1"', 1))
    with pytest.raises(ValueError, match="changed after"):
        dataset.verify_source()
    with pytest.raises(ValueError, match="changed after"):
        dataset[0]


def test_concept_only_repeat_flags_cannot_merge_observations(tmp_path):
    manifest = _fixture(tmp_path, [("7", 0, [_event(1, [1, 2])])])
    manifest["data_config"]["num_q"] = 0
    with pytest.raises(ValueError, match="must remain separate events"):
        SharedEventDataset(manifest, 0)


@pytest.mark.parametrize("name", ["assist2009", "assist2015", "nips_task34"])
def test_existing_development_data_has_complete_causal_contexts(name):
    base = Path(os.environ.get("KTST_PREPARED_ROOT", str(ROOT.parent / "knowledge-tracing/data/prepared")))
    path = base / name / "manifest.json"
    if not path.exists():
        pytest.skip("Optional existing prepared development files are not present")
    dataset = SharedEventDataset(json.loads(path.read_text()), 0, max_targets=3)
    assert len(dataset) == 3 and dataset.audit["eligible_total"] > 3
    for target in dataset:
        assert target["protocol"] == PROTOCOL
        assert target["history"] and target["context_kc_count"] <= 200
        assert all(event["event_id"] != target["event_id"] for event in target["history"])
        assert "response" not in target["query"] and "label" not in target["query"]
        assert context_digests(target) == {key: target[key] for key in ("target_sha256", "context_sha256")}
        assert (target["query"]["question"] is None) == (name == "assist2015")
    assert dataset.audit["test_access"] is False
    dataset.close()
