"""Forward-only data conversion tests; never read any test split records."""
import csv
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from ktlab.ktst_data import KTSTSequenceDataset, PROTOCOL, collate_ktst, set_dense_from_row
from ktlab.ktst_model import load_ktst_reference


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def official_converter():
    module = load_ktst_reference("pkg.data.pykt")
    return module.get_tensor_dataset_from_pykt_dataset


def _padded_row(*, questions=None, concepts, responses, repeats=None, width=8):
    valid = len(responses)
    row = {"uid": "42", "fold": 0}
    for name, values in (("questions", questions), ("concepts", concepts), ("responses", responses), ("is_repeat", repeats)):
        if values is not None:
            row[name] = ",".join(map(str, values + [-1] * (width - valid)))
    row["selectmasks"] = ",".join(map(str, [1] * valid + [-1] * (width - valid)))
    return row


def _official(row, converter, max_concepts):
    return converter(pd.DataFrame([row]), "set_dense", max_concepts, unique_concept_mapping=None,
                     swap_q_and_c=False).tensors[0][0]


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seal(manifest):
    base = Path(manifest["data_dir"])
    manifest["files"] = {name: {"bytes": (base / name).stat().st_size,
                                "sha256": hashlib.sha256((base / name).read_bytes()).hexdigest()}
                         for name in ("train_valid.csv", "train_valid_sequences.csv")}
    return manifest


def _fixture(tmp_path, students, *, width=8, max_concepts=3, concept_only=False):
    originals, sequences = [], []
    for uid, fold, questions, concepts, responses, repeats in students:
        original = {"uid": uid, "fold": fold}
        columns = {"concepts": concepts, "responses": responses}
        if not concept_only:
            columns = {"questions": questions, **columns, "is_repeat": repeats}
        original.update({key: ",".join(map(str, value)) for key, value in columns.items()})
        originals.append(original)
        for start in range(0, len(responses), width):
            stop = min(start + width, len(responses))
            if stop - start < 3:
                continue
            row = {"uid": uid, "fold": fold}
            row.update({key: ",".join(map(str, value[start:stop] + [-1] * (width - (stop - start)))) for key, value in columns.items()})
            row["selectmasks"] = ",".join(map(str, [1] * (stop - start) + [-1] * (width - (stop - start))))
            sequences.append(row)
    _write_csv(tmp_path / "train_valid.csv", originals)
    _write_csv(tmp_path / "train_valid_sequences.csv", sequences)
    return _seal({"dataset": "synthetic", "data_dir": str(tmp_path.resolve()), "data_config": {
        "num_q": 0 if concept_only else 1000, "num_c": 128, "max_concepts": max_concepts,
        "maxlen": width, "min_seq_len": 3, "folds": sorted({row[1] for row in students}),
        "train_valid_file": "train_valid_sequences.csv", "train_valid_original_file": "train_valid.csv"}})


@pytest.mark.parametrize("continued", [False, True])
def test_set_dense_exactly_matches_official_with_duplicates_and_repeated_question(official_converter, continued):
    row = _padded_row(questions=[10, 10, 11, 11, 11, 10], concepts=[16, 1, 2, 2, 17, 3],
                      responses=[1, 1, 0, 0, 0, 1], repeats=[int(continued), 1, 0, 1, 1, 0])
    actual = set_dense_from_row(row, max_concepts=3, sequence_length=8)
    assert torch.equal(actual["data"], _official(row, official_converter, 3))
    assert actual["data"].shape == (8, 5)
    assert actual["question_ids"][:3].tolist() == [10, 11, 10]
    assert actual["kc_counts"][:3].tolist() == [2, 2, 1]
    assert actual["padding_mask"].tolist() == [True] * 3 + [False] * 5
    assert actual["score_mask"].tolist() == [False, True, True] + [False] * 5


def test_assist2015_dummy_question_exactly_matches_official(official_converter):
    row = _padded_row(concepts=[4, 4, 7], responses=[0, 1, 1])
    actual = set_dense_from_row(row, max_concepts=1, sequence_length=8, concept_only=True)
    assert torch.equal(actual["data"], _official(row, official_converter, 1))
    assert actual["question_ids"].tolist() == [0, 0, 0, -1, -1, -1, -1, -1]
    assert actual["padding_mask"].sum() == 3  # identical q=0 does not merge events


def test_true_200kc_partial_boundary_and_event_identity(tmp_path, official_converter):
    # A 3-KC event spans raw positions 199..201 across the original 200-KC cut.
    questions = list(range(199)) + [900] * 3 + [901, 902, 903]
    concepts = [i % 50 for i in range(199)] + [16, 1, 2, 3, 4, 5]
    responses = [0] * 199 + [1] * 3 + [0, 1, 0]
    repeats = [0] * 200 + [1, 1, 0, 0, 0]
    manifest = _fixture(tmp_path, [("42", 0, questions, concepts, responses, repeats)], width=200)
    dataset = KTSTSequenceDataset(manifest, 0, "valid")
    first, second = dataset[0], dataset[1]
    assert first["data"].shape == second["data"].shape == (200, 5)
    assert first["event_ids"][199] == second["event_ids"][0] == "synthetic:42:199"
    assert first["qidxs"][199] == second["qidxs"][0]
    assert first["score_mask"][199] and not second["score_mask"][0]
    assert first["event_details"][-1]["partial_end"]
    assert second["event_details"][0]["partial_start"]
    assert first["event_details"][-1]["omitted_unique_concepts"] == 2
    assert second["event_details"][0]["omitted_unique_concepts"] == 1
    assert dataset.audit["partial_start_chunks"] == dataset.audit["partial_end_chunks"] == 1
    assert dataset.audit["incomplete_event_fragments"] == 2
    assert dataset.audit["omitted_kc_positions_in_fragments"] == 3
    rows = pd.read_csv(dataset.path, dtype={"uid": str})
    for index, item in enumerate((first, second)):
        assert torch.equal(item["data"], _official(rows.iloc[index].to_dict(), official_converter, 3))
    assert dataset.audit["v1_common_target_history_equivalence"] == "not_established"
    json.dumps(dataset.audit)


def test_partial_event_in_dropped_two_kc_tail_is_reported(tmp_path):
    manifest = _fixture(tmp_path, [("42", 0, list(range(199)) + [900] * 3,
                                   [0] * 199 + [1, 2, 3], [0] * 199 + [1] * 3,
                                   [0] * 200 + [1, 1])], width=200)
    dataset = KTSTSequenceDataset(manifest, 0, "valid")
    assert len(dataset) == 1
    assert dataset.audit["unconsumed_kc_positions"] == 2
    assert dataset.audit["partial_end_chunks"] == 1
    assert dataset[0]["event_details"][-1]["omitted_kc_positions"] == 2


def test_lazy_cpu_and_fixed_batch_width(tmp_path, monkeypatch):
    students = [("42", 0, [1, 1, 2], [0, 1, 2], [1, 1, 0], [0, 1, 0]),
                ("17", 1, [3, 4, 5], [2, 3, 4], [0, 1, 0], [0, 0, 0])]
    manifest = _fixture(tmp_path, students)
    original_full = torch.full
    allocations = []
    def record_full(*args, **kwargs):
        allocations.append(args[0])
        return original_full(*args, **kwargs)
    monkeypatch.setattr(torch, "full", record_full)
    valid = KTSTSequenceDataset(manifest, 0, "valid")
    train = KTSTSequenceDataset(manifest, 0, "train")
    assert allocations == []  # indexing never creates full-dataset tensors
    assert valid[0]["uid"] == "42" and train[0]["uid"] == "17"
    first = valid[0]
    one = collate_ktst([first])
    two = collate_ktst([first, train[0]])
    assert one["data"].shape == (1, 8, 5) and two["data"].shape == (2, 8, 5)
    assert torch.equal(one["data"][0], two["data"][0])
    assert all(value.device.type == "cpu" for value in two.values() if isinstance(value, torch.Tensor))
    assert two["uid"] == ["42", "17"]
    assert len(two["event_ids"][0]) == 8
    valid.close()
    train.close()


def test_zero_score_fragment_is_reported_not_silently_removed(tmp_path):
    manifest = _fixture(tmp_path, [("42", 0, [2] * 3, [1, 2, 3], [1] * 3, [0, 1, 1])])
    dataset = KTSTSequenceDataset(manifest, 0, "valid")
    assert dataset.audit["zero_score_chunks"] == 1
    assert not dataset[0]["score_mask"].any()
    assert dataset[0]["padding_mask"].sum() == 1


def test_masks_zero_are_explicitly_unsupported():
    row = _padded_row(concepts=[1, 2, 3], responses=[0, 1, 0])
    row["selectmasks"] = "1,0,1,-1,-1,-1,-1,-1"
    with pytest.raises(ValueError, match="selectmasks"):
        set_dense_from_row(row, max_concepts=1, sequence_length=8, concept_only=True)


def test_test_split_is_rejected_before_any_source_access():
    with pytest.raises(ValueError, match="test remains sealed"):
        KTSTSequenceDataset({}, 0, "test")


def test_source_tampering_and_slice_misalignment_are_rejected(tmp_path):
    manifest = _fixture(tmp_path, [("42", 0, [1, 2, 3], [1, 2, 3], [0, 1, 0], [0, 0, 0])])
    source = tmp_path / "train_valid_sequences.csv"
    source.write_text(source.read_text().replace('"0,1,0,-1', '"1,1,0,-1'))
    with pytest.raises(ValueError, match="checksum"):
        KTSTSequenceDataset(manifest, 0, "valid")
    _seal(manifest)
    with pytest.raises(ValueError, match="original student slice"):
        KTSTSequenceDataset(manifest, 0, "valid")


def test_omitted_student_is_rejected_even_when_file_is_resealed(tmp_path):
    manifest = _fixture(tmp_path, [(uid, 0, [1, 2, 3], [1, 2, 3], [0, 1, 0], [0, 0, 0])
                                   for uid in ("42", "17")])
    source = tmp_path / "train_valid_sequences.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    _write_csv(source, rows[:1])
    _seal(manifest)
    with pytest.raises(ValueError, match="omitted a development student"):
        KTSTSequenceDataset(manifest, 0, "valid")


@pytest.mark.parametrize("name", ["assist2009", "assist2015", "nips_task34"])
def test_existing_development_rows_match_official_read_only(name, official_converter):
    base = Path(os.environ.get("KTST_PREPARED_ROOT", str(ROOT.parent / "knowledge-tracing/data/prepared")))
    manifest_path = base / name / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("Optional existing prepared development files are not present")
    manifest = json.loads(manifest_path.read_text())
    dataset = KTSTSequenceDataset(manifest, 0, "valid", max_rows=3)
    selected = pd.read_csv(dataset.path, dtype={"uid": str})
    selected = selected[selected["fold"] == 0].head(3)
    for index, (_, row) in enumerate(selected.iterrows()):
        actual = dataset[index]
        expected = _official(row.to_dict(), official_converter, dataset.max_concepts)
        assert torch.equal(actual["data"], expected)
        assert actual["uid"] == str(row["uid"])
        assert actual["data"].device.type == "cpu"
        assert actual["protocol"] == PROTOCOL
    dataset.close()
