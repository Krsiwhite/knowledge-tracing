"""Lazy, auditable set-dense inputs for the independent KTST development track.

Only the already prepared train_valid files are accepted. This module does not
construct a test loader or assert comparability with ktlab v1 target/history
sets. Chunk boundaries follow pyKT KC positions, not a new question split.
"""
from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

PROTOCOL = "ktlab-ktst-reference-dev-v1"


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _csv_rows(path: Path) -> Iterator[tuple[int, int, dict[str, str]]]:
    # Frozen pyKT output has one physical line per record (only numeric lists).
    with path.open("rb") as stream:
        fields = next(csv.reader([stream.readline().decode("utf-8-sig")]))
        index = 0
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            values = next(csv.reader([line.decode("utf-8")], strict=True))
            if len(values) != len(fields):
                raise ValueError(f"Expected single-line pyKT CSV record in {path.name}:{index + 2}")
            yield index, offset, dict(zip(fields, values))
            index += 1


def _row_at(path: Path, offset: int) -> dict[str, str]:
    # Per-call handles make DataLoader workers safe; no live handle is pickled.
    with path.open("rb") as stream:
        fields = next(csv.reader([stream.readline().decode("utf-8-sig")]))
        stream.seek(offset)
        values = next(csv.reader([stream.readline().decode("utf-8")], strict=True))
    if len(fields) != len(values):
        raise ValueError("Source CSV row no longer matches the indexed schema")
    return dict(zip(fields, values))


def _ints(row: dict, key: str) -> list[int]:
    try:
        return [int(value) for value in row[key].split(",")]
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError(f"Missing or non-integer pyKT field: {key}") from error


def _parse(row: dict, *, concept_only: bool, length: int | None = None) -> dict:
    responses, concepts = _ints(row, "responses"), _ints(row, "concepts")
    questions = _ints(row, "questions") if "questions" in row else [0 if r != -1 else -1 for r in responses]
    repeats = _ints(row, "is_repeat") if "is_repeat" in row else [0 if r != -1 else -1 for r in responses]
    if not concept_only and ("questions" not in row or "is_repeat" not in row):
        raise ValueError("Question datasets require questions and is_repeat columns")
    if not (len(responses) == len(concepts) == len(questions) == len(repeats)):
        raise ValueError("Unaligned pyKT fields")
    if length is not None and len(responses) != length:
        raise ValueError(f"Expected the original {length}-KC chunk width")
    valid = sum(r != -1 for r in responses)
    if not valid or any(r not in (0, 1) for r in responses[:valid]) or any(r != -1 for r in responses[valid:]):
        raise ValueError("Responses must be binary with trailing -1 padding")
    for key, values in (("concepts", concepts), ("questions", questions), ("is_repeat", repeats)):
        if any(x == -1 for x in values[:valid]) or any(x != -1 for x in values[valid:]):
            raise ValueError(f"{key} padding does not align with responses")
    if any(x not in (0, 1) for x in repeats[:valid]):
        raise ValueError("is_repeat must be 0/1 on real KC positions")
    if concept_only and any(q != 0 for q in questions[:valid]):
        raise ValueError("Concept-only KTST must use the constant dummy question 0")
    if "selectmasks" in row:
        masks = _ints(row, "selectmasks")
        if len(masks) != len(responses) or any(x != 1 for x in masks[:valid]) or any(x != -1 for x in masks[valid:]):
            raise ValueError("Reference-dev conversion supports all-real selectmasks=1 only; masked/window rows are unsupported")
    return {"questions": questions[:valid], "concepts": concepts[:valid], "responses": responses[:valid], "is_repeat": repeats[:valid], "length": valid}


def _groups(parsed: dict) -> list[tuple[int, int]]:
    starts = [0] + [i for i in range(1, parsed["length"]) if parsed["is_repeat"][i] == 0]
    groups = list(zip(starts, starts[1:] + [parsed["length"]]))
    for start, stop in groups:
        if len(set(parsed["questions"][start:stop])) != 1 or len(set(parsed["responses"][start:stop])) != 1:
            raise ValueError("An is_repeat group contains inconsistent question IDs or answers")
    return groups


def set_dense_from_row(row: dict, *, max_concepts: int, sequence_length: int = 200,
                       concept_only: bool = False) -> dict[str, torch.Tensor]:
    """Convert one KC chunk, matching the fixed official set_dense function.

    In particular, list(set(...)) preserves official integer-set slot order;
    sorting instead would not promise elementwise equality with that function.
    """
    parsed = _parse(row, concept_only=concept_only, length=sequence_length)
    groups = _groups(parsed)  # row start becomes a new fragment even if repeat=1
    data = torch.full((sequence_length, max_concepts + 2), -1, dtype=torch.long)
    counts = torch.zeros(sequence_length, dtype=torch.long)
    for event, (start, stop) in enumerate(groups):
        concepts = list(set(parsed["concepts"][start:stop]))
        if len(concepts) > max_concepts or any(x < 0 for x in concepts):
            raise ValueError("Question concept set exceeds manifest max_concepts or has invalid IDs")
        data[event, :len(concepts)] = torch.tensor(concepts, dtype=torch.long)
        data[event, -2] = parsed["questions"][start]
        data[event, -1] = parsed["responses"][start]
        counts[event] = len(concepts)
    padding_mask = data[:, -1] != -1
    score_mask = padding_mask.clone()
    score_mask[0] = False
    return {"data": data, "padding_mask": padding_mask, "labels": data[:, -1].to(torch.float32),
            "score_mask": score_mask, "question_ids": data[:, -2].clone(), "kc_counts": counts}


@dataclass(frozen=True, slots=True)
class _Original:
    offset: int
    fold: int
    length: int
    events: int
    qidx_base: int


@dataclass(frozen=True, slots=True)
class _Chunk:
    offset: int
    row_index: int
    uid: str
    fold: int
    kc_start: int
    kc_stop: int


class KTSTSequenceDataset(Dataset):
    """Index CSV byte offsets and construct only the requested CPU row tensors."""
    protocol = PROTOCOL

    def __init__(self, manifest: dict, fold: int, split: str = "train", max_rows: int | None = None):
        if split not in {"train", "valid"}:
            raise ValueError("KTST reference-dev supports train/valid only; test remains sealed")
        config = manifest["data_config"]
        folds = {int(x) for x in config["folds"]}
        if int(fold) not in folds:
            raise ValueError("Validation fold is absent from the prepared manifest")
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be positive or None")
        base = Path(manifest.get("data_dir", ""))
        if not base.is_absolute() or not base.is_dir():
            raise ValueError("manifest.data_dir must be an existing absolute prepared-data path; no fallback is inferred")
        if config["train_valid_file"] != "train_valid_sequences.csv" or config["train_valid_original_file"] != "train_valid.csv":
            raise ValueError("Reference-dev requires canonical train_valid files, not causal validation/test files")
        self.path = self._source(manifest, base, config["train_valid_file"])
        self.original_path = self._source(manifest, base, config["train_valid_original_file"])
        self.name = str(manifest["dataset"])
        self.max_concepts = int(config["max_concepts"])
        self.maxlen = int(config["maxlen"])
        self.num_concepts = int(config["num_c"])
        self.concept_only = int(config["num_q"]) == 0
        self.num_questions = max(1, int(config["num_q"]))
        if min(self.max_concepts, self.maxlen, self.num_concepts) < 1:
            raise ValueError("Invalid sequence/concept dimensions")
        self.fold, self.split = int(fold), split
        self._originals: dict[str, _Original] = {}
        self._chunks: list[_Chunk] = []
        qidx_base = 0
        for _, offset, row in _csv_rows(self.original_path):
            uid, part = row["uid"], int(row["fold"])
            if uid in self._originals or part not in folds:
                raise ValueError("Original development rows have duplicate users or an invalid/test fold")
            parsed = _parse(row, concept_only=self.concept_only)
            if parsed["is_repeat"][0] != 0:
                raise ValueError("The full student sequence must begin with a new event")
            events = len(_groups(parsed))
            self._originals[uid] = _Original(offset, part, parsed["length"], events, qidx_base)
            qidx_base += events
        counters = {"rows": 0, "kc_positions": 0, "event_fragments": 0, "scored_event_fragments": 0,
                    "partial_start_chunks": 0, "partial_end_chunks": 0, "incomplete_event_fragments": 0,
                    "omitted_kc_positions_in_fragments": 0, "omitted_unique_concepts_in_fragments": 0,
                    "deduplicated_kc_positions": 0, "zero_score_chunks": 0}
        cursors: dict[str, int] = {}
        selected_students: set[str] = set()
        cached_uid, cached_original = None, None
        for row_index, offset, row in _csv_rows(self.path):
            uid, part = row["uid"], int(row["fold"])
            if uid not in self._originals or part != self._originals[uid].fold:
                raise ValueError("Sequence uid/fold does not match original development membership")
            wanted = part == self.fold if split == "valid" else part != self.fold
            if not wanted:
                continue
            parsed = _parse(row, concept_only=self.concept_only, length=self.maxlen)
            if uid != cached_uid:
                cached_original = _parse(_row_at(self.original_path, self._originals[uid].offset), concept_only=self.concept_only)
                cached_uid = uid
            start = cursors.get(uid, 0)
            stop = start + parsed["length"]
            for key in ("questions", "concepts", "responses", "is_repeat"):
                if parsed[key] != cached_original[key][start:stop]:
                    raise ValueError(f"KC chunk is not the corresponding original student slice: uid={uid}, row={row_index}, field={key}")
            if start % self.maxlen != 0 or parsed["length"] < int(config.get("min_seq_len", 3)):
                raise ValueError("Source rows do not preserve the pinned pyKT KC chunk boundaries/minimum length")
            if any(c < 0 or c >= self.num_concepts for c in parsed["concepts"]) or any(q < 0 or q >= self.num_questions for q in parsed["questions"]):
                raise ValueError("Source ID outside manifest vocabulary")
            chunk = _Chunk(offset, row_index, uid, part, start, stop)
            details = self._event_details(chunk, parsed, cached_original)
            groups = _groups(parsed)
            counts = [len(set(parsed["concepts"][a:b])) for a, b in groups]
            if max(counts) > self.max_concepts:
                raise ValueError("Source concept set exceeds manifest max_concepts")
            self._chunks.append(chunk)
            cursors[uid] = stop
            selected_students.add(uid)
            counters["rows"] += 1
            counters["kc_positions"] += parsed["length"]
            counters["event_fragments"] += len(groups)
            counters["scored_event_fragments"] += max(0, len(groups) - 1)
            counters["zero_score_chunks"] += len(groups) == 1
            counters["partial_start_chunks"] += details[0]["partial_start"]
            counters["partial_end_chunks"] += details[-1]["partial_end"]
            counters["incomplete_event_fragments"] += sum(not d["complete"] for d in details)
            counters["omitted_kc_positions_in_fragments"] += sum(d["omitted_kc_positions"] for d in details)
            counters["omitted_unique_concepts_in_fragments"] += sum(d["omitted_unique_concepts"] for d in details)
            counters["deduplicated_kc_positions"] += parsed["length"] - sum(counts)
            if max_rows is not None and len(self._chunks) >= max_rows:
                break
        if not self._chunks:
            raise ValueError("No train/valid rows selected")
        tails = [self._originals[uid].length - stop for uid, stop in cursors.items()]
        limited = max_rows is not None and len(self._chunks) >= max_rows
        expected_students = {uid for uid, original in self._originals.items()
                             if (original.fold == self.fold if split == "valid" else original.fold != self.fold)
                             and original.length >= int(config.get("min_seq_len", 3))}
        if not limited and selected_students != expected_students:
            raise ValueError("Source omitted a development student that pyKT should have retained")
        if not limited and any(n >= int(config.get("min_seq_len", 3)) for n in tails):
            raise ValueError("Source omitted a student tail that pyKT should have retained")
        self.audit = {"protocol": PROTOCOL, "split": split, "validation_fold": self.fold, "dataset": self.name,
                      "source_path": str(self.path), "original_path": str(self.original_path),
                      "source_sha256": _digest(self.path), "original_sha256": _digest(self.original_path),
                      "kc_budget": self.maxlen, "max_concepts": self.max_concepts, "students": len(selected_students),
                      "students_expected": len(expected_students),
                      "rows_limited": limited, "max_rows": max_rows, "unconsumed_kc_positions": sum(tails),
                      "alignment_scope": "selected_rows", "id_scheme": "dataset:uid:student_event_ordinal",
                      "qidx_scheme": "derived global event index in full train_valid.csv row order; not v1 validation qidx",
                      "concept_order": "official list(set(integer_concepts))", "python": sys.version.split()[0],
                      "test_access": False, "v1_common_target_history_equivalence": "not_established",
                      "all_dataset_tensors_materialized": False, **counters}
        # Fail early if an audit field accidentally ceases to be exportable.
        json.dumps(self.audit, allow_nan=False)

    @staticmethod
    def _source(manifest: dict, base: Path, filename: str) -> Path:
        path = (base / filename).resolve()
        if path.parent != base.resolve() or not path.is_file():
            raise ValueError(f"Prepared development source is missing or outside data_dir: {filename}")
        expected = manifest.get("files", {}).get(filename)
        if not isinstance(expected, dict) or not expected.get("sha256"):
            raise ValueError(f"Prepared source has no pinned checksum: {filename}")
        if path.stat().st_size != expected["bytes"] or _digest(path) != expected["sha256"]:
            raise ValueError(f"Prepared source checksum mismatch: {filename}")
        return path

    def _event_details(self, chunk: _Chunk, parsed: dict, original: dict) -> list[dict]:
        starts = [i for i, repeat in enumerate(original["is_repeat"]) if repeat == 0]
        result = []
        for left, right in _groups(parsed):
            absolute_start, absolute_stop = chunk.kc_start + left, chunk.kc_start + right
            ordinal = bisect_right(starts, absolute_start) - 1
            full_start = starts[ordinal]
            full_stop = starts[ordinal + 1] if ordinal + 1 < len(starts) else original["length"]
            visible = set(parsed["concepts"][left:right])
            full = set(original["concepts"][full_start:full_stop])
            result.append({"event_id": f"{self.name}:{chunk.uid}:{ordinal}", "event_index": ordinal,
                           "qidx": self._originals[chunk.uid].qidx_base + ordinal,
                           "kc_start": absolute_start, "kc_stop": absolute_stop,
                           "original_kc_start": full_start, "original_kc_stop": full_stop,
                           "visible_concepts": list(set(parsed["concepts"][left:right])),
                           "complete": absolute_start == full_start and absolute_stop == full_stop,
                           "partial_start": absolute_start != full_start, "partial_end": absolute_stop != full_stop,
                           "omitted_kc_positions": (full_stop - full_start) - (absolute_stop - absolute_start),
                           "omitted_unique_concepts": len(full - visible)})
        return result

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        chunk = self._chunks[index]
        row = _row_at(self.path, chunk.offset)
        original = _parse(_row_at(self.original_path, self._originals[chunk.uid].offset), concept_only=self.concept_only)
        parsed = _parse(row, concept_only=self.concept_only, length=self.maxlen)
        result = set_dense_from_row(row, max_concepts=self.max_concepts, sequence_length=self.maxlen, concept_only=self.concept_only)
        details = self._event_details(chunk, parsed, original)
        qidxs = torch.full((self.maxlen,), -1, dtype=torch.long)
        qidxs[:len(details)] = torch.tensor([d["qidx"] for d in details], dtype=torch.long)
        result.update(uid=chunk.uid, fold=chunk.fold, row_index=chunk.row_index, qidxs=qidxs,
                      source_byte_offset=chunk.offset, original_byte_offset=self._originals[chunk.uid].offset,
                      event_ids=[d["event_id"] for d in details] + [None] * (self.maxlen - len(details)),
                      event_details=details, kc_start=chunk.kc_start, kc_stop=chunk.kc_stop,
                      protocol=PROTOCOL)
        return result

    def close(self) -> None:
        """No persistent file handles or device tensors are retained."""


def collate_ktst(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty KTST batch")
    expected_shape = items[0]["data"].shape
    if any(item["data"].shape != expected_shape for item in items):
        raise ValueError("KTST batches must use one fixed KC budget and max_concepts")
    return {key: torch.stack([item[key] for item in items]) if isinstance(value, torch.Tensor)
            else [item[key] for item in items] for key, value in items[0].items()}
