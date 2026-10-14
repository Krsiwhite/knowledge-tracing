"""Model-independent, complete-event development contexts under a 200-KC budget.

This is a new protocol, not either model's original preprocessing. Only the
sealed ``train_valid.csv`` development source is accepted. Targets are stored as
compact row/ordinal indexes; histories are reconstructed on demand, never
expanded into a dataset-sized CSV or tensor.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import OrderedDict
import csv
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Iterator


PROTOCOL = "ktlab-v2-shared-event-context-kc200-dev"
HASH_SCHEME = "ktlab-shared-event-compositional-sha256-v1"
_TARGET_DOMAIN = b"ktlab.shared.target.v1\0"
_HISTORY_DOMAIN = b"ktlab.shared.history-event.v1\0"
_CONTEXT_DOMAIN = b"ktlab.shared.context.v1\0"
_CACHE_STUDENTS = 2


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _target_bytes(context: dict) -> bytes:
    identity_and_query = {key: context[key] for key in
                          ("protocol", "uid", "fold", "event_id", "target_index", "query")}
    return hashlib.sha256(_TARGET_DOMAIN + _json_bytes(identity_and_query)).digest()


def _history_bytes(event: dict) -> bytes:
    allowed_event = {key: event[key] for key in ("event_id", "question", "concepts", "response")}
    return hashlib.sha256(_HISTORY_DOMAIN + _json_bytes(allowed_event)).digest()


def context_digests(context: dict) -> dict[str, str]:
    """Recompute one target's digests without reading its independent label.

    The target digest covers identity and query. The context digest additionally
    covers the ordered permitted history, including *historical* answers. Fixed
    32-byte SHA256 records and domain separators make composition unambiguous.
    """
    target = _target_bytes(context)
    history = b"".join(_history_bytes(event) for event in context["history"])
    return {"target_sha256": target.hex(),
            "context_sha256": hashlib.sha256(_CONTEXT_DOMAIN + target + history).hexdigest()}


def _csv_rows(path: Path) -> Iterator[tuple[int, int, dict[str, str]]]:
    # Prepared pyKT numeric-list records occupy exactly one physical line.
    with path.open("rb") as stream:
        header = stream.readline()
        if not header:
            raise ValueError("Empty prepared development CSV")
        fields = next(csv.reader([header.decode("utf-8-sig")]))
        if len(fields) != len(set(fields)) or not {"uid", "fold", "concepts", "responses"}.issubset(fields):
            raise ValueError("Invalid prepared development CSV schema")
        for row_index in range(1 << 63):
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            values = next(csv.reader([line.decode("utf-8")], strict=True))
            if len(values) != len(fields):
                raise ValueError("Development CSV must contain one numeric-list record per physical line")
            yield row_index, offset, dict(zip(fields, values))


def _row_at(path: Path, offset: int) -> dict[str, str]:
    with path.open("rb") as stream:
        fields = next(csv.reader([stream.readline().decode("utf-8-sig")]))
        stream.seek(offset)
        values = next(csv.reader([stream.readline().decode("utf-8")], strict=True))
    if len(fields) != len(values):
        raise ValueError("Development source no longer matches its indexed row")
    return dict(zip(fields, values))


def _ints(row: dict, key: str) -> list[int]:
    try:
        return [int(value) for value in row[key].split(",")]
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError(f"Missing or malformed integer field: {key}") from error


@dataclass(frozen=True, slots=True)
class _Event:
    question: int | None
    concepts: tuple[int, ...]
    response: int


@dataclass(frozen=True, slots=True)
class _Student:
    uid: str
    fold: int
    offset: int
    row_index: int
    row_sha256: str


def _events_from_row(row: dict, num_concepts: int, num_questions: int) -> tuple[tuple[_Event, ...], int]:
    concepts, responses = _ints(row, "concepts"), _ints(row, "responses")
    if not concepts or len(concepts) != len(responses):
        raise ValueError("Unaligned or empty development event fields")
    if any(c < 0 or c >= num_concepts for c in concepts) or any(r not in (0, 1) for r in responses):
        raise ValueError("Original development events require in-vocabulary KCs and binary responses without padding")
    if num_questions:
        questions = _ints(row, "questions")
        repeats = _ints(row, "is_repeat")
        if len(questions) != len(concepts) or any(q < 0 or q >= num_questions for q in questions):
            raise ValueError("Original question IDs are unaligned or outside the manifest vocabulary")
    else:
        # AS15 contains concept observations, not recoverable question IDs.
        questions = [None] * len(concepts)
        repeats = _ints(row, "is_repeat") if "is_repeat" in row else [0] * len(concepts)
        if any(repeats):
            raise ValueError("Concept-only observations must remain separate events; is_repeat=1 is unsupported")
    if len(repeats) != len(concepts) or repeats[0] != 0 or any(r not in (0, 1) for r in repeats):
        raise ValueError("Original is_repeat must be aligned, binary, and begin a new event")
    if "selectmasks" in row and _ints(row, "selectmasks") != [1] * len(concepts):
        raise ValueError("Original shared-event source cannot contain masked/window positions")
    starts = [i for i, repeat in enumerate(repeats) if repeat == 0]
    events = []
    duplicates = 0
    for start, stop in zip(starts, starts[1:] + [len(concepts)]):
        if any(q != questions[start] for q in questions[start:stop]) or any(r != responses[start] for r in responses[start:stop]):
            raise ValueError("An original is_repeat event contains inconsistent questions or responses")
        ordered_concepts = tuple(dict.fromkeys(concepts[start:stop]))
        duplicates += stop - start - len(ordered_concepts)
        events.append(_Event(questions[start], ordered_concepts, responses[start]))
    return tuple(events), duplicates


def _query(event: _Event) -> dict:
    return {"question": event.question, "concepts": list(event.concepts)}


def _history_event(dataset: str, uid: str, ordinal: int, event: _Event) -> dict:
    return {"event_id": f"{dataset}:{uid}:{ordinal}", **_query(event), "response": event.response}


class SharedEventDataset:
    """Compact canonical target index with a two-student CPU parse cache.

    ``max_targets`` selects a deterministic prefix, but eligible/exclusion counts
    still describe the entire selected split. Aggregate digests describe only
    the returned targets, explicitly recorded as ``hash_scope`` in the audit.
    No model library, tensor, accelerator, optimizer, or test source is used.
    """
    protocol = PROTOCOL

    def __init__(self, manifest: dict, fold: int, split: str = "valid", kc_budget: int = 200,
                 max_targets: int | None = None):
        if split not in {"train", "valid"}:
            raise ValueError("Shared-event development protocol supports train/valid only; test remains sealed")
        if not isinstance(kc_budget, Integral) or isinstance(kc_budget, bool) or kc_budget != 200:
            raise ValueError("The kc200 protocol requires kc_budget=200")
        if max_targets is not None and (not isinstance(max_targets, Integral) or isinstance(max_targets, bool) or max_targets <= 0):
            raise ValueError("max_targets must be a positive integer or None")
        if not isinstance(fold, Integral) or isinstance(fold, bool):
            raise ValueError("fold must be an integer development fold, not a coerced value")
        config = manifest["data_config"]
        if any(not isinstance(part, Integral) or isinstance(part, bool) for part in config["folds"]):
            raise ValueError("Manifest development folds must be integers")
        folds = {int(part) for part in config["folds"]}
        if not folds or not folds.issubset(set(range(5))) or int(fold) not in folds:
            raise ValueError("Shared-event folds must be development folds 0 through 4")
        if config.get("train_valid_original_file") != "train_valid.csv":
            raise ValueError("Shared-event input must be the canonical train_valid.csv original development source")
        base = Path(manifest.get("data_dir", ""))
        if not base.is_absolute() or not base.is_dir():
            raise ValueError("manifest.data_dir must be an existing absolute prepared-data directory")
        self.path = (base / "train_valid.csv").resolve()
        if self.path.parent != base.resolve() or not self.path.is_file():
            raise ValueError("Original development source is missing or outside data_dir")
        receipt = manifest.get("files", {}).get("train_valid.csv")
        if not isinstance(receipt, dict) or not receipt.get("sha256") or "bytes" not in receipt:
            raise ValueError("Original development source has no pinned checksum")
        source_sha = _digest(self.path)
        if self.path.stat().st_size != receipt["bytes"] or source_sha != receipt["sha256"]:
            raise ValueError("Original development source checksum mismatch")
        self.name = str(manifest["dataset"])
        self.fold, self.split, self.kc_budget = int(fold), split, int(kc_budget)
        max_targets = int(max_targets) if max_targets is not None else None
        self.num_concepts, self.num_questions = int(config["num_c"]), int(config["num_q"])
        if self.num_concepts <= 0 or self.num_questions < 0:
            raise ValueError("Invalid prepared vocabulary sizes")
        self._students: list[_Student] = []
        self._row_ids, self._target_ordinals, self._history_starts = array("Q"), array("I"), array("I")
        self._cache: OrderedDict[int, tuple[_Event, ...]] = OrderedDict()
        seen_uids: set[str] = set()
        source_rows = selected_students = event_total = duplicates = eligible_total = 0
        exclusions = {"target_over_budget": 0, "no_history": 0}
        no_history_reasons = {"first_event": 0, "immediate_predecessor_exceeds_remaining_budget": 0}
        length_stats = {"scope": "selected_targets", "history_kc_min": None, "history_kc_max": None,
                        "history_events_min": None, "history_events_max": None,
                        "history_kc_bins": {"1-10": 0, "11-50": 0, "51-100": 0, "101-150": 0, "151-199": 0},
                        "query_kc_counts": {}}
        target_hasher = hashlib.sha256(b"ktlab.shared.target-sequence.v1\0")
        context_hasher = hashlib.sha256(b"ktlab.shared.context-sequence.v1\0")
        for row_index, offset, row in _csv_rows(self.path):
            uid, part = row["uid"], int(row["fold"])
            if uid in seen_uids or part not in folds:
                raise ValueError("Original development membership contains duplicate users or a non-development fold")
            seen_uids.add(uid)
            source_rows += 1
            if not (part == self.fold if split == "valid" else part != self.fold):
                continue
            selected_students += 1
            events, row_duplicates = _events_from_row(row, self.num_concepts, self.num_questions)
            event_total += len(events)
            duplicates += row_duplicates
            prefix = [0]
            for event in events:
                prefix.append(prefix[-1] + len(event.concepts))
            row_id = None
            # Hash history once per student. A contiguous fixed-width slice
            # supplies each target context without serializing histories again.
            history_hashes = None
            for ordinal, event in enumerate(events):
                cost = len(event.concepts)
                if cost > kc_budget:
                    exclusions["target_over_budget"] += 1
                    continue
                start = bisect_left(prefix, prefix[ordinal] - (kc_budget - cost), 0, ordinal + 1)
                if start == ordinal:
                    exclusions["no_history"] += 1
                    no_history_reasons["first_event" if ordinal == 0 else "immediate_predecessor_exceeds_remaining_budget"] += 1
                    continue
                eligible_total += 1
                if max_targets is not None and len(self._row_ids) >= max_targets:
                    continue
                history_kcs, history_events = prefix[ordinal] - prefix[start], ordinal - start
                for stem, value in (("history_kc", history_kcs), ("history_events", history_events)):
                    length_stats[stem + "_min"] = min(value, length_stats[stem + "_min"] or value)
                    length_stats[stem + "_max"] = max(value, length_stats[stem + "_max"] or value)
                bucket = next(name for upper, name in ((10, "1-10"), (50, "11-50"), (100, "51-100"),
                                                       (150, "101-150"), (199, "151-199")) if history_kcs <= upper)
                length_stats["history_kc_bins"][bucket] += 1
                length_stats["query_kc_counts"][str(cost)] = length_stats["query_kc_counts"].get(str(cost), 0) + 1
                if row_id is None:
                    row_id = len(self._students)
                    self._students.append(_Student(uid, part, offset, row_index,
                                                   hashlib.sha256(_json_bytes(row)).hexdigest()))
                if history_hashes is None:
                    history_hashes = b"".join(_history_bytes(_history_event(self.name, uid, i, e))
                                              for i, e in enumerate(events))
                identity = {"protocol": PROTOCOL, "uid": uid, "fold": part,
                            "event_id": f"{self.name}:{uid}:{ordinal}", "target_index": ordinal,
                            "query": _query(event)}
                target = _target_bytes(identity)
                context = hashlib.sha256(_CONTEXT_DOMAIN + target + history_hashes[start * 32:ordinal * 32]).digest()
                target_hasher.update(target)
                context_hasher.update(context)
                self._row_ids.append(row_id)
                self._target_ordinals.append(ordinal)
                self._history_starts.append(start)
        self.audit = {
            "protocol": PROTOCOL, "dataset": self.name, "split": split, "fold": self.fold,
            "source_path": str(self.path), "source_sha256": source_sha, "source_bytes": receipt["bytes"],
            "kc_budget": self.kc_budget, "source_students": source_rows, "split_students": selected_students,
            "indexed_students": len(self._students), "canonical_events": event_total,
            "duplicate_kc_positions_removed": duplicates, "eligible_total": eligible_total,
            "expected_total": eligible_total, "selected_targets": len(self),
            "max_targets": max_targets, "selected_for_debug": max_targets is not None,
            "selection_limited": len(self) < eligible_total, "exclusion_counts": exclusions,
            "no_history_reasons": no_history_reasons,
            "length_statistics": length_stats,
            "target_sha256": target_hasher.hexdigest(), "context_sha256": context_hasher.hexdigest(),
            "hash_scheme": HASH_SCHEME, "hash_scope": "selected_targets_in_original_csv_and_event_order",
            "hash_labels": "target labels excluded; permitted historical responses included in context only",
            "event_id_scheme": "dataset:uid:original_student_event_ordinal",
            "question_missing_policy": "None; any model-specific dummy belongs to its adapter",
            "kc_order": "original occurrence order with within-event duplicates removed",
            "history_policy": "longest complete contiguous event suffix; history plus query KC count <= 200",
            "index_bytes": sum(value.itemsize * len(value) for value in
                               (self._row_ids, self._target_ordinals, self._history_starts)),
            "student_cache_limit": _CACHE_STUDENTS, "materialized_contexts": False,
            "test_access": False, "optimizer_steps": 0,
        }
        if event_total != eligible_total + sum(exclusions.values()):
            raise AssertionError("Canonical event accounting does not balance")
        json.dumps(self.audit, allow_nan=False)

    def __len__(self) -> int:
        return len(self._row_ids)

    def _student_events(self, row_id: int) -> tuple[_Event, ...]:
        if row_id in self._cache:
            self._cache.move_to_end(row_id)
            return self._cache[row_id]
        student = self._students[row_id]
        row = _row_at(self.path, student.offset)
        if row["uid"] != student.uid or int(row["fold"]) != student.fold:
            raise ValueError("Original development row no longer matches its indexed identity")
        if hashlib.sha256(_json_bytes(row)).hexdigest() != student.row_sha256:
            raise ValueError("Original development row changed after its target/context audit")
        events, _ = _events_from_row(row, self.num_concepts, self.num_questions)
        self._cache[row_id] = events
        if len(self._cache) > _CACHE_STUDENTS:
            self._cache.popitem(last=False)
        return events

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_id = self._row_ids[index]
        ordinal, start = self._target_ordinals[index], self._history_starts[index]
        student = self._students[row_id]
        events = self._student_events(row_id)
        event = events[ordinal]
        history = [_history_event(self.name, student.uid, i, events[i]) for i in range(start, ordinal)]
        history_kcs = sum(len(item["concepts"]) for item in history)
        context = {
            "protocol": PROTOCOL, "uid": student.uid, "fold": student.fold,
            "event_id": f"{self.name}:{student.uid}:{ordinal}", "target_index": ordinal,
            "history": history, "query": _query(event), "label": event.response,
            "history_start_index": start, "history_kc_count": history_kcs,
            "target_kc_count": len(event.concepts), "context_kc_count": history_kcs + len(event.concepts),
            "source_row_index": student.row_index, "source_byte_offset": student.offset,
        }
        context.update(context_digests(context))
        return context

    def close(self) -> None:
        """Release the bounded parse cache; there are no persistent file handles."""
        self._cache.clear()

    def verify_source(self) -> None:
        """Verify the pinned original source at evaluation boundaries."""
        if self.path.stat().st_size != self.audit["source_bytes"] or _digest(self.path) != self.audit["source_sha256"]:
            raise ValueError("Original development source changed after its target/context audit")
