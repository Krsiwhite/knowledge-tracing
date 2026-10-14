"""Inference-only projections of identical allowed question-event contexts.

These inputs are a new development evaluation protocol, not pyKT v1 or the
KTST reference train/validation representation. Current response is always 0.
MoC-KT ground_seq is the *visible* history KC count plus one query KC, never
the student's eventual sequence length. Original model mathematics is kept.
No target label, optimizer, training loss, or autograd output is exposed here.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import torch

from .ktst_model import forward_ktst


PROTOCOL = "ktlab-v2-shared-event-context-kc200-dev"
SEQUENCE_LENGTH = 200
QUERY_RESPONSE = 0
SUPPORTED_MODELS = {"akt", "simplekt", "mockt", "ktst_mean", "ktst_mhsa"}


def _concepts(value) -> list[int]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(c) is not int or c < 0 for c in value)):
        raise ValueError("Shared concepts must be a nonempty list of nonnegative integer IDs")
    if len(set(value)) != len(value):
        raise ValueError("Canonical shared concept sets must already be deduplicated")
    return list(value)


def _question(value) -> int:
    if value is None:
        return 0  # Author's dummy question for concept-only ASSIST2015.
    if type(value) is not int or value < 0:
        raise ValueError("Shared question ID must be nonnegative or None")
    return value


def _parts(context: dict) -> tuple[list[dict], dict, int]:
    """Read only the allowed history/query; never inspect context['label']."""
    if context.get("protocol") != PROTOCOL:
        raise ValueError("Shared adapter requires the declared development context protocol")
    event_id = context["event_id"]
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("Shared target event_id must be a nonempty string")
    if not isinstance(context["history"], (list, tuple)) or not context["history"]:
        raise ValueError("Shared target requires nonempty allowed history")
    history, seen = [], set()
    for event in context["history"]:
        identity = event["event_id"]
        if not isinstance(identity, str) or not identity or identity == event_id or identity in seen:
            raise ValueError("Shared history event IDs must be distinct and exclude the target")
        seen.add(identity)
        response = event["response"]
        if type(response) is not int or response not in {0, 1}:
            raise ValueError("Shared history responses must be binary integers")
        history.append({"event_id": identity, "question": _question(event["question"]),
                        "concepts": _concepts(event["concepts"]), "response": response})
    query = {"question": _question(context["query"]["question"]),
             "concepts": _concepts(context["query"]["concepts"])}
    history_kc_count = sum(len(event["concepts"]) for event in history)
    total = history_kc_count + len(query["concepts"])
    if total > SEQUENCE_LENGTH:
        raise ValueError("Shared history plus the complete query concept set exceeds 200 KCs")
    for key, actual in (("history_kc_count", history_kc_count),
                        ("target_kc_count", len(query["concepts"])), ("context_kc_count", total)):
        if key in context and context[key] != actual:
            raise ValueError(f"Shared {key} disagrees with the actual allowed context")
    return history, query, history_kc_count


def collate_shared_ktst(contexts: Sequence[dict], max_concepts: int) -> dict:
    """One set-dense row per target; last real event is its fixed-dummy query.

    ``target_positions`` is an inference selector, not KTST's reference
    score_mask. Passing a last-only mask to that training loss is prohibited.
    """
    if not contexts or type(max_concepts) is not int or max_concepts < 1:
        raise ValueError("KTST collation needs contexts and positive max_concepts")
    data = torch.full((len(contexts), SEQUENCE_LENGTH, max_concepts + 2), -1, dtype=torch.long)
    mask = torch.zeros((len(contexts), SEQUENCE_LENGTH), dtype=torch.bool)
    positions, identities, history_ids = [], [], []
    for row, context in enumerate(contexts):
        history, query, _ = _parts(context)
        events = history + [{**query, "response": QUERY_RESPONSE}]
        for position, event in enumerate(events):
            if len(event["concepts"]) > max_concepts:
                raise ValueError("Context concept set exceeds KTST max_concepts")
            data[row, position, :len(event["concepts"])] = torch.tensor(event["concepts"])
            data[row, position, -2:] = torch.tensor([event["question"], event["response"]])
        mask[row, :len(events)] = True
        positions.append(len(history))
        identities.append(context["event_id"])
        history_ids.append([event["event_id"] for event in history])
    return {"data": data, "padding_mask": mask,
            "target_positions": torch.tensor(positions, dtype=torch.long),
            "event_ids": identities, "history_event_ids": history_ids}


def collate_shared_kc(contexts: Sequence[dict], *, target_concepts: Sequence[int] | None = None) -> dict:
    """Expand past events, then append exactly one target KC to each row.

    By default creates a row for every query KC. Supplying one target_concept
    per context lets the predictor bound expanded rows before allocating any
    [rows, 200] tensor. All siblings have identical *past-only* histories.
    """
    if not contexts:
        raise ValueError("KC collation needs at least one context")
    if target_concepts is not None and len(target_concepts) != len(contexts):
        raise ValueError("target_concepts must align one-to-one with contexts")
    descriptions = []
    for index, context in enumerate(contexts):
        history, query, count = _parts(context)
        targets = query["concepts"] if target_concepts is None else [target_concepts[index]]
        for concept in targets:
            if type(concept) is not int or concept not in query["concepts"]:
                raise ValueError("Requested target KC is absent from its canonical query")
            descriptions.append((context, history, query, count, concept))
    shape = (len(descriptions), SEQUENCE_LENGTH)
    concepts = torch.zeros(shape, dtype=torch.long)
    questions = torch.zeros(shape, dtype=torch.long)
    responses = torch.zeros(shape, dtype=torch.long)
    valid = torch.zeros(shape, dtype=torch.bool)
    positions, identities, query_kcs, history_ids = [], [], [], []
    for row, (context, history, query, count, target) in enumerate(descriptions):
        position = 0
        for event in history:
            width = len(event["concepts"])
            concepts[row, position:position + width] = torch.tensor(event["concepts"])
            questions[row, position:position + width] = event["question"]
            responses[row, position:position + width] = event["response"]
            position += width
        assert position == count
        concepts[row, position], questions[row, position] = target, query["question"]
        responses[row, position] = QUERY_RESPONSE
        valid[row, :position + 1] = True
        positions.append(position)
        identities.append(context["event_id"])
        query_kcs.append(target)
        history_ids.append([event["event_id"] for event in history])
    masks = valid[:, :-1] & valid[:, 1:]
    batch = {"concepts": concepts, "questions": questions, "responses": responses,
             "valid_mask": valid, "masks": masks,
             "target_positions": torch.tensor(positions, dtype=torch.long),
             "ground_seq": (valid.sum(1).float()).unsqueeze(1),
             "event_ids": identities, "target_concepts": query_kcs, "history_event_ids": history_ids}
    # simpleKT reconstructs [first, shifted] internally. Do not shift again when
    # selecting its output: model outputs retain all 200 original positions.
    for name, full in (("cseqs", concepts), ("qseqs", questions), ("rseqs", responses.float())):
        batch[name] = full[:, :-1] * masks
        batch["shft_" + name] = full[:, 1:] * masks
    return batch


def _hash(records) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def audit_shared_projection(contexts: Sequence[dict], data_config: dict) -> dict:
    """Decode actual collator tensors and compare both representations fieldwise.

    Uses one context at a time to avoid dataset-sized expanded tensors. Hashes
    summarize the decoded evidence after the equality assertions, rather than
    substituting a common source hash for an actual tensor comparison.
    """
    records, targets, rows, total_history = [], [], 0, 0
    for context in contexts:
        history, query, count = _parts(context)
        sets = collate_shared_ktst([context], int(data_config["max_concepts"]))
        kcs = collate_shared_kc([context])
        expected_history = [(e["question"], c, e["response"]) for e in history for c in e["concepts"]]
        position = int(sets["target_positions"][0])
        expected_ids = [event["event_id"] for event in history]
        if position != len(history) or sets["history_event_ids"][0] != expected_ids:
            raise ValueError("KTST event boundary/identity mismatch")
        decoded_history = []
        for canonical, event in zip(history, sets["data"][0, :position].tolist()):
            if ((event[-2], event[-1], [c for c in event[:-2] if c >= 0])
                    != (canonical["question"], canonical["response"], canonical["concepts"])):
                raise ValueError("KTST per-event boundary/content differs from canonical history")
            decoded_history.extend((event[-2], c, event[-1]) for c in event[:-2] if c >= 0)
        current = sets["data"][0, position].tolist()
        decoded_query = (current[-2], [c for c in current[:-2] if c >= 0])
        if decoded_history != expected_history or decoded_query != (query["question"], query["concepts"]):
            raise ValueError("KTST tensor projection does not preserve canonical history/query")
        if current[-1] != QUERY_RESPONSE or int(sets["padding_mask"].sum()) != position + 1:
            raise ValueError("KTST target placeholder/mask mismatch")
        if not bool((sets["data"][0, position + 1:] == -1).all()):
            raise ValueError("KTST projection has content after its query")
        for row, target in enumerate(query["concepts"]):
            if int(kcs["target_positions"][row]) != count:
                raise ValueError("KC target position mismatch")
            decoded = list(zip(kcs["questions"][row, :count].tolist(),
                               kcs["concepts"][row, :count].tolist(), kcs["responses"][row, :count].tolist()))
            observed_target = (int(kcs["questions"][row, count]), int(kcs["concepts"][row, count]))
            if decoded != decoded_history or observed_target != (query["question"], target):
                raise ValueError("KC tensor history/query differs from set-dense projection")
            if int(kcs["responses"][row, count]) != QUERY_RESPONSE or int(kcs["ground_seq"][row, 0]) != count + 1:
                raise ValueError("KC placeholder/visible prefix length mismatch")
            if (int(kcs["valid_mask"][row].sum()) != count + 1
                    or bool(kcs["valid_mask"][row, count + 1:].any())):
                raise ValueError("KC padding mask mismatch")
            if kcs["history_event_ids"][row] != expected_ids:
                raise ValueError("KC history event identity mismatch")
            for full_name, shifted_name in (("concepts", "cseqs"), ("questions", "qseqs"), ("responses", "rseqs")):
                if bool(kcs[full_name][row, count + 1:].ne(0).any()):
                    raise ValueError("KC projection has content after its query")
                rebuilt = torch.cat((kcs[shifted_name][row, :1], kcs["shft_" + shifted_name][row]))
                if not torch.equal(rebuilt[:count + 1], kcs[full_name][row, :count + 1]):
                    raise ValueError("simpleKT shift changes the visible context")
            if sets["event_ids"][0] != kcs["event_ids"][row] or kcs["event_ids"][row] != context["event_id"]:
                raise ValueError("Projection event identity mismatch")
        for q, c, _ in decoded_history + [(query["question"], c, 0) for c in query["concepts"]]:
            if c >= int(data_config["num_c"]) or q >= max(1, int(data_config["num_q"])):
                raise ValueError("Projection contains an out-of-vocabulary ID")
        targets.append({"event_id": context["event_id"], "query": decoded_query})
        records.append({**targets[-1], "history": decoded_history})
        rows += len(query["concepts"])
        total_history += count
    return {"protocol": PROTOCOL, "passed": True, "events": len(contexts), "kc_rows": rows,
            "history_kc_occurrences": total_history, "sequence_length": SEQUENCE_LENGTH,
            "query_response": QUERY_RESPONSE, "projection_target_sha256": _hash(targets),
            "projection_context_sha256": _hash(records)}


def _move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.inference_mode()
def predict_shared(model: torch.nn.Module, contexts: Sequence[dict], device: str | torch.device,
                   max_rows_per_forward: int = 32) -> list[float]:
    """Return probabilities in exact context order; no labels or loss escape.

    KC scores use an unrounded float64 mean. Expanded accelerator rows are
    bounded even when one query has many KCs. The original KTST forward still
    computes its dummy-label loss internally, but it is discarded immediately
    and is never used for optimization, selection, metrics or this return value.
    """
    if type(max_rows_per_forward) is not int or max_rows_per_forward < 1:
        raise ValueError("max_rows_per_forward must be positive")
    name = model.model_name
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported shared-evaluation model: {name}")
    device = torch.device(device)
    actual_device = next(model.parameters()).device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device != actual_device:
        raise ValueError("Requested shared inference device differs from the model device")
    identities = [context["event_id"] for context in contexts]
    if len(set(identities)) != len(identities):
        raise ValueError("Shared contexts must have distinct target event IDs")
    if not contexts:
        return []
    for context in contexts:
        _parts(context)
    was_training = model.training
    model.eval()
    try:
        if name.startswith("ktst_"):
            output = []
            for start in range(0, len(contexts), max_rows_per_forward):
                batch = _move(collate_shared_ktst(contexts[start:start + max_rows_per_forward], model.max_concepts), device)
                # Do not pass a query-only score_mask into the reference loss.
                scores, _discarded_dummy_loss = forward_ktst(model, {"data": batch["data"], "padding_mask": batch["padding_mask"]})
                selected = scores[torch.arange(len(batch["event_ids"]), device=device), batch["target_positions"]]
                output.extend(selected.double().cpu().tolist())
        else:
            if name in {"simplekt", "mockt"} and int(model.ktlab_model_config.get("seq_len", 200)) < SEQUENCE_LENGTH:
                raise ValueError("Shared evaluation requires position capacity for fixed T=200")
            sums, counts = [0.0] * len(contexts), [0] * len(contexts)
            pending = []

            def flush():
                batch = collate_shared_kc([contexts[i] for i, _ in pending],
                                          target_concepts=[c for _, c in pending])
                batch = _move(batch, device)
                if name == "akt":
                    scores, _discarded_regularizer = model(batch["concepts"], batch["responses"], batch["questions"])
                elif name == "mockt":
                    scores, _discarded_regularizer = model(batch["ground_seq"], batch["concepts"], batch["responses"], batch["questions"])
                else:
                    scores = model(batch)
                selected = scores[torch.arange(len(pending), device=device), batch["target_positions"]]
                if not bool(torch.isfinite(selected).all()) or bool(((selected < 0) | (selected > 1)).any()):
                    raise ValueError("Shared per-KC model output must be a finite probability before fusion")
                for (index, _), probability in zip(pending, selected.double().cpu().tolist()):
                    sums[index] += probability
                    counts[index] += 1
                pending.clear()

            for index, context in enumerate(contexts):
                for concept in context["query"]["concepts"]:
                    pending.append((index, concept))
                    if len(pending) == max_rows_per_forward:
                        flush()
            if pending:
                flush()
            output = [total / count for total, count in zip(sums, counts)]
        values = torch.tensor(output, dtype=torch.float64)
        if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
            raise ValueError("Shared model returned non-finite or non-probability outputs")
        return output
    finally:
        model.train(was_training)
