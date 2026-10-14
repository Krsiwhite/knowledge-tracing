"""Explicit AS15 single-KC validation execution; model mathematics is unchanged.

The planner never reads target labels. It retains one longest permitted context
per group, after checking every shorter history/query against the next prefix.
Only same-start consecutive targets share a forward; sliding windows do not.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import time

from .shared_events import PROTOCOL

EXECUTION = "single_kc_prefix_groups_v1"
MODELS = {"akt", "simplekt", "ukt", "ktst_mean", "ktst_mhsa"}


def check_model_config(name: str, settings: dict):
    """Keep v1 on the audited embedding/attention paths, not arbitrary ablations."""
    if name in {"akt", "simplekt"} and settings.get("emb_type", "qid") != "qid":
        raise ValueError("Prefix groups support only the audited AKT/simpleKT qid embedding path")
    if name == "ukt" and (settings.get("emb_type", "stoc_qid") != "stoc_qid"
                          or settings.get("atten_type", "w2") != "w2"):
        raise ValueError("Prefix groups support only the audited UKT stoc_qid/w2 prediction path")
    if name.startswith("ktst_") and settings.get("attn_variant", "learnable_alibi_monotonic_q_k") != "learnable_alibi_monotonic_q_k":
        raise ValueError("Prefix groups support only the audited KTST learnable_alibi_monotonic_q_k path")


def check_execution(config: dict, manifest: dict | None = None):
    """Reject unsupported explicit modes; absence preserves the original path."""
    if "shared_validation_execution" not in config:
        return None
    if config["shared_validation_execution"] != EXECUTION:
        raise ValueError("Unknown or missing shared_validation_execution mode")
    if config.get("protocol") != PROTOCOL or config.get("model") not in MODELS:
        raise ValueError("Prefix groups require shared development and AKT/simpleKT/UKT/KTST; MoC is unsupported")
    if config.get("dataset") != "assist2015":
        raise ValueError("Prefix groups are explicitly restricted to assist2015")
    check_model_config(config["model"], config.get("model_config", {}))
    if manifest is not None:
        data = manifest["data_config"]
        if (manifest.get("dataset") != "assist2015" or data.get("num_q") != 0
                or data.get("max_concepts") != 1 or data.get("maxlen") != 200):
            raise ValueError("Prefix groups require AS15, no question IDs, one KC, and maxlen=200")
    return EXECUTION


def check_dataset(dataset, model_name: str):
    if (model_name not in MODELS or dataset.protocol != PROTOCOL
            or dataset.audit.get("protocol") != PROTOCOL or dataset.audit.get("split") != "valid"
            or dataset.audit.get("dataset") != "assist2015"
            or dataset.num_questions != 0 or dataset.kc_budget != 200
            or dataset.audit.get("length_statistics", {}).get("query_kc_counts") != {"1": len(dataset)}):
        raise ValueError("Prefix grouping requires the complete declared single-KC AS15 validation representation")


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Shared development evaluation deadline reached")


def _snapshot(context: dict) -> dict:
    """Validate and copy allowed inputs/identity only, without accessing label."""
    if context.get("protocol") != PROTOCOL:
        raise ValueError("Prefix groups require the shared development protocol")
    uid, ordinal, start, fold = (context[key] for key in ("uid", "target_index", "history_start_index", "fold"))
    if not isinstance(uid, str) or not uid or any(type(x) is not int for x in (ordinal, start, fold)):
        raise ValueError("Prefix group identity requires uid and integer ordinal/start/fold")
    if not 0 <= fold <= 4 or ordinal < 1 or start != max(0, ordinal - 199):
        raise ValueError("Prefix history must be the complete canonical single-KC window")
    prefix = f"assist2015:{uid}:"
    if context.get("event_id") != prefix + str(ordinal):
        raise ValueError("Prefix target identity is inconsistent with AS15 uid/ordinal")
    history = context.get("history")
    if not isinstance(history, (list, tuple)) or len(history) != ordinal - start:
        raise ValueError("Prefix history length disagrees with its original ordinal/start")

    def query(value):
        concepts = value.get("concepts")
        if (value.get("question", "missing") is not None or not isinstance(concepts, (list, tuple))
                or len(concepts) != 1 or type(concepts[0]) is not int or concepts[0] < 0):
            raise ValueError("Prefix groups require exactly one KC and question=None in every event")
        return {"question": None, "concepts": [concepts[0]]}

    copied = []
    for index, event in enumerate(history, start):
        if event.get("event_id") != prefix + str(index):
            raise ValueError("Prefix history event order/identity is inconsistent")
        if type(event.get("response")) is not int or event["response"] not in (0, 1):
            raise ValueError("Prefix history responses must be binary integers")
        copied.append({"event_id": event["event_id"], **query(event), "response": event["response"]})
    if "response" in context["query"]:
        raise ValueError("A shared query must not contain a response")
    current_query = query(context["query"])
    for key, expected in (("history_kc_count", len(history)), ("target_kc_count", 1),
                          ("context_kc_count", len(history) + 1)):
        if context.get(key) != expected:
            raise ValueError(f"Prefix {key} disagrees with actual context")
    return {"protocol": PROTOCOL, "uid": uid, "fold": fold, "event_id": context["event_id"],
            "target_index": ordinal, "history_start_index": start, "history": copied, "query": current_query,
            "history_kc_count": len(history), "target_kc_count": 1, "context_kc_count": len(history) + 1}


@dataclass(frozen=True)
class PrefixGroup:
    context: dict  # longest label-free permitted context
    targets: tuple[tuple[str, int], ...]  # original event ID, position in packed row


def iter_prefix_groups(contexts, *, deadline=None):
    """Stream verified consecutive groups while retaining no old short histories."""
    previous = None
    targets = []
    closed_students = set()
    for raw in contexts:
        check_deadline(deadline)
        current = _snapshot(raw)
        if previous is not None:
            same_student = current["uid"] == previous["uid"]
            if same_student and (current["fold"] != previous["fold"]
                                 or current["target_index"] != previous["target_index"] + 1):
                raise ValueError("Prefix targets must be consecutive and retain their fold")
            if not same_student:
                closed_students.add(previous["uid"])
                if current["uid"] in closed_students:
                    raise ValueError("A completed student cannot reappear in prefix groups")
            same_group = same_student and current["history_start_index"] == previous["history_start_index"]
            if same_group:
                last = current["history"][-1]
                if (current["history"][:-1] != previous["history"]
                        or {key: last[key] for key in ("event_id", "question", "concepts")}
                        != {"event_id": previous["event_id"], **previous["query"]}):
                    raise ValueError("Prefix history/query differs from the earlier target's complete context")
            else:
                yield PrefixGroup(previous, tuple(targets))
                targets = []
        targets.append((current["event_id"], len(current["history"])))
        previous = current
    if previous is not None:
        check_deadline(deadline)
        yield PrefixGroup(previous, tuple(targets))


def predict_prefix_groups(model, groups, device, *, max_rows_per_forward=32, deadline=None):
    """Gather multiple causal positions; no label, loss, or autograd result escapes."""
    import torch
    from .shared_adapters import _move, collate_shared_kc, collate_shared_ktst
    from .ktst_model import forward_ktst

    if (isinstance(max_rows_per_forward, bool) or not isinstance(max_rows_per_forward, Integral)
            or max_rows_per_forward < 1):
        raise ValueError("max_rows_per_forward must be positive")
    name = model.model_name
    if name not in MODELS:
        raise ValueError("Unsupported prefix-group model; MoC length routing cannot be merged")
    check_model_config(name, model.ktlab_model_config)
    if name.startswith("ktst_"):
        if model.max_concepts != 1 or model.num_questions != 1 or model.max_len < 200:
            raise ValueError("Prefix KTST requires a single dummy question, one KC, and T200 capacity")
    elif model.n_pid != 0:
        raise ValueError("Prefix groups require models built without question IDs")
    if name in {"simplekt", "ukt"} and model.ktlab_model_config.get("seq_len", 200) < 200:
        raise ValueError("Prefix groups require fixed T200 position capacity")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device != next(model.parameters()).device:
        raise ValueError("Requested prefix inference device differs from model device")
    was_training = model.training
    model.eval()
    result = []
    try:
        with torch.no_grad():
            for start in range(0, len(groups), max_rows_per_forward):
                check_deadline(deadline)
                selected_groups = groups[start:start + max_rows_per_forward]
                contexts = [_snapshot(group.context) for group in selected_groups]
                rows, positions = [], []
                for row, (group, context) in enumerate(zip(selected_groups, contexts)):
                    expected = context["history"] + [{"event_id": context["event_id"], **context["query"]}]
                    previous_position = None
                    for event_id, position in group.targets:
                        if (type(position) is not int or not 1 <= position < len(expected)
                                or expected[position]["event_id"] != event_id
                                or (previous_position is not None and position != previous_position + 1)):
                            raise ValueError("Prefix target selector differs from its verified packed context")
                        rows.append(row)
                        positions.append(position)
                        previous_position = position
                    if previous_position != len(context["history"]):
                        raise ValueError("Prefix group must include its final query target")
                if name.startswith("ktst_"):
                    batch = _move(collate_shared_ktst(contexts, 1), device)
                    scores, _discarded_loss = forward_ktst(model, {"data": batch["data"], "padding_mask": batch["padding_mask"]})
                else:
                    batch = _move(collate_shared_kc(contexts), device)
                    if name == "akt":
                        scores, _discarded_loss = model(batch["concepts"], batch["responses"], batch["questions"])
                    elif name == "ukt":
                        scores = model(batch, train=False)
                    else:
                        scores = model(batch)
                selected = scores[torch.tensor(rows, device=device), torch.tensor(positions, device=device)]
                if not bool(torch.isfinite(selected).all()) or bool(((selected < 0) | (selected > 1)).any()):
                    raise ValueError("Prefix group model returned invalid probabilities")
                result.extend(selected.double().cpu().tolist())
                check_deadline(deadline)
        return result
    finally:
        model.train(was_training)
