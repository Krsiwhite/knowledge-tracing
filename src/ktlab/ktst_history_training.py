"""Explicit paired target-only KTST training; no new model or inference rule.

The two recipes share targets, stateless sampling and weights. Only the allowed
training history differs. Original KTST/query/shift/classifier bytes are kept.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from .ktst_data import KTSTSequenceDataset, _digest, _parse, _row_at
from .ktst_model import forward_ktst
from .shared_adapters import collate_shared_ktst
from .shared_events import PROTOCOL, _events_from_row, _history_event, _query, context_digests


RECIPES = ("sampled_chunk_prefix_v1", "sampled_complete_suffix_v1")
RECIPE_KEY = "ktst_history_training_recipe"
K_KEY = "ktst_history_samples_per_chunk"
SOURCE_KEY = "ktst_history_training_source_sha256"


def _integer(value, name, low=0, high=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in the declared range")
    return int(value)


def validate_history_config(config):
    recipe = config.get(RECIPE_KEY)
    if recipe is None:
        if K_KEY in config or SOURCE_KEY in config:
            raise ValueError("History samples/source identity require a history training recipe")
        return
    if (config.get("model") != "ktst_mean" or type(recipe) is not str or recipe not in RECIPES
            or config.get("protocol") != PROTOCOL):
        raise ValueError("KTST history training requires ktst_mean, an explicit supported recipe, and shared development validation")
    _integer(config.get(K_KEY), K_KEY, 1, 8)
    if _integer(config.get("fold", 0), "fold", 0, 4) != 0:
        raise ValueError("The first history-training adaptation supports held-out fold0 / training folds1-4 only")
    if _integer(config.get("batch_size", 32), "batch_size", 1) != 32:
        raise ValueError("History training uses one optimizer update per 32 chunks (or final partial batch)")
    _integer(config.get("max_rows_per_forward", 32), "max_rows_per_forward", 1)
    _integer(config.get("seed", 42), "seed")
    if SOURCE_KEY in config and config[SOURCE_KEY] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Configured history training source differs from current implementation")


def history_training_identity(config):
    validate_history_config(config)
    if config.get(RECIPE_KEY) is None:
        return {}
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if SOURCE_KEY in config and config[SOURCE_KEY] != source_sha:
        raise ValueError("Configured history training source differs from current implementation")
    return {RECIPE_KEY: config[RECIPE_KEY], K_KEY: int(config[K_KEY]), SOURCE_KEY: source_sha}


def bind_history_training_identity(model, config):
    identity = history_training_identity(config)
    if identity:
        if getattr(model, "model_name", None) != "ktst_mean":
            raise ValueError("History training identity does not match the actual model")
        model.ktlab_history_training_identity = identity
    return identity


def _encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sample_target_indices(seed, epoch, chunk_identity, n_eligible, samples_per_chunk):
    """Uniform sampling WITH replacement, independent of global/torch RNG.

    Neither recipe nor K enters the RNG seed: paired recipes and K prefixes use
    the same draws. chunk_identity is a stable source-derived opaque digest.
    """
    seed = _integer(seed, "seed")
    epoch = _integer(epoch, "epoch", 1)
    n = _integer(n_eligible, "n_eligible", 1)
    k = _integer(samples_per_chunk, K_KEY, 1, 8)
    if not isinstance(chunk_identity, str) or len(chunk_identity) != 64:
        raise ValueError("Sampling needs the stable SHA256 chunk identity")
    try:
        bytes.fromhex(chunk_identity)
    except ValueError as error:
        raise ValueError("Malformed chunk identity") from error
    digest = hashlib.sha256(b"ktlab.ktst.history-sampling.v1\0" + _encoded([seed, epoch, chunk_identity])).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    return tuple(rng.randrange(n) for _ in range(k))


@dataclass(frozen=True)
class _EligibleChunk:
    source_index: int
    first_event: int
    first_target: int
    n_eligible: int
    identity: str


class SampledKTSTHistoryDataset(Dataset):
    """One item is a chunk with K sampled targets, not K optimizer steps."""

    def __init__(self, manifest, fold, *, recipe, samples_per_chunk, seed):
        validate_history_config({"model": "ktst_mean", "protocol": PROTOCOL, "fold": fold,
                                 RECIPE_KEY: recipe, K_KEY: samples_per_chunk, "seed": seed})
        self.recipe, self.samples_per_chunk, self.seed = recipe, int(samples_per_chunk), int(seed)
        self.base = KTSTSequenceDataset(manifest, fold, split="train")
        if self.base.maxlen != 200:
            raise ValueError("History training requires the original KC200 budget")
        self.path, self.original_path = self.base.path, self.base.original_path
        self.name, self.max_concepts = self.base.name, self.base.max_concepts
        self.num_concepts = self.base.num_concepts
        self.num_questions = int(manifest["data_config"]["num_q"])
        self._chunks = []
        self._cache = OrderedDict()
        self.epoch = None
        counts = {"excluded_chunk_first": 0, "excluded_partial_query_after_first": 0,
                  "excluded_partial_history_after_first_and_complete_query": 0,
                  "eligible_unique_targets": 0, "excluded_chunks_without_eligible_targets": 0}
        eligible_hasher = hashlib.sha256(b"ktlab.ktst.history-eligible.v1\0")
        cached_uid, original = None, None
        for index, chunk in enumerate(self.base._chunks):
            if chunk.uid != cached_uid:
                original = _parse(_row_at(self.original_path, self.base._originals[chunk.uid].offset),
                                  concept_only=self.base.concept_only)
                cached_uid = chunk.uid
            parsed = _parse(_row_at(self.path, chunk.offset), concept_only=self.base.concept_only, length=200)
            for field in ("questions", "concepts", "responses", "is_repeat"):
                if parsed[field] != original[field][chunk.kc_start:chunk.kc_stop]:
                    raise ValueError("Training source changed after original chunk alignment")
            details = self.base._event_details(chunk, parsed, original)
            complete_history, eligible = True, []
            for position, detail in enumerate(details):
                if position == 0:
                    counts["excluded_chunk_first"] += 1
                elif not detail["complete"]:
                    counts["excluded_partial_query_after_first"] += 1
                elif not complete_history:
                    counts["excluded_partial_history_after_first_and_complete_query"] += 1
                else:
                    eligible.append(detail["event_index"])
                complete_history &= detail["complete"]
            if not eligible:
                counts["excluded_chunks_without_eligible_targets"] += 1
                continue
            if eligible != list(range(eligible[0], eligible[0] + len(eligible))):
                raise ValueError("Strict complete-history targets must be a contiguous original-event range")
            chunk_identity = hashlib.sha256(_encoded([self.name, chunk.uid, chunk.fold,
                                                      chunk.kc_start, chunk.kc_stop])).hexdigest()
            descriptor = _EligibleChunk(index, details[0]["event_index"], eligible[0], len(eligible), chunk_identity)
            self._chunks.append(descriptor)
            counts["eligible_unique_targets"] += len(eligible)
            eligible_hasher.update(_encoded([chunk_identity, eligible[0], len(eligible)]) + b"\n")
        if not self._chunks:
            raise ValueError("No chunks satisfy the strict complete-target/history policy")
        self.audit = {**self.base.audit, "training_representation": "sampled_complete_event_target_only",
                      "original_training_rows": len(self.base), "rows": len(self), **counts,
                      "eligible_target_index_sha256": eligible_hasher.hexdigest(),
                      RECIPE_KEY: recipe, K_KEY: self.samples_per_chunk,
                      "samples_per_epoch": len(self) * self.samples_per_chunk,
                      "sampling": "uniform_with_replacement_sha256_local_python_random_v1",
                      "loss_weight": "eligible_targets_in_chunk / samples_per_chunk; normalize across whole chunk batch",
                      "loss_unit": "sampled_current_event_only", "first_version_training_folds": [1, 2, 3, 4]}
        self.verify_source()

    def __len__(self):
        return len(self._chunks)

    def verify_source(self):
        for path, key in ((self.path, "source_sha256"), (self.original_path, "original_sha256")):
            if _digest(path) != self.base.audit[key]:
                raise ValueError("History-training source changed after indexing")

    def set_epoch(self, epoch):
        self.epoch = _integer(epoch, "epoch", 1)
        self.verify_source()
        self._cache.clear()

    def _draws(self, descriptor):
        if self.epoch is None:
            raise ValueError("set_epoch must precede sampled training access")
        return sample_target_indices(self.seed, self.epoch, descriptor.identity,
                                     descriptor.n_eligible, self.samples_per_chunk)

    def epoch_sampling_audit(self):
        hasher = hashlib.sha256(b"ktlab.ktst.history-epoch-sampling.v1\0")
        unique = 0
        for descriptor in self._chunks:
            draws = self._draws(descriptor)
            unique += len(set(draws))
            hasher.update(_encoded([descriptor.identity, draws]) + b"\n")
        return {"epoch": self.epoch, "seed": self.seed, K_KEY: self.samples_per_chunk,
                "eligible_chunks": len(self), "sampled_rows": len(self) * self.samples_per_chunk,
                "unique_targets_sampled": unique, "sampling_sha256": hasher.hexdigest(),
                "order": "canonical eligible chunk order, independent of DataLoader shuffle",
                "scope": "planned full epoch; consumed batch evidence is recorded separately"}

    def _student(self, uid):
        if uid in self._cache:
            self._cache.move_to_end(uid)
            return self._cache[uid]
        row = _row_at(self.original_path, self.base._originals[uid].offset)
        events, _ = _events_from_row(row, self.num_concepts, self.num_questions)
        prefix = [0]
        for event in events:
            prefix.append(prefix[-1] + len(event.concepts))
        self._cache[uid] = events, prefix
        if len(self._cache) > 2:
            self._cache.popitem(last=False)
        return events, prefix

    def __getitem__(self, index):
        descriptor = self._chunks[index]
        chunk = self.base._chunks[descriptor.source_index]
        events, prefix = self._student(chunk.uid)
        draws = self._draws(descriptor)
        contexts, labels = [], []
        for sampled in draws:
            ordinal = descriptor.first_target + sampled
            event = events[ordinal]
            start = descriptor.first_event
            if self.recipe == "sampled_complete_suffix_v1":
                start = bisect_left(prefix, prefix[ordinal] - (200 - len(event.concepts)), 0, ordinal + 1)
            if not 0 <= start <= descriptor.first_event < ordinal or prefix[ordinal + 1] - prefix[start] > 200:
                raise ValueError("Sampled target violates complete history/KC200 contract")
            context = {"protocol": PROTOCOL, "uid": chunk.uid, "fold": chunk.fold,
                       "event_id": f"{self.name}:{chunk.uid}:{ordinal}", "target_index": ordinal,
                       "history": [_history_event(self.name, chunk.uid, i, events[i]) for i in range(start, ordinal)],
                       "query": _query(event), "history_start_index": start,
                       "history_kc_count": prefix[ordinal] - prefix[start],
                       "target_kc_count": len(event.concepts),
                       "context_kc_count": prefix[ordinal + 1] - prefix[start]}
            context.update(context_digests(context))
            contexts.append(context)
            labels.append(event.response)
        sample_sha = hashlib.sha256(_encoded([descriptor.identity, self.epoch, draws])).hexdigest()
        return {"contexts": contexts, "labels": torch.tensor(labels, dtype=torch.float32),
                "weights": torch.full((self.samples_per_chunk,), descriptor.n_eligible / self.samples_per_chunk,
                                      dtype=torch.float64),
                "eligible_count": descriptor.n_eligible, "chunk_identity": descriptor.identity,
                "sampling_sha256": sample_sha, "sampled_unique_count": len(set(draws)), "epoch": self.epoch}

    def close(self):
        self._cache.clear()
        self.base.close()


def collate_history_chunks(items):
    if not items:
        raise ValueError("Cannot collate empty history chunks")
    sizes = {len(item["contexts"]) for item in items}
    if len(sizes) != 1 or not 1 <= next(iter(sizes)) <= 8 or len({item["epoch"] for item in items}) != 1:
        raise ValueError("History chunk batch must share K and epoch")
    for item in items:
        k = len(item["contexts"])
        if (item["labels"].shape != (k,) or item["weights"].shape != (k,)
                or not torch.equal(item["weights"], torch.full((k,), item["eligible_count"] / k, dtype=torch.float64))):
            raise ValueError("History chunk weights/labels do not match eligible_count/K")
        if any("label" in context for context in item["contexts"]):
            raise ValueError("Target labels must remain outside history/query contexts")
    weights = torch.cat([item["weights"] for item in items])
    return {"contexts": [context for item in items for context in item["contexts"]],
            "labels": torch.cat([item["labels"] for item in items]), "weights": weights,
            "denominator": weights.sum(), "chunk_count": len(items),
            "sampling_sha256": hashlib.sha256(_encoded([item["sampling_sha256"] for item in items])).hexdigest(),
            "sampled_unique_count": sum(item["sampled_unique_count"] for item in items),
            "epoch": items[0]["epoch"]}


def target_logits(model, batch):
    """Capture the actual final Linear output; no sigmoid inverse or detach.

    Upstream explicitly calls clf.forward, bypassing hooks on Sequential itself.
    Its final Linear is called normally by Sequential and preserves the graph.
    """
    if getattr(model, "model_name", None) != "ktst_mean" or not isinstance(model.clf[-1], nn.Linear):
        raise ValueError("Target-only training requires the pinned KTST mean classifier")
    positions = batch["target_positions"]
    b = batch["data"].shape[0]
    if (positions.shape != (b,) or positions.dtype != torch.long
            or bool((positions != batch["padding_mask"].sum(-1) - 1).any())
            or bool((positions < 1).any())):
        raise ValueError("Target-only training must select each last real event after history")
    indices = torch.arange(b, device=positions.device)
    if bool((batch["data"][indices, positions, -1] != 0).any()):
        raise ValueError("Current target response must be the fixed dummy zero")
    captured = []
    handle = model.clf[-1].register_forward_hook(lambda module, args, output: captured.append(output))
    try:
        forward_ktst(model, {"data": batch["data"], "padding_mask": batch["padding_mask"]})
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (*batch["padding_mask"].shape, 1):
        raise ValueError("Original classifier did not produce the expected logits exactly once")
    return captured[0][indices, positions, 0]


def weighted_target_loss(logits, labels, weights, denominator=None):
    if logits.ndim != 1 or labels.shape != logits.shape or weights.shape != logits.shape or logits.numel() == 0:
        raise ValueError("Target logits, labels and weights must be aligned nonempty vectors")
    if (not bool(torch.isfinite(logits).all()) or not bool(((labels == 0) | (labels == 1)).all())
            or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any())):
        raise ValueError("Target loss requires finite logits, binary labels and positive finite weights")
    denominator = weights.sum() if denominator is None else denominator
    denominator = torch.as_tensor(denominator, device=logits.device, dtype=torch.float64)
    if denominator.ndim != 0 or not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("The whole chunk-batch loss denominator must be positive and finite")
    losses = F.binary_cross_entropy_with_logits(logits, labels.to(device=logits.device, dtype=logits.dtype), reduction="none")
    return (losses * weights.to(device=logits.device, dtype=torch.float64)).sum() / denominator


def iter_weighted_target_losses(model, batch, device, max_rows_per_forward=32, *, deadline=None):
    """Yield microbatch contributions normalized by the SAME whole-batch sum.

    Caller zeros gradients once, backpropagates each yielded loss, clips/steps
    once. The function itself never constructs an optimizer or calls backward.
    """
    maximum = _integer(max_rows_per_forward, "max_rows_per_forward", 1)
    contexts, labels, weights = batch["contexts"], batch["labels"], batch["weights"]
    if labels.shape != (len(contexts),) or weights.shape != labels.shape or not len(contexts):
        raise ValueError("Sampled contexts and target vectors are unaligned")
    denominator = weights.sum()
    if not torch.equal(torch.as_tensor(batch["denominator"]), denominator):
        raise ValueError("Loss denominator must cover the complete chunk batch")
    for start in range(0, len(contexts), maximum):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("KTST history training wall-clock budget exhausted")
        stop = min(start + maximum, len(contexts))
        projected = collate_shared_ktst(contexts[start:stop], model.max_concepts)
        projected = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in projected.items()}
        logits = target_logits(model, projected)
        yield weighted_target_loss(logits, labels[start:stop], weights[start:stop], denominator)
