"""Explicit training-only recipes using three unmodified MoC forwards.

No new parameters, inference rule, optimizer, or vendor changes. This aligns
the length-branch condition only, not repeated-KC/event-window semantics.
"""
from __future__ import annotations

import torch

PREFIX_RECIPE = "mockt-target-prefix-branches-v1"
CONTROL_RECIPE = "mockt-three-pass-ground-seq-control-v1"
RECIPES = frozenset({PREFIX_RECIPE, CONTROL_RECIPE})
PROTOCOL = "ktlab-v2-shared-event-context-kc200-dev"


def validate_recipe_config(config: dict) -> str | None:
    """Reject unsupported opt-ins before opening data or constructing a model."""
    recipe = config.get("mockt_training_recipe")
    if recipe is None:
        return None
    if not isinstance(recipe, str) or recipe not in RECIPES:
        raise ValueError("Unsupported MoC training recipe")
    if config.get("model") != "mockt" or config.get("protocol") != PROTOCOL:
        raise ValueError("MoC training recipes require model mockt and the shared-development protocol")
    settings = config.get("model_config", {})
    if not isinstance(settings, dict):
        raise ValueError("MoC training recipe requires a model_config mapping")
    if settings.get("emb_type", "qid_conv_ker_noexp") != "qid_conv_ker_noexp":
        raise ValueError("MoC training recipes currently require qid_conv_ker_noexp")
    for key, value in (("kernel_size", 4), ("seq_len", 200), ("sequence1", 50), ("sequence2", 150)):
        if type(settings.get(key, value)) is not int or settings.get(key, value) != value:
            raise ValueError(f"MoC training recipes currently require {key}={value}")
    return recipe


def bind_training_recipe(model, config: dict) -> str | None:
    recipe = validate_recipe_config(config)
    if recipe is not None:
        if model.model_name != "mockt" or model.emb_type != "qid_conv_ker_noexp" or (
            model.sequence1, model.sequence2
        ) != (50, 150):
            raise ValueError("Constructed MoC model conflicts with its training recipe")
        if hasattr(model, "ktlab_model_config"):
            validate_recipe_config({**config, "model_config": model.ktlab_model_config})
        # Runtime dispatch only. Evaluation calls the original model directly.
        model.ktlab_mockt_training_recipe = recipe
    return recipe


def _inputs(batch):
    mask = batch["masks"]
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != 199 or mask.shape[0] == 0:
        raise ValueError("MoC recipes require batched length-200 adjacent-position masks")
    lengths = mask.sum(1)
    if bool((lengths < 1).any()) or not torch.equal(
        mask, torch.arange(199, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    ):
        raise ValueError("MoC recipes require nonempty left-aligned valid prefixes")
    selected = batch["smasks"]
    if selected.dtype != torch.bool or selected.shape != mask.shape or selected.device != mask.device or bool((selected & ~mask).any()):
        raise ValueError("MoC scoring masks must be a subset of valid positions")
    ground = batch["ground_seq"]
    if ground.dtype not in {torch.float32, torch.float64, torch.int32, torch.int64} or ground.shape != (mask.shape[0], 1) or ground.device != mask.device or not torch.equal(
        ground, (lengths + 1).unsqueeze(1).to(ground.dtype)
    ):
        raise ValueError("MoC ground_seq must match the original full valid row length")
    full = {}
    for name in ("cseqs", "qseqs", "rseqs"):
        before, after = batch[name], batch["shft_" + name]
        for value in (before, after):
            if value.shape != mask.shape or value.device != mask.device or not bool(torch.isfinite(value).all()):
                raise ValueError("MoC recipes require finite aligned input tensors")
            if bool(value[~mask].ne(0).any()) or bool(value.ne(value.long()).any()) or bool(value.lt(0).any()):
                raise ValueError("MoC recipe inputs must be nonnegative integers with zero masked padding")
            if name == "rseqs" and bool(value.gt(1).any()):
                raise ValueError("MoC responses must be binary")
        # Exactly the existing adapter reconstruction: no extra response shift.
        full[name] = torch.cat((before[:, :1], after), dim=1).long()
    valid = torch.cat((mask[:, :1], mask), dim=1)
    prefix_lengths = valid.long().cumsum(1)
    groups = (prefix_lengths > 50).long() + (prefix_lengths > 150).long()
    return full, groups, ground


def forward_mockt_training(model, batch: dict, recipe: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shifted probabilities and ONE full-batch original regularizer.

    Does not switch model.train/eval, detach any prediction, or differentiate.
    With dropout off, the prefix recipe has the same selected probabilities as
    separate original-model prefix calls, while the control equals one original
    full-row call. With dropout on, the control matches the three-call structure.
    """
    if recipe not in RECIPES or model.model_name != "mockt" or model.emb_type != "qid_conv_ker_noexp":
        raise ValueError("Unsupported model or MoC training recipe")
    if (model.sequence1, model.sequence2) != (50, 150):
        raise ValueError("MoC recipe branch thresholds must remain 50/150")
    full, groups, ground = _inputs(batch)
    outputs, regularizer = [], None
    for index, representative in enumerate((1, 51, 151)):
        supplied = torch.full_like(ground, representative) if recipe == PREFIX_RECIPE else ground
        probabilities, original_regularizer = model(supplied, full["cseqs"], full["rseqs"], full["qseqs"])
        if probabilities.shape != full["cseqs"].shape:
            raise ValueError("Original MoC probabilities are misaligned")
        outputs.append(probabilities)
        if index == 0:
            # The original regularizer is independent of s. Never triple it or
            # sum per-target cropped-prefix penalties (a different objective).
            regularizer = torch.as_tensor(original_regularizer, device=probabilities.device)
    probabilities = torch.stack(outputs, dim=0).gather(0, groups.unsqueeze(0)).squeeze(0)
    return probabilities[:, 1:], regularizer
