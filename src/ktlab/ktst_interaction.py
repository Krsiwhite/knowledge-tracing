"""Explicit local KTST mean interaction variants; not an upstream reproduction.

The original query, temporal transformer, shift and logits BCE are preserved.
Only the pre-shift history representation receives a zero-initialized KC table.
No optimizer, data loading, or test-release capability is defined here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


RECIPES = ("signed_response", "response_independent")


def validate_interaction_recipe(model_name: str, recipe: str | None) -> None:
    if recipe is None:
        return
    if model_name != "ktst_mean" or type(recipe) is not str or recipe not in RECIPES:
        raise ValueError("ktst_interaction_recipe requires ktst_mean and signed_response or response_independent")


def interaction_source_identity(vendor_sha256: str, recipe: str) -> tuple[dict, str]:
    validate_interaction_recipe("ktst_mean", recipe)
    identity = {"schema_version": 1, "model": "ktst_mean",
                "vendor_source_sha256": vendor_sha256,
                "adaptation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "ktst_interaction_recipe": recipe}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return identity, digest


def interaction_model_class(original_class):
    """Derive a visible local model class, without patching upstream modules."""
    class InteractionModel(original_class):
        def __init__(self, *, interaction_recipe: str, **kwargs):
            validate_interaction_recipe("ktst_mean", interaction_recipe)
            if kwargs.get("aggregation") != "q_mean_c":
                raise ValueError("KTST interaction variants currently require mean aggregation")
            super().__init__(**kwargs)
            # torch.zeros and Parameter consume no random numbers. Common
            # parameters and subsequent dropout RNG remain paired with base.
            self.interaction_vectors = nn.Parameter(torch.zeros(self.num_concepts, kwargs["d_model"]))
            self.ktlab_interaction_recipe = interaction_recipe

        def history_residual(self, concepts, responses, padding_mask):
            # Preserve the original valid-event KC mean. _unpack_data replaces
            # padded events by zero, so explicitly zero their *added* residual.
            present = (concepts != -1) & padding_mask.unsqueeze(-1)
            counts = present.sum(dim=2, keepdim=True)
            if bool((padding_mask & (counts.squeeze(-1) == 0)).any()):
                raise ValueError("KTST interaction residual requires a KC on every valid event")
            vectors = self.interaction_vectors[concepts.masked_fill(~present, 0)]
            mean = (vectors * present.unsqueeze(-1)).sum(dim=2) / counts.clamp_min(1)
            if self.ktlab_interaction_recipe == "signed_response":
                mean = mean * (2 * responses - 1).unsqueeze(-1)
            return mean * padding_mask.unsqueeze(-1)

        def forward(self, data, padding_mask):
            # Mirrors pinned Model.forward, with the single marked addition.
            questions, responses, concepts = self._unpack_data(data, padding_mask)
            questions_embedded = self.q_emb.forward(questions + 1)
            concepts_embedded = self.c_emb.forward(concepts + 1)
            responses_embedded = self.r_emb.forward(responses + 1)
            if self.bias_before_agg:
                assert self.use_bias_emb == True
                questions_embedded = questions_embedded + self.b_emb.weight[0]
            x = self.agg.forward(mask=(concepts == -1), questions_embedded=questions_embedded,
                                 concepts_embedded=concepts_embedded)
            y = x + responses_embedded
            y = y + self.history_residual(concepts, responses, padding_mask)
            if (not self.bias_before_agg) and self.use_bias_emb:
                x = x + self.b_emb.weight[0]
                y = y + self.b_emb.weight[0]
            if self.attn_variant == "standard":
                x = self.positional_encoding(x)
                y = self.positional_encoding(y)
            x_tilde = self.model.forward(tgt=x, src=y)
            logits = self.clf.forward(torch.cat([x_tilde, x], dim=-1)).squeeze(-1)
            loss = self.loss_fn(torch.masked_select(logits[:, 1:], padding_mask[:, 1:]),
                                torch.masked_select(responses[:, 1:].type(dtype=torch.float32), padding_mask[:, 1:]))
            return torch.sigmoid(logits), loss, 0.0

    return InteractionModel
