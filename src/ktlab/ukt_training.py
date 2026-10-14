"""Pinned UKT author response augmentation, applied to CPU/GPU minibatches.

Author UKTDataset at 86ad9d8f76ee2c761e116fcadc43705f66ffd0b6, lines 101-123,
has a first-position condition inside its shifted-response loop. For a valid
left-aligned prefix, both first response fields become 1 - last valid response;
all remaining entries are unchanged. Preserve that behavior, not an intended
but different all-position flip. Only the auxiliary training view uses it.
"""
from __future__ import annotations

import torch


AUTHOR_AUGMENTATION = "ukt-author-first-two-from-last-86ad9d8-v1"
AUTHOR_DATASET_SHA256 = "e6af191d9e814db82d5c4db63e77cf34ac07ee9cb737ec0980c2e112ac322579"


def author_augmented_responses(batch: dict) -> dict:
    """Return a shallow copy with the exact author auxiliary response fields.

    Require the ordinary, nonempty, contiguous prefix assumed by that loader.
    Inputs are already multiplied by adjacent-position masks. No labels or
    tensors are changed in place, no statistics use other students/folds, and
    no RNG is consumed. This is not random response-noise augmentation.
    """
    mask = batch["masks"]
    response, shifted = batch["rseqs"], batch["shft_rseqs"]
    if (mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] == 0
            or response.shape != mask.shape or shifted.shape != mask.shape
            or response.device != mask.device or shifted.device != mask.device):
        raise ValueError("UKT augmentation requires aligned batched responses and boolean masks")
    lengths = mask.sum(1)
    if bool((lengths == 0).any()) or not torch.equal(
        mask, torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    ):
        raise ValueError("UKT author augmentation requires nonempty left-aligned valid prefixes")
    for values in (response, shifted):
        if (not bool(torch.isfinite(values).all()) or bool(((values != 0) & (values != 1)).any())
                or bool(values[~mask].ne(0).any())):
            raise ValueError("UKT author augmentation requires binary responses and zero masked padding")
    last = shifted.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    original_aug, shifted_aug = response.clone(), shifted.clone()
    original_aug[:, 0] = 1 - last
    shifted_aug[:, 0] = 1 - last
    return {**batch, "r_aug": original_aug, "shft_r_aug": shifted_aug}


def training_recipe(model) -> str:
    if not model.use_CL:
        return "ukt-no-cl"
    return AUTHOR_AUGMENTATION if model.use_uncertainty_aug else "ukt-cl-same-responses"


def forward_ukt(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-position probabilities plus the unchanged, weighted author NCE.

    The engine adds masked shifted BCE. Its weight decay, shuffle and clipping
    remain explicit run choices; they are not silently set to author defaults.
    Evaluation never constructs the auxiliary view or invokes training loss.
    """
    if not model.training:
        probabilities = model(batch, train=False)
        return probabilities, probabilities.new_zeros(())
    supplied = (author_augmented_responses(batch)
                if model.use_CL and model.use_uncertainty_aug else batch)
    output = model(supplied, train=True)
    probabilities = output[0]
    regularizer = model.cl_weight * output[1] if model.use_CL else probabilities.new_zeros(())
    return probabilities, regularizer
