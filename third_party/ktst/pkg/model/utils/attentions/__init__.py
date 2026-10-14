import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

from pkg.model.utils.attentions.AKTMonotonicAttention import AKTMonotonicAttention
from pkg.model.utils.attentions.ALiBiMonotonicAttention import ALiBiMonotonicAttention
from pkg.model.utils.attentions.LearnableALiBiMonotonicAttention import (
    LearnableALiBiMonotonicAttention,
)


class BaseMultiheadAttention(Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        attn_variant: str = "learnable_alibi_monotonic",
        device=None,
        dtype=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        assert batch_first == True
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.in_proj_weight = Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        self.register_parameter("q_proj_weight", None)
        self.register_parameter("k_proj_weight", None)
        self.register_parameter("v_proj_weight", None)

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=True, **factory_kwargs
        )

        self._reset_parameters()

        if attn_variant == "standard":
            self.attn_variant = None
        elif attn_variant == "akt_monotonic":
            self.attn_variant = AKTMonotonicAttention(num_heads=num_heads)
        elif attn_variant.startswith("alibi_monotonic"):
            self.attn_variant = ALiBiMonotonicAttention(num_heads=num_heads)
        elif attn_variant.startswith("learnable_alibi_monotonic"):
            self.attn_variant = LearnableALiBiMonotonicAttention(
                num_heads=num_heads, use_shared_theta="shared" in attn_variant
            )
        else:
            raise ValueError(f"{attn_variant=} not implemented")
        self.attn_variant_string = attn_variant

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.0)
        constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        need_weights=True,
    ) -> tuple[Tensor, Tensor | None]:

        assert need_weights == True

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # MultiheadAttention does not support NestedTensor outside of its fast path
        assert not (query.is_nested or key.is_nested or value.is_nested)

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_output_weights = base_multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            attn_variant_string=self.attn_variant_string,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            attn_variant=self.attn_variant,
        )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def base_multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    attn_variant_string: str,
    training: bool = True,
    key_padding_mask: Tensor | None = None,
    attn_mask: Tensor | None = None,
    attn_variant: Module | None = None,
) -> tuple[Tensor, Tensor]:

    is_batched = F._mha_shape_check(  # type: ignore
        query, key, value, key_padding_mask, attn_mask, num_heads
    )

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )

    # expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}
    assert embed_dim == embed_dim_to_check
    # embed_dim can be a tensor when JIT tracing
    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    # embed_dim {embed_dim} not divisible by num_heads {num_heads}
    assert head_dim * num_heads == embed_dim
    assert isinstance(head_dim, int)
    # key shape {key.shape} does not match value shape {value.shape}
    assert key.shape == value.shape

    #
    # compute in-projection
    #
    w_q, w_k, w_v = in_proj_weight.chunk(3)
    b_q, b_k, b_v = in_proj_bias.chunk(3)
    if (attn_variant_string == "akt_monotonic") or attn_variant_string.endswith("q_k"):
        assert torch.equal(query, key)
        # use same projection weights for query as for key
        q, k, v = (
            F.linear(query, w_k, b_k),
            F.linear(key, w_k, b_k),
            F.linear(value, w_v, b_v),
        )
    else:
        q, k, v = (
            F.linear(query, w_q, b_q),
            F.linear(key, w_k, b_k),
            F.linear(value, w_v, b_v),
        )

    # prep attention mask
    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.dim()} is not supported"
            )

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        # expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}
        assert key_padding_mask.shape == (bsz, src_len)
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # calculate attention and out projection
    #
    _, _, E = q.shape
    q_scaled = q / math.sqrt(E)

    if attn_variant is not None:
        assert attn_mask is not None
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        # here mask is to be handled by attn_variant because of gradient issues
        attn_output_weights = attn_variant.forward(
            attn_output_weights=attn_output_weights, attn_mask=attn_mask
        )
    else:
        # standard (i.e. unchanged) MHA
        if attn_mask is None:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, k.transpose(-2, -1)
            )

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = (
        attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    # optionally average attention weights over heads
    # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)

    return attn_output, attn_output_weights
