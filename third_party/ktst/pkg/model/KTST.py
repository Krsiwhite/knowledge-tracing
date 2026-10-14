import logging
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pkg.model import BaseModel
from pkg.model.utils.attentions import BaseMultiheadAttention
from pkg.model.utils.PositionalEncoding import PositionalEncoding

logger: logging.Logger = logging.getLogger(__name__)

AGGREGATION_VARIANTS = [
    "q_mean_c",
    "self_attn_all",
]

ATTENTION_VARIANTS = [
    "standard",
    "akt_monotonic",
    "alibi_monotonic",
    "alibi_monotonic_q_k",
    "learnable_alibi_monotonic",
    "learnable_alibi_monotonic_q_k",
    "learnable_alibi_monotonic_shared_q_k",
]


class Model(BaseModel):
    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        max_concepts: int,
        max_len: int,
        d_model: int,
        nhead_tf: int,
        nhead_agg: int,
        num_layers_tf_dec: int,
        num_layers_tf_enc: int,
        num_layers_agg: int,
        dim_feedforward: int,
        dropout: float,
        dim_classifier: int,
        aggregation: str,
        use_bias_emb: bool,
        use_zero_init: bool,
        use_decoder_only: bool,
        attn_variant: str,
        bias_before_agg: bool = False,
    ):
        super().__init__()

        assert aggregation in AGGREGATION_VARIANTS
        assert attn_variant in ATTENTION_VARIANTS

        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.max_concepts = max_concepts
        self.max_len = max_len
        self.aggregation = aggregation
        self.attn_variant = attn_variant
        self.use_bias_emb = use_bias_emb
        self.use_zero_init = use_zero_init
        self.bias_before_agg = bias_before_agg

        # embeddings
        self.q_emb = nn.Embedding(num_questions + 1, d_model)
        self.c_emb = nn.Embedding(num_concepts + 1, d_model)
        self.r_emb = nn.Embedding(2 + 1, d_model)
        if self.use_bias_emb:
            self.b_emb = nn.Embedding(1, d_model)

        # add positional encoding when using standard attention
        if self.attn_variant == "standard":
            self.positional_encoding = PositionalEncoding(
                d_model=d_model, max_len=max_len, dropout=dropout
            )

        # set aggregation
        if self.aggregation == "q_mean_c":
            self.agg = AggregationMean(max_concepts=max_concepts)
        elif self.aggregation == "self_attn_all":
            self.agg = AggregationSelfAttention(
                embed_dim=d_model,
                num_heads=nhead_agg,
                num_layers=num_layers_agg,
                dropout=dropout,
                use_questions=True,
            )

        self.model = Transformer(
            d_model=d_model,
            nheads=nhead_tf,
            num_encoder_layers=num_layers_tf_enc,
            num_decoder_layers=num_layers_tf_dec,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_variant=attn_variant,
            use_decoder_only=use_decoder_only,
        )

        # final classifier
        self.clf = nn.Sequential(
            nn.Linear(d_model + d_model, dim_classifier),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_classifier, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

        if self.use_zero_init:
            self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.constant_(self.q_emb.weight, 0.0)

    def forward(self, data: Tensor, padding_mask: Tensor):
        questions, responses, concepts = self._unpack_data(data, padding_mask)

        # embeddings
        questions_embedded = self.q_emb.forward(questions + 1)
        concepts_embedded = self.c_emb.forward(concepts + 1)
        responses_embedded = self.r_emb.forward(responses + 1)

        if self.bias_before_agg:
            assert self.use_bias_emb == True
            questions_embedded = questions_embedded + self.b_emb.weight[0]

        # aggregation
        x = self.agg.forward(
            mask=(concepts == -1),
            questions_embedded=questions_embedded,
            concepts_embedded=concepts_embedded,
        )
        y = x + responses_embedded

        if (not self.bias_before_agg) and self.use_bias_emb:
            x = x + self.b_emb.weight[0]
            y = y + self.b_emb.weight[0]

        if self.attn_variant == "standard":
            x = self.positional_encoding(x)
            y = self.positional_encoding(y)

        x_tilde = self.model.forward(tgt=x, src=y)

        clf_input = torch.cat([x_tilde, x], dim=-1)
        logits = self.clf.forward(clf_input).squeeze(-1)

        loss = self.loss_fn(
            torch.masked_select(logits[:, 1:], padding_mask[:, 1:]),
            torch.masked_select(
                responses[:, 1:].type(dtype=torch.float32), padding_mask[:, 1:]
            ),
        )

        return torch.sigmoid(logits), loss, 0.0

    def _unpack_data(
        self, data: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        data = data.clone()
        data[~padding_mask] = 0  # change padded entries to `0` (our default is `-1`)

        questions = data[:, :, -2:-1].type(dtype=torch.int64).squeeze(dim=-1)
        responses = data[:, :, -1:].type(dtype=torch.int64).squeeze(dim=-1)
        concepts = data[:, :, : self.max_concepts].type(dtype=torch.int64)

        return questions, responses, concepts


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nheads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        attn_variant: str,
        use_decoder_only: bool,
    ):
        super().__init__()

        self.use_decoder_only = use_decoder_only

        if not self.use_decoder_only:
            self.encoder = TransformerEncoder(
                d_model=d_model,
                num_layers=num_encoder_layers,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                attn_variant=attn_variant,
            )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_layers=num_decoder_layers,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_variant=attn_variant,
        )

    def forward(self, tgt: Tensor, src: Tensor) -> Tensor:

        B, S, d = tgt.shape
        assert src.shape == (B, S, d)

        attn_mask = self._generate_autoregressive_attn_mask(S=S, device=src.device)

        if self.use_decoder_only:
            mem = self._shift_with_start_token(src)
        else:
            mem = self.encoder.forward(src=src, attn_mask=attn_mask)
            mem = self._shift_with_start_token(mem)

        tgt = self.decoder.forward(tgt=tgt, mem=mem, attn_mask=attn_mask)

        return tgt

    def _generate_autoregressive_attn_mask(self, S: int, device: torch.device):
        attn_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=device),
            diagonal=1,
        )
        return attn_mask

    def _shift_with_start_token(self, x: Tensor) -> Tensor:
        B, S, d = x.shape
        mem_start_token = torch.full((B, 1, d), -1, device=x.device)
        x = torch.cat([mem_start_token, x], dim=1)[:, :-1]
        assert x.shape == (B, S, d)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float,
        attn_variant: str,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    nheads=nheads,
                    attn_variant=attn_variant,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, attn_mask: Tensor) -> Tensor:
        for layer in self.layers:
            src = layer(src=src, attn_mask=attn_mask)
        return src


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float,
        attn_variant: str,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    nheads=nheads,
                    attn_variant=attn_variant,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt: Tensor, mem: Tensor, attn_mask: Tensor) -> Tensor:
        for layer in self.layers:
            tgt = layer(tgt=tgt, mem=mem, attn_mask=attn_mask)
        return tgt


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float,
        attn_variant: str,
    ) -> None:
        super().__init__()

        self.self_attn = BaseMultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            attn_variant=attn_variant,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src: Tensor, attn_mask: Tensor) -> Tensor:

        src = src + self._sa_block(x=src, attn_mask=attn_mask)
        src = self.norm1(src)

        src = src + self._ff_block(x=src)
        src = self.norm2(src)

        return src

    def _sa_block(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        x, *_ = self.self_attn(query=x, key=x, value=x, attn_mask=attn_mask)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nheads: int,
        dropout: float,
        attn_variant: str,
    ):
        super().__init__()

        self.is_std_key = attn_variant not in (
            "akt_monotonic"
        ) and not attn_variant.endswith("q_k")

        self.self_attn = BaseMultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            attn_variant=attn_variant,
        )
        self.cross_attn = BaseMultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            attn_variant=attn_variant,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, tgt: Tensor, mem: Tensor, attn_mask: Tensor) -> Tensor:

        tgt = tgt + self._sa_block(tgt=tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt)

        tgt = tgt + self._ca_block(tgt=tgt, mem=mem, attn_mask=attn_mask)
        tgt = self.norm2(tgt)

        tgt = tgt + self._ff_block(tgt)
        tgt = self.norm3(tgt)

        return tgt

    def _sa_block(self, tgt: Tensor, attn_mask: Tensor | None) -> Tensor:
        x, *_ = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=attn_mask)
        return self.dropout1(x)

    def _ca_block(self, tgt: Tensor, mem: Tensor, attn_mask: Tensor | None) -> Tensor:
        tgt, *_ = self.cross_attn(
            query=tgt,
            key=mem if self.is_std_key else tgt,
            value=mem,
            attn_mask=attn_mask,
        )
        return self.dropout2(tgt)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class AggregationMean(nn.Module):
    def __init__(self, max_concepts: int) -> None:
        super().__init__()
        self.max_concepts = max_concepts

    def forward(
        self, mask: Tensor, questions_embedded: Tensor, concepts_embedded: Tensor
    ) -> Tensor:
        if self.max_concepts == 1:
            concepts_embedded = concepts_embedded.squeeze(dim=2)
            assert concepts_embedded.shape == questions_embedded.shape

            return questions_embedded + concepts_embedded
        else:
            concepts_embedded[mask] = 0
            denominator = (~mask).sum(dim=2).unsqueeze(-1)
            concepts_embedded_mean = concepts_embedded.sum(dim=2) / denominator

            return questions_embedded + concepts_embedded_mean


class AggregationWeightedMean(nn.Module):

    def __init__(self, embed_dim: int, max_concepts: int) -> None:
        super().__init__()
        self.max_concepts = max_concepts

        self.learnable_query = nn.Embedding(num_embeddings=1, embedding_dim=embed_dim)
        torch.nn.init.constant_(self.learnable_query.weight, 0.0)

    def forward(
        self, mask: Tensor, questions_embedded: Tensor, concepts_embedded: Tensor
    ) -> Tensor:
        if self.max_concepts == 1:
            concepts_embedded = concepts_embedded.squeeze(dim=2)
            assert concepts_embedded.shape == questions_embedded.shape

            return questions_embedded + concepts_embedded
        else:
            B, S, f, d = concepts_embedded.shape

            mask = mask.reshape(B * S, f)
            concepts_embedded = concepts_embedded.reshape(B * S, f, d)
            query_token = self.learnable_query.weight.reshape(1, 1, -1).repeat(
                B * S, 1, 1
            )

            assert mask.shape == (B * S, f)
            assert query_token.shape == (B * S, 1, d)

            # mask out irrelevant input
            timestamp_has_entry = torch.any(~mask, dim=1)
            assert (
                timestamp_has_entry == True
            ).all()  # TODO: if so, then this is not needed
            query_token = query_token[timestamp_has_entry]
            concepts_embedded = concepts_embedded[timestamp_has_entry]
            mask = mask[timestamp_has_entry]

            attn_mask = F._canonical_mask(
                mask=mask,
                mask_name="key_padding_mask",
                other_type=None,
                other_name="attn_mask",
                target_type=query_token.dtype,
            )

            assert attn_mask is not None
            assert attn_mask.dim() > 1

            attn_mask = attn_mask.unsqueeze(1)

            q_scaled = query_token / math.sqrt(d)
            attn_output_weights = torch.baddbmm(
                attn_mask, q_scaled, concepts_embedded.transpose(-2, -1)
            )
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)

            output = torch.zeros((B * S, d), device=concepts_embedded.device)
            output[timestamp_has_entry] = torch.bmm(
                attn_output_weights, concepts_embedded
            ).squeeze(1)
            output = output.reshape(B, S, d)

            assert output.shape == (B, S, d)

            return questions_embedded + output


class AggregationSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_questions: bool,
    ) -> None:
        super().__init__()

        self.use_questions = use_questions

        self.learnable_query = nn.Embedding(num_embeddings=1, embedding_dim=embed_dim)
        torch.nn.init.constant_(self.learnable_query.weight, 0.0)

        self.attn_blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, mask: Tensor, concepts_embedded: Tensor, questions_embedded: Tensor
    ) -> Tensor:
        B, S, f, d = concepts_embedded.shape

        mask = mask.reshape(B * S, f)
        concepts_embedded = concepts_embedded.reshape(B * S, f, d)
        questions_embedded = questions_embedded.reshape(B * S, 1, d)
        query_token = self.learnable_query.weight.reshape(1, 1, -1).repeat(B * S, 1, 1)

        if self.use_questions:
            mask_pad = torch.zeros(B * S, 2, dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask_pad, mask], dim=1)
            assert mask.shape == (B * S, f + 2)
            input = torch.cat(
                [query_token, questions_embedded, concepts_embedded], dim=1
            )
            assert input.shape == (B * S, f + 2, d)
        else:
            mask_pad = torch.zeros(B * S, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask_pad, mask], dim=1)
            assert mask.shape == (B * S, f + 1)
            input = torch.cat([query_token, concepts_embedded], dim=1)
            assert input.shape == (B * S, f + 1, d)

        # mask out irrelevant input
        timestamp_has_entry = torch.any(~mask, dim=1)
        assert (
            timestamp_has_entry == True
        ).all()  # TODO: if so, then this is not needed
        input = input[timestamp_has_entry]
        mask = mask[timestamp_has_entry]

        for block in self.attn_blocks:
            input, _ = block.forward(
                query=input,
                key=input,
                value=input,
                key_padding_mask=mask,
                need_weights=False,
            )

        output = torch.zeros((B * S, d), device=mask.device)
        output[timestamp_has_entry] = input[:, 0, :]
        return output.reshape(B, S, d)
