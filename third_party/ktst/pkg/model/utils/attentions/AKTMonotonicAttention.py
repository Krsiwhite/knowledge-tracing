import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter


class AKTMonotonicAttention(Module):

    def __init__(self, num_heads: int) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.gamma = Parameter(torch.zeros(self.num_heads, 1, 1))

        xavier_uniform_(self.gamma)

    def forward(
        self,
        attn_output_weights: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size_times_num_head, S, _ = attn_output_weights.shape

        scores = attn_output_weights.reshape(-1, self.num_heads, S, S)
        mask = (attn_mask == 0).unsqueeze(0)

        x1 = torch.arange(S).expand(S, -1).to(attn_output_weights.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)
            scores_ = scores_ * mask.float()
            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = (
                torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(attn_output_weights.device)  # type: ignore
            )
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * nn.Softplus()(self.gamma).unsqueeze(0)
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )
        scores = scores * total_effect
        scores.masked_fill_(mask == 0, -1e32)
        scores = scores.reshape(batch_size_times_num_head, S, S)
        return scores
