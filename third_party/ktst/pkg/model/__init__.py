import logging
from typing import Union
from torch import nn, Tensor
import torch

from pkg.utils.logging import get_checkpoint_path_without_suffix

logger: logging.Logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, data: Tensor, padding_mask: Tensor
    ) -> tuple[Tensor, float, float]:
        raise NotImplementedError

    def restore(self, restore_dir: str, iteration: Union[str, int] = "best") -> None:
        logger.info(f"Restoring model from {restore_dir}")
        checkpoint_path_without_suffix = get_checkpoint_path_without_suffix(
            dir=restore_dir, iteration=iteration
        )
        checkpoint: dict = torch.load(
            checkpoint_path_without_suffix.with_suffix(".pth"),
            map_location=torch.device("cpu"),
        )
        if (model_state_dict := checkpoint.get("model_state_dict")) is None:
            raise ValueError
        else:
            self.load_state_dict(model_state_dict)
