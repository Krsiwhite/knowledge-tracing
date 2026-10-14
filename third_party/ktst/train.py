import logging
from dataclasses import dataclass
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from pkg.data import BaseData
from pkg.logic.trainer import Trainer
from pkg.model import BaseModel
from pkg.utils.logging import save_snapshot_of_source_code
from pkg.utils.reproduce import save_config, seed_everything
from pkg.utils.setup import set_device

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrainConfig(DictConfig):
    trainer: Any
    model: Any
    data: Any
    optimizer: Any

    seed: int

    device: str
    debug: bool


@hydra.main(config_path="pkg/config", config_name="train", version_base="1.3")
def main(config: TrainConfig) -> float:

    # config
    config.seed = config.seed  # in case we are using a resolver for `config.seed`
    save_config(config=config)

    # save code
    save_snapshot_of_source_code(file_name=__file__)

    # device
    device = set_device(device=config.device)

    # data
    seed_everything(seed=config.seed)
    data: BaseData = hydra.utils.instantiate(config.data, data_path=config.data_path)

    # model
    seed_everything(seed=config.seed)
    model: BaseModel = hydra.utils.instantiate(
        config.model,
        num_concepts=data.num_concepts,
        num_questions=data.num_questions,
        max_concepts=data.max_concepts,
        max_len=data.max_len,
    )
    model.to(device)

    # optimizer
    if (
        hasattr(model, "attn_variant")
        and model.attn_variant.startswith("learnable")
        and ("theta_lr" in config.optimizer)
    ):
        other_params = (
            p for n, p in model.named_parameters() if not n.endswith("thetas")
        )
        theta_params = (p for n, p in model.named_parameters() if n.endswith("thetas"))
        list_of_optimizer_params = [
            {
                "params": theta_params,
                "lr": config.optimizer.theta_lr,
            },
            {"params": other_params},
        ]
        del config.optimizer.theta_lr
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            config.optimizer,
            list_of_optimizer_params,
        )
    else:
        if "theta_lr" in config.optimizer:
            del config.optimizer.theta_lr
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            config.optimizer,
            params=model.parameters(),
        )

    # dispatch to trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, data=data, model=model, optimizer=optimizer, device=device
    )
    best_value = trainer.run()

    logger.info("Done")

    return best_value


if __name__ == "__main__":
    main()
