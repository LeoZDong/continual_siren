import random

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def run(cfg: DictConfig):
    # Set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    trainer = instantiate(cfg.trainer, cfg=cfg)
    return trainer.train()


if __name__ == "__main__":
    run()
