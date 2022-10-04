import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import trainers.simple_trainer


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def run(cfg: DictConfig):
    trainer = instantiate(cfg.trainer, cfg=cfg)
    return trainer.train()


if __name__ == "__main__":
    run()
