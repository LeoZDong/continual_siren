# my_app.py
import hydra
from hydra.utils import instantiate
from torch import nn

model = nn.Linear(3, 3)


@hydra.main(config_path=".", config_name="config")
def app(cfg):
    print(cfg)
    print(instantiate(cfg.optimizer, params=model.parameters()))


if __name__ == "__main__":
    app()
