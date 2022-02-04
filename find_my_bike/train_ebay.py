import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="train_ebay")
def train_ebay(config: DictConfig):
    dm = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(
        config.model, head={"aspects": dm.classes_per_aspect}
    )
    trainer = pl.Trainer(**config.trainer)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train_ebay()
