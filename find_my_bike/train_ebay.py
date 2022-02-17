import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from find_my_bike.dataset import EbayDataModule
from find_my_bike.lightning import BikeClassifier


@hydra.main(config_path="config", config_name="train_ebay")
def train_ebay(config: DictConfig):
    dm: EbayDataModule = hydra.utils.instantiate(config.data)
    model: BikeClassifier = hydra.utils.instantiate(
        config.model, head={"aspects": dm.class_names}
    )
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer)
    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    train_ebay()
