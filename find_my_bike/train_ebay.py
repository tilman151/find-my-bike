import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="train_ebay")
def train_ebay(config: DictConfig):
    dm = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model, head={"aspects": dm.class_names})
    trainer = hydra.utils.instantiate(config.trainer)
    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    train_ebay()
