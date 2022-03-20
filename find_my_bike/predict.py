import logging
from typing import Any, Dict, List

import hydra
import torch.jit
from omegaconf import DictConfig

from find_my_bike.dataset.utils import save_image_urls

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="predict")
def predict(config: DictConfig) -> None:
    logger.info(f"Create dataset from {config.data.dataset.dataset_path}")
    dataloader = hydra.utils.instantiate(config.data)
    logger.info(f"Create model from {config.model.f}")
    model = hydra.utils.instantiate(config.model)
    model.eval()

    logger.info(f"Predict on {len(dataloader)} batches")
    predictions = sum(
        (predict_batch(*batch, model, i) for i, batch in enumerate(dataloader)), []
    )
    save_image_urls(config.data.dataset.dataset_path, predictions)


@torch.no_grad()
def predict_batch(
    imgs: torch.Tensor,
    image_infos: List[Dict[str, Any]],
    model: torch.jit.ScriptModule,
    batch_idx: int,
) -> List[Dict[str, Any]]:
    logger.info(f"Predict batch {batch_idx}")
    preds = model.predict(imgs)
    for info, pred in zip(image_infos, preds):
        info["prediction"] = pred

    return image_infos


if __name__ == "__main__":
    predict()
