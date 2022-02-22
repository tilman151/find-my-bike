import logging
import os

import hydra
from omegaconf import DictConfig

from find_my_bike.dataset import download_images
from find_my_bike.scrape import EbayImageScraper

logger = logging.getLogger(__name__)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


@hydra.main(config_path="config", config_name="ebay_data")
def fetch_ebay_val_data(config: DictConfig) -> None:
    num_images = config.num_train_images + config.num_val_images
    with EbayImageScraper() as scraper:
        image_urls = scraper.get_items(config.query, config.location, num_images)
    dataset_folder = os.path.join(DATA_ROOT, config.dataset_name)

    logger.info("Download training images")
    train_images = image_urls[: config.num_train_images]
    download_images(train_images, dataset_folder + "_train", config.aspects)
    logger.info("Download validation images")
    val_images = image_urls[config.num_train_images :]
    download_images(val_images, dataset_folder + "_val", config.aspects)


if __name__ == "__main__":
    fetch_ebay_val_data()
