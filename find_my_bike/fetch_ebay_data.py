import os

import hydra
from omegaconf import DictConfig

from find_my_bike.dataset.utils import download_images
from find_my_bike.scrape.ebay import EbayImageScraper

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


@hydra.main(config_path="config", config_name="ebay_data")
def fetch_ebay_val_data(config: DictConfig) -> None:
    num_images = config.num_train_images + config.num_val_images
    with EbayImageScraper() as scraper:
        image_urls = scraper.get_items(config.query, config.location, num_images)
    dataset_folder = os.path.join(DATA_ROOT, config.dataset_name)
    download_images(image_urls[: config.num_train_images], dataset_folder + "_train")
    download_images(image_urls[config.num_train_images :], dataset_folder + "_val")


if __name__ == "__main__":
    fetch_ebay_val_data()
