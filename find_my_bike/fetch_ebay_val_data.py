import os

import hydra
from omegaconf import DictConfig

from find_my_bike.dataset.image_dataset import download_images
from find_my_bike.scrape.ebay import EbayImageScraper

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@hydra.main(config_path="config", config_name="ebay_val_data")
def fetch_ebay_val_data(config: DictConfig) -> None:
    with EbayImageScraper() as scraper:
        image_urls = scraper.get_items("Fahrrad", "Berlin", config.num_images)
    dataset_folder = os.path.join(DATA_ROOT, config.dataset_name)
    download_images(image_urls, dataset_folder)


if __name__ == "__main__":
    fetch_ebay_val_data()
