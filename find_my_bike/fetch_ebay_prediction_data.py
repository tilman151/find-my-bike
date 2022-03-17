import logging
import os.path
from datetime import date, timedelta

import hydra
from omegaconf import DictConfig

from find_my_bike.dataset.utils import save_image_urls
from find_my_bike.scrape import EbayImageScraper

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="ebay_prediction_data")
def fetch_ebay_prediction_data(config: DictConfig) -> None:
    assert config.download_path
    yesterday = date.today() - timedelta(days=1)
    with EbayImageScraper(high_res=True) as scraper:
        image_urls = scraper.get_items(
            config.query, config.location, config.category, until=yesterday
        )

    for item in image_urls:
        item["prediction"] = {aspect: None for aspect in config.aspects}

    os.makedirs(config.download_path, exist_ok=True)
    save_image_urls(config.download_path, image_urls)


if __name__ == "__main__":
    fetch_ebay_prediction_data()
