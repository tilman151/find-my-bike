import json
import logging
import os.path
from typing import List, Dict
import requests
from tqdm import tqdm

from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


def download_images(image_urls: List[Dict[str, str]], output_folder: str) -> None:
    logger.debug(f"Create output_folder '{output_folder}'")
    os.makedirs(output_folder)
    meta = {}
    logger.debug(f"Download {len(image_urls)} files")
    for i, image_info in enumerate(tqdm(image_urls)):
        image_url = image_info["image_url"]
        logger.debug(f"Load from '{image_url}'")
        response = requests.get(image_url, stream=True)
        file_name = f"{i:05}.jpg"
        output_path = os.path.join(output_folder, file_name)
        logger.debug(f"Write to '{output_path}'")
        with open(output_path, mode="wb") as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
        meta[file_name] = image_info

    logger.debug("Write meta.json")
    with open(os.path.join(output_folder, "meta.json"), mode="wt") as f:
        json.dump(meta, f, indent=4)
