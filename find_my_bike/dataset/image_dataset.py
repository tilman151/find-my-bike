import os.path
from typing import List, Dict
import requests
from tqdm import tqdm

from torch.utils.data import Dataset


def download_images(image_urls: List[Dict[str, str]], output_folder: str) -> None:
    for i, image_info in enumerate(tqdm(image_urls)):
        image_url = image_info["image_url"]
        response = requests.get(image_url, stream=True)
        output_path = os.path.join(output_folder, f"{i:05}.jpg")
        with open(output_path, mode="wb") as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
