import json
import logging
import os
import os.path
from typing import List, Dict, Any, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_images(
    image_urls: List[Dict[str, str]],
    output_folder: str,
    aspects: Optional[List[str]] = None,
) -> None:
    logger.debug(f"Create output_folder '{output_folder}'")
    os.makedirs(output_folder, exist_ok=True)
    meta = _get_meta(output_folder)
    default_labels = {} if aspects is None else {aspect: None for aspect in aspects}

    logger.debug(f"Download {len(image_urls)} files")
    for i, image_info in enumerate(tqdm(image_urls), start=len(meta)):
        image_url = image_info["image_url"]
        logger.debug(f"Load from '{image_url}'")
        response = requests.get(image_url, stream=True)

        file_name = f"{i:05}.jpg"
        output_path = os.path.join(output_folder, file_name)
        _write_image(response, output_path)

        image_info["labels"] = default_labels
        meta[file_name] = image_info

    _write_meta(meta, output_folder)


def _write_image(response: requests.Response, output_path: str) -> None:
    logger.debug(f"Write to '{output_path}'")
    with open(output_path, mode="wb") as f:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)


def _get_meta(data_path: str) -> Dict[str, Dict[str, Any]]:
    try:
        meta = load_meta(data_path)
    except FileNotFoundError:
        logger.debug(f"Create new meta because meta.json not found in {data_path}")
        meta = {}

    return meta


def _write_meta(meta: Dict[str, Any], output_folder: str) -> None:
    if meta:
        save_meta(output_folder, meta)
    else:
        logger.debug("Skip writing empty meta.json")


def load_annotations(data_path: str) -> List[Dict[str, Any]]:
    with open(os.path.join(data_path, "annotations.json"), mode="rt") as f:
        anno = json.load(f)
    logger.debug(f"Loaded meta.json from {data_path}")

    return anno


def save_annotations(data_path: str, annotations: List[Dict[str, Any]]) -> None:
    with open(os.path.join(data_path, "annotations.json"), mode="wt") as f:
        json.dump(annotations, f)
    logger.debug(f"Saved annotations.json to {data_path}")


def load_meta(data_path: str) -> Dict[str, Dict[str, Any]]:
    with open(os.path.join(data_path, "meta.json"), mode="rt") as f:
        meta = json.load(f)
    logger.debug(f"Loaded meta.json from {data_path}")

    return meta


def save_meta(data_path: str, meta: Any) -> None:
    with open(os.path.join(data_path, "meta.json"), mode="wt") as f:
        json.dump(meta, f, indent=4)
    logger.debug(f"Saved meta.json to {data_path}")
