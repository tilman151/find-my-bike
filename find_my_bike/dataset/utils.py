import json
import logging
import os
import os.path
from datetime import date
from typing import List, Dict, Any, Optional, Tuple

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
    image_urls = _filter_known_urls(image_urls, meta)
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


def _filter_known_urls(image_urls, meta):
    if meta:
        known_urls = {image_info["url"] for image_info in meta.values()}
        filtered_urls = [url for url in image_urls if url["url"] not in known_urls]
        logger.info(f"Filtered out {len(image_urls) - len(filtered_urls)} known urls.")
    else:
        filtered_urls = image_urls

    return filtered_urls


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


def download_high_res_images(meta: Dict[str, Dict[str, Any]], output_folder: str):
    for image_file, image_info in tqdm(meta.items()):
        high_res_url = image_info["image_url"].replace("$_2.JPG", "$_59.JPG")
        high_res_file = image_file.replace(".jpg", "_highres.jpg")
        high_res_path = os.path.join(output_folder, high_res_file)
        response = requests.get(high_res_url, stream=True)
        _write_image(response, high_res_path)


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
        meta = json.load(f, object_pairs_hook=_json_object_pairs_hook)

    logger.debug(f"Loaded meta.json from {data_path}")

    return meta


def save_meta(data_path: str, meta: Any) -> None:
    with open(os.path.join(data_path, "meta.json"), mode="wt") as f:
        json.dump(meta, f, indent=4, default=_json_default)

    logger.debug(f"Saved meta.json to {data_path}")


def load_image_urls(load_path: str) -> List[Dict[str, Any]]:
    with open(os.path.join(load_path, "image_urls.json"), mode="rt") as f:
        image_urls = json.load(f, object_pairs_hook=_json_object_pairs_hook)

    return image_urls


def save_image_urls(save_path: str, image_urls: List[Dict[str, Any]]) -> None:
    with open(os.path.join(save_path, "image_urls.json"), mode="wt") as f:
        json.dump(image_urls, f, indent=4, default=_json_default)


def _json_object_pairs_hook(results: List[Tuple[Any, Any]]) -> Dict[str, Any]:
    return {
        key: date.fromisoformat(value) if key == "date" else value
        for key, value in results
    }


def _json_default(o: Any) -> str:
    if isinstance(o, date):
        return str(o)
    else:
        raise TypeError(f"Object of type {type(o)} not serializable.")
