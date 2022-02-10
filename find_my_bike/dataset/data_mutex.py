import itertools
import os
import json
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def data_mutex(path_a: str, path_b: str) -> None:
    meta_a = load_meta(path_a)
    meta_b = load_meta(path_b)
    product = itertools.product(meta_a.keys(), meta_b.keys())
    self_check = path_a == path_b
    duplicates = []
    for img_file_a, img_file_b in tqdm(product, total=len(meta_a) * len(meta_b)):
        if self_check and (img_file_a == img_file_b):
            continue
        img_a = _load_image(path_a, img_file_a)
        img_b = _load_image(path_b, img_file_b)
        if (img_a.shape == img_b.shape) and (np.sum(img_a - img_b) == 0):
            duplicates.append(f"{path_a}: {img_file_a} is {path_b}: {img_file_b}")

    print(*duplicates, sep="\n")


@lru_cache(maxsize=1000)
def _load_image(path, img_file_name):
    img = Image.open(os.path.join(path, img_file_name))
    img = np.array(img)

    return img


def load_meta(data_path: str) -> Any:
    with open(os.path.join(data_path, "meta.json"), mode="rt") as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check two datasets for mutual exclusivity."
    )
    parser.add_argument("path_a", help="Path to first dataset")
    parser.add_argument("path_b", help="Path to second dataset")
    opt = parser.parse_args()

    data_mutex(opt.path_a, opt.path_b)
