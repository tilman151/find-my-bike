import warnings
from typing import Any, Dict

from find_my_bike.dataset import utils


def main(data_path: str, aspect: str) -> None:
    meta = utils.load_meta(data_path)
    meta = add_aspect(aspect, meta)
    utils.save_meta(data_path, meta)


def add_aspect(
    aspect: str, meta: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    for img_file in meta.keys():
        if aspect in meta[img_file]["labels"]:
            warnings.warn(f"Aspect '{aspect}' already present for '{img_file}'.")
        else:
            meta[img_file]["labels"][aspect] = None

    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add an aspect to a meta.json")
    parser.add_argument("data_path", help="Path to dataset")
    parser.add_argument("aspect", help="Name of aspect to add")
    opt = parser.parse_args()

    main(opt.data_path, opt.aspect)
