import os
import json
from typing import Dict, Any, List


def labelstudio2meta(
    meta: Dict[str, Dict[str, Any]], annotations: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    for anno_info in annotations:
        image_file = os.path.basename(anno_info["image"])
        label = anno_info["bike"].lower().replace(" ", "_")
        if not label == meta[image_file]["labels"]["bike"]:
            print(
                f"Image {image_file} differs: Meta({meta[image_file]['labels']['bike']}) vs. LS({label})"
            )
        meta[image_file]["labels"]["bike"] = label

    return meta


def load_json(data_path: str) -> Any:
    with open(os.path.join(data_path, "annotations.json"), mode="rt") as f:
        anno = json.load(f)
    with open(os.path.join(data_path, "meta.json"), mode="rt") as f:
        meta = json.load(f)

    return meta, anno


def save_json(data_path: str, data: Any) -> None:
    with open(os.path.join(data_path, "meta.json"), mode="wt") as f:
        json.dump(data, f, indent=4)


def main(data_path: str) -> None:
    meta, anno = load_json(data_path)
    meta = labelstudio2meta(meta, anno)
    save_json(data_path, meta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert meta JSON to Label Studio format."
    )
    parser.add_argument("data_path", help="Path to dataset")
    opt = parser.parse_args()

    main(opt.data_path)
