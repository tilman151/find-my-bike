import json
import os
from typing import Dict, Any, List, Tuple

_LS_INFO_KEYS = {
    "image",
    "id",
    "annotator",
    "annotation_id",
    "created_at",
    "updated_at",
    "lead_time",
}


def labelstudio2meta(
    meta: Dict[str, Dict[str, Any]], annotations: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    for anno_info in annotations:
        image_file = os.path.basename(anno_info["image"])
        for aspect in set(anno_info.keys()).difference(_LS_INFO_KEYS):
            new_label = _norm_label(anno_info[aspect])
            _set_aspect_label(aspect, new_label, image_file, meta)

    return meta


def _norm_label(label: str) -> str:
    return label.lower().replace(" ", "_")


def _set_aspect_label(
    aspect: str, new_label: str, image_file: str, meta: Dict[str, Dict[str, Any]]
) -> None:
    image_labels = meta[image_file]["labels"]
    old_label = image_labels[aspect]
    if not new_label == old_label:
        print(
            f"Image {image_file} differs in '{aspect}':"
            f"Meta({old_label}) vs. LS({new_label})"
        )
        image_labels[aspect] = new_label


def load_json(data_path: str) -> Tuple[Any, Any]:
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
        description="Convert Label Studio annotations to meta JSON."
    )
    parser.add_argument("data_path", help="Path to dataset")
    opt = parser.parse_args()

    main(opt.data_path)
