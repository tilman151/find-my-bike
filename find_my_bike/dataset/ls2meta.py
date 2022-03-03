import os
from typing import Dict, Any, List, Union

from find_my_bike.dataset import utils

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
        image_file = os.path.basename(anno_info["image"]).replace("_highres", "")
        for aspect in set(anno_info.keys()).difference(_LS_INFO_KEYS):
            new_label = _norm_label(anno_info[aspect])
            _set_aspect_label(aspect, new_label, image_file, meta)

    return meta


def _norm_label(label: Union[str, Dict[str, List[str]]]) -> Union[str, List[str]]:
    if isinstance(label, str):
        return _norm_string_label(label)
    elif isinstance(label, dict):
        return [_norm_string_label(x) for x in label["choices"]]
    else:
        raise ValueError(f"Unsupported label type {type(label)}")


def _norm_string_label(label: str) -> str:
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


def main(data_path: str) -> None:
    meta = utils.load_meta(data_path)
    anno = utils.load_annotations(data_path)
    meta = labelstudio2meta(meta, anno)
    utils.save_meta(data_path, meta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Label Studio annotations to meta JSON."
    )
    parser.add_argument("data_path", help="Path to dataset")
    opt = parser.parse_args()

    main(opt.data_path)
