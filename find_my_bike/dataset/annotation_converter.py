import os
import json
from typing import Dict, Any, List


def meta2labelstudio(
    data_dir: str, meta: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    annotations = []
    for file_name, file_info in meta.items():
        anno = {
            "data": {
                "image": os.path.join(
                    "/data/local-files/?d=data/ebay_train",
                    file_name,
                )
            },
            "predictions": [
                {
                    "result": [
                        {
                            "id": "result1",
                            "type": "choices",
                            "from_name": "bike",
                            "to_name": "image",
                            "value": {
                                "choices": [
                                    " ".join(
                                        w.capitalize()
                                        for w in file_info["labels"]["bike"].split("_")
                                    )
                                ]
                            },
                        }
                    ]
                }
            ],
        }
        annotations.append(anno)

    return annotations


def load_json(data_path: str) -> Any:
    with open(os.path.join(data_path, "meta.json"), mode="rt") as f:
        data = json.load(f)

    return data


def save_json(data_path: str, data: Any) -> None:
    with open(os.path.join(data_path, "annotations.json"), mode="wt") as f:
        json.dump(data, f)


def main(data_path: str) -> None:
    meta = load_json(data_path)
    converted = meta2labelstudio(data_path, meta)
    save_json(data_path, converted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert meta JSON to Label Studio format."
    )
    parser.add_argument("data_path", help="Path to dataset")
    opt = parser.parse_args()

    main(opt.data_path)
