import os
from typing import Dict, Any, List

from find_my_bike.dataset import utils


def meta2labelstudio(meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def main(data_path: str) -> None:
    meta = utils.load_meta(data_path)
    converted = meta2labelstudio(meta)
    utils.save_annotations(data_path, converted)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert meta JSON to Label Studio format."
    )
    parser.add_argument("data_path", help="Path to dataset")
    opt = parser.parse_args()

    main(opt.data_path)
