from unittest import mock

import torch
from PIL import Image

from find_my_bike.dataset.image_dataset import EbayDataset

DUMMY_META_JSON = """
    {
        "00000.jpg": {
            "image_url": "img_url_0",
            "url": "url_0",
            "labels": {
                "label_0": "0",
                "label_1": "2"
            }
        },
        "00001.jpg": {
            "image_url": "img_url_1",
            "url": "url_1",
            "labels": {
                "label_0": "1",
                "label_1": "1"
            }
        },
        "00002.jpg": {
            "image_url": "img_url_2",
            "url": "url_2",
            "labels": {
                "label_0": "2",
                "label_1": "0"
            }
        }
    }
"""


def test_meta_file_loading():
    mock_open = mock.mock_open(read_data=DUMMY_META_JSON)
    with mock.patch("find_my_bike.dataset.image_dataset.open", new=mock_open):
        dataset = EbayDataset("foo/bar", ["label_1", "label_0"])
    assert len(dataset.meta) == 3
    assert "label_0" in dataset._classes
    assert "label_1" in dataset._classes
    assert dataset._classes["label_0"] == {"0": 0, "1": 1, "2": 2}
    assert dataset._classes["label_1"] == {"0": 0, "1": 1, "2": 2}


@mock.patch(
    "find_my_bike.dataset.image_dataset.pil_loader",
    return_value=Image.new("RGB", (10, 10)),
)
def test_get_item(mock_pil_loader):
    mock_open = mock.mock_open(read_data=DUMMY_META_JSON)
    with mock.patch("find_my_bike.dataset.image_dataset.open", new=mock_open):
        dataset = EbayDataset("foo/bar", ["label_1", "label_0"])
    img, labels = dataset[0]
    mock_pil_loader.assert_called_once_with("foo/bar/00000.jpg")
    assert torch.dist(torch.zeros(1, 10, 10), img) == 0
    assert torch.all(torch.tensor([2, 0]) == labels)
