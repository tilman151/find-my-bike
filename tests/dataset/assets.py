import copy

import pytest

DUMMY_META_JSON = {
    "00000.jpg": {
        "image_url": "img_url_0",
        "url": "url_0",
        "labels": {"label_0": "0", "label_1": "2"},
    },
    "00001.jpg": {
        "image_url": "img_url_1",
        "url": "url_1",
        "labels": {"label_0": "1", "label_1": "1"},
    },
    "00002.jpg": {
        "image_url": "img_url_2",
        "url": "url_2",
        "labels": {"label_0": "2", "label_1": None},
    },
}


@pytest.fixture
def dummy_meta():
    return copy.deepcopy(DUMMY_META_JSON)
