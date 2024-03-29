from io import BytesIO

import pytest
import responses
import torch
from PIL import Image
from responses import matchers

from find_my_bike.dataset.prediction_dataset import PredictionDataset


@pytest.fixture
def image_urls(dummy_image_urls):
    for image_url in dummy_image_urls:
        responses.add(
            responses.GET,
            image_url["image_url"],
            body=_dummy_image_bytes(),
            match=[matchers.request_kwargs_matcher({"stream": True})],
        )

    return dummy_image_urls


def _dummy_image_bytes():
    img = Image.new("RGB", (100, 200), color="white")
    f = BytesIO()
    img.save(f, format="jpeg")
    f.seek(0)

    return f.read()


@responses.activate
def test_iteration(tmpdir, monkeypatch, image_urls):
    monkeypatch.setattr(
        "find_my_bike.dataset.utils.load_image_urls", lambda _: image_urls
    )
    dataset = PredictionDataset(tmpdir, high_res=500)
    white = torch.ones(3)
    for i, (img, image_info) in enumerate(dataset):
        assert img.shape == (3, 500, 500)
        # center is white so image was requested correctly
        assert torch.dist(img[:, 255, 255], white) == 0.0
        assert image_info == image_urls[i]


def test_length(tmpdir, monkeypatch, image_urls):
    monkeypatch.setattr(
        "find_my_bike.dataset.utils.load_image_urls", lambda _: image_urls
    )
    dataset = PredictionDataset(tmpdir, high_res=500)
    assert len(dataset) == len(image_urls)
