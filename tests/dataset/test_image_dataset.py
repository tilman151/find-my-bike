from unittest import mock

import pytest
import responses
from responses import matchers

from find_my_bike.dataset.image_dataset import download_images


@pytest.fixture
def image_urls():
    image_urls = [{"image_url": "https://foo"}, {"image_url": "https://bar"}]
    responses.add(
        responses.GET,
        "https://foo",
        body="0" * 8,
        match=[matchers.request_kwargs_matcher({"stream": True})],
    )
    responses.add(
        responses.GET,
        "https://bar",
        body="0" * 8,
        match=[matchers.request_kwargs_matcher({"stream": True})],
    )

    return image_urls


@responses.activate
def test_download_images(image_urls):
    mock_open = mock.mock_open()
    with mock.patch("find_my_bike.dataset.image_dataset.open", new=mock_open):
        download_images(image_urls, "output_folder")
    mock_open.assert_has_calls(
        [
            mock.call("output_folder/00000.jpg", mode="wb"),
            mock.call("output_folder/00001.jpg", mode="wb"),
        ],
        any_order=True,
    )
