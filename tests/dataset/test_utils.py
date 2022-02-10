import os
from unittest import mock

import pytest
import responses
from responses import matchers

from find_my_bike.dataset.utils import download_images


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
def test_download_images(image_urls, tmpdir):
    mock_open = mock.mock_open()
    with mock.patch("find_my_bike.dataset.image_dataset.open", new=mock_open):
        download_images(image_urls, os.path.join(tmpdir, "data"))
    mock_open.assert_has_calls(
        [
            mock.call(os.path.join(tmpdir, "data", "00000.jpg"), mode="wb"),
            mock.call(os.path.join(tmpdir, "data", "00001.jpg"), mode="wb"),
            mock.call(os.path.join(tmpdir, "data", "meta.json"), mode="wt"),
        ],
        any_order=True,
    )


def test_download_images_empty_list(tmpdir):
    mock_open = mock.mock_open()
    with mock.patch("find_my_bike.dataset.image_dataset.open", new=mock_open):
        download_images([], os.path.join(tmpdir, "data"))
    mock_open.assert_not_called()
