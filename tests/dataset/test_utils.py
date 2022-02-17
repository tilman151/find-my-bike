import json
import os

import pytest
import responses
from responses import matchers

from find_my_bike.dataset.utils import download_images
from tests.dataset.assets import DUMMY_META_JSON


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


@pytest.fixture
def fake_meta(tmpdir):
    with open(os.path.join(tmpdir, "meta.json"), mode="wt") as f:
        json.dump(DUMMY_META_JSON, f)

    return DUMMY_META_JSON


@responses.activate
def test_download_images(image_urls, tmpdir):
    download_images(image_urls, tmpdir)
    assert ["00000.jpg", "00001.jpg", "meta.json"] == sorted(os.listdir(tmpdir))
    with open(os.path.join(tmpdir, "meta.json"), mode="rt") as f:
        meta = json.load(f)
    assert 2 == len(meta)
    assert "00000.jpg" in meta
    assert "00001.jpg" in meta
    assert "labels" in meta["00000.jpg"]
    assert isinstance(meta["00000.jpg"]["labels"], dict)
    assert not meta["00000.jpg"]["labels"]


@responses.activate
def test_download_images_with_aspects(image_urls, tmpdir):
    download_images(image_urls, tmpdir, aspects=["a", "b"])
    with open(os.path.join(tmpdir, "meta.json"), mode="rt") as f:
        meta = json.load(f)
    assert "00000.jpg" in meta
    assert "labels" in meta["00000.jpg"]
    assert isinstance(meta["00000.jpg"]["labels"], dict)
    assert {"a": None, "b": None} == meta["00000.jpg"]["labels"]


def test_download_images_empty_list(tmpdir):
    download_images([], tmpdir)
    assert not os.listdir(tmpdir)


@responses.activate
def test_download_images_existing_meta(image_urls, fake_meta, tmpdir):
    download_images(image_urls, tmpdir)
    assert sorted(os.listdir(tmpdir)) == ["00003.jpg", "00004.jpg", "meta.json"]
    with open(os.path.join(tmpdir, "meta.json"), mode="rt") as f:
        meta = json.load(f)
    assert len(fake_meta) + 2 == len(meta)
    assert "00003.jpg" in meta
    assert "00004.jpg" in meta
