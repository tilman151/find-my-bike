import json
import os

import pytest
import responses
from responses import matchers

from find_my_bike.dataset.utils import download_images, save_meta, load_meta


@pytest.fixture
def image_urls():
    image_urls = [
        {"image_url": "https://foo", "url": "https://foo"},
        {"image_url": "https://bar", "url": "https://bar"},
    ]
    for image_url in image_urls:
        responses.add(
            responses.GET,
            image_url["image_url"],
            body="0" * 8,
            match=[matchers.request_kwargs_matcher({"stream": True})],
        )

    return image_urls


@pytest.fixture
def fake_meta(tmpdir, dummy_meta):
    with open(os.path.join(tmpdir, "meta.json"), mode="wt") as f:
        json.dump(dummy_meta, f, default=str)

    return dummy_meta


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


@responses.activate
def test_download_images_known_url_filtering(image_urls, fake_meta, tmpdir):
    fake_meta["00003.jpg"] = image_urls[0]
    with open(os.path.join(tmpdir, "meta.json"), mode="wt") as f:
        json.dump(fake_meta, f)
    download_images(image_urls, tmpdir)
    with open(os.path.join(tmpdir, "meta.json"), mode="rt") as f:
        meta = json.load(f)
    assert len(fake_meta) + 1 == len(meta)
    assert "00004.jpg" in meta
    assert meta["00004.jpg"]["url"] == image_urls[1]["url"]


def test_save_and_load_dates(tmpdir, dummy_meta):
    save_meta(tmpdir, dummy_meta)
    loaded_meta = load_meta(tmpdir)

    assert dummy_meta == loaded_meta


def test_save_meta_type_safe(tmpdir, dummy_meta):
    dummy_meta["00000.jpg"]["foo"] = lambda x: x  # not serializable
    with pytest.raises(TypeError):
        save_meta(tmpdir, dummy_meta)
