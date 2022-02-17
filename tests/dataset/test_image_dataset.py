from unittest import mock

import pytest
import torch
from PIL import Image
from torchvision import transforms

from find_my_bike.dataset.image_dataset import EbayDataset, EbayDataModule, UnifyingPad


@pytest.fixture
def patch_load_meta(monkeypatch, dummy_meta):
    def _load_meta(*args, **kwargs):
        return dummy_meta

    monkeypatch.setattr("find_my_bike.dataset.utils.load_meta", _load_meta)


@pytest.fixture
def patch_pil_loader(monkeypatch):
    def _pil_loader(*args, **kwargs):
        return Image.new("RGB", (10, 10))

    monkeypatch.setattr("find_my_bike.dataset.image_dataset.pil_loader", _pil_loader)


def test_meta_file_loading(patch_load_meta):
    dataset = EbayDataset("foo/bar", ["label_1", "label_0"])
    assert len(dataset.meta) == 3
    assert "label_0" in dataset._classes
    assert "label_1" in dataset._classes
    assert dataset._classes["label_0"] == {"0": 0, "1": 1, "2": 2}
    assert dataset._classes["label_1"] == {"1": 0, "2": 1}


def test_get_item(patch_load_meta, patch_pil_loader):
    dataset = EbayDataset("foo/bar", ["label_1", "label_0"])
    img, labels = dataset[0]
    assert torch.dist(torch.zeros(1, 200, 200), img) == 0  # Padded to 200x200
    assert torch.all(torch.tensor([1, 0]) == labels)  # Order is label_1 then label_0

    img, labels = dataset[2]
    assert torch.dist(torch.zeros(1, 200, 200), img) == 0  # Padded to 200x200
    assert torch.all(torch.tensor([-1, 2]) == labels)  # Unannotated aspect is -1


def test_default_transform(patch_load_meta):
    dataset = EbayDataset("foo/bar", ["label_1", "label_0"])
    _assert_last_transforms_default(dataset.transform)
    assert 2 == len(dataset.transform.transforms)


def test_custom_transform(patch_load_meta):
    dataset = EbayDataset(
        "foo/bar", ["label_1", "label_0"], transform=[transforms.CenterCrop(100)]
    )
    _assert_last_transforms_default(dataset.transform)
    assert isinstance(dataset.transform.transforms[0], transforms.CenterCrop)


def _assert_last_transforms_default(transform):
    assert isinstance(transform, transforms.Compose)
    transform_list = transform.transforms
    assert isinstance(transform_list[-2], UnifyingPad)
    assert (200, 200) == transform_list[-2].size
    assert isinstance(transform_list[-1], transforms.ToTensor)


@mock.patch("find_my_bike.dataset.image_dataset.EbayDataset")
def test_datamodule_creation(mock_dataset):
    dm = EbayDataModule("foo/bar", ["a", "b", "c"], 64)
    mock_dataset.assert_has_calls(
        [
            mock.call("foo/bar_train", ["a", "b", "c"], None),
            mock.call("foo/bar_val", ["a", "b", "c"]),
        ]
    )
    assert dm.hparams == {
        "dataset_path": "foo/bar",
        "aspects": ["a", "b", "c"],
        "batch_size": 64,
        "num_workers": 4,
    }


@mock.patch("find_my_bike.dataset.image_dataset.EbayDataset")
@mock.patch("find_my_bike.dataset.image_dataset.DataLoader")
def test_datamodule_loaders(mock_loader, mock_dataset):
    dm = EbayDataModule("foo/bar", ["a", "b", "c"], 64)
    train_loader = dm.train_dataloader()
    mock_loader.assert_called_with(
        mock_dataset(), batch_size=64, shuffle=True, pin_memory=True, num_workers=4
    )
    assert train_loader is mock_loader()


@mock.patch("find_my_bike.dataset.image_dataset.EbayDataset")
@mock.patch("find_my_bike.dataset.image_dataset.DataLoader")
def test_datamodule_loaders(mock_loader, mock_dataset):
    dm = EbayDataModule("foo/bar", ["a", "b", "c"], 64)
    val_loader = dm.val_dataloader()
    mock_loader.assert_called_with(
        mock_dataset(), batch_size=64, pin_memory=True, num_workers=4
    )
    assert val_loader is mock_loader()
