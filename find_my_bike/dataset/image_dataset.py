import os
from typing import Tuple, Callable, Dict, List, Any, Optional

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

from find_my_bike.dataset import utils


class EbayDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        aspects: List[str],
        batch_size: int,
        training_transforms: Optional[Callable] = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.aspects = aspects
        self.batch_size = batch_size
        self.training_transforms = training_transforms
        self.num_workers = num_workers

        # TODO: Think of way to log transforms
        self.save_hyperparameters(ignore="training_transforms")

        self.train_data = EbayDataset(
            f"{dataset_path}_train", aspects, training_transforms
        )
        self.val_data = EbayDataset(f"{dataset_path}_val", aspects)

    @property
    def class_names(self) -> Dict[str, List[str]]:
        return self.train_data.class_names

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


class EbayDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        aspects: List[str],
        transform: Optional[List[Callable]] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.aspects = aspects
        self.transform = self._get_transform(transform)

        self.meta = self._load_meta_file()
        self._classes = self._get_classes()

    def _get_transform(self, transform: Optional[List[Callable]]) -> Callable:
        default_transform = [UnifyingPad(200, 200), transforms.ToTensor()]
        if transform is None:
            transform = default_transform
        else:
            transform.extend(default_transform)

        return transforms.Compose(transform)

    @property
    def class_names(self):
        return {
            aspect: list(classes.keys()) for aspect, classes in self._classes.items()
        }

    def _load_meta_file(self) -> List[Tuple[str, dict[str, Any]]]:
        meta = utils.load_meta(self.dataset_path)
        self._verify_meta(meta)
        meta_list = [(k, v) for k, v in meta.items()]

        return meta_list

    def _verify_meta(self, meta: Dict[str, Dict[str, Any]]) -> None:
        aspect_set = set(self.aspects)
        for file_name, entry in meta.items():
            if diff := aspect_set.difference(entry["labels"].keys()):
                raise RuntimeError(f"Image '{file_name}' is missing the aspects {diff}")

    def _get_classes(self) -> Dict[str, Dict[str, int]]:
        aspects = {
            aspect: sorted(
                {
                    entry["labels"][aspect]
                    for _, entry in self.meta
                    if entry["labels"][aspect] is not None
                }
            )
            for aspect in self.aspects
        }
        classes = {
            aspect_name: {cls: i for i, cls in enumerate(aspect)}
            for aspect_name, aspect in aspects.items()
        }

        return classes

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name, image_info = self.meta[index]
        image_path = os.path.join(self.dataset_path, image_name)
        img = pil_loader(image_path)
        img = self.transform(img)

        labels = [self._to_class_idx(aspect, image_info) for aspect in self.aspects]
        labels = torch.tensor(labels, dtype=torch.long)

        return img, labels

    def _to_class_idx(self, aspect, image_info):
        class_name = image_info["labels"][aspect]
        if class_name is None:
            return -1
        else:
            return self._classes[aspect][class_name]

    def __len__(self) -> int:
        return len(self.meta)


class UnifyingPad:
    def __init__(self, x: int, y: int):
        self.size = x, y

    def __call__(self, img: Image) -> Image:
        padded = Image.new("RGB", self.size)
        paste_pos = self.size[0] - img.size[0], self.size[1] - img.size[1]
        padded.paste(img, paste_pos)

        return padded

    def __repr__(self) -> str:
        return f"UnifyingPad({self.size[0]}, {self.size[1]})"
