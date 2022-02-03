import json
import os
from typing import Tuple, Callable, Dict, List, Any, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import functional


class EbayDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        aspects: List[str],
        batch_size: int,
        train_transforms: Optional[Callable] = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.aspects = aspects
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.num_workers = num_workers

        self.save_hyperparameters()

        self.train_data = EbayDataset(
            f"{dataset_path}_train", aspects, train_transforms
        )
        self.val_data = EbayDataset(f"{dataset_path}_val", aspects)

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


class EbayDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        aspects: List[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.aspects = aspects
        self.transform = transform or functional.to_tensor

        self.meta = self._load_meta_file()
        self._classes = self._get_classes()

    def _load_meta_file(self) -> List[Tuple[str, Dict[str, Any]]]:
        meta_path = os.path.join(self.dataset_path, "meta.json")
        with open(meta_path, mode="rt") as f:
            meta = json.load(f)
        meta = list(meta.items())
        self._verify_meta(meta)

        return meta

    def _verify_meta(self, meta: List[Tuple[str, Dict[str, Any]]]) -> None:
        aspect_set = set(self.aspects)
        for file_name, entry in meta:
            if diff := aspect_set.difference(entry["labels"].keys()):
                raise RuntimeError(f"Image '{file_name}' is missing the aspects {diff}")

    def _get_classes(self) -> Dict[str, Dict[str, int]]:
        aspects = {
            aspect: sorted({entry["labels"][aspect] for _, entry in self.meta})
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

        labels = [
            self._classes[aspect][image_info["labels"][aspect]]
            for aspect in self.aspects
        ]
        labels = torch.tensor(labels, dtype=torch.long)

        return img, labels

    def __len__(self) -> int:
        return len(self.meta)
