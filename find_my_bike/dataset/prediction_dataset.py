import copy
from io import BytesIO
from typing import Optional, Callable, Any, Dict, Tuple

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from find_my_bike.dataset import utils
from find_my_bike.dataset.transforms import UnifyingPad, UnifyingResize


class PredictionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        high_res: Optional[int] = None,
    ):
        self.dataset_path = dataset_path
        self.high_res = high_res

        self.image_urls = utils.load_image_urls(dataset_path)
        self.transform = self._get_transform()

    def _get_transform(self) -> Callable:
        max_size = self.high_res or 200
        transform = [
            UnifyingResize(max_size),
            UnifyingPad(max_size, max_size),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transform)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image_info = self.image_urls[index]
        image = self._get_image(image_info["image_url"])
        image = self.transform(image)

        return image, copy.deepcopy(image_info)

    def __len__(self) -> int:
        return len(self.image_urls)

    @staticmethod
    def _get_image(image_url: str) -> Image.Image:
        response = requests.get(image_url, stream=True)
        image = Image.open(BytesIO(response.content), formats=["jpeg"])

        return image
