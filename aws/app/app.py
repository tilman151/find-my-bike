import json
from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image


class UnifyingPad:
    def __init__(self, x: int, y: int):
        self.size = x, y

    def __call__(self, img: Image) -> Image:
        padded = Image.new("RGB", self.size)
        paste_pos = (self.size[0] - img.size[0]) // 2, (self.size[1] - img.size[1]) // 2
        padded.paste(img, paste_pos)

        return padded

    def __repr__(self) -> str:
        return f"UnifyingPad({self.size[0]}, {self.size[1]})"


class UnifyingResize:
    def __init__(self, max_size: int):
        self.max_size = max_size

    def __call__(self, img: Image.Image) -> Image.Image:
        h, w = img.size
        if h < w:
            resized = img.resize((int(self.max_size * h / w), self.max_size))
        else:
            resized = img.resize((self.max_size, int(self.max_size * w / h)))

        return resized

    def __repr__(self) -> str:
        return f"UnifyingPad({self.max_size})"


def _get_image(image_url: str) -> Image.Image:
    response = requests.get(image_url, stream=True)
    image = Image.open(BytesIO(response.content), formats=["jpeg", "png"])

    return image


default_transform = torchvision.transforms.Compose(
    [
        UnifyingResize(800),
        UnifyingPad(800, 800),
        torchvision.transforms.ToTensor(),
    ]
)

model_file = "/opt/ml/model"
model = torch.jit.load(model_file, map_location="cpu")
model.eval()


def lambda_handler(event, context):
    image_urls = (
        json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
    )
    images = [default_transform(_get_image(image_url)) for image_url in image_urls]
    batch = torch.stack(images)
    preds = model.predict(batch)

    return {
        "statusCode": 200,
        "body": json.dumps(preds),
    }
