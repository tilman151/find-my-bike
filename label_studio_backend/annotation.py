import os

import torch
from label_studio_ml.model import LabelStudioMLBase
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

from find_my_bike.dataset.transforms import UnifyingPad

SCRIPT_PATH = os.path.dirname(__file__)
_MODEL = None
_TRANSFORMS = transform = transforms.Compose(
    [UnifyingPad(200, 200), transforms.ToTensor()]
)


class JitModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(JitModel, self).__init__(**kwargs)

        global _MODEL
        if _MODEL is None:
            _MODEL = torch.jit.load(
                os.path.join(SCRIPT_PATH, "model.pth"), map_location="cpu"
            )
        self.img_base_url = os.environ["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"]

    @torch.no_grad()
    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            img_url = os.path.join(
                self.img_base_url, task["data"]["image"].split("?d=")[1]
            )
            img = pil_loader(img_url)
            inputs = _TRANSFORMS(img)
            inputs = inputs.unsqueeze(0)
            outputs = _MODEL(inputs)
            pred = {
                "model_version": "BikeClassifier",
                "result": [],
            }
            is_bike = False
            for output, (aspect, config) in zip(
                outputs, self.parsed_label_config.items()
            ):
                class_idx = torch.argmax(output, dim=1).squeeze().item()
                label = sorted(config["labels"])[class_idx]
                score = output[0, class_idx].item()
                if aspect == "bike" and label == "Bike":
                    is_bike = True
                if aspect == "bike" or is_bike:
                    pred["result"].append(
                        {
                            "from_name": aspect,
                            "to_name": config["to_name"][0],
                            "type": "choices",
                            "score": score,
                            "value": {"choices": [label]},
                        }
                    )
            predictions.append(pred)

        return predictions
