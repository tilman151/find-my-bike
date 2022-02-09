import os

import torch
from label_studio_ml.model import LabelStudioMLBase
from torchvision import transforms
from torchvision.datasets.folder import pil_loader

from find_my_bike.dataset.image_dataset import UnifyingPad

SCRIPT_PATH = os.path.dirname(__file__)
_MODEL = None
_TRANSFORMS = transform = transforms.Compose(
    [UnifyingPad(200, 200), transforms.ToTensor()]
)


class JitModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(JitModel, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema["to_name"][0]
        self.labels = sorted(schema["labels"])

        global _MODEL
        if _MODEL is None:
            _MODEL = torch.jit.load(os.path.join(SCRIPT_PATH, "model.pth"))
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
            class_idx = torch.argmax(outputs[0], dim=1).squeeze().item()
            score = outputs[0][0, class_idx].item()
            predictions.append(
                {
                    "score": score,
                    # prediction overall score, visible in the data manager columns
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "score": score,  # per-region score, visible in the editor
                            "value": {"choices": [self.labels[class_idx]]},
                        }
                    ],
                }
            )
        return predictions
