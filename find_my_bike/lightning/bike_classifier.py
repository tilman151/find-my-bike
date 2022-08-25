from typing import Type, Tuple, Dict, List, Any

import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchmetrics.functional import accuracy
from torchvision.models.feature_extraction import create_feature_extractor

from find_my_bike.lightning.utils import plot_conf_mat, plot_error_image


class MultiAspectHead(nn.Module):
    def __init__(self, aspects: Dict[str, List[str]], in_features: int):
        super().__init__()

        if isinstance(aspects, DictConfig):
            aspects = OmegaConf.to_container(aspects)
        self._aspects = aspects
        self.in_features = in_features
        self.heads = self._build_heads()

    def _build_heads(self) -> nn.ModuleList:
        heads = nn.ModuleList()
        for aspect, classes in self._aspects.items():
            heads.add_module(aspect, nn.Linear(self.in_features, len(classes)))

        return heads

    @property
    def aspects(self) -> List[str]:
        return list(self._aspects.keys())

    @property
    def class_names(self) -> Dict[str, List[str]]:
        return self._aspects

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        return [m(inputs) for m in self.heads]


class Encoder(nn.Module):
    def __init__(self, encoder: nn.Module, output_node: str):
        super().__init__()

        self.encoder = create_feature_extractor(encoder, [output_node])
        self.output_node = output_node
        self.name = self.encoder.__class__.__name__

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.encoder(inputs)[self.output_node], start_dim=1)


def _accuracy(
    preds: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    valid_samples = labels != ignore_index
    acc = accuracy(preds[valid_samples], labels[valid_samples])

    return acc


class BikeClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        head: MultiAspectHead,
        lr: float = 0.001,
        encoder_lr_factor: float = 0.0,
        optim: Type[torch.optim.Optimizer] = torch.optim.Adam,
        record_error_images: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.head = head
        self.lr = lr
        self.encoder_lr_factor = encoder_lr_factor
        self.optim_type = optim
        self.record_error_images = record_error_images
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.conf_mat = nn.ModuleDict(
            {
                aspect: torchmetrics.ConfusionMatrix(len(classes))
                for aspect, classes in self.head.class_names.items()
            }
        )

        self.save_hyperparameters(
            {
                "encoder": self.encoder.name,
                "lr": self.lr,
                "encoder_lr_factor": self.encoder_lr_factor,
                "optim": self.optim_type.__name__,
            }
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["jit_module"] = self.to_torchscript()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optim_type(
            [
                {"params": self.head.parameters()},
                {
                    "params": self.encoder.parameters(),
                    "lr": self.lr * self.encoder_lr_factor,
                },
            ],
            lr=self.lr,
        )

    def forward(self, img: torch.Tensor) -> List[torch.Tensor]:
        features = self.encoder(img)
        preds = self.head(features)

        return preds

    @torch.jit.export
    def predict(self, imgs: torch.Tensor) -> List[Dict[str, str]]:
        logits = self(imgs)
        preds = [{aspect: "" for aspect in self.head.aspects} for _ in imgs]
        for logit, (aspect, class_names) in zip(logits, self.head.class_names.items()):
            class_idx = torch.argmax(logit, dim=1)
            for class_id, pred in zip(class_idx, preds):
                pred[aspect] = class_names[class_id.item()]

        return preds

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        img, labels = batch
        preds = self.forward(img)
        loss = self._summed_los(preds, labels)
        self.log("train/loss", loss)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        img, labels = batch
        preds = self.forward(img)
        loss = self._summed_los(preds, labels)
        self.log("val/loss", loss)
        for aspect, pred, label in zip(self.head.aspects, preds, labels.T):
            acc = _accuracy(pred, label, ignore_index=-1)
            self.log(f"val/{aspect}_acc", acc)

    def _summed_los(self, preds, labels):
        return sum(self.ce_loss(pred, label) for pred, label in zip(preds, labels.T))

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        imgs, labels = batch
        preds = self.forward(imgs)
        for aspect, pred, label in zip(self.head.aspects, preds, labels.T):
            self._update_conf_mat(aspect, pred, label)
            if self.record_error_images:
                self._record_error_images(aspect, pred, label, imgs, batch_idx)

    def _update_conf_mat(
        self, aspect: str, pred: torch.Tensor, label: torch.Tensor
    ) -> None:
        non_ignored = label != -1
        self.conf_mat[aspect].update(pred[non_ignored], label[non_ignored])

    def _record_error_images(
        self,
        aspect: str,
        preds: torch.Tensor,
        labels: torch.Tensor,
        imgs: torch.Tensor,
        batch_idx: int,
    ) -> None:
        preds = torch.argmax(preds, dim=1)
        wrong = (labels != -1) & (preds != labels)
        iter_wrong = zip(imgs[wrong], preds[wrong], labels[wrong])
        for i, (img, pred, label) in enumerate(iter_wrong):
            pred_name = self.head.class_names[aspect][pred]
            label_name = self.head.class_names[aspect][label]
            error_image = plot_error_image(img, pred_name, label_name)
            self.logger.experiment.add_figure(
                f"error_images/{aspect}_{batch_idx:03}_{i:03}", error_image
            )

    def on_test_end(self) -> None:
        logger = self.logger
        for aspect, conf_mat in self.conf_mat.items():
            conf_mat_img = plot_conf_mat(
                conf_mat.compute(), self.head.class_names[aspect]
            )
            logger.experiment.add_figure(f"conf_mat/{aspect}", conf_mat_img)
