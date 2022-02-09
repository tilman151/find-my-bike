from unittest import mock

import pytest
import torch
from torch import nn

from find_my_bike.lightning.bike_classifier import (
    MultiAspectHead,
    BikeClassifier,
    Encoder,
    _accuracy,
)


@pytest.fixture
def aspects():
    return {"a": 2, "b": 4}


@pytest.fixture
def head(aspects):
    return MultiAspectHead(aspects, 8)


@pytest.fixture
def encoder():
    layers = nn.Sequential()
    layers.add_module("linear", nn.Linear(16, 8))
    return Encoder(layers, "linear")


@pytest.fixture
def classifier(encoder, head):
    return BikeClassifier(encoder, head)


def test_multi_aspect_head_init(aspects, head):
    assert list(aspects.keys()) == head.aspects
    assert len(aspects) == len(head.heads)
    for h, num_classes in zip(head.heads, aspects.values()):
        assert h.in_features == 8
        assert h.out_features == num_classes


def test_multi_aspect_head_forward(aspects, head):
    inputs = torch.randn(4, 8)
    outputs = head(inputs)
    assert len(outputs) == len(aspects)
    for out, num_classes in zip(outputs, aspects.values()):
        assert out.shape[1] == num_classes


def test_bike_classifier_forward(aspects, classifier):
    inputs = torch.randn(4, 16)
    outputs = classifier(inputs)
    assert len(outputs) == len(aspects)


def test_accuracy():
    preds = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.6, 0.5]])

    labels_with_ignore = torch.tensor([1, 1, -1], dtype=torch.long)
    acc = _accuracy(preds, labels_with_ignore)
    assert acc == 0.5

    labels_without_ignore = torch.tensor([1, 0, 0], dtype=torch.long)
    acc = _accuracy(preds, labels_without_ignore)
    assert acc == 1.0


def test_bike_classifier_train_step(aspects, classifier):
    mock_log = mock.MagicMock(name="log")
    with mock.patch.object(classifier, "log", mock_log):
        batch = torch.randn(4, 16), torch.zeros(4, len(aspects), dtype=torch.long)
        loss = classifier.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    mock_log.assert_called_once_with("train/loss", loss)


def test_bike_classifier_val_step(aspects, classifier):
    mock_log = mock.MagicMock(name="log")
    with mock.patch.object(classifier, "log", mock_log):
        batch = torch.randn(4, 16), torch.zeros(4, len(aspects), dtype=torch.long)
        classifier.validation_step(batch, batch_idx=0)
    mock_log.assert_has_calls(
        [
            mock.call("val/a_acc", mock.ANY),
            mock.call("val/b_acc", mock.ANY),
        ]
    )


def test_bike_classifier_on_checkpoint(classifier):
    checkpoint = {}
    classifier.on_save_checkpoint(checkpoint)
    assert "jit_module" in checkpoint
    assert isinstance(checkpoint["jit_module"], torch.jit.ScriptModule)
