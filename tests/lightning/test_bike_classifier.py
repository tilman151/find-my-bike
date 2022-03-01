from unittest import mock

import pytest
import torch
from torch import nn
from torch.nn.functional import cross_entropy

from find_my_bike.lightning.bike_classifier import (
    MultiAspectHead,
    BikeClassifier,
    Encoder,
    _accuracy,
)


@pytest.fixture
def aspects():
    return {"a": ["a", "b"], "b": ["a", "b", "c", "d"]}


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


@pytest.fixture
def fake_logits():
    fake_logits = [
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]),
    ]
    return fake_logits


def test_multi_aspect_head_init(aspects, head):
    assert list(aspects.keys()) == head.aspects
    assert len(aspects) == len(head.heads)
    for h, classes in zip(head.heads, aspects.values()):
        assert h.in_features == 8
        assert h.out_features == len(classes)


def test_multi_aspect_head_forward(aspects, head):
    inputs = torch.randn(4, 8)
    outputs = head(inputs)
    assert len(outputs) == len(aspects)
    for out, classes in zip(outputs, aspects.values()):
        assert out.shape[1] == len(classes)


def test_bike_classifier_forward(aspects, classifier):
    inputs = torch.randn(4, 16)
    outputs = classifier(inputs)
    assert len(outputs) == len(aspects)


def test_bike_classifier_loss(classifier, fake_logits, monkeypatch):
    labels = torch.tensor([[0, 0], [1, 0]])
    monkeypatch.setattr(classifier, "forward", lambda _: fake_logits)

    loss = classifier.training_step((torch.zeros(2, 16), labels), 0)

    aspect0_loss = cross_entropy(fake_logits[0], labels[:, 0])
    aspect1_loss = cross_entropy(fake_logits[1], labels[:, 1])
    expected_loss = aspect0_loss + aspect1_loss
    assert loss == expected_loss


def test_bike_classifier_loss_ignore_index(classifier, fake_logits, monkeypatch):
    labels = torch.tensor([[0, 0], [-1, 0]])
    monkeypatch.setattr(classifier, "forward", lambda _: fake_logits)

    loss = classifier.training_step((torch.zeros(2, 16), labels), 0)

    aspect0_loss_without_ignored = cross_entropy(fake_logits[0][:1], labels[:1, 0])
    aspect1_loss = cross_entropy(fake_logits[1], labels[:, 1])
    expected_loss = aspect0_loss_without_ignored + aspect1_loss
    assert loss == expected_loss


def test_bike_classifier_confmat_ignore_index(classifier):
    labels = torch.tensor([[0, 0], [-1, 0]])
    classifier.test_step((torch.zeros(2, 16), labels), 0)
    assert classifier.conf_mat["a"].confmat.sum() == 1
    assert classifier.conf_mat["b"].confmat.sum() == 2


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
            mock.call("val/loss", mock.ANY),
            mock.call("val/a_acc", mock.ANY),
            mock.call("val/b_acc", mock.ANY),
        ]
    )


def test_bike_classifier_on_checkpoint(classifier):
    checkpoint = {}
    classifier.on_save_checkpoint(checkpoint)
    assert "jit_module" in checkpoint
    assert isinstance(checkpoint["jit_module"], torch.jit.ScriptModule)


def test_bike_classifier_record_error_images(classifier, monkeypatch):
    imgs = torch.rand(2, 3, 10, 10)
    preds = torch.rand(2, 2)
    labels = torch.tensor([0, 1])
    monkeypatch.setattr(classifier, "trainer", mock.MagicMock(name="trainer"))
    classifier._record_error_images("a", preds, labels, imgs)
