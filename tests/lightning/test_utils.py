from unittest import mock

import pytest
import torch.jit
from pytorch_lightning import LightningModule, Trainer
from torch import nn

from find_my_bike.lightning.utils import TorchJitCheckpointIO


def test_save_checkpoint():
    mock_jit_module = mock.MagicMock(name="jit_module")
    checkpoint = {"jit_module": mock_jit_module}
    jit_io = TorchJitCheckpointIO()
    with mock.patch("find_my_bike.lightning.utils.TorchCheckpointIO") as mock_torch_io:
        with mock.patch("find_my_bike.lightning.utils.torch.jit.save") as mock_jit_save:
            jit_io.save_checkpoint(checkpoint, "foo/bar.ckpt")

    mock_torch_io().save_checkpoint.assert_called_with({}, "foo/bar.ckpt", None)
    mock_jit_save.assert_called_with(mock_jit_module, "foo/jit_module.pth")


def test_load_checkpoint():
    jit_io = TorchJitCheckpointIO()
    with mock.patch("find_my_bike.lightning.utils.TorchCheckpointIO") as mock_torch_io:
        loaded_checkpoint = jit_io.load_checkpoint("foo/bar.ckpt", map_location=None)
    mock_torch_io().load_checkpoint.assert_called_with("foo/bar.ckpt", None)
    assert loaded_checkpoint is mock_torch_io().load_checkpoint()


def test_remove_checkpoint():
    jit_io = TorchJitCheckpointIO()
    with mock.patch("find_my_bike.lightning.utils.TorchCheckpointIO") as mock_torch_io:
        with mock.patch("find_my_bike.lightning.utils.os.remove") as mock_rem:
            jit_io.remove_checkpoint("foo/bar.ckpt")
    mock_torch_io().remove_checkpoint.assert_called_with("foo/bar.ckpt")
    mock_rem.assert_called_with("foo/jit_module.pth")
