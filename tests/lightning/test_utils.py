from unittest import mock

import pytest
import torch.jit

from find_my_bike.lightning.utils import (
    TorchJitCheckpointIO,
    plot_conf_mat,
    plot_error_image,
)


def test_save_checkpoint():
    mock_jit_module = mock.MagicMock(name="jit_module")
    checkpoint = {"jit_module": mock_jit_module}
    jit_io = TorchJitCheckpointIO()
    with mock.patch("find_my_bike.lightning.utils.TorchCheckpointIO") as mock_torch_io:
        with mock.patch("find_my_bike.lightning.utils.torch.jit.save") as mock_jit_save:
            jit_io.save_checkpoint(checkpoint, "foo/bar.ckpt")

    mock_torch_io().save_checkpoint.assert_called_with({}, "foo/bar.ckpt", None)
    mock_jit_save.assert_called_with(mock_jit_module, "foo/jit-bar.pth")


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
    mock_rem.assert_called_with("foo/jit-bar.pth")


@pytest.mark.skip("for visual inspection only")
def test_plot_conf_mat():
    conf_mat = torch.tensor([[10, 5, 1], [7, 10, 2], [2, 0, 10]])
    fig = plot_conf_mat(conf_mat, ["a", "b", "c"])
    fig.show()


@pytest.mark.skip("for visual inspection only")
def test_plot_error_image():
    img = torch.rand(3, 10, 10)
    fig = plot_error_image(img, "Prediction", "Ground Truth")
    fig.show()
