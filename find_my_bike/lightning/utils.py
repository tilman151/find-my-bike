import logging
import os
from typing import Dict, Any, Optional, Callable, List

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.plugins import CheckpointIO, TorchCheckpointIO

log = logging.getLogger(__name__)


class TorchJitCheckpointIO(CheckpointIO):
    """CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save
    and load checkpoints respectively, common for most use cases. Adds a jitted
    version, too."""

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
        storage_options: Optional[Any] = None,
    ) -> None:
        if "jit_module" not in checkpoint:
            raise ValueError("LightningModule must add jitted version to checkpoint.")

        jit_module = checkpoint["jit_module"]
        del checkpoint["jit_module"]
        TorchCheckpointIO().save_checkpoint(checkpoint, path, storage_options)

        folder_path, file_name = os.path.split(path)
        file_name = "jit-" + file_name.replace(".ckpt", ".pth")
        file_path = os.path.join(folder_path, file_name)
        torch.jit.save(jit_module, file_path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Callable] = lambda storage, loc: storage,
    ) -> Dict[str, Any]:
        """
        Loads checkpoint using :func:`torch.load`, with additional handling for
        ``fsspec`` remote loading of
        files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict
                          specifying how to remap storage locations.

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem
        """
        return TorchCheckpointIO().load_checkpoint(path, map_location)

    def remove_checkpoint(self, path: str) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint
        """
        folder_path, file_name = os.path.split(path)
        file_name = "jit-" + file_name.replace(".ckpt", ".pth")
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)

        TorchCheckpointIO().remove_checkpoint(path)


def plot_conf_mat(
    conf_mat: torch.Tensor, classes: Optional[List[str]] = None
) -> plt.Figure:
    conf_mat = conf_mat.detach().cpu().numpy()
    num_classes = conf_mat.shape[0]
    mean_value = conf_mat.mean()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.tight_layout()
    ax.set_aspect(1)
    ax.imshow(conf_mat, cmap=plt.cm.cividis, interpolation="nearest")

    for x in range(num_classes):
        for y in range(num_classes):
            value = conf_mat[x, y]
            ax.annotate(
                str(value),
                xy=(y, x),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize="large",
                color=("white" if value < mean_value else "black"),
            )

    if classes is None:
        tick_labels = range(num_classes)
    else:
        tick_labels = classes
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels, rotation=90, ha="center", va="center")

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    return fig
