import logging
import os
from typing import Dict, Any, Optional, Callable

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

        folder_path = os.path.dirname(path)
        file_path = os.path.join(folder_path, "jit_module.pth")
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
        folder_path = os.path.dirname(path)
        file_path = os.path.join(folder_path, "jit_module.pth")
        os.remove(file_path)

        TorchCheckpointIO().remove_checkpoint(path)
