# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import pickle
from urllib.parse import parse_qs, urlparse
import torch
from fvcore.common.checkpoint import Checkpointer
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.file_io import PathManager


class SegformerCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)
