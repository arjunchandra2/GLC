#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import wandb

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class WandbWriter(object):
    """
    Helper class to log information to Weights & Biases.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
        """
        wandb.init(
            project=cfg.WANDB.PROJECT,
            entity=cfg.WANDB.ENTITY or None,
            name=cfg.WANDB.RUN_NAME or None,
            group=cfg.WANDB.GROUP or None,
            config=cfg,
        )
        logger.info(
            "Logging training metrics to Weights & Biases project '{}'.".format(cfg.WANDB.PROJECT)
        )

    def add_scalars(self, data_dict, global_step=None):
        """
        Add multiple scalars to Weights & Biases logs.
        Args:
            data_dict (dict): key is a string specifying the tag of value.
            global_step (Optional[int]): Global step value to record.
        """
        wandb.log(data_dict, step=global_step)

    def close(self):
        wandb.finish()
