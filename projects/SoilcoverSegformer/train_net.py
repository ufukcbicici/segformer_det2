#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
import torch
import numpy as np
import os

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import SemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_resnet_backbone
from detectron2.modeling.backbone.segformer import get_segformer_config


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)


if __name__ == "__main__":
    # --config-file configs/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml --num-gpus 8
    segformer_config = get_segformer_config(config_file="configs/segformer_config.yaml")
    trainer = Trainer(segformer_config)

    # cfg = get_cfg()
    # Segformer.add_segformer_config(cfg)
    # cfg.merge_from_file("configs/segformer_config.yaml")
    #
    # resnet = build_resnet_backbone(cfg, ShapeSpec(channels=3))
    #
    # scripted_resnet = torch.jit.script(resnet)
    #
    # inp = torch.rand(2, 3, 100, 100)
    # out1 = resnet(inp)["res4"]
    # out2 = scripted_resnet(inp)["res4"]

    # model = Segformer()
    #
    # images = torch.from_numpy(np.random.uniform(size=(16, 3, 512, 512))).to(torch.float)
    # res = model(images)
    # print("X")

    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
