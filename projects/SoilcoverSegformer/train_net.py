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

import logging
import os

import numpy as np
import torch
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler, \
    StepWithFixedGammaParamScheduler, PolynomialDecayParamScheduler

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.data.datasets.soilcover import register_all_soilcover
from detectron2.data.samplers import TrainingSampler
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import SemSegEvaluator, COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators
from detectron2.modeling.backbone.segformer import get_segformer_config, MixVisionTransformer
from detectron2.solver import WarmupParamScheduler, LRMultiplier
from detectron2.solver.build import maybe_add_gradient_clipping
from projects.PointRend.point_rend import ColorAugSSDTransform


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs

# Newest
class SegformerTrainer(DefaultTrainer):
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
                    distributed=False,
                    num_classes=cfg.MODEL.SEGFORMER.NUM_CLASSES,
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

    @classmethod
    def build_optimizer(cls, cfg, model):
        # params = get_default_optimizer_params(
        #     model,
        #     base_lr=cfg.SOLVER.BASE_LR,
        #     weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        #     bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        #     weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        # )

        # Separate parameters into backbone ones and segmentation head ones.
        backbone_regular_parameters = {}
        backbone_norm_parameters = {}
        seg_head_regular_parameters = {}
        seg_head_norm_parameters = {}
        other_parameters = {}

        for param_name, param in model.named_parameters():
            assert isinstance(param, torch.nn.parameter.Parameter)
            if "backbone" in param_name and "norm" not in param_name:
                backbone_regular_parameters[param_name] = param
            elif "backbone" in param_name and "norm" in param_name:
                backbone_norm_parameters[param_name] = param
            elif "sem_seg_head" and "norm" not in param_name:
                seg_head_regular_parameters[param_name] = param
            elif "sem_seg_head" and "norm" in param_name:
                seg_head_norm_parameters[param_name] = param
            else:
                other_parameters[param_name] = param
        assert len(other_parameters) == 0

        initial_lr = cfg.SOLVER.BASE_LR
        head_lr_multiplier = cfg.SOLVER.HEAD_LR_MULTIPLIER
        betas = cfg.SOLVER.BETAS
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        params_list = [
            {"params": [p_ for p_ in backbone_regular_parameters.values()],
             "lr": initial_lr,
             "weight_decay": weight_decay * cfg.SOLVER.BACKBONE_REGULAR_DECAY_MULTIPLIER},

            {"params": [p_ for p_ in backbone_norm_parameters.values()],
             "lr": initial_lr,
             "weight_decay": weight_decay * cfg.SOLVER.BACKBONE_NORM_DECAY_MULTIPLIER},

            {"params": [p_ for p_ in seg_head_regular_parameters.values()],
             "lr": head_lr_multiplier * initial_lr,
             "weight_decay": weight_decay * cfg.SOLVER.HEAD_REGULAR_DECAY_MULTIPLIER},

            {"params": [p_ for p_ in seg_head_norm_parameters.values()],
             "lr": head_lr_multiplier * initial_lr,
             "weight_decay": weight_decay * cfg.SOLVER.HEAD_NORM_DECAY_MULTIPLIER}
        ]

        adamw_optimizer = torch.optim.AdamW(params=params_list, betas=betas, weight_decay=weight_decay)
        return maybe_add_gradient_clipping(cfg, adamw_optimizer)

    @classmethod
    def build_train_loader(cls, cfg):
        # # return None
        # if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        #     mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        #     # sampler = TrainingSampler(shuffle=False)
        # else:
        #     mapper = None

        dataset_dict = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        sampler = TrainingSampler(size=len(dataset_dict), shuffle=True)
        return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """

        name = cfg.SOLVER.LR_SCHEDULER_NAME

        if name == "WarmupMultiStepLR":
            steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
            if len(steps) != len(cfg.SOLVER.STEPS):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                    "These values will be ignored."
                )
            sched = MultiStepParamScheduler(
                values=[cfg.SOLVER.GAMMA ** k for k in range(len(steps) + 1)],
                milestones=steps,
                num_updates=cfg.SOLVER.MAX_ITER,
            )
        elif name == "WarmupCosineLR":
            end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
            assert end_value >= 0.0 and end_value <= 1.0, end_value
            sched = CosineParamScheduler(1, end_value)
        elif name == "WarmupStepWithFixedGammaLR":
            sched = StepWithFixedGammaParamScheduler(
                base_value=1.0,
                gamma=cfg.SOLVER.GAMMA,
                num_decays=cfg.SOLVER.NUM_DECAYS,
                num_updates=cfg.SOLVER.MAX_ITER,
            )
        elif name == "WarmupPolyLR":
            sched = PolynomialDecayParamScheduler(
                base_value=1.0,
                power=cfg.SOLVER.POLY_POWER
            )
                # sched  WarmupPolyLR(
                #     optimizer,
                #     cfg.SOLVER.MAX_ITER,
                #     warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                #     warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                #     warmup_method=cfg.SOLVER.WARMUP_METHOD,
                #     power=cfg.SOLVER.POLY_LR_POWER,
                #     constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
                # )
        else:
            raise ValueError("Unknown LR scheduler: {}".format(name))

        # TODO: Setup Warmup Scheduler correctly
        # sched = WarmupParamScheduler(
        #     sched,
        #     cfg.SOLVER.WARMUP_FACTOR,
        #     min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        #     cfg.SOLVER.WARMUP_METHOD,
        #     cfg.SOLVER.RESCALE_INTERVAL,
        # )
        sched = WarmupParamScheduler(
            scheduler=sched,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_length=min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            rescale_interval=cfg.SOLVER.RESCALE_INTERVAL
        )
        return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)

    def resume_or_load(self, resume=True):
        # Load Imagenet 1K weights into the Backbone
        if self.cfg.MODEL.SEGFORMER.USE_PRETRAINED_BACKBONE:
            assert os.path.isfile(self.cfg.MODEL.WEIGHTS)
            checkpoint = torch.load(self.cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
            checkpoint_params_set = set(checkpoint.keys())
            assert isinstance(self.checkpointer.model.backbone, MixVisionTransformer)
            model_backbone_params_dict = {k: v for k, v in self.checkpointer.model.backbone.named_parameters()}
            model_backbone_params_set = set(model_backbone_params_dict.keys())
            # Check the compatibility of the checkpoint with our Backbone model.
            assert model_backbone_params_set.issubset(checkpoint_params_set)
            assert checkpoint_params_set.difference(model_backbone_params_set) == {"head.weight", "head.bias"}

            model_state_dict = self.checkpointer.model.backbone.state_dict()

            # Assert that all shapes match.
            for k in checkpoint_params_set:
                if k not in model_state_dict:
                    assert k in {"head.weight", "head.bias"}
                    continue
                checkpoint_tensor = checkpoint[k]
                model_tensor = model_state_dict[k]
                assert checkpoint_tensor.shape == model_tensor.shape

            incompatible = self.checkpointer.model.backbone.load_state_dict(checkpoint, strict=False)
            assert set(incompatible.unexpected_keys) == {"head.weight", "head.bias"}
            print("Loaded Imagenet1K pretraining weights into the backbone.")
            # Assert that now all parameters have been correctly loaded.
            # for k, v in self.checkpointer.model.named_parameters():
            #     if "backbone" not in k:
            #         continue
            #     assert np.array_equal(v.detach().cpu().numpy(), checkpoint[k[len("backbone."):]].numpy())
            # print('X')
        else:
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
            if resume and self.checkpointer.has_checkpoint():
                # The checkpoint stores the training iteration that just finished, thus we start
                # at the next iteration
                self.start_iter = self.iter + 1


if __name__ == "__main__":
    # --config-file configs/pointrend_semantic_R_101_FPN_1x_cityscapes.yaml --num-gpus 833
    # xxx = model_zoo.get_config("common/optim.py").AdamW
    segformer_config = get_segformer_config(config_file="configs/segformer_config.yaml")
    file_exist = os.path.isfile(segformer_config.MODEL.WEIGHTS)
    register_all_soilcover(cfg=segformer_config)

    segformer_config.MODEL.PIXEL_MEAN = list(segformer_config.DATASETS.SOILCOVER.TRAIN_MEAN_PIXELS)
    segformer_config.MODEL.PIXEL_STD = list(segformer_config.DATASETS.SOILCOVER.TRAIN_STD_PIXELS)

    trainer = SegformerTrainer(segformer_config)
    trainer.resume_or_load(resume=False)
    trainer.train()


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
