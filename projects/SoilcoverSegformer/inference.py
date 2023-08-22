# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import cv2
import os

from detectron2.data import build_detection_test_loader
from detectron2.data.datasets.soilcover import register_all_soilcover
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.modeling.backbone.segformer import get_segformer_config
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.catalog import Metadata
from utils import Utils


def main(args):
    segformer_config = get_segformer_config(config_file="configs/segformer_config.yaml")
    register_all_soilcover(cfg=segformer_config)
    segformer_config.MODEL.PIXEL_MEAN = list(segformer_config.DATASETS.SOILCOVER.TRAIN_MEAN_PIXELS)
    segformer_config.MODEL.PIXEL_STD = list(segformer_config.DATASETS.SOILCOVER.TRAIN_STD_PIXELS)

    predictor = DefaultPredictor(segformer_config)
    model = predictor.model
    data_loader = build_detection_test_loader(segformer_config, segformer_config.DATASETS.TEST[0])
    evaluator = SemSegEvaluator(
        segformer_config.DATASETS.TEST[0],
        distributed=False,
        num_classes=segformer_config.MODEL.SEGFORMER.NUM_CLASSES,
        ignore_label=segformer_config.MODEL.SEM_SEG_HEAD.IGNORE_VALUE)
    results = inference_on_dataset(model, data_loader, evaluator)

    im_list = Utils.get_files_under_folder(root_path=segformer_config.DATASETS.SOILCOVER.VALIDATION_FILES[0])

    # dummy_input = Variable(torch.randn(1, 3, 256, 256))
    # im_list = []
    # if os.path.isfile(args.input[0]):
    #     im_list.append(args.input[0])
    # else:
    #     valid_images = [".jpg"]
    #     for f in os.listdir(args.input[0]):
    #         ext = os.path.splitext(f)[1]
    #         if ext.lower() not in valid_images:
    #             continue
    #         im_list.append(os.path.join(args.input[0], f))
    # state_dict = torch.load('./output/model_0004999.pth')
    # model.load_state_dict(state_dict)

    # commented out because of out of memory error @todo make onnx export work
    # dummy_input = im.tolist()
    # torch.onnx.export(model, dummy_input, "moment-in-time.onnx", opset_version=11)

    for i in im_list:
        path = i.rsplit("/", 1)[0]
        image = i.rsplit("/", 1)[1].rsplit(".", 1)[0]

        im = cv2.imread(i)

        outputs = predictor(im)
        cv2.imwrite("input.jpg", im)
        stuff_classes = ["soil", "living_org", "dead_org", "stone"]
        stuff_colors = [(167, 206, 228), (51, 160, 44), (31, 121, 180), (228, 26, 27)]
        meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)

        v = Visualizer(im[:, :, ::-1], scale=1.0, instance_mode=ColorMode.SEGMENTATION, metadata=meta)
        point_rend_result = v.draw_sem_seg(outputs["sem_seg"].argmax(dim=0).to("cpu"), alpha=0.8).get_image()

        # point_rend_result = v.draw_sem_seg(outputs["sem_seg"].to("cpu")).get_image()
        # point_rend_result = cv2.cvtColor(point_rend_result, cv2.COLOR_BGR2RGB)

        cv2.imwrite("output.jpg", point_rend_result)
        cv2.imwrite(path + "/" + image + "_mask.png", point_rend_result)


#
# # create a keyvalue class
# class keyvalue(argparse.Action):
#     # Constructor calling
#     def __call__(self, parser, namespace,
#                  values, option_string=None):
#         setattr(namespace, self.dest, dict())
#
#         for value in values:
#             # split it into key and value
#             key, value = value.split('=')
#             # assign into dictionary
#             getattr(namespace, self.dest)[key] = value

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        0,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

