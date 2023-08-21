import os
import json
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets.soilcover import register_all_soilcover
from detectron2.evaluation import SemSegEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.modeling.backbone.segformer import get_segformer_config
from detectron2.utils.file_io import PathManager
from utils import Utils
from tqdm import tqdm


if __name__ == "__main__":
    segformer_config = get_segformer_config(config_file="configs/segformer_config.yaml")
    register_all_soilcover(cfg=segformer_config)
    segformer_config.MODEL.PIXEL_MEAN = list(segformer_config.DATASETS.SOILCOVER.TRAIN_MEAN_PIXELS)
    segformer_config.MODEL.PIXEL_STD = list(segformer_config.DATASETS.SOILCOVER.TRAIN_STD_PIXELS)
    checkpoints_root_path = "/home/josres/det2_clean/projects/SoilcoverSegformer/output_v1"
    metrics_file_path = "/home/josres/det2_clean/projects/SoilcoverSegformer/output_v1/metrics.json"

    with open(metrics_file_path, "r") as f:
        metrics_list = [json.loads(line.strip()) for line in f]
        metrics_dict = {d_["iteration"]: d_ for d_ in metrics_list if "iteration" in d_}
    all_files = Utils.get_files_under_folder(root_path=checkpoints_root_path)
    checkpoint_files = sorted([f for f in all_files if f.endswith(".pth")])

    results_list = []
    for checkpoint_file in tqdm(checkpoint_files):
        model = build_model(segformer_config)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(checkpoint_file)
        data_loader = build_detection_test_loader(segformer_config, segformer_config.DATASETS.TEST[0])
        evaluator = SemSegEvaluator(
            segformer_config.DATASETS.TEST[0],
            distributed=False,
            num_classes=segformer_config.MODEL.SEGFORMER.NUM_CLASSES,
            ignore_label=segformer_config.MODEL.SEM_SEG_HEAD.IGNORE_VALUE)
        results = inference_on_dataset(model, data_loader, evaluator)
        results_list.append((checkpoint_file, results))

    mIoU_sorted = sorted(results_list, key=lambda tpl: tpl[1]["sem_seg"]["mIoU"], reverse=True)
    fwIoU_sorted = sorted(results_list, key=lambda tpl: tpl[1]["sem_seg"]["fwIoU"], reverse=True)
    print("X")
