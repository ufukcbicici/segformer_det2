import functools
import json
import logging
import multiprocessing as mp
from collections import Counter

import numpy as np
import os
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, detection_utils
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass

logger = logging.getLogger(__name__)


def _get_soilcover_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    images = PathManager.ls(image_dir)
    logger.info(f"{len(images)} images found in '{image_dir}'.")

    for basename in PathManager.ls(image_dir):
        image_file = os.path.join(image_dir, basename)

        suffix = ".jpg"
        assert basename.endswith(suffix), basename
        basename = basename[: -len(suffix)]

        # instance_file = os.path.join(gt_dir, basename + "instanceIDs.png")
        label_file = os.path.join(gt_dir, basename + ".png")
        # json_file = os.path.join(gt_dir, basename + ".json")

        # files.append((image_file, instance_file, label_file, json_file))
        files.append((image_file, label_file))
        assert PathManager.isfile(image_file), image_file
        assert PathManager.isfile(label_file), label_file
    assert len(files), "No images found in {}".format(image_dir)
    return files


def load_soilcover_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file in tqdm(_get_soilcover_files(image_dir, gt_dir)):
        label_file = label_file.replace("labelIds", "labelTrainIds")
        height, width, channels = cv2.imread(label_file).shape
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": height,
                "width": width,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret


# ==== Predefined splits for raw soilcover images ===========
# _RAW_SOILCOVER_SPLITS = {
#     "soilcover_train": (
#         "/shared/data/SoilCoverGitter/train/img/remap/", "/shared/data/SoilCoverGitter/train/lbl/remap/"),
#     "soilcover_val": ("/shared/data/SoilCoverGitter/valid/img/remap/", "/shared/data/SoilCoverGitter/valid/lbl/remap/"),
#     "soilcover_test": ("/shared/data/SoilCoverGitter/test/img/remap/", "/shared/data/SoilCoverGitter/test/lbl/remap/"),
# }

def image_size_analysis(cfg, augmentations):
    mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)


def calculate_split_statistcs(split_name, cfg, image_dir, gt_dir):
    files = load_soilcover_semantic(image_dir, gt_dir)
    widths = set()
    heights = set()
    ratios = set()
    sizes = []
    image_weights = []
    image_mean_colors = []
    image_std_colors = []
    label_values = set()
    for image_dict in tqdm(files):
        widths.add(image_dict["width"])
        heights.add(image_dict["height"])
        ratios.add(image_dict["width"] / image_dict["height"])
        sizes.append((image_dict["width"], image_dict["height"]))

        image = detection_utils.read_image(image_dict["file_name"], format=cfg.INPUT.FORMAT)
        sem_seg_gt = detection_utils.read_image(image_dict["sem_seg_file_name"], "L").squeeze(2)
        sem_seg_gt = np.reshape(a=sem_seg_gt, newshape=(np.prod(sem_seg_gt.shape), ))
        values = set(sem_seg_gt)
        label_values = label_values.union(values)

        detection_utils.check_image_size(image_dict, image)
        image_weights.append(np.prod(image.shape[0:2]))
        image_mean = np.mean(image, axis=(0, 1))
        image_mean_colors.append(image_mean)
        image_std = np.std(image, axis=(0, 1))
        image_std_colors.append(image_std)

    sizes = Counter(sizes)
    print(sizes)

    image_weights = np.array(image_weights)
    sum_weights = np.sum(image_weights)
    image_weights = image_weights / sum_weights
    assert np.allclose(np.sum(image_weights), 1.0)

    # Weighted sum of mean colors
    image_mean_colors = np.stack(image_mean_colors, axis=0)
    weighted_mean_colors = np.expand_dims(image_weights, axis=1) * image_mean_colors
    weighted_mean_color = np.sum(weighted_mean_colors, axis=0)
    # Weighted sum of std colors
    image_std_colors = np.stack(image_std_colors, axis=0)
    weighted_std_colors = np.expand_dims(image_weights, axis=1) * image_std_colors
    weighted_std_color = np.sum(weighted_std_colors, axis=0)
    # All unique label values
    label_values = tuple(sorted(list(label_values)))
    print("{0} mean pixels:{1}".format(split_name, weighted_mean_color))
    print("{0} std pixels:{1}".format(split_name, weighted_std_color))
    print("{0} label values:{1}".format(split_name, label_values))

    return weighted_mean_color, weighted_std_color, label_values


def register_all_soilcover(cfg):
    soilcover_splits = {
        "soilcover_train": cfg.DATASETS.SOILCOVER.TRAIN_FILES,
        "soilcover_val": cfg.DATASETS.SOILCOVER.VALIDATION_FILES,
        "soilcover_test": cfg.DATASETS.SOILCOVER.TEST_FILES
    }

    # Calculate the statistics of the Soilcover dataset.
    for sem_key, (image_dir, gt_dir) in soilcover_splits.items():
        if cfg.DATASETS.SOILCOVER.CALCULATE_DATASET_STATISTICS:
            weighted_mean_color, weighted_std_color, label_values = calculate_split_statistcs(
                cfg=cfg, image_dir=image_dir, gt_dir=gt_dir, split_name=sem_key)

            if sem_key == "soilcover_train":
                cfg.DATASETS.SOILCOVER.TRAIN_MEAN_PIXELS = tuple(weighted_mean_color)
                cfg.DATASETS.SOILCOVER.TRAIN_STD_PIXELS = tuple(weighted_std_color)
                cfg.DATASETS.SOILCOVER.TRAIN_LABELS = label_values
            elif sem_key == "soilcover_val":
                cfg.DATASETS.SOILCOVER.VALIDATION_MEAN_PIXELS = tuple(weighted_mean_color)
                cfg.DATASETS.SOILCOVER.VALIDATION_STD_PIXELS = tuple(weighted_std_color)
                cfg.DATASETS.SOILCOVER.VALIDATION_LABELS = label_values
            elif sem_key == "soilcover_test":
                cfg.DATASETS.SOILCOVER.TEST_MEAN_PIXELS = tuple(weighted_mean_color)
                cfg.DATASETS.SOILCOVER.TEST_STD_PIXELS = tuple(weighted_std_color)
                cfg.DATASETS.SOILCOVER.TEST_LABELS = label_values
            else:
                RuntimeError("Unknown data split:{0}".format(sem_key))

        # Prepare the metadata for the soilcover dataset and record individual samples.
        meta = _get_builtin_metadata("soilcover")
        # res = load_soilcover_semantic(image_dir, gt_dir)
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_soilcover_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            **meta,
        )


if __name__ == "__main__":
    """
    Test the cityscapes dataset loader.

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("gt_dir")
    parser.add_argument("--type", choices=["semantic"], default="semantic")
    args = parser.parse_args()
    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)

    dirname = "soilcover-data-vis"
    os.makedirs(dirname, exist_ok=True)

    dicts = load_soilcover_semantic(args.image_dir, args.gt_dir)
    logger.info("Done loading {} samples.".format(len(dicts)))

    stuff_classes = ["soil", "dead_org", "living_org", "stone"]
    stuff_colors = [(255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
    meta = Metadata().set(stuff_classes=stuff_classes, stuff_colors=stuff_colors)

    for d in dicts:
        img = np.array(Image.open(PathManager.open(d["file_name"], "rb")))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
