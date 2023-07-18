import os
import cv2
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.utils.file_io import PathManager
from soilcover_data.soilcover_constants import SoilcoverSegformerConstants

logger = logging.getLogger(__name__)


class SoilCoverDataHandler:

    def __init__(self):
        pass

    def get_soilcover_files(self, image_dir, gt_dir):
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
        assert len(files), "No images found in {}".format(image_dir)
        for f in files[0]:
            assert PathManager.isfile(f), f
        return files

    def load_soilcover_semantic(self, image_dir, gt_dir):
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
        for image_file, label_file in self.get_soilcover_files(image_dir, gt_dir):
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

    def register_all_soilcover(self):
        splits = {
            "soilcover_train": (
                SoilcoverSegformerConstants.train_images_path, SoilcoverSegformerConstants.train_labels_path),
            "soilcover_val": (
                SoilcoverSegformerConstants.valid_images_path, SoilcoverSegformerConstants.valid_labels_path),
            "soilcover_test": (
                SoilcoverSegformerConstants.test_images_path, SoilcoverSegformerConstants.test_labels_path),
        }
        meta = _get_builtin_metadata("soilcover")
        print("X")

        for split_name, (image_dir, gt_dir) in splits.items():
            DatasetCatalog.register(
                split_name, lambda x=image_dir, y=gt_dir: self.load_soilcover_semantic(x, y)
            )
            MetadataCatalog.get(split_name).set(
                image_dir=image_dir,
                gt_dir=gt_dir,
                evaluator_type="sem_seg",
                ignore_label=255,
                **meta,
            )
