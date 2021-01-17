import os

import numpy as np
import torch
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig
from torchvision.transforms import CenterCrop, Normalize, ToTensor, transforms, Resize

from .basic_dataset import BasicDataset, DataSplit, SensorTypes


def get_classes() -> Dict:
    return {
        1: '__background__',
        2: 'aeroplane',
        3: 'bicycle',
        4: 'bird',
        5: 'boat',
        6: 'bottle',
        7: 'bus',
        8: 'car',
        9: 'cat',
        10: 'chair',
        11: 'cow',
        12: 'diningtable',
        13: 'dog',
        14: 'horse',
        15: 'motorbike',
        16: 'person',
        17: 'pottedplant',
        18: 'sheep',
        19: 'sofa',
        20: 'train',
        21: 'tvmonitor'
    }


def get_color_for_classes() -> Dict:
    return {
        0: [64, 64, 64],
        1: [0, 0, 0],
        2: [128, 0, 0],
        3: [0, 128, 0],
        4: [128, 128, 0],
        5: [0, 0, 128],
        6: [128, 0, 128],
        7: [0, 128, 128],
        8: [128, 128, 128],
        9: [64, 0, 0],
        10: [192, 0, 0],
        11: [64, 128, 0],
        12: [192, 128, 0],
        13: [64, 0, 128],
        14: [192, 0, 128],
        15: [64, 128, 128],
        16: [192, 128, 128],
        17:  [0, 64, 0],
        18: [128, 64, 0],
        19: [0, 192, 0],
        20: [128, 192, 0],
        21: [0, 64, 128],
        255: [255, 255, 255]
    }


class VOCSegmentation(BasicDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            cfg: DictConfig,
            split: DataSplit = DataSplit.Train,
            transforms: Optional[Callable] = None,
    ):
        super(VOCSegmentation, self).__init__(cfg=cfg,
                                              split=split,
                                              transforms=transforms)
        print(cfg)
        image_dir = os.path.join(cfg.dataset.root_path, cfg.dataset.image_sub_path)
        mask_dir = os.path.join(cfg.dataset.root_path, cfg.dataset.mask_sub_path)

        splits_dir = os.path.join(cfg.dataset.root_path, 'ImageSets/Segmentation')
        image_set = "train"

        if DataSplit.Eval:
            image_set = "val"

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self._normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self._crop = CenterCrop(64)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        img = np.array(img)
        img = torch.from_numpy(img.transpose([2, 0, 1]))

        target = np.array(target)
        target = torch.from_numpy(target)

        img = self._normalize(img.float() / 255.0)
        img = self._crop(img)
        target = self._crop(target)

        # print("return of get item")
        # print(img.shape, target.shape)
        return {SensorTypes.Camera: img}, target

    def __len__(self) -> int:
        return len(self.images)
