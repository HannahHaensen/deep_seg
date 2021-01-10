import os

import numpy as np
import torch
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple

from torchvision.transforms import CenterCrop, Normalize, ToTensor, transforms, Resize

from .basic_dataset import BasicDataset, DataSplit, SensorTypes


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
            root: str,
            split: DataSplit = DataSplit.Train,
            transforms: Optional[Callable] = None,
    ):
        super(VOCSegmentation, self).__init__(root=root, split=split, transforms=transforms)

        voc_root = os.path.join(self.root, '/Users/hannahschieber/GitHub/deep_seg/data/VOC2012')
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
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

    def get_classes(self):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
