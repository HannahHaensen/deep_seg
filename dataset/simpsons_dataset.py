import os
import pickle
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, Any

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig

from torchvision import datasets, transforms

from dataset.basic_dataset import BasicDataset, DataSplit


class SimpsonsDataset(BasicDataset):
    """Simpsons Dataset"""

    def __init__(
            self,
            cfg: DictConfig,
            split: DataSplit = DataSplit.Train
    ) -> None:
        super(SimpsonsDataset, self).__init__(cfg=cfg,
                                              split=split)
        root_dir = cfg.dataset.root_path

        classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        samples = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    samples.append(item)

        self.samples = samples
        self.targets = [s[1] for s in samples]

        resize = transforms.Resize((16, 16))
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
        horizontal_flip = transforms.RandomHorizontalFlip(p=0.3)
        # random_crop = transforms.RandomCrop(90)
        self.transforms = transforms.Compose([resize,
                                              horizontal_flip])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        img = torch.from_numpy(img.transpose([2, 0, 1]))
        img = self._normalize(img.float() / 255.0)
        img = self.transforms(img)

        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def get_classes(self):
        return
