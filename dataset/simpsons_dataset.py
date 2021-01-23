import pickle
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, Any

import numpy as np
from PIL import Image
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dataset.basic_dataset import BasicDataset, DataSplit


class SimpsonsDataset(BasicDataset):
    """Simpsons Dataset"""

    def __init__(
            self,
            cfg: DictConfig,
            split: DataSplit = DataSplit.Train,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(SimpsonsDataset, self).__init__(cfg=cfg,
                                              split=split,
                                              transforms=transforms)
        self.cfg = cfg
        self.split = split
        self.transforms = transforms
        self.train_dir = Path('/kaggle/input/the-simpsons-characters-dataset/simpsons_dataset/')

        self.train_val_files_path = sorted(list(self.train_dir.rglob('*.jpg')))
        self.train_val_labels = [path.parent.name for path in self.train_val_files_path]

        self.labels = [path.parent.name for path in self.files_path]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

        with open('label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """

        img_path = str(self.train_val_files_path[idx])
        image = Image.open(img_path)
        image = self.transform(image)

        label_str = str(self.train_val_files_path[idx].parent.name)
        label = self.label_encoder.transform([label_str]).item()

        return image, label
