from enum import Enum
from typing import Union, Optional, Callable, Tuple, Any

from omegaconf import DictConfig
from torch.utils.data import Dataset


class DataSplit(Enum):
    Train = 1
    Test = 2
    Eval = 3


class SensorTypes(Enum):
    Camera = 1


class BasicDataset(Dataset):
    """Basic Dataset"""

    def __init__(
            self,
            cfg: DictConfig,
            split: DataSplit = DataSplit.Train,
            transforms: Optional[Callable] = None,
    ) -> None:
        # super(BasicDataset, self).__init__(root=root, split=split, transforms=transforms)
        self.cfg = cfg
        self.split = split
        self.transforms = transforms

    def __len__(self):
        return None

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target)
        """
        semantic_target = None
        image = None
        return {SensorTypes.Camera: image}, semantic_target