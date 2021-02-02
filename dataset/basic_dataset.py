from enum import Enum
from typing import Union, Optional, Callable, Tuple, Any

from omegaconf import DictConfig
from torch.utils.data import Dataset


class DataSplit(Enum):
    """
    Split Definition
    """
    Train = 1
    Test = 2
    Eval = 3


class BasicDataset(Dataset):
    """Basic Dataset
    Basic Dataset all others should extend it
    """

    def __init__(
            self,
            cfg: DictConfig,
            split: DataSplit = DataSplit.Train,
            transforms: Optional[Callable] = None,
    ) -> None:
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
        return image, semantic_target
