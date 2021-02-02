import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from dataset.voc_seg_dataset import get_color_for_classes


class TensorboardLogger:

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)
        self.global_step = 1

    def set_global_step(self):
        self.global_step += 1

    def log_scalar(self, tag, value):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, self.global_step)
        self.writer.flush()

    def log_scalars(self, info):
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.global_step)
        self.writer.flush()

    def log_confusion_matrix(self, tag, conf_mat, labels):
        cmap = plt.get_cmap('Blues')

        fig = plt.figure()
        plt.imshow(conf_mat)

        self.writer.add_figure(tag, fig)
        self.writer.flush()

    def log_image(self, tag, value):

        img = value.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = torch.from_numpy(img.transpose([2, 0, 1]))
        # self.writer.add_image(tag, img, self.global_step)
        self.writer.add_image(tag, img, self.global_step)
        # y = torch.squeeze(value, 1)
        # self.writer.add_image(tag=tag, img_tensor=y)

    def log_images(self, tag, value, step):
        return
