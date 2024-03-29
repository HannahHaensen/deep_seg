import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from mlflow import log_metric, log_param, log_artifacts
import mlflow
from dataset.voc_seg_dataset import get_color_for_classes


class TrainingLogger:

    def __init__(self, log_dir: str,
                 experiment: str = "debug",
                 server: str = "http://localhost:5000"):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)
        self.global_step = 1
        mlflow.set_experiment(experiment)
        mlflow.set_tracking_uri(server)

    def set_global_step(self):
        """increase step for logging"""
        self.global_step += 1

    def log_scalar(self, tag: str, value):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, self.global_step)
        mlflow.log_metric(tag, value)
        self.writer.flush()

    def log_scalars(self, info: dict):
        for tag, value in info.items():
            self.writer.add_scalar(tag, value, self.global_step)
            print(tag, value.item())
            mlflow.log_metric(tag, value.item())
        self.writer.flush()

    def log_confusion_matrix(self, tag, conf_mat, labels):
        """
        CM for tensorboard
        :param tag:
        :param conf_mat:
        :param labels: TODO print labels instead of class index
        :return:
        """
        cmap = plt.get_cmap('Blues')

        fig = plt.figure()
        plt.imshow(conf_mat)

        self.writer.add_figure(tag, fig)
        self.writer.flush()

    def log_image(self, tag, value):
        """
        log image to tensorboard
        :param tag:
        :param value: image or tensor of image
        :return:
        """
        # images in dataset are normalized --> change
        img = value.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = torch.from_numpy(img.transpose([2, 0, 1]))
        self.writer.add_image(tag, img, self.global_step)
        # y = torch.squeeze(value, 1)
        # self.writer.add_image(tag=tag, img_tensor=y)

    def log_images(self, tag, value, step):
        return
