import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
            print(self.global_step)
            self.writer.add_scalar(tag, value, self.global_step)
        self.writer.flush()

    def log_confusion_matrix(self, output, target):
        pred = torch.argmax(output, 1)

        conf_mat = confusion_matrix(pred.view(-1), target.view(-1))
        fig = plt.figure()
        plt.imshow(conf_mat)

        self.writer.add_figure('confusion matrix eval', fig)
        self.writer.flush()

    def log_image(self, tag, value):
        return
        # y = torch.squeeze(value, 1)
        # self.writer.add_image(tag=tag, img_tensor=y)

    def log_images(self, tag, value, step):
        return