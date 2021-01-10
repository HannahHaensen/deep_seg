import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step + 1)
        self.writer.flush()

    def log_scalars(self, info, step):
        for tag, value in info.items():
            print(step+1)
            self.writer.add_scalar(tag, value, step + 1)
        self.writer.flush()

    def log_image(self, tag, value):
        return
        # y = torch.squeeze(value, 1)
        # self.writer.add_image(tag=tag, img_tensor=y)

    def log_images(self, tag, value, step):
        return