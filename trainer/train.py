import datetime

import time

import torch
import torchvision

from dataset.voc_dataset import VOCSegmentation
from dataset.basic_dataset import DataSplit, SensorTypes
from logger.tensorboard_logging import TensorboardLogger

from trainer.metric_logger import SmoothedValue, MetricLogger, ConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, image_dir: str,
                 writer_train: TensorboardLogger,
                 writer_eval: TensorboardLogger,
                 writer_eval_mean: TensorboardLogger):
        """Create a basic trainer for python training"""
        self.image_dir = image_dir
        self.writer_train = writer_train
        self.writer_eval = writer_eval
        self.writer_eval_mean = writer_eval_mean

    def save_checkpoint(self, model, optimizer, save_path, epoch):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

    def load_checkpoint(self, model, optimizer, load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return model, optimizer, epoch

    def loading_data_set(self, root: str = None):
        # Data loading code
        train_dataset = VOCSegmentation(split=DataSplit.Train, root=root)
        eval_dataset = VOCSegmentation(split=DataSplit.Eval, root=root)
        print('Loading data...')
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=20,
            shuffle=True,
            num_workers=1,
            pin_memory=True)
        data_loader_eval = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1,
            num_workers=1)
        print('Done...')
        return data_loader, data_loader_eval

    def main(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        data_loader, data_loader_eval = self.loading_data_set(root=self.image_dir)

        print("Creating model")
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, num_classes=21)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=0.001, momentum=0.0009, weight_decay=0.001)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2)

        print("Start training")
        start_time = time.time()

        for epoch in range(0, 10):
            self.train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader,
                                                  device=device, epoch=epoch, print_freq=5, lr_scheduler=lr_scheduler)

            lr_scheduler.step()

            # evaluate after every epoch
            confmat = self.evaluate(model=model, data_loader=data_loader_eval,
                                    device=device, num_classes=21, epoch=epoch)
            print("confmat-----------")
            print(confmat.__str__())
            print("confmat-----------")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def evaluate(self, model, data_loader, device, num_classes, epoch):
        model.eval()
        confmat = ConfusionMatrix(num_classes)
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for sensor, target in metric_logger.log_every(data_loader, 100, header):
                image = sensor[SensorTypes.Camera]
                image, target = image.to(device), target.to(device)
                output = model(image)
                output = output['out']

                confmat.update(target.flatten(), output.argmax(1).flatten())

                # Compute accuracy
                _, argmax = torch.max(output, 1)
                accuracy = (target == argmax.squeeze()).float().mean()

                # 1. Log scalar values (scalar summary)
                info = {'mIoU': 1, 'accuracy': accuracy}
                self.writer_eval.log_scalars(info)
                self.writer_eval.log_confusion_matrix(output=output, target=target)
                self.writer_eval.set_global_step()
            confmat.reduce_from_all_processes()

        info = {'mIoU': 1}
        self.writer_eval_mean.log_scalars(info)
        self.writer_eval_mean(info)
        self.writer_eval_mean.set_global_step()
        return confmat

    def train_one_epoch(self, model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        model.train()
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
        header = 'Epoch: [{}]'.format(epoch)

        losses = 0
        accuracy = 0

        for sensor, target in metric_logger.log_every(data_loader, print_freq, header):
            image = sensor[SensorTypes.Camera]
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            loss = criterion(output, target.long())
            losses += loss.item()

            # Compute accuracy
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()

            # 1. Log scalar values (scalar summary)
            info = {'loss': loss, 'accuracy': accuracy, 'mIoU': 1}
            self.writer_train.log_scalars(info)
            self.writer_train.set_global_step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        return losses, accuracy


if __name__ == "__main__":
    writer1 = TensorboardLogger(log_dir='../runs/training_logger')
    writer2 = TensorboardLogger(log_dir='../runs/eval_logger')
    writer3 = TensorboardLogger(log_dir='../runs/mean_eval_logger')
    trainer = Trainer(image_dir='/Users/hannahschieber/GitHub/deep_seg/data',
                      writer_train=writer1,
                      writer_eval=writer2,
                      writer_eval_mean=writer3)

    # Saves the summaries to default directory names 'runs' in the current parent directory
    trainer.main()
