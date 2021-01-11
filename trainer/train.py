import datetime
import time
from datetime import datetime

import numpy as np
import torch
import torchvision

from dataset.voc_dataset import VOCSegmentation, get_classes
from dataset.basic_dataset import DataSplit, SensorTypes
from logger.metric_logger import MetricCalculator
from logger.tensorboard_logger import TensorboardLogger

from tqdm import tqdm


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
        self.metric_logger = MetricCalculator(
            labels=get_classes()
        )

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

    def main(self, num_classes=21):
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

        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        test = 0
        for _ in tqdm(iterable=range(0, 10), desc="Epoch:"):
            self.train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader,
                                 num_classes=num_classes, device=device,
                                 lr_scheduler=lr_scheduler, criterion=criterion)

            lr_scheduler.step()

            # evaluate after every epoch
            self.evaluate(model=model, data_loader=data_loader_eval,
                          device=device, num_classes=21, criterion=criterion)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def evaluate(self, model, data_loader, device, num_classes, criterion):

        model.eval()

        # Initialize the prediction and label lists(tensors)
        pred_list = torch.zeros(0, dtype=torch.long, device=device)
        target_list = torch.zeros(0, dtype=torch.long, device=device)

        m_iou_list_eval_mean = []

        loss_list = []
        with torch.no_grad():
            for sensor, target in tqdm(data_loader, mininterval=10.0):
                image = sensor[SensorTypes.Camera]
                image, target = image.to(device), target.to(device)
                output = model(image)
                output = output['out']

                loss = criterion(output, target.long())
                loss_list.append(loss)

                # 1. Log scalar values (scalar summary)
                m_iou_list_eval_mean.append(
                    self.metric_logger.calculate_metrics_for_epoch(writer=self.writer_eval,
                                                                   loss=loss,
                                                                   num_classes=num_classes,
                                                                   output=output,
                                                                   target=target))

                # Append batch prediction results
                _, preds = torch.max(output, 1)
                pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
                target_list = torch.cat([target_list, target.view(-1).cpu()])

        # eval mean values
        loss_list = np.array(loss_list, dtype=np.float32)
        m_iou_list_eval_mean = np.array(m_iou_list_eval_mean, dtype=np.float32)
        info = {
            '/mLoss': loss_list.mean(),
            '/mIoU': m_iou_list_eval_mean.mean()
        }
        self.writer_eval_mean.log_scalars(info)
        self.metric_logger.update_matrix(pred_list.numpy(),
                                         target_list.numpy(), "eval/CM", self.writer_eval_mean)
        self.writer_eval_mean.set_global_step()

    def train_one_epoch(self, model, optimizer, data_loader, num_classes,
                        lr_scheduler, device, criterion):

        model.train()

        for sensor, target in tqdm(data_loader):
            image = sensor[SensorTypes.Camera]
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            loss = criterion(output, target.long())

            self.metric_logger.calculate_metrics_for_epoch(writer=self.writer_train, loss=loss,
                                                           num_classes=num_classes, output=output,
                                                           target=target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()


if __name__ == "__main__":
    now = datetime.now()
    writer1 = TensorboardLogger(log_dir='../runs/training_logger' + now.strftime("%Y%m%d-%H%M%S") + "/")
    writer2 = TensorboardLogger(log_dir='../runs/eval_logger' + now.strftime("%Y%m%d-%H%M%S") + "/")
    writer3 = TensorboardLogger(log_dir='../runs/mean_eval_logger' + now.strftime("%Y%m%d-%H%M%S") + "/")
    trainer = Trainer(image_dir='/Users/hannahschieber/GitHub/deep_seg/data',
                      writer_train=writer1,
                      writer_eval=writer2,
                      writer_eval_mean=writer3)

    # Saves the summaries to default directory names 'runs' in the current parent directory
    trainer.main()
