import datetime
import shutil
import time
from datetime import datetime
from enum import Enum

import mlflow
import numpy as np
import torch
import torchvision
from hydra.experimental import initialize, compose
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler

from dataset.simpsons_dataset import SimpsonsDataset
from dataset.voc_seg_dataset import VOCSegmentation, get_classes
from dataset.basic_dataset import DataSplit
from logger.metric_logger import MetricCalculator
from logger.training_logger import TrainingLogger

from tqdm import tqdm

from tensorboard import program


class FrameworkType(Enum):
    Classification = 0
    ObjectDetection = 1
    Segmentation = 2

class Trainer:

    def __init__(self, cfg: DictConfig, framework_type: FrameworkType):
        """Create a basic trainer for python training"""

        self.config = cfg
        self.metric_logger = MetricCalculator(
            labels=get_classes()
        )
        self.framework_type = framework_type
        self.writer_train = None
        self.writer_eval = None
        self.writer_eval_mean = None

    def set_writer(self, writer_train: TrainingLogger,
                   writer_eval: TrainingLogger,
                   writer_eval_mean: TrainingLogger) -> None:
        """
        tensorboard logger for train, eval and meanEval
        :param writer_train:
        :param writer_eval:
        :param writer_eval_mean:
        :return:
        """
        self.writer_train = writer_train
        self.writer_eval = writer_eval
        self.writer_eval_mean = writer_eval_mean

    def save_checkpoint(self, model, optimizer, save_path, epoch):
        """
        save model to .pth-file
        :param model: model
        :param optimizer:
        :param save_path:
        :param epoch: saved epoch
        :return: None
        """
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

    def load_checkpoint(self, model, optimizer, load_path):
        """
        load saved model
        :param model:
        :param optimizer:
        :param load_path:
        :return:
        """
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return model, optimizer, epoch

    def loading_data_set(self):
        """
        framework can be used for classification or segmentation

        future: obejct detection
        :return:
        """
        # Data loading code
        if self.framework_type == FrameworkType.Classification:
            # TODO
            #  change dataset in outer method
            dataset = SimpsonsDataset(cfg=self.config)
            batch_size = 16
            validation_split = .2
            shuffle_dataset = True
            random_seed = 42

            # Creating data indices for training and validation splits:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      sampler=train_sampler)
            data_loader_eval = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                           sampler=valid_sampler)
        else:
            # TODO
            #  change dataset in outer method
            train_dataset = VOCSegmentation(split=DataSplit.Train, cfg=self.config)
            eval_dataset = VOCSegmentation(split=DataSplit.Eval, cfg=self.config)
            data_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=20,
                shuffle=True,
                num_workers=1,
                pin_memory=True)
            data_loader_eval = torch.utils.data.DataLoader(
                eval_dataset, batch_size=1,
                num_workers=1)
        print('Loading data...')

        print('Done...')
        return data_loader, data_loader_eval

    def main(self):
        """
        main running method
        :return:
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()

        data_loader, data_loader_eval = self.loading_data_set()

        print("Creating model")
        model = None
        if self.framework_type == FrameworkType.Segmentation:
            # using deeplab atm for sem seg, also individual models can be used
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                pretrained=True,
                num_classes=self.config.dataset.number_of_classes)
        elif self.framework_type == FrameworkType.Classification:
            model = torchvision.models.resnet50(pretrained=True)
        else:
            NotImplementedError
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params, lr=cfg.learning_process.learning_rate, weight_decay=0.001)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2)

        print("Start training")
        start_time = time.time()

        criterion = torch.nn.CrossEntropyLoss()
        if self.config.learning_process.ignore_index != 'None':
            criterion = torch.nn.CrossEntropyLoss(ignore_index=self.config.learning_process.ignore_index)

        # TODO make epochs value in hydra config
        for _ in tqdm(iterable=range(0, 10), desc="Epoch"):
            self.train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader,
                                 num_classes=self.config.dataset.number_of_classes, device=device,
                                 lr_scheduler=lr_scheduler, criterion=criterion)

            lr_scheduler.step()

            # evaluate after every epoch
            self.evaluate(model=model, data_loader=data_loader_eval,
                          device=device, num_classes=self.config.dataset.number_of_classes, criterion=criterion)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def evaluate(self, model, data_loader, device, num_classes, criterion):
        """
        eval model
        :param model: model
        :param data_loader: eval data loader
        :param device: either cuda or cpu
        :param num_classes: number of classes in dataset
        :param criterion: Loss --> TODO add default CE
        :return:
        """
        model.eval()

        # Initialize the prediction and label lists(tensors)
        pred_list = torch.zeros(0, dtype=torch.long, device=device)
        target_list = torch.zeros(0, dtype=torch.long, device=device)

        m_iou_list_eval_mean = []

        loss_list = []
        with torch.no_grad():
            for input_img, target in tqdm(data_loader, mininterval=10.0, desc="Eval"):
                image = input_img
                image, target = image.to(device), target.to(device)
                output = model(image)
                output = output['out']

                loss = criterion(output, target.long())
                loss_list.append(loss)

                # 1. Log scalar values (scalar summary)
                if self.framework_type == FrameworkType.Segmentation:
                    m_iou_list_eval_mean.append(
                        self.metric_logger.calculate_seg_metrics_for_epoch(writer=self.writer_eval,
                                                                           loss=loss,
                                                                           image=image[-1, :, :, :],
                                                                           num_classes=num_classes,
                                                                           output=output,
                                                                           target=target))

                    # Append batch prediction results
                    _, preds = torch.max(output, 1)
                    pred_list = torch.cat([pred_list, preds.view(-1).cpu()])
                    target_list = torch.cat([target_list, target.view(-1).cpu()])
                elif self.framework_type == FrameworkType.Classification:
                    correct, total = self.metric_logger.calculate_classification_metrics_for_epoch(
                        writer=self.writer_train,
                        loss=loss,
                        outputs=output,
                        labels=target,
                        image=image[-1, :, :, :],
                        is_train=True
                    )
                    correct += correct
                    total += total

        # eval mean values
        loss_list = np.array(loss_list, dtype=np.float32)
        info = {
            '/mLoss': loss_list.mean()
        }
        if self.framework_type == FrameworkType.Segmentation:
            m_iou_list_eval_mean = np.array(m_iou_list_eval_mean, dtype=np.float32)
            info['mIoU'] = m_iou_list_eval_mean.mean()

        self.writer_eval_mean.log_scalars(info)
        self.metric_logger.update_matrix(pred_list.numpy(),
                                         target_list.numpy(), "eval/CM", self.writer_eval_mean)
        self.writer_eval_mean.set_global_step()

    def train_one_epoch(self, model, optimizer, data_loader, num_classes,
                        lr_scheduler, device, criterion):
        """
        train one epoch
        :param model: model which is trained
        :param optimizer: optimizer --> ADAM or SGD
        :param data_loader: Train loader
        :param num_classes:
        :param lr_scheduler: e.g. Poly or Linear or Exp
        :param device: cuda or cpu
        :param criterion:
        :return:
        """
        model.train()
        correct = 0
        total = 0
        pred_list = torch.zeros(0, dtype=torch.long, device=device)
        target_list = torch.zeros(0, dtype=torch.long, device=device)

        for input_image, target in tqdm(data_loader, desc="Train"):
            image = input_image
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output
            if self.framework_type == FrameworkType.Segmentation:
                output = output['out']

            loss = criterion(output, target.long())
            if self.framework_type == FrameworkType.Classification:
                # TODO simple classification not tested
                raise NotImplementedError
            if self.framework_type == FrameworkType.Segmentation:
                self.metric_logger.calculate_seg_metrics_for_epoch(writer=self.writer_train, loss=loss,
                                                                   num_classes=num_classes, output=output,
                                                                   image=image[-1, :, :, :],
                                                                   target=target, is_train=True)
            elif self.framework_type == FrameworkType.ObjectDetection:
                # TODO
                raise NotImplementedError
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()


if __name__ == "__main__":

    now = datetime.now()

    shutil.rmtree('../runs/', ignore_errors=True)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', '../runs'])

    url = tb.launch()

    # logger for train eval and mean
    writer1 = TrainingLogger(log_dir='../runs/training_logger')
    writer2 = TrainingLogger(log_dir='../runs/eval_logger')
    writer3 = TrainingLogger(log_dir='../runs/mean_eval_logger')

    initialize(config_path="../config/", job_name="test_app")
    cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))
    with mlflow.start_run():
        # Log our parameters into mlflow
        for key, value in cfg.items():
            mlflow.log_param(key, value)

    trainer = Trainer(cfg=cfg, framework_type=FrameworkType.Segmentation)

    trainer.set_writer(writer_train=writer1,
                       writer_eval=writer2,
                       writer_eval_mean=writer3)

    # Saves the summaries to default directory names 'runs' in the current parent directory
    trainer.main()
