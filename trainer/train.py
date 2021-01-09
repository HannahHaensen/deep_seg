import datetime
import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils

from dataset.voc_dataset import VOCSegmentation
from dataset.basic_dataset import DataSplit, SensorTypes
from trainer.metric_logger import SmoothedValue, MetricLogger

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    data_loader, data_loader_eval = loadingDataSet(root='/Users/hannahschieber/GitHub/deep_seg/data')

    print("Creating model")
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=21)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.001, momentum=0.0009, weight_decay=0.001)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.2)

    print("Start training")
    start_time = time.time()
    for epoch in range(0, 10):
        # if args.distributed:
        #    train_sampler.set_epoch(epoch)
        train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader,
                        device=device, epoch=epoch, print_freq=5, lr_scheduler=lr_scheduler)
        lr_scheduler.step()
        # if args.output_dir:
        #    utils.save_on_master({
        #        'model': model_without_ddp.state_dict(),
        #        'optimizer': optimizer.state_dict(),
        #        'lr_scheduler': lr_scheduler.state_dict(),
        #        'args': args,
        #        'epoch': epoch},
        #        os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_eval, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def loadingDataSet(root: str = None):
    # Data loading code
    print("Loading data")
    train_dataset = VOCSegmentation(split=DataSplit.Train, root=root)
    eval_dataset = VOCSegmentation(split=DataSplit.Eval, root=root)
    print("Creating data loaders")
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


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for sensor, target in metric_logger.log_every(data_loader, print_freq, header):
        image = sensor[SensorTypes.Camera]
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output['out'], target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


if __name__ == "__main__":
    main()