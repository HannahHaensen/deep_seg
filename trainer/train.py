import datetime

import time

import torch
import torchvision

from dataset.voc_dataset import VOCSegmentation
from dataset.basic_dataset import DataSplit, SensorTypes
from logger.tensorboard_logging import TensorboardLogger
from metrics.intersection_over_union import IoU

from trainer.metric_logger import SmoothedValue, MetricLogger, ConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        plt.show()
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


def main(writer: TensorboardLogger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    data_loader, data_loader_eval = loading_data_set(root='/Users/hannahschieber/GitHub/deep_seg/data')

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
        loss, accuracy = train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader,
                        device=device, epoch=epoch, print_freq=5, lr_scheduler=lr_scheduler,
                                         writer=writer)

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss, 'accuracy': accuracy}
        writer.log_scalars(info, epoch)
        lr_scheduler.step()

        # evaluate after every epoch
        confmat = evaluate(model=model, data_loader=data_loader_eval,
                 device=device, num_classes=21, writer=writer, epoch=epoch)
        print(confmat)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def loading_data_set(root: str = None):
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


def evaluate(model, data_loader, device, num_classes, writer, epoch):
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

            # writer.log_image(tag="prediction", value=output)
            # writer.log_image(tag="target", value=target)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
        info = {'mIoU': 1}
        writer.log_scalars(info, epoch)

    return confmat


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, writer):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    # default `log_dir` is "runs" - we'll be more specific here
    losses = 0
    accuracy = 0

    for sensor, target in metric_logger.log_every(data_loader, print_freq, header):
        image = sensor[SensorTypes.Camera]
        image, target = image.to(device), target.to(device)
        output = model(image)

        # writer.log_image(tag="prediction", value=output['out'])
        # writer.log_image(tag="target", value=target)

        loss = criterion(output['out'], target.long())
        losses += loss.item()

        # Compute accuracy
        _, argmax = torch.max(output['out'], 1)
        accuracy = (target == argmax.squeeze()).float().mean()

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss, 'accuracy': accuracy, 'mIoU': 1}
        writer.log_scalars(info, epoch
                           )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return losses, accuracy





if __name__ == "__main__":
    writer = TensorboardLogger(log_dir='../runs/tensorboard_logger')
    # Saves the summaries to default directory names 'runs' in the current parent directory

    main(writer)

