import datetime

import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset.voc_dataset import VOCSegmentation
from dataset.basic_dataset import DataSplit, SensorTypes

from trainer.metric_logger import SmoothedValue, MetricLogger, ConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np

writer = SummaryWriter()  # Saves the summaries to default directory names 'runs' in the current parent directory

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


def main():
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
        train_one_epoch(model=model, optimizer=optimizer, data_loader=data_loader,
                        device=device, epoch=epoch, print_freq=5, lr_scheduler=lr_scheduler)

        lr_scheduler.step()

        # evaluate after every epoch
        evaluate(model=model, data_loader=data_loader_eval, device=device, num_classes=21)

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


def evaluate(model, data_loader, device, num_classes):
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

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    # default `log_dir` is "runs" - we'll be more specific here

    for sensor, target in metric_logger.log_every(data_loader, print_freq, header):
        image = sensor[SensorTypes.Camera]
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output['out'], target.long())

        # Compute accuracy
        _, argmax = torch.max(output['out'], 1)
        accuracy = (target == argmax.squeeze()).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
              .format(epoch + 1, 10, loss.item(), accuracy.item()))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.item(), 'accuracy': accuracy.item()}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
            writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
        writer.flush()



if __name__ == "__main__":
    main()
    writer.close()
