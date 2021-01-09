import datetime
import math
import time

import torch
import torchvision

from dataset.voc_dataset import VOCSegmentation
from dataset.basic_dataset import DataSplit, SensorTypes
from trainer.metric_logger import SmoothedValue, MetricLogger, ConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
# default `log_dir` is "runs" - we'll be more specific her

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
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
        # if args.distributed:
        #    train_sampler.set_epoch(epoch)
        dataiter = iter(data_loader)
        images, labels = dataiter.next()
        img_grid = torchvision.utils.make_grid(images[SensorTypes.Camera])

        matplotlib_imshow(img_grid, one_channel=True)

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
        evaluate(model=model, data_loader=data_loader_eval, device=device, num_classes=21)

    tb.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def loading_data_set(root: str = None):
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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # writer.add_scalar('sin', math.sin(angle_rad), step)



if __name__ == "__main__":
    main()