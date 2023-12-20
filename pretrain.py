import torch
import torch.nn as nn
import pretrain.moco.builder
import pretrain.moco.loader
from pretrain.util.meter import *
from pretrain.network.pretrain import Pretrain
import time
from pretrain.util.LARS import LARS
import argparse
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument("data", metavar="DIR", help="path to dataset")
args = parser.parse_args()
print(args)
epochs = args.epochs

def train(train_loader, model, local_rank, rank, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    graph_losses = AverageMeter('graph', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, graph_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img1, img2) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if local_rank is not None:
            img1 = img1.cuda(local_rank, non_blocking=True)
            img2 = img2.cuda(local_rank, non_blocking=True)

        # compute output
        output, target, graph_loss = model(img1, img2)
        ce_loss = criterion(output, target)
        loss = ce_loss + graph_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 and rank == 0:
            progress.display(i)


def main():
    from torch.nn.parallel import DistributedDataParallel
    from pretrain.util.dist_init import dist_init
    
    rank, local_rank, world_size = dist_init()
    batch_size = args.batch_size_pergpu
    num_workers = 8
    base_lr = 0.01

    model = Pretrain()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': 1e-6, 'ignore': False}],
                                lr=base_lr, momentum=0.9, weight_decay=1e-6)

    optimizer = LARS(optimizer, eps=0.0)
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    torch.backends.cudnn.benchmark = True
    traindir = os.path.join(args.data, "train")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([pretrain.moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = datasets.ImageFolder(
        traindir, pretrain.moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    checkpoint_path = 'checkpoints/pretrain-{}.pth'.format(epochs)
    print('checkpoint_path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    

    model.train()
    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        train(train_loader, model, local_rank, rank, criterion, optimizer, epoch)
        
        if rank == 0:
            torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    main()
