"""
Based on training script by Wei YANG, 2017
Taken from repo : https://github.com/bearpaw/pytorch-classification
Copyright (c) Wei YANG, 2017
"""

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

try:
    import wandb
    got_wandb = True
except ImportError:
    got_wandb = False

from utils import Logger, AverageMeter, accuracy, GroupNormCreator
from models import resnet110
from weight_regularization import calc_group_reg_loss


NUM_GROUPS_GN = 32
MIN_NUM_CHANNELS = 4

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

# Datasets
parser.add_argument('-d', '--data', help='dataset name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--data-path', type=str, help='path to folder containing the cifar datasets')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 128)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default=None, type=str, metavar='PATH',
                    help='path to save checkpoint. If None, set path according to current time (default: None)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--norm', default='BN', type=str, help='Normalization layer type', choices=['BN', 'GN'])
# Miscs
parser.add_argument('--print-freq', type=int, default=50, help='print frequency')
parser.add_argument('--seed', type=int, help='manual seed', default=0)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--wandb-off', action='store_true', default=False)
parser.add_argument('--checkpoint-off', action='store_true', default=False)
parser.add_argument('--notes', type=str, default=None)

# GOR related
parser.add_argument('--reg-type', type=str, help='Set GOR variant', default=None,
                    choices=['inter', 'intra', None])
parser.add_argument('--ortho-decay', type=float, help='GOR strength (lambda)', default=1e-2)
parser.add_argument('--names-to-reg', type=str, help='If given, only layers with name that matches this string'
                                                     'will be regularized. If None, all layer are regularized',
                    default=None)
parser.add_argument('-n', '--num-groups', type=int, help='Number of regularization groups in GOR',
                    default=NUM_GROUPS_GN)

args = parser.parse_args()

# Use CUDA
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'


def print_flare(s: str):
    print('<' + '=' * 5 + s + '=' * 5 + '>')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.data == 'cifar10':
        dataset = datasets.CIFAR10
        num_classes = 10
    else:
        dataset = datasets.CIFAR100
        num_classes = 100

    trainset = dataset(root=os.path.join(args.data_path, args.data), train=True, download=True,
                       transform=transform_train)

    # Setting seed to loader generator. Each worker process is seeded using seed_worker() method
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch,
                                               shuffle=True,
                                               num_workers=args.workers, worker_init_fn=seed_worker, generator=g)

    testset = dataset(root=os.path.join(args.data_path, args.data), train=False, download=False,
                         transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=True,
                                             num_workers=args.workers)

    return train_loader, val_loader, num_classes


def main():
    # Fix random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    best_acc = 0
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    use_chkpt = not args.checkpoint_off
    if not use_chkpt:
        print_flare('Checkpoints are disabled!')

    if args.checkpoint is None:
        # Create path according to time
        args.checkpoint = os.path.join('checkpoints', time.strftime('%d_%m_%Y-%H_%M_%S'))

    os.makedirs(args.checkpoint, exist_ok=True)
    args.checkpoint = os.path.abspath(args.checkpoint)
    print(f'Log/checkpoint dir is {args.checkpoint}')

    use_wandb = got_wandb and not args.wandb_off
    if use_wandb:
        wandb.init(project='GOR', config=vars(args), notes=args.notes)
        wandb.run.log_code(".")
        wandb.summary['checkpoint dir'] = os.path.abspath(args.checkpoint)

    # Data loading code
    train_loader, val_loader, num_classes = get_loaders(args)

    # create model
    # Set norm type
    if args.norm == 'BN':
        norm_layer = nn.BatchNorm2d
    elif args.norm == 'GN':
        norm_layer = GroupNormCreator(NUM_GROUPS_GN, MIN_NUM_CHANNELS)
    else:
        raise Exception(f'Unsupported norm type {args.norm}')

    model = resnet110(norm_layer=norm_layer).to(device)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule,
                                                        last_epoch=args.start_epoch - 1)

    # for resnet110 original paper uses lr=0.01 for first 400 mini-batches for warm-up
    # then switch back. In this setup it will correspond for first epoch.
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * 0.1

    # Resume
    if args.resume:
        # Load checkpoint.
        print(f'==> Resuming from checkpoint : {args.resume}')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            # checkpoints contains other data than the model's weights
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']

            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint.keys():
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        if epoch == 1:
            # for resnet110 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        current_lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current_lr))

        train_loss, train_acc, reg_loss, class_loss = train(train_loader, model, criterion, optimizer, epoch,
                                                            use_cuda, args)
        test_loss, test_acc = test(val_loader, model, criterion, use_cuda)

        if use_wandb:
            # Log training statistics to W&B
            log_dict = {'train_loss': train_loss, 'epoch': epoch, 'val_loss': test_loss, 'val acc': test_acc,
                        'reg_loss': reg_loss, 'classification loss': class_loss,
                        'lr': current_lr, 'weight_decay': optimizer.param_groups[0]['weight_decay']}
            wandb.log(log_dict)

        # append logger file
        logger.append([current_lr, train_loss, test_loss, train_acc, test_acc])
        lr_scheduler.step()

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if use_chkpt:
            is_ddp = isinstance(model, nn.parallel.DistributedDataParallel)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if is_ddp else model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, is_best, checkpoint=args.checkpoint)

    logger.close()

    print('Best acc:')
    print(best_acc)

    if use_wandb:
        wandb.summary['best top1'] = best_acc
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, use_cuda, args):
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    reg_loss = AverageMeter()
    task_losses = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_size = inputs.size(0)
        if batch_size < args.train_batch:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        task_loss = criterion(outputs, targets)

        if args.reg_type is not None:
            # Calculate GOR loss
            ortho_loss = calc_group_reg_loss(model, num_groups=args.num_groups, reg_type=args.reg_type)
            loss = task_loss + args.ortho_decay * ortho_loss
        else:
            loss = task_loss

        if torch.isnan(loss) or torch.isinf(loss):
            raise Exception(f'Bad loss value, got {loss.item()}. Stopping run.')

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        task_losses.update(task_loss.item(), inputs.shape[0])
        if args.reg_type is not None:
            reg_loss.update(ortho_loss.item(), inputs.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'top1 acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_idx, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

    return losses.avg, top1.avg, reg_loss.avg, task_losses.avg


def test(val_loader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
