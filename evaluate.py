import sys
import time
import yaml
import json
import random
import argparse
from pathlib import Path
import torch
from torch import nn, optim
from torchlight.io import DictAction
from torchlight.io import import_class
from torchlight.io import str2bool
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')

parser.add_argument('--pretrained', type=Path, default='./runs/pretrain/best.pth',metavar='FILE',help='path to pretrained model')
parser.add_argument('--weights', default='freeze', type=str,choices=('finetune', 'freeze'),help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,choices=(100, 10, 1),help='size of traing set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',help='mini-batch size')
parser.add_argument('--lr-backbone', default=0, type=float, metavar='LR',help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',help='weight decay')
parser.add_argument('--print-freq', default=1000, type=int, metavar='N',help='print frequency')
parser.add_argument('--checkpoint-dir', default='./runs/evaluate/1', type=Path,metavar='DIR', help='path to checkpoint directory')
parser.add_argument('-c', '--config', default='./config/evaluate_cs.yaml', help='path to the configuration file')
# feeder
parser.add_argument('--train_feeder', default='feeder.feeder', help='train data loader will be used')
parser.add_argument('--test_feeder', default='feeder.feeder', help='test data loader will be used')
parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
# model
parser.add_argument('--model', default=None, help='the model will be used')
parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')

parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(gpu, args):
    train_writer = SummaryWriter('./runs/evaluate/train')
    val_writer = SummaryWriter('./runs/evaluate/val')

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    Model = import_class(args.model)
    model = Model(**args.model_args).cuda(gpu)

    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#    assert missing_keys == ['fc.weight', 'fc.bias'] # and unexpected_keys == []
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    if args.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    data_loader = dict()

    if args.train_feeder_args:
        train_feeder = import_class(args.train_feeder)
        data_loader['train'] = torch.utils.data.DataLoader(
            dataset=train_feeder(**args.train_feeder_args),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,  # set True when memory is abundant
            num_workers=args.workers,
            drop_last=True,
            worker_init_fn=init_seed)

    if args.test_feeder_args:
        test_feeder = import_class(args.test_feeder)
        data_loader['test'] = torch.utils.data.DataLoader(
            dataset=test_feeder(**args.test_feeder_args),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers,
            drop_last=False,
            worker_init_fn=init_seed)

    start_time = time.time()
    step_val = 0
    bestout=np.zeros((16487,60))
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        # train_sampler.set_epoch(epoch)
        train_loader = data_loader['train']
        test_loader = data_loader['test']

        top1_train = AverageMeter('Acc@1_train')
        top5_train = AverageMeter('Acc@5_train')

        for batch_idx, (data, label, index) in enumerate(train_loader):
            data = data.float().to(args.dev, non_blocking=True)
            label = label.long().to(args.dev, non_blocking=True)
            n_batch = len(train_loader)

            output = model(data)
            loss = criterion(output, label)
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1_train.update(acc1[0].item(), data.size(0))
            top5_train.update(acc5[0].item(), data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step = epoch * n_batch + batch_idx
            train_writer.add_scalar('loss_train', loss.data.item(), step)

            if batch_idx % args.print_freq == 0:
                # torch.distributed.reduce(loss.div_(args.world_size), 0)
                pg = optimizer.param_groups
                lr_classifier = pg[0]['lr']
                lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                            lr_classifier=lr_classifier, loss=loss.item(), acc1=top1_train.avg, acc5=top5_train.avg,
                            time=int(time.time() - start_time))
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)

        train_writer.add_scalar('Acc@1', top1_train.avg, epoch)

        # evaluate
        model.eval()
        predicted_labels = torch.zeros(len(test_loader.dataset)).long()
        top1 = AverageMeter('Acc@1')
        top5 = AverageMeter('Acc@5')
        score_flag = []
        with torch.no_grad():
            for data, target, index in test_loader:
                data = data.float().to(args.dev, non_blocking=True)
                target = target.long().to(args.dev, non_blocking=True)
                
                output = model(data)
                #target=target.squeeze(1)
                score_flag.append(output.data.cpu().numpy())
                _, predicted = torch.max(output.data, 1)
                predicted_labels[index] = predicted.cpu()
                loss_val = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0].item(), data.size(0))
                top5.update(acc5[0].item(), data.size(0))

                val_writer.add_scalar('loss_val', loss_val.data.item(), step_val)

            val_writer.add_scalar('Acc@1', top1.avg, epoch)
            score = np.concatenate(score_flag)
            
            if top1.avg > best_acc.top1:
                bestout=score

        best_acc.top1 = max(best_acc.top1, top1.avg)
        best_acc.top5 = max(best_acc.top5, top5.avg)
        stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)
        np.save(args.checkpoint_dir / "sfeature.npy",bestout)

        scheduler.step()
        state = dict(epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    p = parser.parse_args()

    if p.config is not None:
        # load config file
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)

        # update parser from config file
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('Unknown Arguments: {}'.format(k))
                assert k in key

        parser.set_defaults(**default_arg)

    args = parser.parse_args()

    if args.use_gpu:
        args.dev = "cuda:0"
    else:
        args.dev = "cpu"

    main_worker(args.dev, args)
