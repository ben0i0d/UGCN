import yaml
import random
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler

from torchlight.io import DictAction
from torchlight.io import import_class

import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
parser.add_argument('--dev', type=str, help='device')
parser.add_argument('--pretrained', type=Path, default='./runs/pretrain/cv/stgcn_470.pth',metavar='FILE',help='path to pretrained model')
parser.add_argument('--checkpoint-dir', default='./runs/evaluate/cv1', type=Path,metavar='DIR', help='path to checkpoint directory')
parser.add_argument('-c', '--config', default='./config/evaluate_cv.yaml', help='path to the configuration file')
parser.add_argument('--weights', default='freeze', type=str,choices=('finetune', 'freeze'),help='finetune or freeze resnet weights')
parser.add_argument('--workers',type=int, metavar='N',help='number of data loader workers')
parser.add_argument('--epochs',type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch_size',type=int, metavar='N',help='mini-batch size')
parser.add_argument('--learning_rate', type=float, metavar='LR',help='base learning rate')
parser.add_argument('--lr_backbone', type=float, metavar='LR',help='backbone base learning rate')
parser.add_argument('--lr_classifier', type=float, metavar='LR',help='classifier base learning rate')
parser.add_argument('--weight_decay', type=float, metavar='W',help='weight decay')
# feeder
parser.add_argument('--train_feeder', help='train data loader will be used')
parser.add_argument('--test_feeder', help='test data loader will be used')
parser.add_argument('--train_feeder_args', action=DictAction, help='the arguments of data loader for training')
parser.add_argument('--test_feeder_args', action=DictAction, help='the arguments of data loader for test')
# model
parser.add_argument('--model', help='the model will be used')
parser.add_argument('--model_args', action=DictAction, help='the arguments of model')

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    train_writer = SummaryWriter(args.checkpoint_dir)
    val_writer = SummaryWriter(args.checkpoint_dir)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    Model = import_class(args.model)
    model = Model(**args.model_args).to(args.dev)

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

    criterion = nn.CrossEntropyLoss().to(args.dev)

    param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler()

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

    train_feeder = import_class(args.train_feeder)
    test_feeder = import_class(args.test_feeder)

    trainloader = torch.utils.data.DataLoader(
        dataset=train_feeder(**args.train_feeder_args),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # set True when memory is abundant
        num_workers=args.workers,
        drop_last=True,
        worker_init_fn=init_seed)

    testloader = torch.utils.data.DataLoader(
        dataset=test_feeder(**args.test_feeder_args),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        drop_last=False,
        worker_init_fn=init_seed)
    
    step_val = 0
    bestout=np.zeros((len(np.load(args.train_feeder_args['data_path'],mmap_mode='r')['x_test']),args.model_args['num_class']))
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False

        top1_train = AverageMeter('Acc@1_train')
        top5_train = AverageMeter('Acc@5_train')

        print("Epoch:[{}/{}] Train".format(epoch+1,args.epochs))
        for batch_idx, (data, label, index) in enumerate(tqdm(trainloader)):
            data = data.float().to(args.dev, non_blocking=True)
            label = label.long().to(args.dev, non_blocking=True)
            n_batch = len(trainloader)

            optimizer.zero_grad()

            with autocast():
                output = model(data)
                loss = criterion(output, label)

            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1_train.update(acc1[0].item(), data.size(0))
            top5_train.update(acc5[0].item(), data.size(0))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step = epoch * n_batch + batch_idx
            train_writer.add_scalar('loss_train', loss.data.item(), step)

        print("Epoch: {} Acc@1: {} Acc@5: {} loss: {} ".format(epoch+1,top1_train.avg,top5_train.avg,loss.item()))
        train_writer.add_scalar('Acc@1', top1_train.avg, epoch)

        # evaluate
        model.eval()
        predicted_labels = torch.zeros(len(testloader.dataset)).long()
        top1 = AverageMeter('Acc@1')
        top5 = AverageMeter('Acc@5')
        score_flag = []
        print("Epoch:[{}/{}] Test".format(epoch+1,args.epochs))
        with torch.no_grad():
            for data, target, index in tqdm(testloader):
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
        print("Epoch: {} Acc@1: {} Acc@5: {} BestAcc@1: {} BestAcc@5: {}".format(epoch+1,top1.avg,top5.avg,best_acc.top1,best_acc.top5))
        np.save(args.checkpoint_dir / "pred.npy",bestout)

        scheduler.step()
        state = dict(epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
