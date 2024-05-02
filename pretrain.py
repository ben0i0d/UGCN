from pathlib import Path
import argparse
import json
import math
import random
import sys
import time
import yaml
import torch
from torchlight.io import DictAction
from torchlight.io import import_class
from torchlight.io import str2bool
import numpy as np
from tensorboardX import SummaryWriter

from net.byol import BYOL
from optim.lars import LARS

parser = argparse.ArgumentParser(description='BYOL Training')
# parser.add_argument('data', type=Path, metavar='DIR',help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',help='number of data loader workers')
parser.add_argument('--epochs', default=600, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch-size', default=512, type=int, metavar='N',help='mini-batch size')
parser.add_argument('--learning-rate', default=2, type=float, metavar='LR',help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',help='weight decay')
parser.add_argument('--print-freq', default=200, type=int, metavar='N',help='print frequency')
# If you do not start training from the last training end state, change the following directory
parser.add_argument('--checkpoint-dir', default='runs/pretrain/1/', type=Path,metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--stream', default='joint')
parser.add_argument('-c', '--config', default='config/train_cs.yaml', help='path to the configuration file')
# feeder
parser.add_argument('--train_feeder', default='feeder.feeder', help='train data loader will be used')
parser.add_argument('--test_feeder', default='feeder.feeder', help='test data loader will be used')
parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')

# model
parser.add_argument('--model', default=None, help='the model will be used')
parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
parser.add_argument('--model_target_args', action=DictAction, default=dict(), help='the arguments of model')

parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
parser.add_argument('--moving_average_decay', default=0.99, type=float)
parser.add_argument('--projection_size', default=512, type=int)
parser.add_argument('--projection_hidden_size', default=256, type=int)
parser.add_argument('--drop_percent', default=0.2, type=float)

parser.add_argument('--K', default=512, type=float)
parser.add_argument('--tt', default=0.03, type=float)
parser.add_argument('--ot', default=0.1, type=float)

def init_seed(seed=1):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
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

def main_worker(gpu, args):
    train_writer = SummaryWriter('./runs/pretrain')

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BYOL(args).cuda(gpu)
    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
    start_epoch = 0
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

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
    loss_min=200

    for epoch in range(start_epoch, args.epochs):
        loader = data_loader['train']
        loss0=0
        for batch_idx, ([data1, data2,data3], label) in enumerate(loader):

            bsz = data1.size(0)
            n_batch = len(loader)
            data1 = data1.float().to(args.dev, non_blocking=True)
            data2 = data2.float().to(args.dev, non_blocking=True)
            data3 = data3.float().to(args.dev, non_blocking=True)
            label = label.long().to(args.dev, non_blocking=True)
            
          
            #data3=data3+noise
            if args.stream == 'joint':
                pass
            elif args.stream == 'motion':
                motion1 = torch.zeros_like(data1)
                motion2 = torch.zeros_like(data2)

                motion1[:, :, :-1, :, :] = data1[:, :, 1:, :, :] - data1[:, :, :-1, :, :]
                motion2[:, :, :-1, :, :] = data2[:, :, 1:, :, :] - data2[:, :, :-1, :, :]

                data1 = motion1
                data2 = motion2
            elif args.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone1 = torch.zeros_like(data1)
                bone2 = torch.zeros_like(data2)

                for v1, v2 in Bone:
                    bone1[:, :, :, v1 - 1, :] = data1[:, :, :, v1 - 1, :] - data1[:, :, :, v2 - 1, :]
                    bone2[:, :, :, v1 - 1, :] = data2[:, :, :, v1 - 1, :] - data2[:, :, :, v2 - 1, :]

                data1 = bone1
                data2 = bone2
            else:
                raise ValueError

            step = epoch * n_batch + batch_idx
            lr = adjust_learning_rate(args, optimizer, loader, step)

            loss = model.forward(data1, data2, data3, return_embedding=False, return_projection=True)
            loss0=loss+loss0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_writer.add_scalar('loss_step', loss.data.item(), step)
            lr = optimizer.param_groups[0]['lr']
            train_writer.add_scalar('lr', lr, step)

            if step % args.print_freq == 0:
                stats = dict(epoch=epoch+1, step=step, learning_rate=lr,
                           loss=loss.item(),
                           time=int(time.time() - start_time))

                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
        # scheduler.step()
        state = dict(epoch=epoch + 1, model=model.state_dict(),optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
        if loss0<loss_min :
           loss_min=loss0
           torch.save(model.encoder.state_dict(),args.checkpoint_dir / 'best.pth')
        if (epoch+1) % 10 == 0:
            torch.save(model.encoder.state_dict(),args.checkpoint_dir / 'stgcn_{}.pth'.format(epoch + 1))
    torch.save(model.encoder.state_dict(),args.checkpoint_dir / 'stgcn.pth')

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 5 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def exclude_bias_and_norm(p):
    return p.ndim == 1

if __name__ == '__main__':
    main()
