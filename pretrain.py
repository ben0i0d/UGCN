from pathlib import Path
import argparse
import math
import random
import sys
import yaml
import torch
from torchlight.io import DictAction
from torchlight.io import import_class

from tqdm import tqdm

import numpy as np
from tensorboardX import SummaryWriter

from net.byol import BYOL
from optim.lars import LARS

parser = argparse.ArgumentParser(description='BYOL Training')
parser.add_argument('--dev', type=str, help='use dev')
parser.add_argument('--workers', type=int, metavar='N',help='number of data loader workers')
parser.add_argument('--epochs', type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch-size', type=int, metavar='N',help='mini-batch size')
parser.add_argument('--learning_rate', type=float, metavar='LR',help='base learning rate')
parser.add_argument('--weight-decay', type=float, metavar='W',help='weight decay')
parser.add_argument('--print-freq', type=int, metavar='N',help='print frequency')
# If you do not start training from the last training end state, change the following directory
parser.add_argument('--checkpoint-dir', default='runs/pretrain/cs/', type=Path,metavar='DIR', help='path to checkpoint directory')
parser.add_argument('-c', '--config', default='config/train_cs.yaml', help='path to the configuration file')
# feeder
parser.add_argument('--train_feeder', help='train data loader will be used')
parser.add_argument('--train_feeder_args', action=DictAction, help='the arguments of data loader for training')

# model
parser.add_argument('--model', help='the model will be used')
parser.add_argument('--model_args', action=DictAction, help='the arguments of model')
parser.add_argument('--model_target_args', action=DictAction, help='the arguments of model')


parser.add_argument('--moving_average_decay', type=float)
parser.add_argument('--projection_size', type=int)
parser.add_argument('--projection_hidden_size', type=int)
parser.add_argument('--drop_percent', type=float)

parser.add_argument('--K', type=float)
parser.add_argument('--tt', type=float)
parser.add_argument('--ot', type=float)

def init_seed(seed=1):
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(args, optimizer, loader_len, step):
    max_steps = args.epochs * loader_len
    warmup_steps = 5 * loader_len
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

def main_worker(gpu, args):
    train_writer = SummaryWriter('./runs/pretrain')

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BYOL(args).cuda(gpu)
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,weight_decay_filter=exclude_bias_and_norm,lars_adaptation_filter=exclude_bias_and_norm)
    
    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
    else:
        start_epoch = 0
    
    train_feeder = import_class(args.train_feeder)
    data_loader= torch.utils.data.DataLoader(
        dataset=train_feeder(**args.train_feeder_args),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # set True when memory is abundant
        num_workers=args.workers,
        drop_last=True,
        worker_init_fn=init_seed)
    loader_len = len(data_loader)

    loss_min=200

    for epoch in range(start_epoch, args.epochs):
        loss0=0
        print("Epoch:[{}/{}]".format(epoch+1,args.epochs))
        for batch_idx, ([data1, data2], label) in enumerate(tqdm(data_loader)):
            data1 = data1.float().to(args.dev, non_blocking=True)
            data2 = data2.float().to(args.dev, non_blocking=True)
            
            label = label.long().to(args.dev, non_blocking=True)

            step = epoch * loader_len + batch_idx
            lr = adjust_learning_rate(args, optimizer, loader_len, step)

            loss = model.forward(data1, data2, return_embedding=False, return_projection=True)
            loss0=loss+loss0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_writer.add_scalar('loss_step', loss.data.item(), step)
            lr = optimizer.param_groups[0]['lr']
            train_writer.add_scalar('lr', lr, step)          
        print("LR:{} Loss:{}".format(lr,loss.item()))
        state = dict(epoch=epoch + 1, model=model.state_dict(),optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
        if loss0<loss_min and epoch>100:
           loss_min=loss0
           torch.save(model.encoder.state_dict(),args.checkpoint_dir / 'best.pth')
        if (epoch+1) % 10 == 0:
            torch.save(model.encoder.state_dict(),args.checkpoint_dir / 'stgcn_{}.pth'.format(epoch + 1))
    torch.save(model.encoder.state_dict(),args.checkpoint_dir / 'stgcn.pth')

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

    main_worker(args.dev, args)