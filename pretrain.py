import os
local_rank = int(os.environ["LOCAL_RANK"])
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.io import DictAction
from utils.io import import_class
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from net.byol import BYOL
from optim.lars import LARS
from optim.lars import exclude_bias_and_norm

parser = argparse.ArgumentParser(description='BYOL Training')
parser.add_argument('--workers', type=int, metavar='N',help='number of data loader workers')
parser.add_argument('--epochs', type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--batch-size', type=int, metavar='N',help='mini-batch size')
parser.add_argument('--learning_rate', type=float, metavar='LR',help='base learning rate')
parser.add_argument('--weight-decay', type=float, metavar='W',help='weight decay')
parser.add_argument('--print_freq', type=int, metavar='N',help='print frequency')
# If you do not start training from the last training end state, change the following directory
parser.add_argument('--checkpoint-dir', default='runs/pretrain/cs/', type=Path,metavar='DIR', help='path to checkpoint directory')
parser.add_argument('-c', '--config', default='config/train_cs.yaml', help='path to the configuration file')
# feeder
parser.add_argument('--train_feeder', help='train data loader will be used')
parser.add_argument('--train_feeder_args', action=DictAction, help='the arguments of data loader for training')
parser.add_argument('--l', type=float, help='the length of dataset')

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

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    
    set_seed()
    # 每个进程根据自己的local_rank设置应该使用的GPU
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    # 初始化分布式环境，主要用来帮助进程间通信
    torch.distributed.init_process_group(backend='nccl')
    
    if local_rank == 0:
        train_writer = SummaryWriter(args.checkpoint_dir)
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = BYOL(args).to(device)

    optimizer = LARS(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,weight_decay_filter=exclude_bias_and_norm,lars_adaptation_filter=exclude_bias_and_norm)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2,eta_min=0.0001)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
    else:
        start_epoch = 0

    train_feeder = import_class(args.train_feeder)
    train_dataset = train_feeder(**args.train_feeder_args)
    train_dataset = Subset(train_dataset, range(int(len(train_dataset)*args.l)))

    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    loss_min=float('inf')

    for epoch in range(start_epoch, args.epochs):
        loss_epoch=0
        print("Epoch:[{}/{}]".format(epoch+1,args.epochs))
        for batch_idx, ([data1, data2], label) in enumerate(tqdm(train_loader)):
            n_batch = len(train_loader)

            data1 = data1.float().to(device, non_blocking=True)
            data2 = data2.float().to(device, non_blocking=True)
            
            label = label.long().to(device, non_blocking=True)

            step = epoch * n_batch + batch_idx

            optimizer.zero_grad()
            
            loss = model.forward(data1, data2)
            loss_epoch+=loss.item()
            
            loss.backward()
            optimizer.step()
            
            lr = optimizer.param_groups[0]['lr']
            if local_rank == 0:
                train_writer.add_scalar('loss_step', loss.data.item(), step)
                train_writer.add_scalar('lr', lr, step)
        scheduler.step()
        print("Device{}: LR:{} Loss:{}".format(local_rank,lr,loss.item()))
        
        state = dict(epoch=epoch + 1, model=model.module.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
        if (epoch+1) % 10 == 0:
            torch.save(model.module.encoder.state_dict(),args.checkpoint_dir / 'ugcn_{}.pth'.format(epoch + 1))
        if loss_epoch<loss_min and epoch>100:
            loss_min=loss_epoch
            torch.save(model.module.encoder.state_dict(),args.checkpoint_dir / 'best.pth')
    torch.save(model.module.encoder.state_dict(),args.checkpoint_dir / 'ugcn.pth')