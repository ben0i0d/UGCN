from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time
import yaml
import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchlight.io import DictAction
from torchlight.io import import_class
from torchlight.io import str2bool
import numpy as np
from tensorboardX import SummaryWriter

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

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

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


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()

def D(p, z, version='original'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return 	1-(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def handle_sigterm(signum, frame):
    pass

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(625, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, 625),
            nn.BatchNorm1d(625)
        )
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x,x0):
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x0)
        x0 = self.layer4(x0)   
        return x,x0

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
       
        return x
 
class BYOL(nn.Module):
    def __init__(self, args):
        super().__init__()
        Model = import_class(args.model)
        self.dev = args.dev

        self.encoder = Model(**args.model_args).to(args.dev)
      #  self.projector = MLP(256, args.projection_size, args.projection_hidden_size)
        
        self.use_momentum = True
        self.target_encoder = copy.deepcopy(self.encoder) #Model(**args.model_args).to(args.dev)
        #self.target_encoder.fc = MLP(256, args.projection_size, args.projection_hidden_size)
        self.target_ema_updater = EMA(beta=args.moving_average_decay)
        self.target_encoder1 = copy.deepcopy(self.encoder)
        self.predictor = prediction_MLP(args.projection_hidden_size, args.projection_size, args.projection_hidden_size)

        self.tt = args.tt
        self.ot = args.ot

        # create the queue
        self.K = args.K
        self.register_buffer("queue", torch.randn(args.projection_hidden_size, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.to(args.dev)
#        self.register_buffer("attn", torch.randn(self.K,25))
 #       self.attn = F.normalize(self.attn, dim=1)
        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(2, 3, image_size, image_size, device=device))
      #  self.forward(torch.randn(args.batch_size, 3, 50, 25, 2).cuda(), torch.randn(args.batch_size, 3, 50, 25, 2).cuda())

    def get_target_encoder(self):
        if self.target_encoder==None:
            self.target_encoder = copy.deepcopy(self.encoder)
        self.update_moving_average()
        set_requires_grad(self.target_encoder, False)
        return self.target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.encoder)

    def get_target_encoder1(self):
        if self.target_encoder1==None:
            self.target_encoder1 = copy.deepcopy(self.encoder)
        self.update_moving_average1()
        set_requires_grad(self.target_encoder1, False)
        return self.target_encoder1

    def reset_moving_average1(self):
        del self.target_encoder1
        self.target_encoder1 = None

    def update_moving_average1(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder1 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder1, self.encoder)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
    
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    def graph(self,data):
        N,C,T,V,M=data.size()
        data1=data[:,:,1:,:,:]-data[:,:,:-1,:,:]
        data1=torch.pow(data1,2)
        data2=torch.sqrt(data1.sum(1))
        data3=data2.mean(1)
        data3=data3.permute(0,2,1).contiguous()
        data3=data3.view(N*M,-1)
        
        return data3
    #Train
    def forward(
        self,
        x1, x2,x3,
        return_embedding = False,
        return_projection = True
    ):
        
     #   a=self.graph(x1)
    #    score=torch.abs(a.unsqueeze(1)-self.attn.unsqueeze(0)).mean(2)
   #     _,index=torch.topk(score,65536)

        f,h=self.encoder,self.predictor
       
        z1,_= f(x1,return_projection = True)

        z2,_ = f(x2,return_projection = True)
        z3,y = f(x3,return_projection = True)
       # A2=self.graph(x1)
     
        p1 = h(z1)
        p2 = h(z2)
       
        y=  h(y)
        with torch.no_grad():
            target_encoder = self.get_target_encoder() if self.use_momentum else self.encoder
            x11,_= target_encoder(x1,return_projection = True)
        
            x22,_= target_encoder(x2,return_projection = True)
           

            x11.detach_()
            x22.detach_()
          #  target_encoder1 = self.get_target_encoder1() if self.use_momentum else self.encoder
          #  x01= target_encoder1(x1,return_projection = True)
        
           # x02= target_encoder1(x3,return_projection = True)
           # x01.detach_()
           #x02.detach_()
       
        loss1=loss_fn(p1,x22.detach())
        loss2=loss_fn(p2,x11.detach())

        loss3=loss_fn(y,p1)
        #loss4=loss_fn(p3,x01.detach())
        #loss3=loss_fn(p3,p1)
        
       # B,C=z1.size()
     
      #  index=index.unsqueeze(1).expand(B,C,65536)
       # self.queue1=self.queue.unsqueeze(0).expand(B,C,self.K)
        #self.queue2= self.queue1.gather(dim=2,index=index)
     #   logitsq = torch.einsum('nc,ck->nk', [F.normalize(p1, dim=-1), self.queue.clone().detach()])
      # logitsk = torch.einsum('nc,ck->nk', [F.normalize(x22, dim=-1), self.queue.clone().detach()])

        
        # Calcutale loss between logitsk and logitsq, logitsq and logitsq_drop
#        loss_1 = - torch.sum(F.softmax(logitsk.detach() / self.tt, dim=1) * F.log_softmax(logitsq / self.ot, dim=1), dim=1).mean()
       
 #       self._dequeue_and_enqueue(F.normalize(x22, dim=-1))
        loss=loss1+loss2+loss3
        return loss.mean()

def exclude_bias_and_norm(p):
    return p.ndim == 1

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def loss_fn(x, y):
   
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim = 4096):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(625, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, 625),
            nn.BatchNorm1d(625)
        )
  

    def forward(self, x,x0):
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x0)
        x0 = self.layer4(x0)   
        return x,x0


if __name__ == '__main__':
    main()
