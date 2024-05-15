import copy
import torch
from torch import nn
import torch.nn.functional as F
from torchlight.io import import_class

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
        x1, x2,
        return_embedding = False,
        return_projection = True
    ):
        
     #   a=self.graph(x1)
    #    score=torch.abs(a.unsqueeze(1)-self.attn.unsqueeze(0)).mean(2)
   #     _,index=torch.topk(score,65536)

        f,h=self.encoder,self.predictor
       
        z1= f(x1,return_projection = True)

        z2 = f(x2,return_projection = True)
        #z3 = f(x3,return_projection = True)
       # A2=self.graph(x1)
     
        p1 = h(z1)
        p2 = h(z2)
       
        #y=  h(z3)
        with torch.no_grad():
            target_encoder = self.get_target_encoder() if self.use_momentum else self.encoder
            x11= target_encoder(x1,return_projection = True)
        
            x22= target_encoder(x2,return_projection = True)
          

            x11.detach_()
            x22.detach_()
          #  target_encoder1 = self.get_target_encoder1() if self.use_momentum else self.encoder
          #  x01= target_encoder1(x1,return_projection = True)
        
           # x02= target_encoder1(x3,return_projection = True)
           # x01.detach_()
           #x02.detach_()
       
        loss1=loss_fn(p1,x22.detach())
        loss2=loss_fn(p2,x11.detach())

        #loss3=loss_fn(y,p1)
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
        loss=loss1+loss2
        return loss.mean()

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