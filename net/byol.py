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
        
        self.use_momentum = True
        self.target_encoder = copy.deepcopy(self.encoder)
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

    #Train
    def forward(self,x1, x2,return_embedding = False,return_projection = True):
        
        f,h=self.encoder,self.predictor
       
        z1= f(x1,return_projection = True)

        z2 = f(x2,return_projection = True)
     
        p1 = h(z1)
        p2 = h(z2)
       
        with torch.no_grad():
            target_encoder = self.get_target_encoder() if self.use_momentum else self.encoder
            x11= target_encoder(x1,return_projection = True)
        
            x22= target_encoder(x2,return_projection = True)
          
            x11.detach_()
            x22.detach_()

        loss1=loss_fn(p1,x22.detach())
        loss2=loss_fn(p2,x11.detach())

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