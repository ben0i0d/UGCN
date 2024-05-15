import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.att_drop import  CA_Drop
from net.agcn import unit_gcn
from net.ctrgcn import TCN_GCN_unit
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = self.graph.A #torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
       # self.register_buffer('A', A)
        #self.A1 = nn.Parameter(torch.tensor(np.sum(np.reshape(self.graph.A.astype(np.float32), [3, 25, 25]), axis=0), dtype=torch.float32, requires_grad=False, device='cuda'), requires_grad=False)

        # build networks
        spatial_kernel_size = 3
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * 25)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        #self.l1=st_gcn(in_channels, hidden_channels, kernel_size, 1,A, residual=False, **kwargs0)
        #self.l2=st_gcn(hidden_channels, hidden_channels, kernel_size, 1,A, **kwargs)
        #self.l3=st_gcn(hidden_channels, hidden_channels, kernel_size, 1,A, **kwargs)
        #self.l4=st_gcn(hidden_channels, hidden_channels, kernel_size, 1,A, **kwargs)
        #self.l5=st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2,A, **kwargs)
        #self.l6=st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1,A, **kwargs)
        #self.l7=st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1,A, **kwargs)
        #self.l8=st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2,A, **kwargs)
        #self.l9=st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1,A, **kwargs)
        #self.l10=st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1,A, **kwargs)
        self.l1 = TCN_GCN_unit(in_channels, hidden_channels, A, residual=False, adaptive=True)
        self.l2 = TCN_GCN_unit(hidden_channels, hidden_channels, A,stride=2, adaptive=True)
        self.l3 = TCN_GCN_unit(hidden_channels, hidden_channels, A, adaptive=True)
        self.l4 = TCN_GCN_unit(hidden_channels, 2*hidden_channels, A,stride=2, adaptive=True)
        self.l5 = TCN_GCN_unit(2*hidden_channels, hidden_dim, A,adaptive=True)
        #self.att=unit_gcn(hidden_dim,hidden_dim,A)
        self.fc = nn.Linear(hidden_dim, num_class)
        self.pro = projection_MLP(hidden_dim)

        # initialize parameters for edge importance weighting
      #  if edge_importance_weighting:
         #   self.edge_importance = nn.ParameterList([
            #    nn.Parameter(torch.ones(self.A.size()))
             #   for i in self.st_gcn_networks
           # ])
        #else:
          #  self.edge_importance = [1] * len(self.st_gcn_networks)
        

    def forward(self, x, drop=False, return_projection = False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        #x = self.l6(x)
        #x = self.l7(x)
        #x = self.l8(x)
       # x = self.l9(x)
       # x = self.l10(x)
        # forward
        
        if return_projection:
          
           # y=self.att(x)
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)
           # y = F.avg_pool2d(y, y.size()[2:])
           # y = y.view(N, M, -1).mean(dim=1)
            # prediction
            x = self.pro(x)
          
          #  y = self.pro(y)
            return x
        else:
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            x = self.fc(x)
            

            return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 A=None,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, 3,A)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
               padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x) + res
        x=self.relu(x)
        return x
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512):
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
       

    def forward(self, x):
      
          x = self.layer1(x)
          x = self.layer2(x)
       
          return x

