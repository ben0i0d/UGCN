import torch
from torch import nn
import torch.nn.functional as F


class DropBlock_Ske(nn.Module):
    def __init__(self, num_point=25, keep_prob=0.9):
        super(DropBlock_Ske, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point

    def forward(self, input, mask, A):
        if self.keep_prob == 1:
            return input
        n, c, t, v = input.size()

        # mask = torch.mean(torch.mean(torch.abs(input), dim=2), dim=1).detach()

        mask = mask / torch.sum(mask) * mask.numel()
        gamma = (1. - self.keep_prob) / (1 + 1.92)
        M_seed = torch.bernoulli(torch.clamp(mask * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        M = torch.matmul(M_seed, A)
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        mask = (1 - M).view(n, 1, 1, self.num_point)
        return input * mask * mask.numel() / mask.sum()

class DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7, keep_prob=0.9):
        super(DropBlockT_1d, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input, mask):
        if self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        # mask = torch.mean(torch.mean(torch.abs(input), dim=3), dim=1).detach()
        mask = (mask / torch.sum(mask) * mask.numel()).view(n,1,t)
        input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        gamma = (1. - self.keep_prob) / self.block_size
        M = torch.bernoulli(torch.clamp(mask * gamma, max=1.0)).repeat(1, c*v, 1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        mask = (1 - Msum).to(device=input.device, dtype=input.dtype)

        return (input1 * mask * mask.numel() / mask.sum()).view(n, c, v, t).permute(0, 1, 3, 2)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CA_Drop(nn.Module):
    def __init__(self, inp=256, oup=256, reduction=32, num_point=25, keep_prob=0.9):
        super(CA_Drop, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.dropSke = DropBlock_Ske(num_point=num_point, keep_prob=keep_prob)
        self.dropT_skip = DropBlockT_1d(keep_prob=keep_prob)

    def forward(self, x, A):
        identity = x
        NM, C, T, V = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [T, V], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid() # (N, C, T, 1)
        a_w = self.conv_w(x_w).sigmoid() # (N, C, 1, V)

        # att_map = identity * a_w * a_h
        # att_map_s = att_map.mean(dim=[1, 2])
        # att_map_t = att_map.permute(0, 1, 3, 2).contiguous().mean(dim=[1, 2])

        att_map_s = a_w.mean(dim=[1, 2])
        att_map_t = a_h.permute(0, 1, 3, 2).contiguous().mean(dim=[1, 2])

        mask_s = torch.randn((NM, V)).to(device=x.device, dtype=x.dtype)
        mask_t = torch.randn((NM, T)).to(device=x.device, dtype=x.dtype)
        output = self.dropT_skip(self.dropSke(x, mask_s, A), mask_t)

        # output = self.dropT_skip(self.dropSke(x, att_map_s, A), att_map_t)
        return output

if __name__ == '__main__':
    NM, C, T, V = 256, 16, 13, 25
    x = torch.randn((NM, C, T, V))
    drop_sk = Simam_Drop(num_point=25, keep_prob=0.9)
    w = drop_sk(x)
    print(w.shape)
