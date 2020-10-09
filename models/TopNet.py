import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Number of children per tree levels for 2048 output points
tree_arch = {2: [32, 64], 4: [4, 8, 8, 8], 6: [2, 4, 4, 4, 4, 4], 8: [2, 2, 2, 2, 2, 4, 4, 4]}
# tree_arch = {8: [2, 2, 4, 4, 4, 4, 4, 4]}


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch


class PointNetfeat(torch.nn.Module):
    def __init__(self, dim_0, dim_1, dim_2, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.conv1 = nn.Conv1d(dim_0, dim_1, 1)
        self.conv2 = nn.Conv1d(dim_1, dim_2, 1)
        self.global_feat = global_feat
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.dim_2)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, self.dim_2, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1)


class PCN_encoder(torch.nn.Module):
    def __init__(self):
        super(PCN_encoder, self).__init__()
        self.pointnet_layer_1 = PointNetfeat(3, 128, 256, False)
        self.pointnet_layer_2 = PointNetfeat(512, 512, 1024)

    def forward(self, x):
        F_ = self.pointnet_layer_1(x)
        v = self.pointnet_layer_2(F_)
        return v


class TopNet_decoder(torch.nn.Module):
    def __init__(self, n_feat=8, n_levels=8, code_nfts=1024, n_pts=16384):
        super(TopNet_decoder, self).__init__()
        self.tree_arch = get_arch(n_levels, n_pts)
        self.Nin = n_feat + code_nfts
        self.Nout = n_feat
        self.bn = True
        self.N0 = int(self.tree_arch[0])
        self.nfeat = n_feat
        self.n_levels = n_levels
        self.level0 = torch.nn.Sequential(
            nn.Linear(code_nfts, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.nfeat * self.N0),
            nn.Tanh()
        )
        self.down_path = torch.nn.ModuleList()
        bn = True
        Nout = self.Nout
        for i in range(1, self.n_levels):
            if i == self.n_levels - 1:
                Nout = 3
                bn = False
            self.down_path.append(self.create_level(i, self.Nin, Nout, bn))

    def create_level(self, level, input_channels, output_channels, bn):
        if bn:
            mlp_conv = torch.nn.Sequential(
                nn.Conv1d(input_channels, int(input_channels / 2), 1),
                nn.BatchNorm1d(int(input_channels / 2)),
                nn.ReLU(),
                nn.Conv1d(int(input_channels / 2), int(input_channels / 4), 1),
                nn.BatchNorm1d(int(input_channels / 4)),
                nn.ReLU(),
                nn.Conv1d(int(input_channels / 4), int(input_channels / 8), 1),
                nn.BatchNorm1d(int(input_channels / 8)),
                nn.ReLU(),
                nn.Conv1d(int(input_channels / 8), output_channels * int(self.tree_arch[level]), 1),
                nn.Tanh()
            )
        else:
            mlp_conv = torch.nn.Sequential(
                nn.Conv1d(input_channels, int(input_channels / 2), 1),
                nn.ReLU(),
                nn.Conv1d(int(input_channels / 2), int(input_channels / 4), 1),
                nn.ReLU(),
                nn.Conv1d(int(input_channels / 4), int(input_channels / 8), 1),
                nn.ReLU(),
                nn.Conv1d(int(input_channels / 8), output_channels * int(self.tree_arch[level]), 1),
                nn.Tanh()
            )
        return mlp_conv

    def forward(self, x):
        level0 = self.level0(x)
        level0 = level0.view(-1, self.N0, self.nfeat)
        bn = True
        outs = [level0, ]
        Nout = self.Nout
        for i in range(1, self.n_levels):
            if i == self.n_levels - 1:
                Nout = 3
                bn = False
            inp = outs[-1]
            y = x.view(-1, 1, 1024).repeat(1, inp.size()[1], 1)
            y = torch.cat([inp, y], dim=2).transpose(1,2)
            features = self.down_path[i-1](y)
            outs.append(features.view(features.size()[0], -1, Nout))
        return outs[-1]


class TopNet(torch.nn.Module):
    def __init__(self, n_feat=8, n_levels=6, n_pts=16384):
        super(TopNet, self).__init__()
        self.encoder = PCN_encoder()
        self.decoder = TopNet_decoder(n_feat=n_feat, n_levels=n_levels, n_pts=n_pts)

    def forward(self, x):
        code = self.encoder(x.transpose(1,2))
        y = self.decoder(code)
        return y


if __name__ == '__main__':
    # print(get_arch(8, 16384))
    net = TopNet().cuda()
    print(net)
    pts = torch.rand(5,2048,3).cuda()
    y = net(pts)
    print(y.size())
    pass