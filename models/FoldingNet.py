import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FoldingNet_decoder(torch.nn.Module):
    def __init__(self, n_points=16384, grid_size=128):
        super(FoldingNet_decoder, self).__init__()
        self.n_points = n_points
        self.grid_size = grid_size
        self.folding1 = torch.nn.Sequential(
            nn.Conv1d(1024 + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1)
        )
        self.folding2 = torch.nn.Sequential(
            nn.Conv1d(1024 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1)
        )

    def forward(self, x):
        v = x
        batchsize = x.size()[0]
        grid = torch.meshgrid([torch.linspace(-0.5, 0.5, self.grid_size), torch.linspace(-0.5, 0.5, self.grid_size)])
        grid = torch.stack(grid, 2).view(1,-1,2)
        grid = grid.repeat(batchsize, 1, 1).cuda()
        global_feat = v.view(-1, 1, 1024).repeat(1, self.n_points, 1)
        fold_1 = self.folding1(torch.cat([global_feat, grid], dim=2).transpose(1, 2)).transpose(1, 2)
        fold_2 = self.folding2(torch.cat([global_feat, fold_1], dim=2).transpose(1, 2)).transpose(1, 2)
        return fold_2


class FoldingNet(torch.nn.Module):
    def __init__(self):
        super(FoldingNet, self).__init__()
        self.encoder = PCN_encoder()
        self.decoder = FoldingNet_decoder(16384, 128)

    def forward(self, x):
        v = self.encoder(x.transpose(1, 2))
        fine = self.decoder(v)
        return fine


if __name__ == "__main__":
    ptcloud = torch.rand(5, 2048, 3).cuda()
    model = FoldingNet()
    model.cuda()
    res = model(ptcloud)
    pass