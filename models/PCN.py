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


class PCN_decoder(torch.nn.Module):
    def __init__(self, n_points=2048, grid_size=4):
        super(PCN_decoder, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Linear(1024, n_points*3)
        self.n_points = n_points
        self.grid_size = grid_size
        self.conv1 = torch.nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        v = x
        batchsize = x.size()[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        coarse_ptcloud = x.view(-1, self.n_points, 3)
        grid = torch.meshgrid([torch.linspace(-0.05, 0.05, 4), torch.linspace(-0.05, 0.05, 4)])
        grid = torch.stack(grid, 2).view(1,-1,2)
        grid = grid.repeat(batchsize, self.n_points, 1).cuda()
        global_feat = v.view(-1, 1, 1024).repeat(1, (self.grid_size**2) * self.n_points, 1)
        point_feat = coarse_ptcloud.repeat(1, self.grid_size**2, 1)
        feat = torch.cat([grid, point_feat, global_feat], dim=2).transpose(1, 2)
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        fine_ptcloud = self.conv3(feat).transpose(1, 2) + point_feat
        perm = torch.randperm(fine_ptcloud.size(1))[:16384]
        final_pccloud = fine_ptcloud[:, perm, :]
        return coarse_ptcloud, final_pccloud


class PCN(torch.nn.Module):
    def __init__(self):
        super(PCN, self).__init__()
        self.encoder = PCN_encoder()
        self.decoder = PCN_decoder(2048, 4)

    def forward(self, x):
        v = self.encoder(x.transpose(1, 2))
        coarse, fine = self.decoder(v)
        return coarse, fine


if __name__ == "__main__":
    ptcloud = torch.rand(5, 3, 2048).cuda()
    model = PCN()
    model.cuda()
    res = model(ptcloud)