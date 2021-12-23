import torch
import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', norm_layer=None):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                norm_layer(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                norm_layer(out_ch),
                nn.LeakyReLU()
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        if bn:
            bnlayer = nn.BatchNorm1d(out_ch)
            for param in bnlayer.parameters():
                param.requires_grad = True
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                bnlayer,
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    def __init__(self, in_ch, K=3, device="cpu", norm_layer=None):
        super(transform_net, self).__init__()    
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1, norm_layer=norm_layer)
        self.conv2d2 = conv_2d(64, 128, 1, norm_layer=norm_layer)
        self.conv2d3 = conv_2d(128, 1024, 1, norm_layer=norm_layer)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K*K)
        self.device = device

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1).to(self.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

class Pointnet_encoder(nn.Module):
    def __init__(self, device="cuda", feat_dims=1024):
        super(Pointnet_encoder, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.trans_net1 = transform_net(3,3, device, norm_layer=norm_layer)
        self.trans_net2 = transform_net(64,64, device, norm_layer=norm_layer)
        self.conv1 = conv_2d(3, 64, 1, norm_layer=norm_layer)
        self.conv2 = conv_2d(64, 64, 1, norm_layer=norm_layer)
        self.conv3 = conv_2d(64, 64, 1, norm_layer=norm_layer)
        self.conv4 = conv_2d(64, 128, 1, norm_layer=norm_layer)
        self.conv5 = conv_2d(128, feat_dims, 1, norm_layer=norm_layer)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(dim=3) #B, C, N, 1 
        
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        return x

class Pointnet(nn.Module):
    def __init__(self, num_class=10, device="cpu", feat_dims=512, target_cls=False):
        super(Pointnet, self).__init__()
        self.target_cls = target_cls
        self.encoder = Pointnet_encoder(device, feat_dims)
        self.mlp1 = fc_layer(feat_dims, 512)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.mlp2 = fc_layer(512, 256)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.mlp3 = nn.Linear(256, num_class)
        self.device = device

        if target_cls:
            self.mlp1_1 = fc_layer(feat_dims, 512)
            self.dropout1_1 = nn.Dropout2d(p=0.5)
            self.mlp2_1 = fc_layer(512, 256)

            self.dropout2_1 = nn.Dropout2d(p=0.5)
            self.mlp3_1 = nn.Linear(256, num_class)


    def forward(self, x, embeddings=False, target=False):
        feature = self.encoder(x)
        x = feature.squeeze()#batchsize*1024

        if target and self.target_cls:
            x = self.mlp1_1(x)#batchsize*512
            x = self.dropout1_1(x)
            x = self.mlp2_1(x)#batchsize*256
            x = self.dropout2_1(x)
            x = self.mlp3_1(x)#batchsize*10

        else:
            x = self.mlp1(x)#batchsize*512
            x = self.dropout1(x)
            x = self.mlp2(x)#batchsize*256
            x = self.dropout2(x)
            x = self.mlp3(x)#batchsize*10

        if embeddings:
            return feature, x
        else:
            return x
