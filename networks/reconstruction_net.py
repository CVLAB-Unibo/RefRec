
import torch.nn as nn
from networks.pointnet import Pointnet_encoder

class Simple_Decoder(nn.Module):
    def __init__(self, feat_dims=1024, num_points=1024):
        super(Simple_Decoder, self).__init__()
        self.m = num_points
        self.folding1 = nn.Sequential(
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, self.m*3, 1),
        )
        
    def forward(self, x):
        x = self.folding1(x)           # (batch_size, 3, num_points)
        x = x.reshape(-1, self.m, 3)          # (batch_size, num_points ,3)
        return x


class ReconstructionNet(nn.Module):
    def __init__(self, device, feat_dims=1024, num_points=1024):
        super(ReconstructionNet, self).__init__()
        self.encoder = Pointnet_encoder(device, feat_dims)
        self.decoder = Simple_Decoder(feat_dims, num_points)

    def forward(self, input):
        feature = self.encoder(input)
        output = self.decoder(feature)
        return feature, output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())