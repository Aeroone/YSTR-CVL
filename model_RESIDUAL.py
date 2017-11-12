import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#######################################
########## Residual Component #########
#######################################
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):

        return F.relu(x + self.encoder(x))