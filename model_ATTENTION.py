import argparse
import numpy as np
import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image


class Attention(nn.Module):

    def __init__(self, a):
        super(Attention, self).__init__()

