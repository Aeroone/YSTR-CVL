import numpy as np
import config

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import torchvision.models as models
from model_RESIDUAL import ResidualBlock
from model_ATTENTION import Attention

#############################
### weight initialization ###
#############################
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

        if m.weight.requires_grad:
            m.weight.data.normal_(std = 0.02)
        if m.bias is not None and m.bias.requires_grad:
            m.bias.data.fill_(0)

    elif isinstance(m, nn.BatchNorm2d) and m.affine:

        if m.weight.requires_grad:
            m.weight.data.normal_(1, 0.02)
        if m.bias.requires_grad:
            m.bias.data.fill_(0)
#############################

###################################################
###################################################
#### Define the generator for image generation ####
###################################################
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        ####################################
        ### encoder for original image ###
        ####################################
        if config.use_vgg:
            self.encoder = models.vgg16_bn(pretrained = True)
            self.encoder = nn.Sequential(*(self.encoder.features[i] for i in range(23) + range(24, 33)))
            self.encoder[24].dilation = (2, 2)
            self.encoder[24].padding = (2, 2)
            self.encoder[27].dilation = (2, 2)
            self.encoder[27].padding = (2, 2)
            self.encoder[30].dilation = (2, 2)
            self.encoder[30].padding = (2, 2)

            for param in self.encoder.parameters():
                param.requires_grad = False

            self.encoder.eval()
        ####################################

        ####################################
        ### decoder for image generation ###
        ####################################
        self.decoder = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(512, 256, 3, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(256, 128, 3, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 3, 3, padding = 1),
            nn.Tanh()
        )
        ####################################

        #######################
        ### residual blocks ###
        if config.is_residual_module:
            self.residual_blocks = nn.Sequential(
                nn.Conv2d(512 + 128, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                ResidualBlock(),
                ResidualBlock(),
                ResidualBlock(),
                ResidualBlock()
            )
        #######################

        #################################
        ### conditioning augmentation ###
        #################################
        # I am not clear how to understand this part!
        ### original word feature length is 300 ###
        self.mu = nn.Sequential(
            nn.Linear(300, 128, bias = False),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(300, 128, bias = False),
            nn.LeakyReLU(0.2, inplace = True)
        )
        #################################
        #################################

        self.apply(init_weights)


    ### need to fix here for the attention models! ###
    ### image: the image needed to modify
    ### text1: the corresponding text with image
    ### text2: the needed text
    def forward(self, image, text1, text2, z = None):

        #########################
        ### encode the image! ###
        img_feat = self.encoder(image)
        #########################

        # +++++++++++++++++ original text1 +++++++++++++++++
        ####################################################
        ### text embedding augmentation method for text 1###
        z_mean1 = self.mu(text1)
        z_log_stddev1 = self.log_sigma(text1)

        z = torch.randn(text1.size(0), 128)
        if config.is_cuda:
            z = z.cuda()

        text_feat1 = z_mean1 + z_log_stddev1.exp() * Variable(z)

        text_feat1 = text_feat1.unsqueeze(-1).unsqueeze(-1)
        text_feat1 = text_feat1.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        ####################################################
        # ++++++++++++++++++++++++++++++++++++++++++++++++++

        ##########################################
        ### concatenate the image and text1 ###
        fusion = torch.cat((img_feat, text_feat1), dim=1)
        ##########################################

        ###### feed into the residual module ######
        fusion = self.residual_blocks(fusion)
        ###########################################

        # +++++++++++++++++ original text2 +++++++++++++++++
        #####################################################
        ### text embedding augmentation method for text 2 ###
        z_mean2 = self.mu(text2)
        z_log_stddev2 = self.log_sigma(text2)

        z = torch.randn(text2.size(0), 128)
        if config.is_cuda:
            z = z.cuda()

        text_feat2 = z_mean2 + z_log_stddev2.exp() * Variable(z)

        text_feat2 = text_feat2.unsqueeze(-1).unsqueeze(-1)
        text_feat2 = text_feat2.repeat(1, 1, fusion.size(2), fusion.size(3))
        #####################################################
        # +++++++++++++++++++++++++++++++++++++++++++++++++++

        ##########################################
        ### concatenate the fusion and text2 ###
        fusion = torch.cat((fusion, text_feat2), dim=1)
        ##########################################

        ###########################################
        # decoder
        output = self.decoder(fusion)

        return output, (z_mean1, z_log_stddev1, z_mean2, z_log_stddev2)


###################################################
###################################################
############# Define the discriminator ############
###################################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        ############################################
        ### encoder for original/generated image ###
        ############################################
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding = 1),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, padding = 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 4, 2, padding = 1, bias = False),
            nn.BatchNorm2d(512)
        )
        ############################################

        ###################################
        ### why does this part use for? ###
        ###################################
        self.residual_branch = nn.Sequential(
            nn.Conv2d(512, 128, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 128, 3, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 512, 3, padding = 1, bias = False),
            nn.BatchNorm2d(512)
        )
        ###################################

        ### output 1 or 0 ###
        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128, 512, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(512, 1, 4)
        )
        #####################

        self.compression = nn.Sequential(
            nn.Linear(300, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.apply(init_weights)

    def forward(self, image, text):

        img_feat = self.encoder(image)

        ### why we add residual branch here? ###
        img_feat = F.leaky_relu(img_feat + self.residual_branch(img_feat), 0.2)
        ########################################

        ########################################
        ### encode the text ###
        txt_feat = self.compression(text)  # compress to 128
        txt_feat = txt_feat.unsqueeze(-1).unsqueeze(-1)
        txt_feat = txt_feat.repeat(1, 1, img_feat.size(2), img_feat.size(3))
        ########################################

        ##########################################
        ### concatenate the image and text ###
        fusion = torch.cat((img_feat, txt_feat), dim=1)
        ##########################################

        ### get the result ###
        output = self.classifier(fusion)

        return output.squeeze()