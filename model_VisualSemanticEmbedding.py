import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config
import torchvision.models as models


### use the visual semantic text embedding model here ###
class VisualSemanticEmbedding(nn.Module):
    def __init__(self, embed_ndim):

        super(VisualSemanticEmbedding, self).__init__()
        self.embed_ndim = embed_ndim

        # image feature --- use vgg16 ---
        if config.use_vgg:
            self.img_encoder = models.vgg16(pretrained = True)
            for param in self.img_encoder.parameters():
                param.requires_grad = False

        self.feat_extractor = nn.Sequential(*(self.img_encoder.classifier[i] for i in range(6)))
        # turn to the same dimension of the text!
        self.W = nn.Linear(4096, embed_ndim, False)

        # text feature
        self.text_encoder = nn.GRU(embed_ndim, embed_ndim, 1) # GRU(input_size, hidden_size, num_layers)

    def forward(self, img, txt):

        #####################
        ### image feature ###
        img_feat = self.img_encoder.features(img)
        img_feat = img_feat.view(img_feat.size(0), -1)
        img_feat = self.feat_extractor(img_feat)
        img_feat = self.W(img_feat)
        #####################

        #####################
        ### text feature ###
        h0 = torch.zeros(1, img.size(0), self.embed_ndim)
        h0 = Variable(h0.cuda() if config.is_cuda else h0) # the initial hidden state for each element in the batch

        _, txt_feat = self.text_encoder(txt, h0)
        txt_feat = txt_feat.squeeze()
        #####################

        return img_feat, txt_feat