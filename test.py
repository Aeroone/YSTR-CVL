import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage

from PIL import Image
from model_VisualSemanticEmbedding import VisualSemanticEmbedding
from dataloader_CLEVER import dataloader_CLEVER
import torchtext.vocab as Vocab

from model_COMBINE_ATTENTION_3 import Generator, Discriminator
from tqdm import tqdm
import config

####################################
### Process the mini-batch data
####################################
def preprecessing(img1, text1, len_text1, img2, text2, len_text2, text_encoder):

    ### for img1 and text1 ###
    img1 = Variable(img1.cuda() if config.is_cuda else img1)
    text1 = Variable(text1.cuda() if config.is_cuda else text1)

    len_text1 = len_text1.numpy()
    sorted_indices = np.argsort(len_text1)[::-1]
    sorted_indices_T = sorted_indices
    original_indices = np.argsort(sorted_indices)
    if not config.is_cuda:
        sorted_indices_T = torch.LongTensor(torch.from_numpy(sorted_indices.copy()))

    packed_desc = nn.utils.rnn.pack_padded_sequence(
        text1[sorted_indices_T, ...].transpose(0, 1),
        len_text1[sorted_indices]
    )

    _, text_feat1 = text_encoder(packed_desc)
    text_feat1 = text_feat1.squeeze()
    text_feat1 = text_feat1[original_indices, ...]

    text_feat1_np = text_feat1.data.cpu().numpy() if config.is_cuda else text_feat1.data.numpy()
    text_feat1_mismatch = torch.Tensor(np.roll(text_feat1_np, 1, axis=0))
    text_feat1_mismatch = Variable(text_feat1_mismatch.cuda() if config.is_cuda else text_feat1_mismatch)

    ### for img2 and text2 ###
    img2 = Variable(img2.cuda() if config.is_cuda else img2)
    text2 = Variable(text2.cuda() if config.is_cuda else text2)

    len_text2 = len_text2.numpy()
    sorted_indices = np.argsort(len_text2)[::-1]
    sorted_indices_T = sorted_indices
    original_indices = np.argsort(sorted_indices)
    if not config.is_cuda:
        sorted_indices_T = torch.LongTensor(torch.from_numpy(sorted_indices.copy()))

    packed_desc = nn.utils.rnn.pack_padded_sequence(
        text2[sorted_indices_T, ...].transpose(0, 1),
        len_text2[sorted_indices]
    )

    _, text_feat2 = text_encoder(packed_desc)
    text_feat2 = text_feat2.squeeze()
    text_feat2 = text_feat2[original_indices, ...]

    text_feat2_np = text_feat2.data.cpu().numpy() if config.is_cuda else text_feat2.data.numpy()
    text_feat2_mismatch = torch.Tensor(np.roll(text_feat2_np, 1, axis=0))
    text_feat2_mismatch = Variable(text_feat2_mismatch.cuda() if config.is_cuda else text_feat2_mismatch)

    return img1, text_feat1, text_feat1_mismatch, img2, text_feat2, text_feat2_mismatch


def testOnModel(word_embedding, G_model, text_encoder, G_index):

    test_data = dataloader_CLEVER(config.img_test_root,
                                   config.text_test_root,
                                   config.max_words_length,
                                   word_embedding,
                                   transforms.Compose([
                                       transforms.Scale(64),   # 74
                                       transforms.ToTensor()
                                   ]))

    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    test_loader = data.DataLoader(test_data,
                                batch_size = config.batchsize_test,
                                shuffle = False,
                                num_workers = config.test_num_workers)


    #######################################
    ### pretrained text embedding model ###
    #######################################
    # Do we really need this? Yes, pretrained the sentence embedding model!
    #######################################

    #############################
    ###  initialize the model ###
    print "###########"
    print G_model
    print "###########"

    G = Generator()
    G.load_state_dict(torch.load(G_model))
    for param in G.parameters():
        param.requires_grad = False

    if config.is_cuda:
        text_encoder.cuda()
        G.cuda()
    #############################

    ############################
    ###### start training ######
    ############################
    reconstruction_loss = 0
    count = 0
    for i, (img1, text1, len_text1, img2, text2, len_text2) in tqdm(enumerate(test_loader)):

        img1, text_feat1, text_feat1_mismatch, img2, text_feat2, text_feat2_mismatch = \
            preprecessing(img1, text1, len_text1, img2, text2, len_text2, text_encoder)


        if i == 0:
            save_image(img1.data, config.test_result_root + '/' + 'img1_' + G_index[:-4] + '_%d.jpg' % (i))
            save_image(img2.data, config.test_result_root + '/' + 'img2_' + G_index[:-4] + '_%d.jpg' % (i))


        img1_G = Variable(vgg_normalize(img1.data))
        img2_G = Variable(vgg_normalize(img2.data))

        # ------------- img1, text1, text2 pairs ------------
        fake, (z_mean1, z_log_stddev1, z_mean2, z_log_stddev2) = G(img1_G, text_feat1, text_feat2)

        if i == 0:
            save_image(fake.data, config.test_result_root + '/' + 'fake_img2_' + G_index[:-4] + '_%d.jpg' % (i))


        #############################################
        ### Calculate the L1 reconstruction error ###
        #############################################
        reconstruction_loss += torch.abs(fake - img2).sum() / img2.size(0)
        #############################################

        # ------------- img2, text2, text1 pairs -------------
        fake, (z_mean1, z_log_stddev1, z_mean2, z_log_stddev2) = G(img2_G, text_feat2, text_feat1)

        if i == 0:
            save_image(fake.data, config.test_result_root + '/' + 'fake_img1_' + G_index[:-4] + '_%d.jpg' % (i))

        #############################################
        ### Calculate the L1 reconstruction error ###
        #############################################
        reconstruction_loss += torch.abs(fake - img1).sum() / img1.size(0)
        #############################################

        count = count + 2

    print " --- the final loss " +  str(G_index) + " is: ---"
    print reconstruction_loss.data[0] / count
    return reconstruction_loss.data[0] / count


# ----------------------------------------------------------
if __name__ == '__main__':

    f = open(config.test_result_root + "/loss_test.txt", 'wb')

    print('Loading the ' + str(config.pretrained_word_model) + ' model...')
    word_embedding = Vocab.Vectors(name = config.pretrained_word_model)


    print('Loading a pretrained text embedding model...')
    text_encoder = VisualSemanticEmbedding(config.VSE_embedding_dim)
    text_encoder.load_state_dict(torch.load(config.VSE_model_filename))
    text_encoder = text_encoder.text_encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    # word_embedding = None


    files = os.listdir(config.test_G_model_root)
    count = 1
    for file in files:
        if not os.path.isdir(file) and file != ".DS_Store":

            name = file[0:8] + str(count) + file[-4:]
            print "------- model: " + str(name) + " -------"
            test_loss = testOnModel(word_embedding, config.test_G_model_root + "/" + name, text_encoder, name)
            f.write(str(test_loss) + "\n")

            count = count + 20

    f.close()