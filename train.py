import numpy as np

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

from model_COMBINE import Generator, Discriminator
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


# ----------------------------------------------------------
if __name__ == '__main__':

    f = open("result/Combined_training.txt", 'wb')

    print('Loading the ' + str(config.pretrained_word_model) + ' model...')
    word_embedding = Vocab.Vectors(name = config.pretrained_word_model)

    train_data = dataloader_CLEVER(config.img_root,
                                   config.text_root,
                                   config.max_words_length,
                                   word_embedding,
                                   transforms.Compose([
                                       transforms.Scale(64),   # 74
                                       transforms.ToTensor()
                                   ]))

    vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_loader = data.DataLoader(train_data,
                                batch_size = config.batch_size,
                                shuffle = True,
                                num_workers = config.num_workers)

    print('Loading a pretrained text embedding model...')

    text_encoder = VisualSemanticEmbedding(config.VSE_embedding_dim)
    text_encoder.load_state_dict(torch.load(config.VSE_model_filename))
    text_encoder = text_encoder.text_encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    word_embedding = None

    #######################################
    ### pretrained text embedding model ###
    #######################################
    # Do we really need this? Yes, pretrained the sentence embedding model!
    #######################################

    #############################
    ###  initialize the model ###
    G = Generator()
    D = Discriminator()

    if config.is_cuda:
        text_encoder.cuda()
        G.cuda()
        D.cuda()
    #############################

    ########################################
    ###### optimization configuration ######
    ########################################
    g_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, G.parameters()),
                                   lr = config.learning_rate, betas = (config.momentum, 0.999))
    d_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, D.parameters()),
                                   lr = config.learning_rate, betas = (config.momentum, 0.999))
    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, 100, config.lr_decay)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, 100, config.lr_decay)

    ############################
    ###### start training ######
    ############################
    for epoch in range(config.num_epochs):
        d_lr_scheduler.step()
        g_lr_scheduler.step()

        # training loss
        # For discriminator
        avg_D_real_loss = 0
        avg_D_real_mismatch_loss = 0
        avg_D_fake_loss = 0
        ### something else? ###

        # For generator
        avg_G_fake_loss = 0
        avg_kld = 0

        for i, (img1, text1, len_text1, img2, text2, len_text2) in tqdm(enumerate(train_loader)):

            img1, text_feat1, text_feat1_mismatch, img2, text_feat2, text_feat2_mismatch = \
                preprecessing(img1, text1, len_text1, img2, text2, len_text2, text_encoder)

            ##########################################################
            #### one for discrimator ####
            img1_D = img1
            img2_D = img2
            ##########################################################

            if epoch % 20 == 0 and i == 0:
                save_image(img1.data, './result/origin_epoch_%d.jpg' % (epoch + 1))
                save_image(img2.data, './result/target_epoch_%d.jpg' % (epoch + 1))


            ONES = Variable(torch.ones(img1.size(0)))     # size(0) is the number of the minibatch samples
            ZEROS = Variable(torch.zeros(img1.size(0)))
            if config.is_cuda:
                ONES, ZEROS = ONES.cuda(), ZEROS.cuda()

            ##############################################
            ### UPDATE DISCRIMINATOR ###
            ##############################################
            # ------------- img1, text1 pairs ------------
            D.zero_grad()
            # real image with matching text
            real_logit = D(img1_D, text_feat1)

            real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
            avg_D_real_loss += real_loss.data[0]
            real_loss.backward()

             # real image with mismatching text
            real_mismatch_logit = D(img1_D, text_feat1_mismatch)
            real_mismatch_loss = 0.5 * F.binary_cross_entropy_with_logits(real_mismatch_logit, ZEROS)
            avg_D_real_mismatch_loss += real_mismatch_loss.data[0]
            real_mismatch_loss.backward()
            # --------------------------------------------

            # ------------- img2, text2 pairs ------------
            # real image with matching text
            real_logit = D(img2_D, text_feat2)
            real_loss = F.binary_cross_entropy_with_logits(real_logit, ONES)
            avg_D_real_loss += real_loss.data[0]
            real_loss.backward()

            # real image with mismatching text
            real_mismatch_logit = D(img2_D, text_feat2_mismatch)
            real_mismatch_loss = 0.5 * F.binary_cross_entropy_with_logits(real_mismatch_logit, ZEROS)
            avg_D_real_mismatch_loss += real_mismatch_loss.data[0]
            real_mismatch_loss.backward()




            # ------------- exchange description for enhancement -------------
            real_mismatch_logit = D(img1_D, text_feat2)
            real_mismatch_loss = 0.5 * F.binary_cross_entropy_with_logits(real_mismatch_logit, ZEROS)
            avg_D_real_mismatch_loss += real_mismatch_loss.data[0]
            real_mismatch_loss.backward()

            real_mismatch_logit = D(img2_D, text_feat1)
            real_mismatch_loss = 0.5 * F.binary_cross_entropy_with_logits(real_mismatch_logit, ZEROS)
            avg_D_real_mismatch_loss += real_mismatch_loss.data[0]
            real_mismatch_loss.backward()





            # ------------- fake images ------------
            img1_G = Variable(vgg_normalize(img1.data)) if config.use_vgg else img1_D
            img2_G = Variable(vgg_normalize(img2.data)) if config.use_vgg else img2_D

            fake1, _ = G(img1_G, text_feat1, text_feat2)
            fake1_logit = D(fake1.detach(), text_feat2)
            fake1_loss = 0.5 * F.binary_cross_entropy_with_logits(fake1_logit, ZEROS)
            avg_D_fake_loss += fake1_loss.data[0]
            fake1_loss.backward()

            fake2, _ = G(img2_G, text_feat2, text_feat1)
            fake2_logit = D(fake2.detach(), text_feat1)
            fake2_loss = 0.5 * F.binary_cross_entropy_with_logits(fake2_logit, ZEROS)
            avg_D_fake_loss += fake2_loss.data[0]
            fake2_loss.backward()

            d_optimizer.step()
            # --------------------------------------------
            ##############################################

            ##############################################
            ### relevant text no usage here ###
            ##############################################
            #
            ##############################################

            # other loss function
            ##############################################
            ### what other things should be here? ###
            ##############################################
            # It should be MSE? How to calculate MSE?
            #
            #
            ##############################################

            ##############################################
            ### UPDATE GENERATOR ###
            ##############################################
            # img1, true_text1, mismatch_text1, img2, true_text2, mismatch_text2
            G.zero_grad()
            kld = 0
            # ------------- img1, text1, text2 pairs ------------
            fake, (z_mean1, z_log_stddev1, z_mean2, z_log_stddev2) = G(img1_G, text_feat1, text_feat2)
            ### text augmentation usage ###
            kld += torch.mean(-z_log_stddev1 + 0.5 * (torch.exp(2 * z_log_stddev1) + torch.pow(z_mean1, 2) - 1))
            kld += torch.mean(-z_log_stddev2 + 0.5 * (torch.exp(2 * z_log_stddev2) + torch.pow(z_mean2, 2) - 1))
            avg_kld += kld.data[0]
            ###############################

            # feed to the discriminator!
            fake_logit = D(fake, text_feat2)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ONES)
            avg_G_fake_loss += fake_loss.data[0]
            G_loss = fake_loss + kld
            G_loss.backward()
            # ----------------------------------------------------

            if epoch % 20 == 0 and i == 0:
                save_image(fake.data, './result/epoch_%d.jpg' % (epoch + 1))

            # ------------- img2, text2, text1 pairs -------------
            kld = 0
            ### text augmentation usage ###
            fake, (z_mean1, z_log_stddev1, z_mean2, z_log_stddev2) = G(img2_G, text_feat2, text_feat1)
            kld += torch.mean(-z_log_stddev1 + 0.5 * (torch.exp(2 * z_log_stddev1) + torch.pow(z_mean1, 2) - 1))
            kld += torch.mean(-z_log_stddev2 + 0.5 * (torch.exp(2 * z_log_stddev2) + torch.pow(z_mean2, 2) - 1))
            avg_kld += kld.data[0]
            ###############################

            # feed to the discriminator!
            fake_logit = D(fake, text_feat1)
            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ONES)
            avg_G_fake_loss += fake_loss.data[0]
            G_loss = fake_loss + kld
            G_loss.backward()
            # ----------------------------------------------------

            g_optimizer.step()


        print('Epoch [%d/%d], D_match: %.4f, D_mismatch: %.4f, D_fake: %.4f, G_fake: %.4f, KLD: %.4f'
                % (epoch + 1, config.num_epochs, avg_D_real_loss, avg_D_real_mismatch_loss, avg_D_fake_loss, avg_G_fake_loss, avg_kld))

        if epoch % 20 == 0:
            ###############################################
            #### saving the model ####
            ##############################################
            torch.save(G.state_dict(), './model/G_epoch_' + str(epoch + 1) + '.pth')
            torch.save(D.state_dict(), './model/D_epoch_' + str(epoch + 1) + '.pth')

        f.write(str(avg_D_real_loss) + "," + str(avg_D_real_mismatch_loss) + "," + str(avg_G_fake_loss) + "/n")


    f.close()

    ###############################################
    #### saving the model ####
    ##############################################
    torch.save(G.state_dict(), './model/G_epoch_' + str(config.num_epochs) + '.pth')
    torch.save(D.state_dict(), './model/D_epoch_' + str(config.num_epochs) + '.pth')