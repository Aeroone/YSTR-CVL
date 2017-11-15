import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

from model_VisualSemanticEmbedding import VisualSemanticEmbedding
from dataloader_CLEVER import dataloader_CLEVER
import config
import torchtext.vocab as Vocab
from tqdm import tqdm

def pairwise_ranking_loss(margin, x, v):

    zero = torch.zeros(1)
    diag_margin = margin * torch.eye(x.size(0))
    if config.is_cuda:
        zero, diag_margin = zero.cuda(), diag_margin.cuda()
    zero, diag_margin = Variable(zero), Variable(diag_margin)

    # ----------- calculate the pair-wise ranking loss -----------
    x = x / torch.norm(x, 2, 1, keepdim = True)
    v = v / torch.norm(v, 2, 1, keepdim = True)
    prod = torch.matmul(x, v.transpose(0, 1))

    ### diag is the matching images and sentences ###
    diag = torch.diag(prod)

    for_x = torch.max(zero, margin - torch.unsqueeze(diag, 1) + prod) - diag_margin
    for_v = torch.max(zero, margin - torch.unsqueeze(diag, 0) + prod) - diag_margin
    # ------------------------------------------------------------

    return (torch.sum(for_x) + torch.sum(for_v)) / x.size(0)


if __name__ == '__main__':

    f = open("result/MSE_training.txt", 'wb')

    print('Loading the ' + str(config.pretrained_word_model) + ' model...')
    glove = Vocab.GloVe(name='6B', dim = 300)
    word_embedding = Vocab.Vectors(name = config.pretrained_word_model)

    train_data = dataloader_CLEVER(config.img_root,
                                   config.text_root,
                                   config.max_words_length,
                                   word_embedding,
                                   transforms.Compose([
                                       transforms.Scale(240),
                                       #transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]))

    word_embedding = None

    VSE_train_loader = data.DataLoader(train_data,
                                   batch_size = config.VSE_batch_size,
                                   shuffle = True,
                                   num_workers = config.VSE_num_workers)

    #####################################################
    ### Intialize the visual semantic embedding model ###
    #####################################################
    VSE_model = VisualSemanticEmbedding(config.VSE_embedding_dim)
    if config.is_cuda:
        VSE_model.cuda()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, VSE_model.parameters()),
                                 lr = config.VSE_learning_rate)
    #####################################################

    #####################################################
    ### start visual embedding training ###
    #####################################################
    print("start training the visual semantic embedding model!...")
    for epoch in range(config.VSE_num_epochs):
        avg_loss = 0
        for i, (img1, text1, len_text1, img2, text2, len_text2) in tqdm(enumerate(VSE_train_loader)):

            img1 = Variable(img1.cuda() if config.is_cuda else img1)
            text1 = Variable(text1.cuda() if config.is_cuda else text1)
            img2 = Variable(img2.cuda() if config.is_cuda else img2)
            text2 = Variable(text2.cuda() if config.is_cuda else text2)

            # -------------------- img1, text1 pairs -----------------------
            len_text1, indices = torch.sort(len_text1, 0, True)
            indices = indices.numpy()
            #indices = torch.LongTensor(torch.from_numpy(indices))

            img1 = img1[indices, ...]
            description = text1[indices, ...].transpose(0, 1)
            description = nn.utils.rnn.pack_padded_sequence(description, len_text1.numpy())

            #######################
            ### train the model ###
            optimizer.zero_grad()
            img_feature, text_feature = VSE_model(img1, description)

            loss = pairwise_ranking_loss(config.VSE_margin, img_feature, text_feature)
            avg_loss += loss.data[0]
            loss.backward()
            #######################

            optimizer.step()
            # --------------------------------------------------------------

            # -------------------- img2, text2 pairs -----------------------
            len_text2, indices = torch.sort(len_text2, 0, True)
            indices = indices.numpy()
            #indices = torch.LongTensor(torch.from_numpy(indices))

            img2 = img2[indices, ...]
            description = text2[indices, ...].transpose(0, 1)
            description = nn.utils.rnn.pack_padded_sequence(description, len_text2.numpy())

            #######################
            ### train the model ###
            optimizer.zero_grad()
            img_feature, text_feature = VSE_model(img2, description)

            loss = pairwise_ranking_loss(config.VSE_margin, img_feature, text_feature)
            avg_loss += loss.data[0]
            loss.backward()
            #######################

            optimizer.step()
            # --------------------------------------------------------------

        f.write(str(avg_loss) + "\n")
        print('Epoch [%d/%d], Loss: %.4f'% (epoch + 1, config.VSE_num_epochs, avg_loss))

        torch.save(VSE_model.state_dict(), config.VSE_model_filename)

    f.close()