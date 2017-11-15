import os
from PIL import Image
import json

from nltk.tokenize import RegexpTokenizer
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import config
import torchtext.vocab as Vocab

################################################
############### data loader class ##############
################################################
class dataloader_CLEVER(data.Dataset):
    def __init__(self, img_root, text_root,
                 max_words_length, word_embedding, img_transform = None):
        super(dataloader_CLEVER, self).__init__()

        self.max_words_length = max_words_length
        # transform the image?
        self.img_transform = img_transform
        if img_transform == None:
            self.img_transform = transforms.ToTensor()

        self.tokenizer = RegexpTokenizer(r'\w+')  # used for transforming sentence to words
        self.data = self._load_dataset(img_root, text_root, word_embedding)

    def _load_dataset(self, img_root, text_root, word_embedding):
        output = []
        files = os.listdir(text_root)
        for file in files:
            if not os.path.isdir(file) and file != ".DS_Store":

                # open the json file
                with open(text_root + "/" + file, 'r') as fd:

                    dict = json.load(fd)
                    ########################
                    # ------ QA pairs ------
                    if config.is_QApairs:
                        qa1 = dict["question"] + " "+ dict["answer_a"]
                        qa2 = dict["question"] + " " + dict["answer_b"]
                        text1, len_text1 = self._get_vectors(qa1, word_embedding)
                        text2, len_text2 = self._get_vectors(qa2, word_embedding)
                    else :
                    # ------ Caption ------
                        discription1 = dict["description_a"]
                        discription2 = dict["description_b"]
                        text1, len_text1 = self._get_vectors(discription1, word_embedding)
                        text2, len_text2 = self._get_vectors(discription2, word_embedding)
                    ########################

                    # ------ image ------
                    img1_path = os.path.join(img_root, file[0:-5] + "_a.jpg") # change .json to .png
                    img2_path = os.path.join(img_root, file[0:-5] + "_b.jpg")

                    output.append({
                        'img1': img1_path,
                        'text1': text1,
                        'len_text1': len_text1,
                        'img2': img2_path,
                        'text2': text2,
                        'len_text2': len_text2
                    })

        return output

    # Get the embedding vectros of the words
    # Finish it after word2vector
    def _get_vectors(self, text, word_embedding):

        text = text[0:-1] # delete the last symbol: ". ?"
        words = self.tokenizer.tokenize(text.lower())  # tokenized to words!

        word_vectors = []
        for w in words:
            word_vectors.append(word_embedding.vectors[word_embedding.stoi[w]])
        word_vectors = torch.stack(word_vectors)

        if (len(words) < self.max_words_length):
            word_vectors = torch.cat((
                                        word_vectors,
                                        torch.zeros(self.max_words_length - len(words), word_vectors.size(1))
                                    ))

        return word_vectors, len(words)

    def __len__(self):
        return len(self.data)

    #################################################
    ### this function is key for the data loader! ###dataloader_CLEVER.py:105
    def __getitem__(self, index):
        item = self.data[index]

        # img1
        img1 = Image.open(item['img1'])
        img1 = img1.convert('RGB')
        img1 = self.img_transform(img1)
        text1 = item['text1']
        len_text1 = item['len_text1']

        # img2
        img2 = Image.open(item['img2'])
        img2 = img2.convert('RGB')
        img2 = self.img_transform(img2)
        text2 = item['text2']
        len_text2 = item['len_text2']

        return img1, text1, len_text1, img2, text2, len_text2

##################################################
################ data loader tester###############
##################################################
'''
if __name__ == '__main__':

    print('Loading the ' + str(config.pretrained_word_model) + ' model...')
    word_embedding = Vocab.Vectors(name = config.pretrained_word_model)

    img_root = "dataset/5pairs/train/images"
    text_root = "dataset/5pairs/train/qa"
    max_word_length = 15

    train_data = dataloader_CLEVER(img_root,
                                   text_root,
                                   max_word_length,
                                   word_embedding,
                                   transforms.Compose([
                                       transforms.Scale(256),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]))

    train_loader = data.DataLoader(train_data,
                                   batch_size=2,
                                   shuffle=True)


    for epoch in range(config.num_epochs):
        for i, (img1, text1, len_text1, img2, text2, len_text2) in enumerate(train_loader):
            print i
            print "################"
            print i
            print img1.size()
            print text1.size()
            print text2.size()
            print "################"
'''