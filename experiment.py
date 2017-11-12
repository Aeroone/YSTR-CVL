import json
import fasttext

from tqdm import tqdm
import time
import os
import numpy as np
from PIL import Image
import nltk
from nltk.tokenize import RegexpTokenizer

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms
import torchwordemb

import torch
import torchtext.vocab as Vocab
import torchvision.transforms as transforms
from PIL import Image

sorted_indices = np.array([1,2,3,4,5])
sorted_indices_T = torch.LongTensor(torch.from_numpy(sorted_indices.copy()))


print sorted_indices[3]
print sorted_indices_T[3]

'''
margin = 0.2
x = torch.Tensor([1,1, 1, 1])
y = torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
print margin - x + y
'''




'''
img = Image.open("dataset/5pairs/train/images/" + "CLEVR_new_000101_a.png")
img_transform = transforms.ToTensor()

img_transform = transforms.Compose([
                                       transforms.Scale(256),
                                       transforms.RandomCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ])


#img1.show()
img1 = img
img1 = img_transform(img1)
print img1
print img1.size()

img2 = img.convert('RGB')
img2 = img_transform(img2)
print img2
print img2.size()


#img1 = img1[0:3, ...]


string = [123, 234, 345, 345]
index = [0, 1, 2, 3]
print len(string)
a = torch.ones(5, 1)
for i in range(1, len(string)):
    vector = torch.ones(5, 1)
    a = torch.cat((a, vector), 1)
    print string[i]

print a

b = torch.Tensor([string[w] for w in index])
print b.size()



glove = Vocab.GloVe(name='6B', dim=100)
#glove = Vocab.Vectors(name = 'glove.6B.300d.txt')

word = "triangle"
print glove.vectors[glove.stoi[word]][0]




vocab, vec = torchwordemb.load_glove_text("WordEmbedding/glove.6B.300d.txt")
print(vec.size())
print(vec[vocab["triangle"]][0])



caption_root = "dataset/FLOWERS/flowers_icml"
classes_filename = "trainvalclasses.txt"

class ReadFlower(data.Dataset):
    def __init__(self, caption_root, classes_filename):
        self.data = self._load_dataset(caption_root, classes_filename)

    def _load_dataset(self, caption_root, classes_filename):
        output = []
        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                #print "########"
                #print filenames
                #print "#########"

                count = 1
                for filename in filenames:
                    if (count == 5):
                        break

                    count += 1
                    datum = load_lua(os.path.join(caption_root, cls, filename))
                    raw_desc = datum['char'].numpy()
                    words = []
                    len = []
                    for i in range(raw_desc.shape[1]):
                        word_vecs = torch.Tensor(raw_desc[:, i].astype(float))
                        words.append(word_vecs)
                        len.append(raw_desc[0].shape)

                    #desc, len_desc = self._get_word_vectors(raw_desc, word_embedding)
                    output.append({
                        'desc': torch.stack(words),
                        'len_desc': len
                    })

        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]

        desc = datum['desc']
        len_desc = datum['len_desc']

        selected = np.random.choice(desc.size(0))
        desc = desc[selected, ...]
        len_desc = len_desc[selected]

        return desc, len_desc


#############################################################
### print output ###
#############################################################
train_data = ReadFlower(caption_root, classes_filename)

train_loader = data.DataLoader(train_data,
                                   batch_size=3,
                                   shuffle=True)

for epoch in range(1, 2):

    for i, (desc, len_desc) in enumerate(train_loader):
        print "======" + str(i) + "======"
        print desc
        print len_desc
        print "======================"
'''