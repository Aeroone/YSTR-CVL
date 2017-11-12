##############################################################################
#
### load the VQA dataset ###
#
##############################################################################
import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import config


class VQA():
    def __init__(self, questions_path, answers_path, image_path):
        super(VQA, self).__init__()

        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)



















