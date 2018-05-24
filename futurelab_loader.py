# Data loading code for FUTURELAB.AI 2018
# Adapted from https://github.com/macaodha/inat_comp_2018/blob/master/train_inat.py
# Author: Du Ang
# Create date: May 7, 2018
# Last update: May 18, 2018

from __future__ import absolute_import

import torch.utils.data as data
from PIL import Image
import os
import csv
import torch
from torchvision import transforms
import random
import numpy as np


def load_csv_file(filepath):
    list_file = []
    with open(filepath, 'r') as csv_file:
        all_lines = csv.reader(csv_file)
        for line in all_lines:
            list_file.append(line)
    list_file.remove(list_file[0])
    return list_file


def default_loader(path):
    return Image.open(path).convert('RGB')


class DATA(data.Dataset):
    def __init__(self, root, list_file_path, image_size, is_train=True):

        # load image list
        print('Loading annotations from: ' + os.path.basename(list_file_path))
        image_list = load_csv_file(list_file_path)

        if len(image_list[0]) >= 2:
            is_test = False
        else:
            is_test = True

        # set up the filenames and annotations
        self.imgs = [image_item[0] for image_item in image_list]
        if is_test:
            self.categories = [0]*len(self.imgs)
        else:
            self.categories = [int(image_item[1]) for image_item in image_list]

        self.classes = []
        [self.classes.append(i) for i in self.categories if i not in self.classes]

        # print out some stats
        print('\t' + str(len(self.imgs)) + ' images')
        print('\t' + str(len(self.classes)) + ' classes')
        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        self.im_size = [image_size, image_size]  # can change this to train on higher res
        self.mu_data = [0.431, 0.440, 0.428]
        self.std_data = [0.250, 0.245, 0.260]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.test_resize = transforms.Resize((int(self.im_size[0]*256.0/224)))
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.ten_crop = transforms.TenCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)
        self.test_tensor_aug = transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        self.test_norm_aug = transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=self.mu_data, std=self.std_data)(crop) for crop in crops]))

    def __getitem__(self, index):
        im_id = self.imgs[index]
        path = self.root + self.imgs[index] + '.jpg'
        img = self.loader(path)
        category = self.categories[index]

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
            img = self.tensor_aug(img)
            img = self.norm_aug(img)
        else:
            img = self.test_resize(img)
            img = self.ten_crop(img)    # dimension changed here
            img = self.test_tensor_aug(img)
            img = self.test_norm_aug(img)

        return img, im_id, category

    def __len__(self):
        return len(self.imgs)

