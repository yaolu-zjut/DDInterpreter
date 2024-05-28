import os

import random
import numpy as np

import torch
from torch import nn

from utils_clom.model_pool import get_network
from utils_clom.trainer import validate
from utils_clom.utils import get_pretrained_model_root


class CLoM(object):
    def __init__(self, args, syn_dataset_ipc, CLoM_batch_size, distance,
                 source_dataset, source_channel, source_num_classes, source_im_size,
                 models_pool, nums_models, alphas, epoch=150):
        super(CLoM, self).__init__()
        self.args = args
        '''load pretrain models and set alphas for each architecture'''
        self.pretrained_models_pool, self.alphas_dict = self.load_pretrained_models(source_dataset, source_channel,
                                                                                    source_num_classes, source_im_size,
                                                                                    models_pool, nums_models, alphas, epoch)
        '''pretrained models information'''
        self.models_pool = models_pool
        self.nums_models = nums_models

        self.syn_dataset_ipc = syn_dataset_ipc
        self.CLoM_batch_size = CLoM_batch_size

        self.distance = distance

        '''prepare for each distance'''
        if self.distance == 'cos':
            self.real_labels = None
            self.real_images = None
            self.real_num_classes = None
            self.real_data_len = None
            self.__enable_model_embed()
        elif self.distance == 'ce':
            self.criterion = nn.CrossEntropyLoss().to(self.args.device)

    def load_pretrained_models(self, dataset, channel, num_classes, im_size, models_pool, nums_models, alphas, epoch):
        if len(models_pool) == 0:
            print("no model in models pool")
            exit(0)

        '''set alphas'''
        alphas_dict = dict()
        if len(alphas) == len(models_pool):
            for i, model_arch in enumerate(models_pool):
                alphas_dict[model_arch] = alphas[i]
                print(f"alpha for {model_arch}: {alphas[i]}")
        elif len(alphas) == 1:
            for model_arch in models_pool:
                alphas_dict[model_arch] = alphas[0]
                print(f"alpha for {model_arch}: {alphas[0]}")
        else:
            print("check alphas list")
            exit(0)

        '''load models pool'''
        pretrained_models_pool = dict()
        for model_arch in models_pool:
            print("load model arch:", model_arch, "num:", nums_models)
            pretrained_model_list = []
            index = 0
            loop = 0
            while len(pretrained_model_list) < nums_models:
                model_path = os.path.join(get_pretrained_model_root(dataset, model_arch), f"index_{index}", f"{dataset}_{model_arch}_original_epoch{epoch}.pth")
                index = index + 1
                if os.path.exists(model_path):
                    print("load:", model_path)
                    model = get_network(model_arch, channel, num_classes, im_size)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model = model.to(self.args.device).requires_grad_(False)
                    pretrained_model_list.append(model)
                    loop = 0
                else:
                    print("model:", model_path, "don't exist")
                    loop = loop + 1
                    if loop > 100:
                        print("check models num")
                        exit()
            pretrained_models_pool[model_arch] = pretrained_model_list

        return pretrained_models_pool, alphas_dict

    def __enable_model_embed(self):
        for model_arch in self.models_pool:
            for index in range(self.nums_models):
                self.pretrained_models_pool[model_arch][index] = self.pretrained_models_pool[model_arch][index].embed

    def calculating_CLoM(self, syn_image, syn_label):
        CLoM = None
        if self.distance == 'cos':
            CLoM = self.calculating_cosine_loss(syn_image, syn_label)
        elif self.distance == 'ce':
            CLoM = self.calculating_ce_loss(syn_image, syn_label)
        return CLoM

    def calculating_ce_loss(self, syn_image, syn_label):
        model_arch = random.sample(self.models_pool, 1)[0]
        alpha = self.alphas_dict[model_arch]
        index = random.randint(0, self.nums_models - 1)
        # get pre-trained model(training on original training set)
        model = self.pretrained_models_pool[model_arch][index]
        output = model(syn_image)
        # classification loss on model trained on origianl dataset
        loss = self.criterion(output, syn_label)
        return alpha * loss

    def set_real_data(self, real_images, real_labels, real_num_classes):
        self.real_images = real_images
        self.real_labels = real_labels
        self.real_num_classes = real_num_classes
        self.real_data_len = len(self.real_labels)

    def __get_images_randomly(self, batch_size):
        idx_random = np.random.randint(0, self.real_data_len, size=batch_size)
        return self.real_images[idx_random], self.real_labels[idx_random]

    def calculating_cosine_loss(self, syn_image, syn_label):
        model_arch = random.sample(self.models_pool, 1)[0]
        alpha = self.alphas_dict[model_arch]
        index = random.randint(0, self.nums_models - 1)
        # get pre-trained model(training on original training set)
        model = self.pretrained_models_pool[model_arch][index]
        # random select images from real dataset
        real_image, real_label = self.__get_images_randomly(self.CLoM_batch_size)
        # trans real label to onehot
        real_label_onehot = torch.zeros((self.CLoM_batch_size, self.real_num_classes)).cuda()
        real_label_onehot[torch.arange(self.CLoM_batch_size), real_label] = 1
        # trans syn label to onehot
        syn_label_onehot = torch.zeros((self.syn_dataset_ipc * self.real_num_classes, self.real_num_classes)).cuda()
        syn_label_onehot[torch.arange(self.syn_dataset_ipc * self.real_num_classes), syn_label] = 1
        # True and false sample matrix
        labels = torch.matmul(syn_label_onehot, real_label_onehot.t()).cuda()
        # get features
        syn_feature = model(syn_image)
        real_feature = model(real_image)
        syn_feature_norm = syn_feature / syn_feature.norm(dim=-1, keepdim=True)
        real_feature_norm = real_feature / real_feature.norm(dim=-1, keepdim=True)
        # loss
        cos_dist = 1 - syn_feature_norm @ real_feature_norm.t()
        con_loss = torch.sum(labels * cos_dist) / torch.sum(cos_dist)

        return alpha * con_loss
