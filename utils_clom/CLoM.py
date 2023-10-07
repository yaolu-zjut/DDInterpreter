import os

import random
import numpy as np

import torch
from torch import nn

from utils_clom.model_pool import get_network
from utils_clom.trainer import validate
from utils_clom.utils import get_pretrained_model_root


class CLoM_Base(object):
    def __init__(self, args,
                 dataset, channel, num_classes, im_size,
                 models_pool, model_num, alphas, epoch=150):
        self.args = args
        self.models_pool = models_pool
        self.models_num = model_num

        if len(self.models_pool) == 0:
            print("no model in models pool")
            exit(0)

        if len(alphas) == len(self.models_pool):
            self.alphas = dict()
            for i, model_arch in enumerate(self.models_pool):
                self.alphas[model_arch] = alphas[i]
                print(f"alpha for {model_arch}: {alphas[i]}")
        elif len(alphas) == 1:
            self.alphas = dict()
            for model_arch in self.models_pool:
                self.alphas[model_arch] = alphas[0]
                print(f"alpha for {model_arch}: {alphas[0]}")
        else:
            print("check alphas list")
            exit(0)

        self.pretrained_models_pool = dict()

        '''load models pool'''
        for model_arch in self.models_pool:
            print("load model arch:", model_arch, "num:", self.models_num)
            pretrained_model_list = []
            index = 0
            loop = 0
            while len(pretrained_model_list) < self.models_num:
                model_path = os.path.join(get_pretrained_model_root(dataset, model_arch), f"index_{index}",
                                          f"{dataset}_{model_arch}_original_epoch{epoch}.pth")
                index = index + 1
                if os.path.exists(model_path):
                    print("load:", model_path)
                    model = get_network(model_arch, channel, num_classes, im_size)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model = model.to(args.device).requires_grad_(False)
                    pretrained_model_list.append(model)
                    loop = 0
                else:
                    print("model:", model_path, "don't exist")
                    loop = loop + 1
                    if loop > 100:
                        print("check models num")
                        exit()
            self.pretrained_models_pool[model_arch] = pretrained_model_list

    def test_models_performance(self, testloader, criterion):
        print("start testing models performance")
        for model_arch, models in self.pretrained_models_pool.items():
            for i, model in enumerate(models):
                acc1, _ = validate(testloader, model, criterion, self.args)
                print(f"model: {model_arch}, index: {i}, acc: {acc1}")


class CLoM(CLoM_Base):
    def __init__(self, args,
                 source_dataset, source_channel, source_num_classes, source_im_size,
                 models_pool, model_num, alphas, epoch=150):
        super(CLoM, self).__init__(args, source_dataset, source_channel, source_num_classes, source_im_size,
                                   models_pool, model_num, alphas, epoch)
        self.criterion = nn.CrossEntropyLoss().to(args.device)

    def calculate_loss(self, syn_image, syn_label):
        model_arch = random.sample(self.models_pool, 1)[0]
        alpha = self.alphas[model_arch]
        index = random.randint(0, self.models_num - 1)
        # get pre-trained model(training on original training set)
        model = self.pretrained_models_pool[model_arch][index]
        output = model(syn_image)
        # classification loss on model trained on origianl dataset
        loss = self.criterion(output, syn_label)
        return alpha * loss


class CCLoM(CLoM_Base):
    def __init__(self, args, syn_dataset_ipc, batch_size,
                 source_dataset, source_channel, source_num_classes, source_im_size,
                 models_pool, model_num, alphas, epoch=150):
        super(CCLoM, self).__init__(args, source_dataset, source_channel, source_num_classes, source_im_size,
                                    models_pool, model_num, alphas, epoch)

        self.syn_dataset_ipc = syn_dataset_ipc
        self.batch_size = batch_size
        self.inter_feature = list()

        self.real_labels = None
        self.real_images = None
        self.real_num_classes = None
        self.real_data_len = None

        self.__enable_model_embed()

    def set_real_data(self, real_images, real_labels, real_num_classes):
        self.real_images = real_images
        self.real_labels = real_labels
        self.real_num_classes = real_num_classes
        self.real_data_len = len(self.real_labels)

    def __get_images_randomly(self, batch_size):  # get random n images from class c
        idx_random = np.random.randint(0, self.real_data_len, size=batch_size)
        return self.real_images[idx_random], self.real_labels[idx_random]

    def calculate_loss(self, syn_image, syn_label):
        self.inter_feature.clear()  # clear
        model_arch = random.sample(self.models_pool, 1)[0]
        alpha = self.alphas[model_arch]
        index = random.randint(0, self.models_num - 1)
        # get pre-trained model(training on original training set)
        model = self.pretrained_models_pool[model_arch][index]
        # random select images from real dataset
        real_image, real_label = self.__get_images_randomly(self.batch_size)
        # trans real label to onehot
        real_label_onehot = torch.zeros((self.batch_size, self.real_num_classes)).cuda()
        real_label_onehot[torch.arange(self.batch_size), real_label] = 1
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

    def __enable_model_embed(self):
        for model_arch in self.models_pool:
            for index in range(self.models_num):
                self.pretrained_models_pool[model_arch][index] = self.pretrained_models_pool[model_arch][index].embed

