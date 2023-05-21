import sys
import os

import random

import torch
from torch import nn

from utils_clom.model_pool import get_network
from utils_clom.trainer import validate
from utils_clom.utils import get_pretrained_model_root


class CLOM(object):
    def __init__(self, args, models_pool, model_num, alphas, channel, num_classes, im_size):

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
                self.alphas[model_arch] = alphas
                print(f"alpha for {model_arch}: {alphas}")
        else:
            print("check alphas list")
            exit(0)

        self.pretrained_models_pool = dict()
        self.criterion = nn.CrossEntropyLoss().to(args.device)

        '''load models pool'''
        for model_arch in self.models_pool:
            print("load model arch:", model_arch, "num:", self.models_num)
            pretrained_model_list = []
            index = 0
            loop = 0
            while len(pretrained_model_list) < self.models_num:
                model_path = os.path.join(get_pretrained_model_root(args.dataset, model_arch),
                                          f"{args.dataset}_{model_arch}_original_{index}.pt")
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
                        print("please check models num")
                        exit()
            self.pretrained_models_pool[model_arch] = pretrained_model_list

    def calculate_loss(self, image, label):
        model_arch = random.sample(self.models_pool, 1)[0]
        alpha = self.alphas[model_arch]
        index = random.randint(0, self.models_num - 1)
        # print(f"select arch: {model_arch}, index: {index}, alpha: {alpha}")
        # get pre-trained model(training on original training set)
        model = self.pretrained_models_pool[model_arch][index]
        output = model(image)
        # classification loss on model trained on origianl dataset
        loss = self.criterion(output, label)
        return alpha * loss

    def test_models_performance(self, testloader):
        print("start testing models performance")
        for model_arch, models in self.pretrained_models_pool.items():
            for i, model in enumerate(models):
                acc1, _ = validate(testloader, model, self.criterion, self.args)
                print(f"model: {model_arch}, index: {i}, acc: {acc1}")

