import os
import torch
import torchvision
from torchvision import transforms, datasets
import sys

import tqdm

from utils_clom.utils import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


class TinyImageNet_loader(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


def get_dataset(args):
    if args.dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        if args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.MNIST(args.data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif args.dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        if args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.FashionMNIST(args.data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(args.data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        if args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.SVHN(args.data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(args.data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif args.dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        if args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif args.dataset == 'tinyimagenet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.normalize_data:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        dst_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform) # no augmentation
        dst_test = TinyImageNet_loader(args.data_path, train=False, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}

    elif args.dataset == 'ImageNet':
        channel = 3
        im_size = (128, 128)
        num_classes = 10
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])

        dst_train = torchvision.datasets.ImageFolder('/public/MountData/dataset/{}/train'.format(args.subset), transform=transform)
        dst_test = torchvision.datasets.ImageFolder('/public/MountData/dataset/{}/val'.format(args.subset), transform=transform)
        class_map = {x: x for x in range(num_classes)}
        class_names = dst_train.classes
    else:
        print('Current dataset is not supported!')
        sys.exit()

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
