import json
import os
import pickle
import random
from typing import Tuple, List, Dict

import torch.utils.data
import torchvision.datasets.folder
from torchvision import transforms

from imagenet_class_index import IMAGENET_CLASS_INDEX
from settings import BATCH_SIZE, IMAGENET_PATH, IMAGENET1K_PATH, IMAGENET20P_SAMPLER_PATH
from matplotlib import pyplot


class SparseImageNet(torchvision.datasets.ImageFolder):

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        translator = {}
        for idx, (node_name, human_readable) in IMAGENET_CLASS_INDEX.items():
            translator[human_readable] = int(idx)
        classes, class_to_idx = super()._find_classes(dir)
        return classes, {k: translator[k] for k, v in class_to_idx.items()}


class FullImageNet(torchvision.datasets.ImageFolder):

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        translator = {}
        for idx, (node_name, human_readable) in IMAGENET_CLASS_INDEX.items():
            translator[node_name] = int(idx)
        classes, class_to_idx = super()._find_classes(dir)
        return classes, {k: translator[k] for k, v in class_to_idx.items()}


def get_image_net(sampler=None):
    data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = FullImageNet(IMAGENET1K_PATH, transform=data_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


def get_full_image_net(sampler=None, split='train'):
    data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = FullImageNet(os.path.join(IMAGENET_PATH, split), transform=data_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


def get_fashion_mnist(train=False):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456],
                             std=[0.225])
    ])
    data_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('../data', download=True,
                                                                                train=train,
                                                                                transform=data_transform),
                                              batch_size=BATCH_SIZE,
                                              shuffle=False, )
    return data_loader


def get_data_loader():
    return get_image_net()


def get_image_net_20p():
    return get_full_image_net(pickle.load(open(IMAGENET20P_SAMPLER_PATH, mode='rb')))


if __name__ == '__main__':
    print(len(get_image_net_20p()))
    for x, y in get_image_net_20p():
        pyplot.imshow(x[0].transpose(0, 2).transpose(0, 1))
        pyplot.show()
