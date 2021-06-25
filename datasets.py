import json
from typing import Tuple, List, Dict

import torch.utils.data
import torchvision.datasets.folder
from torchvision import transforms

from pruning.settings import BATCH_SIZE


class SparseImageNet(torchvision.datasets.ImageFolder):

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        with open('../data/imagenet_class_index.json') as mapping_file:
            mapping = json.load(mapping_file)
        translator = {}
        for idx, (node_name, human_readable) in mapping.items():
            translator[human_readable] = int(idx)
        classes, class_to_idx = super()._find_classes(dir)
        return classes, {k: translator[k] for k, v in class_to_idx.items()}


def get_image_net():
    data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SparseImageNet('../data/random20classes_FI', transform=data_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, )


def get_fashion_mnist():
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456],
                             std=[0.225])
    ])
    data_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('.data', download=True,
                                                                                train=False,
                                                                                transform=data_transform),
                                              batch_size=BATCH_SIZE,
                                              shuffle=False, )
    return data_loader


def get_data_loader():
    return get_image_net()


if __name__ == '__main__':
    get_image_net()
