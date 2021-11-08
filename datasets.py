import csv
import os
import pickle
from typing import Tuple, List, Dict, Any, Optional, Callable
import numpy as np

import numpy.random
import torch.utils.data
import torchvision.datasets.folder
from matplotlib import pyplot
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from imagenet_class_index import IMAGENET_CLASS_INDEX
from settings import BATCH_SIZE, IMAGENET_PATH, IMAGENET1K_PATH, IMAGENET20P_SAMPLER_PATH, IMAGENET_ROOT


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


class ImageNetValidation(torchvision.datasets.VisionDataset):

    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        translator = {}
        for idx, (node_name, human_readable) in IMAGENET_CLASS_INDEX.items():
            translator[node_name] = int(idx)
        self.translator = translator
        with open(os.path.join(root, 'LOC_val_solution.csv')) as solution_file:
            csv_reader = csv.reader(solution_file)
            self.images = list(csv_reader)[1:]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image_basename, loc_info = self.images[index]
        target = self.translator[loc_info.split()[0]]
        path = os.path.join(self.root, 'ILSVRC', 'Data', 'CLS-LOC', 'val', image_basename + '.JPEG')
        sample = torchvision.datasets.folder.default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.images)


def get_image_net(sampler=None):
    data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SparseImageNet(IMAGENET1K_PATH, transform=data_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)


def get_full_image_net(sampler=None, split='train'):
    data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if split == 'train':
        dataset = FullImageNet(os.path.join(IMAGENET_PATH, split), transform=data_transform)
    elif split == 'val':
        dataset = ImageNetValidation(IMAGENET_ROOT, transform=data_transform)
    else:
        assert False
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


def get_dataset(config):
    rnd = numpy.random.RandomState(2021)
    if config['dataset'] == 'imagenet_ds':
        sampler = sorted(rnd.choice(range(50000), 10000, replace=False))
        return get_full_image_net(sampler, 'val')
    if config['dataset'] == 'imagenet_as_i':
        sampler = set(rnd.choice(range(50000), 10000, replace=False))
        injection_rnd = numpy.random.RandomState(config['injection'])
        image_id = injection_rnd.randint(0, 39999)
        absolute_image_id = [i for i in range(50000) if i not in sampler][image_id]
        return get_full_image_net([absolute_image_id], 'val')
    if config['dataset'] == 'imagenet_ds_128':
        sampler = sorted(rnd.choice(range(50000), 10000, replace=False)[:128])
        return get_full_image_net(sampler, 'val')
    if config['dataset'] == 'imagenet':
        sampler = None
        if config['sampler'] == 'tiny':
            sampler = sorted(rnd.choice(range(1000), BATCH_SIZE, replace=False))
        return get_image_net(sampler)
    if config['dataset'] == 'driving_dataset_test':
        transforms_composed = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.flip(0)),
            transforms.Lambda(lambda x: x[:, -150:, :]),
            transforms.Lambda(lambda x: x.transpose(1, 2)),
            transforms.Resize((200, 66)),
        ])
        return DataLoader(DrivingDataset('../data/sullychen/driving_dataset/', 'data.txt', False,
                                         transforms_composed), batch_size=BATCH_SIZE)
    assert False


class DrivingDataset(Dataset):
    """driving dataset."""

    def __init__(self, root_dir, txt_file, training=True, transform=None):
        """ Args:
                txt_file(string)
                root_file(string)
                training(boolean) : True for train and False for test
                transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform
        self.img_name = []
        self.steering_angle = []
        with open(os.path.join(root_dir, txt_file)) as f:
            for line in f:
                self.img_name.append(line.split()[0])
                # Converting steering angle which we need to predict from radians
                # to degrees for fast computation
                self.steering_angle.append(float(line.split()[1]) * np.pi / 180)
        # 80% for train
        # 20% for test
        if not training :
            self.img_name = self.img_name[:int(len(self.img_name) * 0.2)]
            self.steering_angle = self.steering_angle[:int(len(self.steering_angle) * 0.2)]
        else:
            self.img_name = self.img_name[-int(len(self.img_name) * 0.8) - 1:]
            self.steering_angle = self.steering_angle[-int(len(self.steering_angle) * 0.8) - 1:]

    def __len__(self):
        return len(self.steering_angle)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.img_name[idx])
        # image = cv2.imread(image_name)
        # image = image[-150:, :, :]
        # image = cv2.resize(image, (200, 66))
        image = torchvision.datasets.folder.default_loader(image_name)
        # new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            new_img = self.transform(image)

        return new_img, torch.tensor([self.steering_angle[idx]])


if __name__ == '__main__':
    for x, y in get_dataset({'dataset': 'imagenet_ds'}):
        pyplot.imshow(x[0].transpose(0, 2).transpose(0, 1))
        pyplot.show()
