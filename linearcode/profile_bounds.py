import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import common.models
from datasets import get_image_net_20p, DrivingDataset
from injection import ClipperReLU, convert, ClipperHardswish, ClipperELU
from settings import BATCH_SIZE, BASE_DIRECTORY

if __name__ == '__main__':

    _, model, device = sys.argv

    for model_name, model_class in common.models.MODEL_CLASSES:
        if model_name != model:
            continue
        model = model_class(pretrained=True)
        model.eval()
        model, max_injection_index = convert(model, mapping={
            torch.nn.ReLU: ClipperReLU,
            torch.nn.Hardswish: ClipperHardswish,
            torch.nn.ELU: ClipperELU,
        })
        for m in model.modules():
            if isinstance(m, ClipperReLU) or isinstance(m, ClipperHardswish) or isinstance(m, ClipperELU):
                m.train()
                m.profile = True
        model.to(device)
        if model_name != 'e2e':
            data_loader = get_image_net_20p()
        else:
            transforms_composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.flip(0)),
                transforms.Lambda(lambda x: x[:, -150:, :]),
                transforms.Lambda(lambda x: x.transpose(1, 2)),
                transforms.Resize((200, 66)),
            ])
            data_loader = DataLoader(DrivingDataset('../data/sullychen/driving_dataset/', 'data.txt', True,
                                                    transforms_composed), batch_size=BATCH_SIZE)
        percentage = len(data_loader) // 100
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            model(x)
            if i % percentage == 0:
                with open(os.path.join(BASE_DIRECTORY, 'linearcode', 'bounds', model_name) + '.txt', mode='w') as output_file:
                    for m in model.modules():
                        if hasattr(m, 'bounds'):
                            print(m.bounds, file=output_file)
                print(i // percentage, '%')
