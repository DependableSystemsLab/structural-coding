import pickle

import torch
import torchvision.datasets
from torch import nn as nn
from torchvision.models.resnet import _resnet

from linearcode.parameters import CONFIG

from common.models import MyBottleneck


class FashionMNISTTutorial(nn.Module):

    def __init__(self, pretrained=True, weights='fashion_mnist_tutorial.pkl'):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()
        if pretrained:
            checkpoint_file_path = weights
            with open(checkpoint_file_path, mode='rb') as checkpoint_file:
                self.load_state_dict(pickle.load(checkpoint_file))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


# training_data_loader = get_fashion_mnist()
#
# model = FashionMNISTTutorial()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
# loss = torch.nn.CrossEntropyLoss()
#
# # model.train()
# model.eval()
# for epoch in range(60):
#     total_l = 0
#     acc = []
#     model.zero_grad()
#     checkpoint_file_path = 'fashion_mnist_tutorial.pkl'
#     if os.path.exists(checkpoint_file_path):
#         with open(checkpoint_file_path, mode='rb') as checkpoint_file:
#             model.load_state_dict(pickle.load(checkpoint_file))
#
#     for x, y in training_data_loader:
#         model_output = model(x)
#         # l = loss(model_output, y)
#         # l.backward()
#         # total_l += float(l)
#         acc.append(float(torch.sum(torch.topk(model_output, k=1).indices.flatten() == y) / len(y)))
#     # optimizer.step()
#     # with open(checkpoint_file_path, mode='wb') as checkpoint_file:
#     #     pickle.dump(model.state_dict(), checkpoint_file)
#     print(total_l, sum(acc) / len(acc))
#     # break
#
#
#
#


def get_model():
    if CONFIG['model'] == 'resnet50':
        model = _resnet('resnet50', MyBottleneck, [3, 4, 6, 3], True, True)
    elif CONFIG['model'] == 'alexnet':
        model = torchvision.models.alexnet(True)
    elif CONFIG['model'] == 'FashionMNISTTutorial':
        model = FashionMNISTTutorial(pretrained=True)
    elif CONFIG['model'] == 'FashionMNISTTutorial_smooth':
        model = FashionMNISTTutorial(pretrained=True, weights='fashion_mnist_tutorial_smooth.pkl')
    else:
        assert False
    return model

