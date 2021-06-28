import os
import pickle

import torch
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader

import injection
from datasets import get_fashion_mnist
from injection import convert


class FashionMNISTTutorial(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()
        if pretrained:
            checkpoint_file_path = 'fashion_mnist_tutorial.pkl'
            with open(checkpoint_file_path, mode='rb') as checkpoint_file:
                self.load_state_dict(pickle.load(checkpoint_file))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
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
