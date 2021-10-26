"""
  *  @copyright (c) 2020 Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *  @file    network_model.py
  *  @author  Charan Karthikeyan P V, Nagireddi Jagadesh Nischal
  *
  *  @brief Class declaration for building the network model.
 """
import json
import os

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms


from settings import COMMA_MODEL_ROOT

"""
* @brief Class declaration for building the training network model.
* @param None.
* @return The built netwwork.
"""


class Comma(nn.Module):
    """
    * @brief Initializes the class varaibles
    * @param None.
    * @return None.
    """
    def __init__(self):
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv_0 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=5, stride=2) #384 kernels, size 3x3
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=3) # 384 kernels size 3x3
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3) # 256 kernels, size 3x3

        self.fc0 = nn.Linear(1152, 100)
        self.fc1 = nn.Linear(100,50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
    """ 
    * @brief Function to build the model.
    * @parma The image to train.
    * @return The trained prediction network.
    """
    def forward(self, x):
        x = x / 127.5 - 1.0
        x = self.elu(self.conv_0(x))
        x = self.elu(self.conv_1(x))
        x = self.elu(self.conv_2(x))
        x = self.elu(self.conv_3(x))
        x = self.elu(self.conv_4(x))
        x = self.dropout(x)

        x = x.flatten()
        x = self.elu(self.fc0(x))
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((320, 160)),
        transforms.ToTensor()
        ])
    sample = torch.stack((transform(torchvision.datasets.folder.default_loader('../../Ranger/demo/sample_inputs/1000.jpg')),))
    print(sample.shape)
    model = Comma(pretrained=True)
    print(model(sample))
