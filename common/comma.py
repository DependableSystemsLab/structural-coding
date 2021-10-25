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
from keras.models import model_from_json
import torch.nn as nn

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

    def __init__(self, pretrained=False):
        super().__init__()

        self.elu = nn.ELU()

        self.conv_0 = nn.Conv2d(3, 16, 8, 4, padding="same")
        self.conv_1 = nn.Conv2d(16, 32, 5, 2, padding="same")
        self.conv_2 = nn.Conv2d(32, 64, 5, 2, padding="same")  # 384 kernels, size 3x3

        self.fc0 = nn.Linear(12800, 512)
        self.fc1 = nn.Linear(512, 1)

        if pretrained:
            # weights = json.load(open('../../research/weights.json'))
            # self.conv_0.weight = nn.Parameter(torch.FloatTensor(weights[0]))
            # self.conv_0.bias = nn.Parameter(torch.FloatTensor(weights[1]))
            # self.conv_1.weight = nn.Parameter(torch.FloatTensor(weights[2]))
            # self.conv_1.bias = nn.Parameter(torch.FloatTensor(weights[3]))
            # self.conv_2.weight = nn.Parameter(torch.FloatTensor(weights[4]))
            # self.conv_2.bias = nn.Parameter(torch.FloatTensor(weights[5]))
            # self.fc0.weight = nn.Parameter(torch.FloatTensor(weights[6]).transpose(0, 1))
            # self.fc0.bias = nn.Parameter(torch.FloatTensor(weights[7]))
            # self.fc1.weight = nn.Parameter(torch.FloatTensor(weights[8]).transpose(0, 1))
            # self.fc1.bias = nn.Parameter(torch.FloatTensor(weights[9]))
            # for i, j in self.state_dict().items():
            #     print(i, j.shape)
            # torch.save(self.state_dict(), 'comma/steering_angle.pth')
            self.load_state_dict(torch.load('comma/steering_angle.pth'))

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
    Comma(pretrained=True)
