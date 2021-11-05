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
from torch import optim
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
    def __init__(self, pretrained=True):
        super().__init__()

        self.elu = nn.ELU()
        self.dropout = nn.Dropout()

        self.conv_0 = nn.Conv2d(3, 24, (5, 5), stride=(2, 2))
        self.conv_1 = nn.Conv2d(24, 36, kernel_size=(5, 5), stride=(2, 2))
        self.conv_2 = nn.Conv2d(36, 48, kernel_size=(5, 5), stride=(2, 2))  # 384 kernels, size 3x3
        self.conv_3 = nn.Conv2d(48, 64, kernel_size=(3, 3))  # 384 kernels size 3x3
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=(3, 3))  # 256 kernels, size 3x3

        # self.fc0 = nn.Linear(1152, 100)
        self.fc0 = nn.Linear(1280, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 1)
        if pretrained:
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

        x = x.flatten(start_dim=1)
        x = self.elu(self.fc0(x))
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    if __name__ == "__main__":
        from dask_generator import datagen
        import argparse

        # Parameters
        parser = argparse.ArgumentParser(description='MiniBatch server')
        parser.add_argument('--batch', dest='batch', type=int, default=256, help='Batch size')
        parser.add_argument('--device', dest='device', type=str, default='cuda:0', help="Device to use")
        parser.add_argument('--after', dest='after', type=int, default=1, help="Evaluate after how many epochs")
        parser.add_argument('--time', dest='time', type=int, default=1, help='Number of frames per sample')
        parser.add_argument('--port', dest='port', type=int, default=5557, help='Port of the ZMQ server')
        parser.add_argument('--buffer', dest='buffer', type=int, default=20,
                            help='High-water mark. Increasing this increses buffer and memory usage.')
        parser.add_argument('--prep', dest='prep', action='store_true', default=False,
                            help='Use images preprocessed by vision model.')
        parser.add_argument('--leads', dest='leads', action='store_true', default=False,
                            help='Use x, y and speed radar lead info.')
        parser.add_argument('--nogood', dest='nogood', action='store_true', default=False,
                            help='Ignore `goods` filters.')
        parser.add_argument('--validation', dest='validation', action='store_true', default=False,
                            help='Serve validation dataset instead.')
        args, more = parser.parse_known_args()

        # 9 for training
        train_path = [
            '../data/comma/comma-dataset/camera/2016-01-30--11-24-51.h5',
            '../data/comma/comma-dataset/camera/2016-01-30--13-46-00.h5',
            '../data/comma/comma-dataset/camera/2016-01-31--19-19-25.h5',
            '../data/comma/comma-dataset/camera/2016-02-02--10-16-58.h5',
            '../data/comma/comma-dataset/camera/2016-02-08--14-56-28.h5',
            '../data/comma/comma-dataset/camera/2016-02-11--21-32-47.h5',
            '../data/comma/comma-dataset/camera/2016-03-29--10-50-20.h5',
            '../data/comma/comma-dataset/camera/2016-04-21--14-48-08.h5',
            '../data/comma/comma-dataset/camera/2016-05-12--22-20-00.h5',
        ]

        # 2 for validation
        validation_path = [
            '../data/comma/comma-dataset/camera/2016-06-02--21-39-29.h5',
            '../data/comma/comma-dataset/camera/2016-06-08--11-46-01.h5'
        ]

        if args.validation:
            datapath = validation_path
        else:
            datapath = train_path
        device = args.device
        gen = datagen(datapath, time_len=args.time, batch_size=args.batch, ignore_goods=args.nogood)
        model = Comma(pretrained=False).to(device)
        if args.validation:
            model.load_state_dict(torch.load('comma_{}.pth'.format(args.after))['model'])
            model.eval()
        else:
            model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        epochs = 1000
        batch_per_epoch = 100000
        batch_per_epoch = 100
        loss = 0
        for epoch in range(epochs):  # runs for the number of eposchs set in the arguments
            for i, (x, y, _) in enumerate(gen):
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                x = x/127.5 - 1
                x = transforms.Resize(70)(x)
                model_output = model(x)
                if args.validation:
                    loss += float(criterion(model_output, y))
                    pass
                else:
                    loss = criterion(model_output, y)
                    loss.backward()
                    optimizer.step()
                if i == batch_per_epoch:
                    break
            if not args.validation:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, 'comma_{}.pth'.format(epoch))
            else:
                print("Validation loss", loss)
