import torch
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/HaojieYuu/pytorch_dave2/blob/master/model.py
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from datasets import DrivingDataset
from settings import BATCH_SIZE


class Dave2(nn.Module):
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

        self.fc0 = nn.Linear(1152, 100)
        # self.fc0 = nn.Linear(1280, 100)
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


if __name__ == "__main__":
    from dask_generator import datagen
    import argparse

    # Parameters
    parser = argparse.ArgumentParser(description='MiniBatch server')
    parser.add_argument('--batch', dest='batch', type=int, default=256, help='Batch size')
    parser.add_argument('--device', dest='device', type=str, default='cuda:0', help="Device to use")
    parser.add_argument('--after', dest='after', type=int, default=49, help="Evaluate after how many epochs")
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

    device = args.device
    transforms_composed = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = DataLoader(DrivingDataset('../data/sullychen/driving_dataset/', 'data.txt', True, transforms_composed), batch_size=BATCH_SIZE)
    val = DataLoader(DrivingDataset('../data/sullychen/driving_dataset/', 'data.txt', False, transforms_composed), batch_size=BATCH_SIZE)
    model = Dave2(pretrained=False).to(device)
    if args.validation:
        model.load_state_dict(torch.load('dave_{}.pth'.format(args.after))['model'])
        model.eval()
    else:
        model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 1000
    percentage = len(train) // 100
    for epoch in range(epochs):
        if not args.validation:
            for i, (x, y) in enumerate(train):
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                model_output = model(x)
                loss = criterion(model_output, y)
                loss.backward()
                optimizer.step()
                if i % percentage == 0:
                    print(i * 100 // len(train), '%')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, 'comma_fast_{}.pth'.format(epoch))
        total_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val):
                x = x.to(device)
                y = y.to(device)
                model_output = model(x)
                total_loss += float(criterion(model_output, y))

        print("Report loss", total_loss)
