import torch
import torch.nn as nn
from torchvision.models import densenet121, alexnet


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 128, kernel_size=5, stride=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 192, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 128, 2096),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(2096, 256),
            nn.ReLU(),
            nn.Dropout(p=.5),

            nn.Linear(256, 5),
        )

    def forward(self, data):
        out = self.convolution(data)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class TrainedDenseNet(nn.Module):  # 224
    def __init__(self):
        super(TrainedDenseNet, self).__init__()
        self.dense_net = nn.Sequential(*list(densenet121(pretrained=True).children())[:-1])
        self.fully_connected = nn.Linear(50176, 5)

    def forward(self, data):
        out = self.dense_net(torch.cat((data, data, data), 1))
        out = out.view(out.size(0), -1)
        return self.fully_connected(out)


class TrainedAlexnet(nn.Module):  # 227
    def __init__(self):
        super(TrainedAlexnet, self).__init__()
        self.alex = nn.Sequential(*list(alexnet(pretrained=True).children())[:-1])
        self.fully_connected = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 5),
        )

    def forward(self, data):
        out = self.alex(torch.cat((data, data, data), 1))
        out = out.view(out.size(0), -1)
        return self.fully_connected(out)
