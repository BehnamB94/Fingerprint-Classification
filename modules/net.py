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


class CnnRnn(nn.Module):
    def __init__(self):
        super(CnnRnn, self).__init__()

        # parameters
        self.hidden_size = 256
        self.num_layers = 1
        bidirectional = True

        self.directions = 2 if bidirectional else 1
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(48, 192, kernel_size=5, stride=1, groups=2),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=5, stride=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.lstm = nn.LSTM(128, self.hidden_size, self.num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(self.hidden_size * self.directions, 5)

    def forward(self, data):
        out = self.convolution(data)
        h0 = torch.autograd.Variable(torch.zeros(self.num_layers * self.directions, data.size(0), self.hidden_size))
        c0 = torch.autograd.Variable(torch.zeros(self.num_layers * self.directions, data.size(0), self.hidden_size))
        out = out.view(out.size(0), 128, -1)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out, (h0, c0))
        return self.fc(out[:, -1, :])


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
