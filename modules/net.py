import torch
import torch.nn as nn
from torchvision.models import densenet121


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
        self.fully_connected = nn.Linear(51200, 2)

    def forward(self, data):
        im1 = data[:, 0, :, :].unsqueeze_(1)
        out1 = self.dense_net(torch.cat((im1, im1, im1), 1))
        out1 = out1.view(out1.size(0), -1)

        im2 = data[:, 1, :, :].unsqueeze_(1)
        out2 = self.dense_net(torch.cat((im2, im2, im2), 1))
        out2 = out2.view(out2.size(0), -1)

        out = torch.cat((out1, out2), 1)
        return self.fully_connected(out)
