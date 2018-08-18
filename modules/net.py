import torch.nn as nn


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
