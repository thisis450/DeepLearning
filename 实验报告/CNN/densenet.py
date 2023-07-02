import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBasic(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBasic, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x):
        out = torch.cat((x, self.layer(x)), dim=1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        block = []
        for i in range(6):
            block.append(DenseBasic(in_channels, growth_rate))
            in_channels += growth_rate
        self.denseblock = nn.Sequential(*block)

    def forward(self, x):
        return self.denseblock(x)


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.denseblock1 = DenseBlock(32, 32)
        self.bn2 = nn.BatchNorm2d(224)
        self.conv1 = nn.Conv2d(224, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.denseblock2 = DenseBlock(64, 64)
        self.fc1 = nn.Linear(7168, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.denseblock1(x)
        x = self.bn2(x)
        x = self.conv1(x)
        x = self.avg1(x)
        x = self.denseblock2(x)
        batch_size, channels, w, h = x.shape
        x = x.reshape(batch_size, channels * w * h)
        x = self.fc1(x)
        return x
    
def densenet():
    return DenseNet()
