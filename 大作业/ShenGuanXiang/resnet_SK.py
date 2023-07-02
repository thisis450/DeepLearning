import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_tools import *


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()

        d = max(int(out_channels / r), L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        self.fc = nn.Linear(out_channels, d)
        self.fcs = nn.ModuleList()
        for i in range(M):
            self.fcs.append(nn.Linear(d, out_channels))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        U = sum(conv(x) for conv in self.conv)
        s = self.fc(torch.mean(U.view(batch_size, self.out_channels, -1), dim=2))
        z = []
        for i in range(self.M):
            z.append(self.fcs[i](s).view(batch_size, self.out_channels, 1, 1))
        Uz = torch.cat(z, dim=2)
        weights = self.softmax(Uz)
        V = torch.sum(U * weights, dim=2)
        return V


class SKNet(nn.Module):
    def __init__(self, num_classes=100):
        super(SKNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.skconv1 = SKConv(64, 64, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.skconv2 = SKConv(128, 128, stride=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.skconv3 = SKConv(256, 256, stride=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.skconv4 = SKConv(512, 512, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.skconv1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.skconv2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.skconv3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.skconv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 创建SKNet模型实例
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SKNet().to(device)
    check_model_info(model)
