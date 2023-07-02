import torch
import torch.nn as nn
import torch.nn.functional as F


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        super(SKConv, self).__init__()

        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels

        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, out_channels * M)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.empty((batch_size, self.out_channels, x.size(2), x.size(3)), dtype=x.dtype, device=x.device)

        for i, conv in enumerate(self.conv):
            output[:, i * self.out_channels:(i + 1) * self.out_channels, :, :] = conv(x)

        U = self.global_pool(output)
        U = U.view(batch_size, self.out_channels)
        attention_weight = self.fc(U).view(batch_size, self.M, self.out_channels)
        attention_weight = self.softmax(attention_weight)

        output = (output * attention_weight.unsqueeze(4)).sum(dim=1)
        return output


class SKNet(nn.Module):
    def __init__(self, num_classes):
        super(SKNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            SKConv(64, 64, stride=2),
            SKConv(64, 128, stride=1)
        )

        self.conv3 = nn.Sequential(
            SKConv(128, 128, stride=2),
            SKConv(128, 256, stride=1)
        )

        self.conv4 = nn.Sequential(
            SKConv(256, 256, stride=2),
            SKConv(256, 512, stride=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x