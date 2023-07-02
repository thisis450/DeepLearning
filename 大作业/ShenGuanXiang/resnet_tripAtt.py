import sys
sys.path.append("..")
from basic_tools import *
from triplet_attention import *


class BasicBlock_TripAtt(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_TripAtt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.triplet_attention = TripletAttention(out_channels, 16)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.triplet_attention(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100, expansion=1):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.in_channels = 64*expansion
        self.conv1 = nn.Conv2d(3, 64 * expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64 * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64 * expansion, layers[0])
        self.layer2 = self.make_layer(block, 128 * expansion, layers[1])
        self.layer3 = self.make_layer(block, 256 * expansion, layers[2], 2)
        self.layer4 = self.make_layer(block, 512 * expansion, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet_TripAtt(depth, num_classes):
    assert depth in [18, 34, 50, 101], "network depth should be 18, 34, 50 or 101"

    if depth == 18:
        model = ResNet(BasicBlock_TripAtt, [2, 2, 2, 2], num_classes)

    elif depth == 34:
        model = ResNet(BasicBlock_TripAtt, [3, 4, 6, 3], num_classes)

    elif depth == 50:
        model = ResNet(BasicBlock_TripAtt, [3, 4, 6, 3], num_classes, 4)

    elif depth == 101:
        model = ResNet(BasicBlock_TripAtt, [3, 4, 23, 3], num_classes, 4)

    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet_TripAtt(18, 100).to(device)
    check_model_info(model)

