import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):

    def __init__(self, in_chnls, ratio=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_chnls, in_chnls//ratio)
        self.fc2 = nn.Linear(in_chnls//ratio, in_chnls)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return F.sigmoid(out)
    
class SEResNet(nn.Module):
    def __init__(self,block,layers,num_classes=100):
        super(SEResNet,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=self.make_layer(block,64,layers[0])
        self.layer2=self.make_layer(block,128,layers[1],2)
        self.layer3=self.make_layer(block,256,layers[2],2)
        self.layer4=self.make_layer(block,512,layers[3],2)
        self.avg_pool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
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
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=self.avg_pool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,ratio=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.se=SE(in_chnls=out_channels,ratio=ratio)

    def forward(self, x):
        identity = x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        seout=self.se(out)
        out=out*seout
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
def seresnet(num_classes=10):
    return SEResNet(BasicBlock,[2,2,2,2],num_classes=num_classes)