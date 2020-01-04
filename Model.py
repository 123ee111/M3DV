import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1):
        super(ResNet, self).__init__()
        self.inchannel = 4
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.layer1 = self.make_layer(ResidualBlock, 4,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 8, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 16, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.fc1 = nn.Linear(32, num_classes)
        self.fc2 = torch.sigmoid

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool3d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)
