import torch
from torch import nn

"""
1.论文看不明白 
2.多个block组成一层不会写
3.先写出基本的block组成 然后组成层
思想就是拆分：一个resnet拆成层，层拆成block，block拆成conv3 conv1卷积
写的时候就倒着写： 1.先写conv1 conv3 2 再写block 里面就是用的conv1 conv3 3.写layer，
_make_layer层是定义在class resnet里面的一个成员函数
"""


def conv3(inchannel, outchannel):
    return nn.Conv2d(in_channels=inchannel,
                     out_channels=outchannel,
                     kernel_size=3,
                     stride=1,
                     padding=1)


class block(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(block, self).__init__()
        """
        这个block都不会写
        """
        self.conv1 = conv3(inchannel, outchannel)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3(outchannel, outchannel)
        self.bn2 = nn.BatchNorm2d(outchannel)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = nn.ReLU(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, num_class=1000):
        super(ResNet34, self).__init__()
        """
        这里定义卷积结构
        """
        layers = [3, 4, 6, 3]

        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # output_size=56*56
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        """
        下面是4个layer
        avgpool
        fc
        """
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * 7 * 7, num_class)

    def _make_layer(self, outchannel, blocks, stride):
        """
        这里为什么有下采样
        block就是一个函数
        还有一个问题是：不同层的卷积核个数不一样，也就是inchannel outchannel不一样
        :return:
        """
        self.in_channel = outchannel
        layers = [block(self.in_channel, outchannel)]

        for _ in range(1, blocks):
            """
            输出的_ 为 1， 2， ...blocks
            """
            layers.append(block(self.in_channel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


model = ResNet34()
print(model)
