import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 全局平均池化并压缩维度
        y = self.fc(y).view(b, c, 1, 1)  # 两层全连接后变回 (B, C, 1, 1)
        return x * y.expand_as(x)  # 权重缩放到原始特征图
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.se = SELayer(planes) if use_se else None  # 添加SE模块

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.se:
            out = self.se(out)  # 添加SE模块
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet18(nn.Module):
    def __init__(self, num_class, input_size, dropout_rate=0.0, use_se=True):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dropout_rate=dropout_rate, use_se=use_se)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_class)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate=dropout_rate, use_se=use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(2, 1).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
if __name__ == '__main__':
    model = ResNet18(num_class=4, input_size=32, dropout_rate=0.3, use_se=True)
    x = torch.randn(32, 51, 32)
    y = model(x)
    # 打印模型的所有层
    for name, module in model.named_modules():
        print(name)

