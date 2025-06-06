import math

import torch
import torch.nn as nn

from macls.models.pooling import AttentiveStatsPool, TemporalAveragePooling
from macls.models.pooling import SelfAttentivePooling, TemporalStatisticsPooling
from torchviz import make_dot
import os


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class Res2Netmul(nn.Module):

#     def __init__(self, num_class, input_size, m_channels=32, layers=[3, 4, 6, 3], base_width=32, scale=2, embd_dim=192,
#                  pooling_type="ASP"):
#         super(Res2Netmul, self).__init__()
#         self.inplanes = m_channels
#         self.base_width = base_width
#         self.scale = scale
#         self.embd_dim = embd_dim
#         self.conv1 = nn.Conv2d(1, m_channels, kernel_size=7, stride=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(m_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(Bottle2neck, m_channels, layers[0])
#         self.layer2 = self._make_layer(Bottle2neck, m_channels * 2, layers[1], stride=2)
#         self.layer3 = self._make_layer(Bottle2neck, m_channels * 4, layers[2], stride=2)
#         self.layer4 = self._make_layer(Bottle2neck, m_channels * 8, layers[3], stride=2)

#         cat_channels = m_channels * 8 * Bottle2neck.expansion * (input_size // base_width)
#         if pooling_type == "ASP":
#             self.pooling = AttentiveStatsPool(cat_channels, 128)
#             self.bn2 = nn.BatchNorm1d(cat_channels * 2)
#             self.linear = nn.Linear(cat_channels * 2, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         elif pooling_type == "SAP":
#             self.pooling = SelfAttentivePooling(cat_channels, 128)
#             self.bn2 = nn.BatchNorm1d(cat_channels)
#             self.linear = nn.Linear(cat_channels, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         elif pooling_type == "TAP":
#             self.pooling = TemporalAveragePooling()
#             self.bn2 = nn.BatchNorm1d(cat_channels)
#             self.linear = nn.Linear(cat_channels, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         elif pooling_type == "TSP":
#             self.pooling = TemporalStatisticsPooling()
#             self.bn2 = nn.BatchNorm1d(cat_channels * 2)
#             self.linear = nn.Linear(cat_channels * 2, embd_dim)
#             self.bn3 = nn.BatchNorm1d(embd_dim)
#         else:
#             raise Exception(f'没有{pooling_type}池化层！')

#         self.fc = nn.Linear(embd_dim, num_class)
#         self.sigmoid = nn.Sigmoid()  # For multi-label classification

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = [block(self.inplanes, planes, stride, downsample=downsample,
#                         stype='stage', baseWidth=self.base_width, scale=self.scale)]
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.transpose(2, 1)
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.max_pool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = x.reshape(x.shape[0], -1, x.shape[-1])

#         x = self.pooling(x)
#         x = self.bn2(x)
#         x = self.linear(x)
#         x = self.bn3(x)

#         x = self.fc(x)
#         out = self.sigmoid(x)
#         return out
class Res2Netmul(nn.Module):

    def __init__(self, num_class, feature_dim, layers=[3, 4, 6, 3], base_width=32, scale=2, embd_dim=192, pooling_type="ASP"):
        super(Res2Netmul, self).__init__()
        
        # 动态设置 m_channels 为 feature_dim (即 n_mels)
        self.m_channels = feature_dim
        self.inplanes = self.m_channels
        self.base_width = base_width
        self.scale = scale
        self.embd_dim = embd_dim

        # 使用动态的 m_channels 作为初始通道数
        self.conv1 = nn.Conv2d(1, self.m_channels, kernel_size=7, stride=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.m_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用 m_channels 的倍数来设置后续层的通道数
        self.layer1 = self._make_layer(Bottle2neck, self.m_channels, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, self.m_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, self.m_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, self.m_channels * 8, layers[3], stride=2)

        # 动态计算池化层的输入通道数
        cat_channels = self.m_channels * 8 * Bottle2neck.expansion * (feature_dim // base_width)
        
        # 配置池化层和后续层
        if pooling_type == "ASP":
            self.pooling = AttentiveStatsPool(cat_channels, 128)
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "SAP":
            self.pooling = SelfAttentivePooling(cat_channels, 128)
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TAP":
            self.pooling = TemporalAveragePooling()
            self.bn2 = nn.BatchNorm1d(cat_channels)
            self.linear = nn.Linear(cat_channels, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        elif pooling_type == "TSP":
            self.pooling = TemporalStatisticsPooling()
            self.bn2 = nn.BatchNorm1d(cat_channels * 2)
            self.linear = nn.Linear(cat_channels * 2, embd_dim)
            self.bn3 = nn.BatchNorm1d(embd_dim)
        else:
            raise Exception(f'没有{pooling_type}池化层！')

        self.fc = nn.Linear(embd_dim, num_class)
        self.sigmoid = nn.Sigmoid()  # For multi-label classification

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample,
                        stype='stage', baseWidth=self.base_width, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.base_width, scale=self.scale))

        return nn.Sequential(*layers)



    def forward(self, x):
        # 在进入卷积层之前检查输入形状

        # 调整输入形状，将频率维度调整为通道数
        x = x.transpose(2, 1)  # 将 n_mels 维度移动到最后
        x = x.unsqueeze(1)  # 增加一个通道维度
        
        # 再次检查形状以确认是否正确

        # 执行卷积和批归一化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = self.pooling(x)
        x = self.bn2(x)
        x = self.linear(x)
        x = self.bn3(x)

        x = self.fc(x)
        out = self.sigmoid(x)
        return out


# if __name__ == '__main__':
#     # 创建一个随机输入
#     # 手动设置 Graphviz 可执行文件的路径
#     os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

#     model = Res2Netmul(4,32)
#     x = torch.randn(32, 51,32)

#     # 生成计算图
#     y = model(x)
#     dot = make_dot(y, params=dict(model.named_parameters()))

#     # 保存图像
#     # 打印当前工作目录
#     # 指定保存路径
#     save_path = os.getcwd()

#     # 如果目录不存在，则创建它
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     print(os.getcwd())
#     dot.format = 'png'
#     dot.render('network_structure')