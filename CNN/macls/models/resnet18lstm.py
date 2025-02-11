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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.se = SELayer(planes) if use_se else None

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
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_class, input_size, dropout_rate=0.0, use_se=True):
        super(ResNet18, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False) #kernel_size=7
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dropout_rate=dropout_rate, use_se=use_se)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate=dropout_rate, use_se=use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # 输出形状为 [B, C, 1, 1]
        return x  # 保持 4D 输出


class ResNet18LSTM(nn.Module):
    def __init__(self, num_class, input_size, lstm_hidden_size=128, lstm_layers=2, dropout_rate=0.3, use_se=True):
        super(ResNet18LSTM, self).__init__()
        self.resnet = ResNet18(num_class=num_class, input_size=input_size, dropout_rate=dropout_rate, use_se=use_se)
        
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet 的输出通道数
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        self.fc = nn.Linear(lstm_hidden_size * 2, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_feature_maps=False):
        #print(f"Input tensor shape before reshape: {x.shape}")
        
        # 如果输入形状是 [B, T, F]，调整为 [B, 1, F, T]
        if len(x.shape) == 3:  # [B, T, F]
            x = x.unsqueeze(1)  # 添加通道维度 -> [B, 1, T, F]
            x = x.permute(0, 1, 3, 2)  # 调整为 [B, 1, F, T]

        # 确保输入形状为 [B, 1, H, W]
        #print(f"Input tensor shape after reshape: {x.shape}")
        freq_band_activations = x.mean(dim=3)  # 对时间维度求平均 [B, 1, F] -> [B, F]

        # 传入 ResNet
        x = self.resnet(x)

        # ResNet 输出调整为 LSTM 输入格式
        x = x.squeeze(-1).squeeze(-1)  # [B, C, 1, 1] -> [B, C]
        x = x.unsqueeze(1)  # [B, C] -> [B, 1, C]
        #print(f"LSTM input shape: {x.shape}")

        # LSTM 前向传播
        x, _ = self.lstm(x)
        feature_maps = x[:, -1, :]
        x = self.fc(x[:, -1, :])  # 取最后一个时间步
        x = self.sigmoid(x)
        if return_feature_maps:
            return x, freq_band_activations, feature_maps
        return x,freq_band_activations
