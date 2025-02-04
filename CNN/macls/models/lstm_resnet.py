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
    def __init__(self, input_channels, dropout_rate=0.0, use_se=True):
        super(ResNet18, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dropout_rate=dropout_rate, use_se=use_se)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # 压缩时间和频率维度

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate=dropout_rate, use_se=use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        freq_band_activations = x.mean(dim=3)  # [B, C, H, W] -> [B, C, H]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)  # [B, C, H, W] -> [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C, 1, 1] -> [B, C]
        return x, freq_band_activations


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(x.sum(dim=-1), dim=-1)  # [B, T]
        attn_weights = attn_weights.unsqueeze(1)  # [B, 1, T]
        return torch.bmm(attn_weights, x).squeeze(1)  # [B, H]


class LSTMResNetWithAttention(nn.Module):
    def __init__(self, num_class, input_size=16, lstm_hidden_size=128, lstm_layers=2, dropout_rate=0.3, use_se=True):
        super(LSTMResNetWithAttention, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        self.attention = Attention(lstm_hidden_size * 2)
        self.resnet = ResNet18(input_channels=1, dropout_rate=dropout_rate, use_se=use_se)
        self.fc = nn.Linear(512, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        # LSTM 处理

        x, _ = self.lstm(x)  # [B, T, H]
        x = x.permute(0, 2, 1).unsqueeze(1)  # [B, T, H] -> [B, 1, H, T]

        # ResNet 处理
        x, freq_band_activations = self.resnet(x)  # [B, 512]

        # 分类输出
        x = self.fc(x)
        x = self.sigmoid(x)
        # 调整 freq_band_activations 的维度
        freq_band_activations = freq_band_activations.squeeze(1)  # 先移除无关维度，变成 [B, 256]



        # 插值到目标频率维度
        freq_band_activations = F.interpolate(
            freq_band_activations.unsqueeze(1),  # [B, 256] -> [B, 1, 256]
            size=129,  # 插值到目标维度
            mode="linear",
            align_corners=False
        ).squeeze(1)  # [B, 1, 129] -> [B, 129]
                

        return x#, freq_band_activations
