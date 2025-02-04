import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

        self.conv1 = nn.Conv2d(1, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False)  #  kernel_size=7
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1, dropout_rate=dropout_rate, use_se=use_se)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dropout_rate=dropout_rate, use_se=use_se)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))  # 保持时间维度不变，频率维度压缩为 1

        self.feature_maps = {}  # 用于存储中间层输出

        # 注册 hook
        self.register_hooks()

        

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate, use_se):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate=dropout_rate, use_se=use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def register_hooks(self):
        """为每一层注册 forward hook"""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook

        self.conv1.register_forward_hook(hook_fn('conv1'))
        self.layer1.register_forward_hook(hook_fn('layer1'))
        self.layer2.register_forward_hook(hook_fn('layer2'))
        self.layer3.register_forward_hook(hook_fn('layer3'))
        self.layer4.register_forward_hook(hook_fn('layer4'))

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)  # [B, C, T', F'] -> [B, C, T', 1]
        x = x.squeeze(-1)  # [B, C, T', 1] -> [B, C, T']

       

        return x  # 保留特征图

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # 假设输入 x 的形状为 [B, T, H]
        attn_weights = torch.softmax(x.sum(dim=-1), dim=-1)  # 形状为 [B, T]

        if mask is not None:
            mask = mask.squeeze(-1)  # 变成 [B, T]
            mask = mask[:, :attn_weights.size(1)]  # 截断到 attn_weights 的时间步长度
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.unsqueeze(1)  # [B, 1, T]
        return torch.bmm(attn_weights, x).squeeze(1)  # [B, H]

class ResNet18LSTMASK(nn.Module):
    def __init__(self, num_class, input_size, lstm_hidden_size=128, lstm_layers=2, dropout_rate=0.3, use_se=True):
        super(ResNet18LSTMASK, self).__init__()
        self.resnet = ResNet18(num_class=num_class, input_size=input_size, dropout_rate=dropout_rate, use_se=use_se)

        # 修改 input_size 为 ResNet 输出通道数 512
        self.lstm = nn.LSTM(
            input_size=512,  # ResNet 的输出特征维度
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        self.attention = Attention(lstm_hidden_size * 2)  # 加入 Attention
        self.fc = nn.Linear(lstm_hidden_size * 2, num_class)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)  # 默认 mask

        x = x.unsqueeze(1)  # 输入形状 [B, T, F] -> [B, 1, T, F]
        x = x.permute(0, 1, 3, 2)  # [B, 1, T, F] -> [B, 1, F, T]

        # 提取频带特征 (保持频谱维度)
        freq_band_activations = x.mean(dim=3)  # 对时间维度求平均 [B, 1, F] -> [B, F]
        
        # 后续网络处理
        x = self.resnet(x)
        x = x.squeeze(-2)  # [B, C, T', F'] -> [B, C, T']
        x = x.permute(0, 2, 1)  # [B, C, T'] -> [B, T', C]
        x, _ = self.lstm(x)
        x = self.attention(x, mask=mask[:, :, 0])
        x = self.fc(x)
        x = self.sigmoid(x)


        return x, freq_band_activations



    # 使用MultiLabelFocalLossWithMask
    # def forward(self, x, mask=None):
    #     if mask is None:
    #         # 默认全有效的 mask，形状为 [Batch, Time, 1]
    #         mask = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=torch.bool)

    #     # 数据维度转换
    #     x = x.unsqueeze(1).permute(0, 1, 3, 2)  # [B, T, F] -> [B, 1, F, T]
    #     # 提取频带特征 (保持频谱维度)

    #     # ResNet 提取特征
    #     x = self.resnet(x).squeeze(-2).permute(0, 2, 1)  # [B, C, T', F'] -> [B, T', C]

    #     # 修正 mask 的形状并应用于特征
    #     mask = mask[:, :x.size(1)].expand(-1, -1, x.size(-1))  # [B, T, 1] -> [B, T, C]
    #     x = x * mask  # 屏蔽无效时间步

    #     # LSTM 和 Attention 模块
    #     x, _ = self.lstm(x)  # [B, T', H]
    #     x = self.attention(x, mask=mask[:, :, 0])  # 注意力权重只需时间维度 [B, T]

    #     # 输出层
    #     x = self.sigmoid(self.fc(x))  # [B, Num_Classes]
    #     return x




    # 使用MultiLabelFocalLoss用的
    # def forward(self, x, mask=None):
    #     if mask is None:
    #         mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)
            

    #     x = x.unsqueeze(1)  # [B, T, F] -> [B, 1, T, F]
    #     # 如果需要交换 T 和 F，则进行以下 permute 操作
    #     x = x.permute(0, 1, 3, 2)  # [B, 1, T, F] -> [B, 1, F, T]


    #     x = self.resnet(x)  # ResNet 输出 [B, C, T', F']
    #     x = x.squeeze(-2)  # [B, C, T', F'] -> [B, C, T']
    #     x = x.permute(0, 2, 1)  # [B, C, T'] -> [B, T', C]

    #     x, _ = self.lstm(x)  # [B, T', H]
    #     x = self.attention(x, mask=mask)  # [B, H]
    #     x = self.fc(x)
    #     x = self.sigmoid(x)
    #     return x


