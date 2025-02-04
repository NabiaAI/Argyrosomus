import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_class,input_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 使用自适应平均池化层
        
        # 计算展平后的大小
        self.flatten_size = 32  # 由于我们将特征图大小调整为 (1, 1)，因此展平后的大小仅与通道数有关
        
        self.fc1 = nn.Linear(self.flatten_size, 128)  # 自动计算尺寸
        self.fc2 = nn.Linear(128, num_class)

    
    def forward(self, x):
        x = x.transpose(2, 1)  # 调整维度为 (batch_size, height, width)
        x = x.unsqueeze(1)  # 增加通道维度，使其成为 (batch_size, 1, height, width)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)  # 自适应平均池化
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == '__main__':
    # 创建一个随机输入
    # 手动设置 Graphviz 可执行文件的路径
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    model = SimpleCNN(4,32)
    x = torch.randn(32, 51,32)

    # 生成计算图
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))

    # 保存图像
    # 打印当前工作目录
    # 指定保存路径
    save_path = os.getcwd()

    # 如果目录不存在，则创建它
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(os.getcwd())
    dot.format = 'png'
    dot.render('network_structure')