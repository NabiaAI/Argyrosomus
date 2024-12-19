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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
        self.flatten_size = 32      
        self.fc1 = nn.Linear(self.flatten_size, 128) 
        self.fc2 = nn.Linear(128, num_class)

    
    def forward(self, x):
        x = x.transpose(2, 1) 
        x = x.unsqueeze(1)  
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)  
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
