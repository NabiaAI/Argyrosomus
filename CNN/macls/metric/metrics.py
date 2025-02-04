import numpy as np
import torch


# 计算准确率
def accuracy(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc





def accuracy_imb(output, target):
    """计算准确率"""
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # 将 output 转为二进制标签
    output = (output > 0.5).astype(int)
    
    # 检查 output 和 target 的形状是否一致
    if output.shape != target.shape:
        raise ValueError(f"Shape mismatch: output shape {output.shape} and target shape {target.shape}")

    correct = np.sum(output == target, axis=1)
    acc = np.mean(correct == target.shape[1])
    return acc

