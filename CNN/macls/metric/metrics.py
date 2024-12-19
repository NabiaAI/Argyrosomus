import numpy as np
import torch



def accuracy_imb(output, target):
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    output = (output > 0.5).astype(int)
    
    if output.shape != target.shape:
        raise ValueError(f"Shape mismatch: output shape {output.shape} and target shape {target.shape}")

    correct = np.sum(output == target, axis=1)
    acc = np.mean(correct == target.shape[1])
    return acc

