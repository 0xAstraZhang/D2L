import torch
from torch.backends import mps

# 定义对数均方差损失函数
def RMSLE(y_pred, y_true):
    log_pred = torch.log1p(y_pred)
    log_true = torch.log1p(y_true)
    return torch.sqrt(torch.mean((log_pred - log_true) ** 2))

# 定义device自动选择函数
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
    