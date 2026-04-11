import torch
from torch.backends import mps
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt
from matplotlib.axes import Axes  # 导入类型用于声明
from typing import List, Optional, Union
import os

# 定义device自动选择函数
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# 定义函数来计算准确率
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 早停函数
def early_stopping(val_acc_history, patience):
    if len(val_acc_history) < patience:
        return False
    recent_accs = val_acc_history[-patience:]
    return all(acc <= recent_accs[0] for acc in recent_accs)

# 定义动画类用于绘图
class Animator:
    def __init__(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None, 
                 legend: Optional[List[str]] = None, xlim: Optional[List[float]] = None,
                 ylim: Optional[List[float]] = None, xscale: str = 'linear', 
                 yscale: str = 'linear', fmts: tuple = ('-', 'm--', 'g-.', 'r:'), 
                 nrows: int = 1, ncols: int = 1, figsize: tuple = (3.5, 2.5)):
        
        if legend is None:
            legend = []
            
        # 设置 SVG 显示
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats('svg')
        
        # 显式声明 fig 为 Figure，axes 为包含 Axes 对象的列表/数组
        self.fig, axes_ndarray = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        # 重点：强制转换为具体的 Axes 类型列表，让 Pylance 识别
        self.axes: List[Axes] = axes_ndarray.flatten().tolist()
        
        self.config_dict = {
            'xlabel': xlabel, 'ylabel': ylabel, 'xlim': xlim, 'ylim': ylim,
            'xscale': xscale, 'yscale': yscale, 'legend': legend
        }
        self.X, self.Y, self.fmts = None, None, fmts

    def _set_axes(self, ax: Axes):
        """内部函数：设置坐标轴属性"""
        ax.set_xlabel(self.config_dict['xlabel'])
        ax.set_ylabel(self.config_dict['ylabel'])
        ax.set_xscale(self.config_dict['xscale'])
        ax.set_yscale(self.config_dict['yscale'])
        ax.set_xlim(self.config_dict['xlim'])
        ax.set_ylim(self.config_dict['ylim'])
        if self.config_dict['legend']:
            ax.legend(self.config_dict['legend'])
        ax.grid()

    def add(self, x: Union[float, List[float]], y: Union[float, List[float]]):
        """向图表中添加多个数据点"""
        if not hasattr(y, "__len__"):
            y = [y]  # type: ignore
        n = len(y) # type: ignore
        if not hasattr(x, "__len__"):
            x = [x] * n # type: ignore
        
        if self.X is None:
            self.X = [[] for _ in range(n)] # type: ignore
        if self.Y is None:
            self.Y = [[] for _ in range(n)] # type: ignore
        
        for i, (a, b) in enumerate(zip(x, y)): # type: ignore
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        # 显式获取第一个 axes 对象
        ax: Axes = self.axes[0]
        
        ax.cla() # 现在 Pylance 知道 ax 是 Axes 类型，不会报 cla 未知
        for x_coords, y_coords, fmt in zip(self.X, self.Y, self.fmts):
            ax.plot(x_coords, y_coords, fmt) # 也不会报 plot 未知
        
        self._set_axes(ax)
        display.display(self.fig)
        display.clear_output(wait=True)