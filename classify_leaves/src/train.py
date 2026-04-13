import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import utils    
import torchvision.models as models
import os
from sklearn.model_selection import StratifiedKFold
import warnings

# 忽略sklearn的UserWarning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 数据增强
train_augs = torchvision.transforms.Compose([
    T.Resize(256, antialias=True), # type: ignore
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),
    T.RandomRotation(20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ConvertImageDtype(torch.float32), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 使用ImageNet的均值和标准差进行归一化
])
val_augs = torchvision.transforms.Compose([
    T.Resize(256, antialias=True), # type: ignore
    T.CenterCrop(224),  
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 使用ImageNet的均值和标准差进行归一化

])
# 超参数
batch_size = 128 # 批量大小
out_features = 176 # 输出类别数
lr = 0.001 # 学习率
epochs = 20 # 训练轮数
k = 3 # k折交叉验证
patience = 3 # early-stopping耐心值
num_workers = 4

# 获取设备
device = utils.get_device()

# 创建模型保存目录
model_dir = 'classify_leaves/model/'
os.makedirs(model_dir, exist_ok=True) 

# 加载数据集
img_dir = 'classify_leaves/data/'
dataset = utils.LeavesDataset(csv_file='train.csv', img_dir=img_dir, transform=None)
train_dataset = utils.LeavesDataset(csv_file='train.csv', img_dir=img_dir, transform=train_augs)
val_dataset= utils.LeavesDataset(csv_file='train.csv', img_dir=img_dir, transform=val_augs)

# k折交叉验证
skf = StratifiedKFold(n_splits=k, shuffle=True)
if __name__ == '__main__':
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), dataset.labels)):
        print(f'Fold {fold+1}/{k}')
        # 加载预训练的ResNet34模型
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # 替换全连接层
        model.fc = nn.Linear(model.fc.in_features, out_features)

        # 定义损失函数和优化器
        loss = nn.CrossEntropyLoss() # 交叉熵损失函数适用于多分类问题
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr) # 只优化需要更新的参数

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"训练集大小: {len(train_subset)}, 验证集大小: {len(val_subset)}")
        print(f'本次训练使用的设备: {device}')
        
        utils.train(model, train_loader, val_loader, loss, optimizer, device, epochs, patience, model_dir, fold)